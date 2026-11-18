"""Run only the approved real-graph pilot with unseen cue instances."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl
import torch
import yaml
from torch.nn import functional as F

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.cue_families import choose_cap, make_cues, overlap_statistics, second_order
from mbtopology.experiment_io import (
    SessionYield,
    cached_result,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
    write_result,
)
from mbtopology.paired_pilot import PILOT_SEEDS, paired_schedule
from mbtopology.registration import require_registered
from mbtopology.runtime import (
    Exclusive,
    Run,
    atomic_json,
    fingerprint,
    provenance,
    read_json,
    require,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def load_pilot_data(bundle, config, seed):
    task, model = bundle["task"], bundle["models"]
    counts = config["family_design"]
    data = gnn_training.frozen_inputs(
        ROOT, task, task["source_manifest"], seed, bundle["nulls"], model, "cuda", pilot=True
    )
    pairs = task["task"]["odor_pairs"]
    trials = config["exploration_trials"]
    pn_count = int((data["roles"] == 1).sum())
    pn, kc, pool = make_cues(counts, seed, pn_count, int(data["kc"].numel()), trials, pairs)
    ordered_config = deepcopy(task)
    ordered_config["task"]["trials_max"] = trials * len(pairs)
    order = paired_schedule(ordered_config, seed).reshape(trials, -1)
    report = {
        "layout_config": counts,
        "seed": seed,
        "n_pn": pn_count,
        "n_kc": int(data["kc"].numel()),
        "active_pns_per_cue": pn[0].shape[-1],
        "active_kcs_per_cue": kc[0].shape[-1],
        "train_cues": trials * len(pairs) * 2,
        "held_out_cues": trials * len(pairs) * 2,
        "exact_pn_overlap_train_heldout": 0,
        "exact_kc_overlap_train_heldout": 0,
        "overlap_statistics_first_blocks": min(256, trials),
        "pn_overlap": overlap_statistics(pn[0][:256], pn[1][:256], pairs),
        "kc_overlap": overlap_statistics(kc[0][:256], kc[1][:256], pairs),
        "pn_second_order": second_order(counts, pool),
        "pn_train_heldout_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in pn],
        "kc_train_heldout_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in kc],
        "cue_order_sha256": fingerprint(order.tolist()),
    }
    train = np.take_along_axis(kc[0], order[:, :, None], axis=1)
    data["training"] = torch.tensor(train, device="cuda", dtype=torch.int32)
    data["heldout"] = torch.tensor(kc[1], device="cuda", dtype=torch.int32)
    data["labels"] = torch.tensor(
        np.isin(order, task["task"]["cs_plus"]), device="cuda", dtype=torch.float32
    )
    data["base"] = models.node_features(data["roles"], data["kc"], data["kc"][:0])
    return data, report


def values(model, data, active):
    features = data["base"].expand(active.shape[0], -1, -1).clone()
    features[
        torch.arange(active.shape[0], device=active.device)[:, None], data["kc"][active.long()], 4
    ] = 1
    output = model(features, data["graph"])[:, data["mbon"]]
    return (
        (output[:, data["avoidance"]].mean(1) - output[:, data["approach"]].mean(1) + 1) / 2
    ).clamp(0, 1)


def train(model, optimizer, data, task, cap, explore, directory, signature, progress, deadline):
    threshold, consecutive = task["readout"]["criterion"], task["readout"]["consecutive_trials"]
    plus, minus = task["task"]["cs_plus"], task["task"]["cs_minus"]
    history, reached, streak, elapsed = [], None, 0, 0.0
    saved = load_training_checkpoint(directory, signature)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, reached, streak, elapsed = (
            saved[k] for k in ("history", "reached_at", "streak", "elapsed_seconds")
        )
    require(len(history) <= cap, "phase-3 checkpoint exceeds cap")
    started = last_save = monotonic()
    in_trial = False

    def checkpoint():
        save_training_checkpoint(
            directory,
            signature,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "cursor": len(history),
                "reached_at": reached,
                "streak": streak,
                "elapsed_seconds": elapsed + monotonic() - started,
            },
        )
        progress.save(trial=len(history))

    checkpoint()
    try:
        while len(history) < cap and (explore or reached is None):
            if monotonic() >= deadline:
                raise SessionYield("phase-3 pilot checkpointed at session deadline")
            t = len(history)
            in_trial = True
            optimizer.zero_grad(set_to_none=True)
            model.train()
            train_values = values(model, data, data["training"][t])
            loss = F.binary_cross_entropy(train_values, data["labels"][t])
            require(bool(torch.isfinite(loss)), "nonfinite phase-3 BCE")
            loss.backward()
            require(
                all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                ),
                "nonfinite phase-3 gradient",
            )
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probe = values(model, data, data["heldout"][t]).double()
                score = float(0.5 * (probe[plus].mean() + 1 - probe[minus].mean()))
                require(np.isfinite(score), "nonfinite held-out score")
            streak = streak + 1 if score >= threshold else 0
            if reached is None and streak >= consecutive:
                reached = t + 1
            history.append(
                {
                    "trial": t + 1,
                    "train_presentations": (t + 1) * 8,
                    "held_out_presentations": (t + 1) * 8,
                    "train_bce": float(loss.detach()),
                    "discrimination": score,
                    "criterion_streak": streak,
                }
            )
            in_trial = False
            if (t + 1) % 500 == 0:
                print("PHASE3_PROGRESS", directory.name, t + 1, score, reached, flush=True)
            if monotonic() - last_save >= task["checkpoint_seconds"]:
                checkpoint()
                last_save = monotonic()
    finally:
        if not in_trial:
            checkpoint()
    return history, reached, elapsed + monotonic() - started


def eligibility(runs, agg, cap, config, min_ttc):
    group = [r for r in runs if r["model"] in ("GCN", f"GraphSAGE-{agg}")]
    if cap is None:
        return {"pass": False, "reason": "no_data_driven_cap"}
    ttcs = [(r["seed"], r["model"], r["criterion_trial"]) for r in group]
    if len(group) != 4 or any(x[2] is None or not min_ttc < x[2] < cap for x in ttcs):
        return {"pass": False, "reason": "miss_at_cap_or_at_floor", "ttcs": ttcs}
    median = float(np.median([x[2] for x in ttcs]))
    distinct_seeds = any(a[0] != b[0] and a[1] == b[1] and a[2] != b[2] for a in ttcs for b in ttcs)
    low, high = config["cap_selection"]["target_median_ttc"]
    return {
        "pass": distinct_seeds and low <= median <= high,
        "ttcs": ttcs,
        "median": median,
        "seed_variation": distinct_seeds,
    }


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    evidence = require_registered(ROOT)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "need one Molab GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    task, model_cfg = bundle["task"], bundle["models"]
    cfg_path = ROOT / "configs/phase3_pilot.yaml"
    config = yaml.safe_load(cfg_path.read_text())
    require(
        config["scope"] == "EXPLORATORY_HELD_OUT_PILOT_ONLY"
        and tuple(config["pilot_seeds"]) == PILOT_SEEDS
        and config["exploration_trials"] == 5000
        and not set(PILOT_SEEDS) & set(bundle["seeds"])
        and model_cfg["optimizer"]["learning_rate"] == 0.001
        and task["readout"]["criterion"] == 0.8
        and task["readout"]["consecutive_trials"] == 20,
        "phase-3 frozen pilot constraints differ",
    )
    common = {
        "bundle_sha256": bundle["bundle_sha256"],
        "config_file_sha256": sha256(cfg_path),
        "script_sha256": sha256(__file__),
        "graph_manifest_sha256": sha256(ROOT / task["source_manifest"]),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    prefix = fingerprint(common)[:12]
    output = ROOT / "results" / "phase3_pilot" / prefix
    ckpt = ROOT / "checkpoints" / "phase3_pilot" / prefix
    trace = provenance(ROOT, config)
    deadline = monotonic() + bundle["experiment"]["session_hours"] * 3600

    def run_one(data, seed, architecture, aggregation, cap, explore, progress):
        model_name = f"{architecture}-{aggregation}" if aggregation else architecture
        name = f"{model_name}-real-seed{seed}"
        job = {
            **common,
            "seed": seed,
            "model": model_name,
            "cap": cap,
            "explore_through_cap": explore,
            "cue_family_config_sha256": trace["config_sha256"],
        }
        signature = fingerprint(job)
        path = output / f"{name}.parquet"
        if cached_result(path, signature, bundle["bundle_sha256"]):
            report = read_json(path.with_suffix(".manifest.json"))["phase3_run"]
            return report, pl.read_parquet(path)["discrimination"].to_numpy()
        print("PHASE3_START", prefix, name, cap, flush=True)
        progress.save(job=name)
        model = make_benchmark_model(
            architecture, model_cfg, seed, "cuda", aggregation=aggregation or "mean"
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=model_cfg["optimizer"]["weight_decay"]
        )
        history, reached, elapsed = train(
            model, optimizer, data, task, cap, explore, ckpt / name, signature, progress, deadline
        )
        require(len(history) == cap or reached == len(history), "incomplete phase-3 pilot job")
        frame = pl.DataFrame(history)
        scores = frame["discrimination"].to_numpy()
        windows = [float(scores[i : i + 100].mean()) for i in range(0, len(scores), 100)]
        report = {
            **job,
            "config_sha256": signature,
            "trials": len(scores),
            "criterion_trial": reached,
            "maximum": float(scores.max()),
            "last100_mean": float(scores[-100:].mean()),
            "elapsed_seconds": elapsed,
            "windows100": windows,
        }
        write_result(
            ROOT,
            path,
            frame,
            {
                **trace,
                **job,
                "run_config_sha256": signature,
                "phase3_run": report,
                "scope": config["scope"],
                "registration": evidence,
            },
        )
        print("PHASE3_RESULT", json.dumps(report), flush=True)
        return report, scores

    with Exclusive(ROOT), Run(ROOT, bundle["experiment"], "phase3-pilot") as progress:
        data, families1337 = load_pilot_data(bundle, config, 1337)
        atomic_json(output / "families-seed1337.json", families1337)
        exploration, scores = run_one(data, 1337, "GCN", None, 5000, True, progress)
        chosen_cap, stable = choose_cap(
            scores,
            exploration["criterion_trial"],
            task["readout"]["criterion"],
            config["cap_selection"],
        )
        print("PHASE3_CAP", chosen_cap, stable, flush=True)
        cap = chosen_cap or 5000
        runs = [exploration]
        for seed in PILOT_SEEDS:
            if seed != 1337:
                data, families2026 = load_pilot_data(bundle, config, seed)
                atomic_json(output / "families-seed2026.json", families2026)
                runs.append(run_one(data, seed, "GCN", None, cap, False, progress)[0])
            for aggregation in ("max", "mean"):
                runs.append(run_one(data, seed, "GraphSAGE", aggregation, cap, False, progress)[0])
        decisions = {agg: eligibility(runs, agg, chosen_cap, config, 20) for agg in ("mean", "max")}
        passes = [agg for agg, outcome in decisions.items() if outcome["pass"]]
        selected = (
            min(passes, key=lambda a: (decisions[a]["median"], a != "mean")) if passes else None
        )
        summary = {
            **common,
            "scope": config["scope"],
            "git_commit": trace["git_commit"],
            "config_sha256": trace["config_sha256"],
            "status": "COMPLETE",
            "selected_cap": chosen_cap,
            "stability_window_end": stable,
            "diagnostic_cap_if_not_selected": 5000,
            "eligibility": decisions,
            "aggregator": selected,
            "passed": selected is not None,
            "runs": runs,
        }
        atomic_json(output / "summary.manifest.json", summary)
    print("PHASE3_COMPLETE", prefix, selected, flush=True)


if __name__ == "__main__":
    main()
