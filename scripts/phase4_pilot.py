"""Run the real-graph A+/A- pilot on Molab, using only the two pilot seeds."""

import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl
import torch
import yaml
from torch.nn import functional as F

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.experiment_io import (
    SessionYield,
    cached_result,
    load_training_checkpoint,
    restore_rng,
    save_training_checkpoint,
    write_result,
)
from mbtopology.paired_pilot import PILOT_SEEDS, learning_trend
from mbtopology.real_cues import binary_cues, check_projected_disjoint, cue_audit, project_cues
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
from mbtopology.task_inputs import load_source

ROOT = Path(__file__).resolve().parents[1]


def load_real_cues(task, config, seed):
    _, nodes, graph = load_source(ROOT, task["source_manifest"])
    roles = nodes["role"].to_numpy()
    pn = np.flatnonzero(roles == "PN")
    kc = np.flatnonzero(roles == "KC")
    pn_ids = nodes["root_id"].to_numpy()[pn]
    adjacency = graph[pn][:, kc].tocsr()
    adjacency.sort_indices()
    patterns, cores, pool = binary_cues(
        adjacency, seed, config["trials"], config["pairs_per_trial"], config["family_design"]
    )
    drives = []
    audit = {"trial": list(range(1, config["trials"] + 1))}
    for label, cues in zip(("train", "heldout"), patterns, strict=True):
        drive, degree, weight = project_cues(adjacency, cues)
        drives.append(drive)
        audit.update(cue_audit(cues, pn_ids, degree, weight, label))
    check_projected_disjoint(*drives)
    metadata = {
        "seed": seed,
        "pn_total": len(pn),
        "kc_total": len(kc),
        "pn_with_kc_edges": int(np.count_nonzero(adjacency.getnnz(axis=1))),
        "kc_with_pn_edges": int(np.count_nonzero(adjacency.getnnz(axis=0))),
        "pn_kc_edges": adjacency.nnz,
        "pn_kc_synapses": int(adjacency.sum()),
        "core_pn_root_ids_including_shared": pn_ids[cores].tolist(),
        "variable_pn_root_ids": pn_ids[pool].tolist(),
        "presentation_order": ["A+", "A-"] * config["pairs_per_trial"],
        "exact_train_heldout_pn_overlap": 0,
        "exact_train_heldout_kc_drive_overlap": 0,
        "kc_drive": "sum_active_real_pn_synapse_count / sum_all_real_pn_synapse_count",
    }
    return drives, pl.DataFrame(audit), metadata


def valences(model, inputs, drive, trial, presentations):
    low, high = drive.indptr[[trial * presentations, (trial + 1) * presentations]]
    row_counts = np.diff(drive.indptr[trial * presentations : (trial + 1) * presentations + 1])
    device = inputs["roles"].device
    row = torch.from_numpy(np.repeat(np.arange(presentations), row_counts)).to(device)
    col = torch.from_numpy(drive.indices[low:high]).to(device)
    activity = torch.from_numpy(drive.data[low:high]).to(device)
    features = inputs["base"].expand(presentations, -1, -1).clone()
    features[row, inputs["kc"][col], 4] = activity
    mbon = model(features, inputs["graph"])[:, inputs["mbon"]]
    return (
        (mbon[:, inputs["avoidance"]].mean(1) - mbon[:, inputs["approach"]].mean(1) + 1) / 2
    ).clamp(0, 1)


def train(model, optimizer, inputs, drives, config, checkpoint_dir, signature, progress, deadline):
    history = []
    elapsed = 0.0
    saved = load_training_checkpoint(checkpoint_dir, signature)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, elapsed = saved["history"], saved["elapsed_seconds"]
        require(
            saved["cursor"] == len(history) <= config["trials"], "invalid phase-4 checkpoint cursor"
        )
    started = last_save = monotonic()
    in_trial = False
    presentations = config["pairs_per_trial"] * 2
    labels = torch.tensor([1.0, 0.0] * config["pairs_per_trial"], device=inputs["roles"].device)

    def checkpoint():
        save_training_checkpoint(
            checkpoint_dir,
            signature,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "cursor": len(history),
                "elapsed_seconds": elapsed + monotonic() - started,
            },
        )
        progress.save(trial=len(history))

    checkpoint()
    try:
        for t in range(len(history), config["trials"]):
            if monotonic() >= deadline:
                raise SessionYield("phase-4 pilot checkpointed at session deadline")
            in_trial = True
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy(
                valences(model, inputs, drives[0], t, presentations), labels
            )
            require(bool(torch.isfinite(loss)), "nonfinite phase-4 BCE")
            loss.backward()
            require(
                all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                ),
                "nonfinite phase-4 gradient",
            )
            optimizer.step()
            model.eval()
            with torch.no_grad():
                probe = valences(model, inputs, drives[1], t, presentations).double()
                a_plus, a_minus = float(probe[::2].mean()), float(probe[1::2].mean())
                score = (a_plus + 1 - a_minus) / 2
                require(np.isfinite(score), "nonfinite phase-4 held-out score")
            history.append(
                {
                    "trial": t + 1,
                    "discrimination": score,
                    "heldout_a_plus": a_plus,
                    "heldout_a_minus": a_minus,
                }
            )
            in_trial = False
            if (t + 1) % 500 == 0:
                print("PHASE4_PROGRESS", checkpoint_dir.name, t + 1, score, flush=True)
            if monotonic() - last_save >= config["checkpoint_seconds"]:
                checkpoint()
                last_save = monotonic()
    finally:
        if not in_trial:
            checkpoint()
    return history, elapsed + monotonic() - started


def score_run(history, threshold):
    trend = learning_trend(history)
    first = next((row["trial"] for row in history if row["discrimination"] >= threshold), None)
    return {
        **trend,
        "first_threshold_trial": first,
        "passed": bool(first is not None and trend["rising"]),
    }


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    evidence = require_registered(ROOT)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "need one Molab GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    task, model_config = bundle["task"], bundle["models"]
    config_path = ROOT / "configs/phase4_pilot.yaml"
    config = yaml.safe_load(config_path.read_text())
    require(
        config["scope"] == "EXPLORATORY_REAL_PN_BINARY_PILOT_ONLY"
        and tuple(config["pilot_seeds"]) == PILOT_SEEDS
        and tuple(config["models"]) == ("GCN", "GraphSAGE-max", "GraphSAGE-mean")
        and config["trials"] == 2000
        and config["threshold"] == 0.75
        and config["pairs_per_trial"] == 4
        and 0 < config["checkpoint_seconds"] <= 900
        and not set(PILOT_SEEDS) & set(bundle["seeds"]),
        "invalid phase-4 pilot constraints",
    )
    trace = provenance(ROOT, config)
    common = {
        "scope": config["scope"],
        "bundle_sha256": bundle["bundle_sha256"],
        "config_file_sha256": sha256(config_path),
        "script_sha256": sha256(__file__),
        "graph_manifest_sha256": sha256(ROOT / task["source_manifest"]),
        "source_sha256": trace["source_sha256"],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    run_id = fingerprint(common)[:12]
    output = ROOT / "results" / "phase4_pilot" / run_id
    checkpoints = ROOT / "checkpoints" / "phase4_pilot" / run_id
    deadline = monotonic() + 3600 * bundle["experiment"]["session_hours"]
    print("PHASE4_START", run_id, trace["git_commit"], trace["config_sha256"], flush=True)
    results = []
    with Exclusive(ROOT), Run(ROOT, config, "phase4-pilot") as progress:
        for seed in PILOT_SEEDS:
            inputs = gnn_training.frozen_inputs(
                ROOT,
                task,
                task["source_manifest"],
                seed,
                bundle["nulls"],
                model_config,
                "cuda",
                pilot=True,
            )
            inputs["base"] = models.node_features(inputs["roles"], inputs["kc"], inputs["kc"][:0])
            drives, cue_frame, cue_info = load_real_cues(task, config, seed)
            cue_path = output / f"cues-seed{seed}.parquet"
            cue_signature = fingerprint({**common, "seed": seed, "artifact": "cue_audit"})
            cue_manifest = write_result(
                ROOT,
                cue_path,
                cue_frame,
                {**trace, **common, "run_config_sha256": cue_signature, "cue_protocol": cue_info},
            )
            for name in config["models"]:
                path = output / f"{name}-real-seed{seed}.parquet"
                job = {
                    **common,
                    "seed": seed,
                    "model": name,
                    "cue_parquet_sha256": cue_manifest["parquet_sha256"],
                }
                signature = fingerprint(job)
                if cached_result(path, signature, job["bundle_sha256"]):
                    results.append(read_json(path.with_suffix(".manifest.json"))["phase4_run"])
                    continue
                progress.save(job=path.stem)
                architecture, _, aggregation = name.partition("-")
                model = make_benchmark_model(
                    architecture, model_config, seed, "cuda", aggregation=aggregation or "mean"
                )
                opt = model_config["optimizer"]
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"]
                )
                history, elapsed = train(
                    model,
                    optimizer,
                    inputs,
                    drives,
                    config,
                    checkpoints / path.stem,
                    signature,
                    progress,
                    deadline,
                )
                result = {
                    "seed": seed,
                    "model": name,
                    "trials": len(history),
                    "elapsed_seconds": elapsed,
                    **score_run(history, config["threshold"]),
                }
                write_result(
                    ROOT,
                    path,
                    pl.DataFrame(history),
                    {
                        **trace,
                        **job,
                        "run_config_sha256": signature,
                        "phase4_run": result,
                        "registration": evidence,
                    },
                )
                results.append(result)
                print("PHASE4_RESULT", json.dumps(result), flush=True)
        by_model = {
            name: all(r["passed"] for r in results if r["model"] == name)
            for name in config["models"]
        }
        summary = {
            **trace,
            **common,
            "status": "COMPLETE",
            "all_runs_passed": all(by_model.values()),
            "both_seeds_pass_by_model": by_model,
            "runs": results,
        }
        atomic_json(output / "summary.manifest.json", summary)
    print("PHASE4_COMPLETE", run_id, json.dumps(by_model), flush=True)


if __name__ == "__main__":
    main()
