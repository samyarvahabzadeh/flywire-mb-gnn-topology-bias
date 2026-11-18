"""Ordered real-graph phase-2 pilots. This program cannot execute a sweep."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

from mbtopology import gnn_training, runner
from mbtopology.batched import precompute_features
from mbtopology.benchmark_models import make_benchmark_model
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
from mbtopology.runtime import Exclusive, Run, atomic_json, fingerprint, read_json, require, sha256

ROOT = Path(__file__).resolve().parents[1]
CAP = 500
ALPHA = 0.001
PAIRS_PER_UPDATE = 4
ROSTER = (("GCN", None), ("GraphSAGE", "mean"), ("GraphSAGE", "max"))


def cue_batches(task, seed, batch_pairs):
    config = deepcopy(task)
    config["task"]["trials_max"] = CAP * batch_pairs
    pairs = paired_schedule(config, seed)
    return pairs.reshape(CAP, 2 * batch_pairs)


def label(architecture, aggregation):
    return architecture if aggregation is None else f"{architecture}-{aggregation}"


def init_model(model_cfg, arch, aggregation, seed):
    return make_benchmark_model(arch, model_cfg, seed, "cuda", aggregation=aggregation or "mean")


def describe_curve(history, criterion):
    values = np.asarray([row["discrimination"] for row in history], dtype=np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), "invalid phase-2 curve")
    return {
        "trials": len(history),
        "criterion_trial": criterion,
        "passed": criterion is not None,
        "maximum": float(values.max()),
        "last50_mean": float(values[-50:].mean()),
        "last_discrimination": float(values[-1]),
        "tail100_slope_per_trial": float(
            np.polyfit(np.arange(min(100, len(values))), values[-100:], 1)[0]
        )
        if len(values) > 1
        else None,
        "max_criterion_streak": max(row["criterion_streak"] for row in history),
        "window_means_25": [float(values[i : i + 25].mean()) for i in range(0, len(values), 25)],
    }


def train_batched(model, optimizer, data, task, table, directory, signature, progress, deadline):
    readout = task["readout"]
    features = precompute_features(data)
    y = features.new_zeros(task["task"]["n_odors"])
    y[task["task"]["cs_plus"]] = 1
    history, reached, streak = [], None, 0
    saved = load_training_checkpoint(directory, signature)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history, reached, streak = (saved[key] for key in ("history", "reached_at", "streak"))
    require(len(history) <= CAP, "invalid batch-pilot cursor")

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
            },
        )
        progress.save(trial=len(history), cue_presentations=len(history) * table.shape[1])

    last_save = monotonic()
    in_trial = False
    checkpoint()
    try:
        while len(history) < CAP and reached is None:
            if monotonic() >= deadline:
                raise SessionYield("phase-2 pilot checkpointed at session deadline")
            trial = len(history) + 1
            cue = table[trial - 1]
            in_trial = True
            model.train()
            optimizer.zero_grad(set_to_none=True)
            cue_ids = torch.as_tensor(cue, device=features.device)
            outputs = model(features.index_select(0, cue_ids), data["graph"])[:, data["mbon"]]
            avoid = outputs[:, data["avoidance"]].mean(dim=1)
            approach = outputs[:, data["approach"]].mean(dim=1)
            loss = F.binary_cross_entropy(((avoid - approach + 1) / 2).clamp(0, 1), y[cue_ids])
            require(bool(torch.isfinite(loss)), "nonfinite phase-2 BCE")
            loss.backward()
            require(
                all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                ),
                "phase-2 nonfinite gradient",
            )
            optimizer.step()
            score = gnn_training.probe(model, data, task)
            streak = streak + 1 if score >= readout["criterion"] else 0
            if streak >= readout["consecutive_trials"]:
                reached = trial
            history.append(
                {
                    "trial": trial,
                    "cue_presentations": trial * table.shape[1],
                    "trial_cues": cue.tolist(),
                    "training_loss": float(loss.detach().cpu()),
                    "discrimination": score,
                    "criterion_streak": streak,
                }
            )
            in_trial = False
            if monotonic() - last_save >= task["checkpoint_seconds"]:
                checkpoint()
                last_save = monotonic()
    finally:
        if not in_trial:
            checkpoint()
    return history, reached


@torch.no_grad()
def mbon_matrix(model, data):
    model.eval().requires_grad_(False)
    rows = model(precompute_features(data), data["graph"])[:, data["mbon"]]
    return rows.cpu().numpy().astype(np.float64)


def train_ridge(features, task, table):
    """Refit over accumulated cue repeats, reusing the exact weighted dual ridge solution."""
    plus, minus = task["task"]["cs_plus"], task["task"]["cs_minus"]
    readout = task["readout"]
    target = np.zeros(features.shape[0])
    target[plus] = 1
    offset = features.mean(axis=0)
    centered = features - offset
    gram = centered @ centered.T
    n_cues = target.size
    identity = np.eye(n_cues)
    history, reached, streak = [], None, 0
    for trial, cues in enumerate(table, start=1):
        require(
            np.array_equal(np.sort(cues), np.arange(n_cues)), "Ridge needs identical cue counts"
        )
        coef = centered.T @ np.linalg.solve(
            gram + identity * (ALPHA / trial), target - target.mean()
        )
        intercept = float(target.mean() - offset @ coef)
        fitted = features @ coef + intercept
        clipped = np.clip(fitted, 0, 1)
        score = float(0.5 * (clipped[plus].mean() + (1 - clipped[minus]).mean()))
        streak = streak + 1 if score > readout["criterion"] else 0
        history.append(
            {
                "trial": trial,
                "cue_presentations": trial * n_cues,
                "discrimination": score,
                "criterion_streak": streak,
            }
        )
        if streak >= readout["consecutive_trials"]:
            reached = trial
            break
    residual = fitted - target
    fit = {
        "alpha": ALPHA,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r_squared": float(1 - np.sum(residual**2) / np.sum((target - target.mean()) ** 2)),
        "intercept": intercept,
        "coefficient_l2": float(np.linalg.norm(coef)),
        "fitted_per_cue": fitted.tolist(),
        "feature_columns": features.shape[1],
    }
    return history, reached, fit


def read_result(path, signature, bundle_hash):
    if cached_result(path, signature, bundle_hash):
        return read_json(path.with_suffix(".manifest.json"))["phase2_run"]
    return None


def save_run(path, history, criterion, job, signature, evidence, commit, started, extra=None):
    require(criterion == len(history) or len(history) == CAP, "incomplete phase-2 history")
    report = {
        **job,
        **describe_curve(history, criterion),
        **(extra or {}),
        "cue_presentations": history[-1]["cue_presentations"],
        "wall_seconds": monotonic() - started,
        "config_sha256": signature,
    }
    write_result(
        ROOT,
        path,
        pl.DataFrame(history),
        {
            "scope": "EXPLORATORY_PHASE2_PILOT_ONLY",
            "git_commit": commit,
            "run_config_sha256": signature,
            "bundle_sha256": job["bundle_sha256"],
            "phase2_run": report,
            "registration": evidence,
        },
    )
    return report


def print_run(report):
    print("PHASE2_RESULT", json.dumps(report), flush=True)


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from mbtopology.runtime import provenance

    evidence = require_registered(ROOT)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "need the Molab GPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    task, cfg = bundle["task"], bundle["models"]
    require(
        cfg["optimizer"]["name"] == "Adam"
        and cfg["optimizer"]["learning_rate"] == 0.001
        and cfg["dropout"] == 0
        and task["readout"]["criterion"] == 0.8
        and task["readout"]["consecutive_trials"] == 20
        and not set(PILOT_SEEDS) & set(bundle["seeds"]),
        "frozen phase-2 pilot constraints changed",
    )
    graph = task["source_manifest"]
    provenance_record = provenance(ROOT, {"scope": "phase2", "seeds": list(PILOT_SEEDS)})
    commit = provenance_record["git_commit"]
    common = {
        "bundle_sha256": bundle["bundle_sha256"],
        "script_sha256": sha256(__file__),
        "real_graph_manifest_sha256": sha256(ROOT / graph),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    prefix = fingerprint(common)[:12]
    out = ROOT / "results" / "phase2_pilot" / prefix
    ckpt = ROOT / "checkpoints" / "phase2_pilot" / prefix
    deadline = monotonic() + bundle["experiment"]["session_hours"] * 3600
    report = {"a": [], "b": [], "c": []}
    frozen = {}

    def pilot_data(seed):
        return gnn_training.frozen_inputs(
            ROOT, task, graph, seed, bundle["nulls"], cfg, "cuda", pilot=True
        )

    def train_gradient(phase, seed, arch, aggr, batch_pairs, data, progress):
        table = cue_batches(task, seed, batch_pairs)
        name = f"{phase}-{label(arch, aggr)}-seed{seed}-pairs{batch_pairs}"
        job = {
            **common,
            "stage": phase,
            "seed": seed,
            "model": label(arch, aggr),
            "pairs_per_trial": batch_pairs,
            "table_sha256": fingerprint(table.tolist()),
        }
        sig = fingerprint(job)
        artifact = out / f"{name}.parquet"
        directory = ckpt / name
        if (cached := read_result(artifact, sig, bundle["bundle_sha256"])) is not None:
            return cached, directory, sig
        print("PHASE2_START", name, flush=True)
        started = monotonic()
        model = init_model(cfg, arch, aggr, seed)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=cfg["optimizer"]["weight_decay"]
        )
        history, reached = train_batched(
            model, optimizer, data, task, table, directory, sig, progress, deadline
        )
        return (
            save_run(artifact, history, reached, job, sig, evidence, commit, started),
            directory,
            sig,
        )

    def successful(rows):
        core = [r for r in rows if r["model"] in ("GCN", "GraphSAGE-mean")]
        return len(core) == 4 and all(r["passed"] for r in core)

    with Exclusive(ROOT), Run(ROOT, bundle["experiment"], "phase2-pilot") as progress:
        for seed in PILOT_SEEDS:
            data = pilot_data(seed)
            for arch, aggr in ROSTER:
                run, directory, signature = train_gradient("a", seed, arch, aggr, 4, data, progress)
                frozen[seed, arch, aggr] = (directory, signature)
                report["a"].append(run)
                print_run(run)
        if not successful(report["a"]):
            for seed in PILOT_SEEDS:
                data = pilot_data(seed)
                table = cue_batches(task, seed, PAIRS_PER_UPDATE)
                for arch, aggr in ROSTER:
                    name = label(arch, aggr)
                    directory, signature = frozen[seed, arch, aggr]
                    body_hash = read_json(directory / "checkpoint.json")["sha256"]
                    job = {
                        **common,
                        "stage": "b",
                        "seed": seed,
                        "model": name,
                        "pairs_per_trial": PAIRS_PER_UPDATE,
                        "table_sha256": fingerprint(table.tolist()),
                        "body_checkpoint_sha256": body_hash,
                        "ridge_alpha": ALPHA,
                    }
                    sig = fingerprint(job)
                    artifact = out / f"b-{name}-seed{seed}.parquet"
                    result = read_result(artifact, sig, bundle["bundle_sha256"])
                    if result is None:
                        require(monotonic() < deadline, "resume phase-2 pilot in a new session")
                        started = monotonic()
                        model = init_model(cfg, arch, aggr, seed)
                        state = load_training_checkpoint(directory, signature)
                        model.load_state_dict(state["model"])
                        matrix = mbon_matrix(model, data)
                        history, reached, fit = train_ridge(matrix, task, table)
                        result = save_run(
                            artifact,
                            history,
                            reached,
                            job,
                            sig,
                            evidence,
                            commit,
                            started,
                            {"ridge_final_fit": fit},
                        )
                    report["b"].append(result)
                    print_run(result)
        winner = "a" if successful(report["a"]) else ("b" if successful(report["b"]) else None)
        if winner is None:
            baseline = next(r for r in report["a"] if r["model"] == "GCN" and r["seed"] == 1337)
            report["c"].append({**baseline, "stage": "c", "reused_stage_a": True})
            data = pilot_data(1337)
            for n in (8, 16):
                run, _, _ = train_gradient("c", 1337, "GCN", None, n, data, progress)
                report["c"].append(run)
                print_run(run)
        summary = {
            **report,
            **common,
            "status": "COMPLETE",
            "scope": "EXPLORATORY_PHASE2_PILOT_ONLY",
            "winner": winner,
            "git_commit": commit,
            "config_sha256": fingerprint({"roster": ROSTER, "cap": CAP, "alpha": ALPHA}),
        }
        atomic_json(out / "summary.manifest.json", summary)
    print("PHASE2_COMPLETE", prefix, winner, flush=True)


if __name__ == "__main__":
    main()
