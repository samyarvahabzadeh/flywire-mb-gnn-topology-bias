"""One exploratory CUDA benchmark, with exactly 200 trials per job at seed 1337."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
BENCH_TRIALS = 200


def roster(k):
    return [
        (architecture, "real" if index == 0 else ("A" if index % 2 else "B"), index)
        for architecture in ("GCN", "GAT")
        for index in range(k // 2)
    ]


def prepare_graphs(bundle, wave, progress, deadline):
    from mbtopology.runner import materialize_null

    return {
        variant: materialize_null(
            ROOT, bundle, variant, 1337 + wave, progress, deadline, scope="batch_pilot_nulls"
        )
        for variant in ("A", "B")
    }


def run_wave(bundle, graphs, k, wave, started, deadline, evidence):
    import torch

    from mbtopology import batched, gnn_training, models
    from mbtopology.runtime import atomic_json, fingerprint, provenance, read_json, require, sha256

    output = ROOT / "results" / "molab_batch_rate" / bundle["bundle_sha256"][:12]
    report_path = output / f"K{k}.manifest.json"
    task, model_cfg = bundle["task"], bundle["models"]
    config = {"k": k, "seed": batched.PILOT_SEED, "trials": BENCH_TRIALS, "wave": wave}
    if (previous := read_json(report_path)) is not None:
        require(
            previous["status"] == "PASS"
            and previous["bundle_sha256"] == bundle["bundle_sha256"]
            and previous["config_sha256"] == fingerprint(config)
            and previous["completed_jobs"] == k
            and previous["total_completed_trials"] == k * BENCH_TRIALS,
            f"existing K={k} benchmark manifest differs",
        )
        print("BATCH_RESULT", json.dumps(previous), flush=True)
        return
    meta = provenance(ROOT, config)
    graph_paths = {"real": task["source_manifest"], **graphs}
    inputs = {
        graph: gnn_training.frozen_inputs(
            ROOT, task, path, batched.PILOT_SEED, bundle["nulls"], model_cfg, "cuda", pilot=True
        )
        for graph, path in graph_paths.items()
    }
    features = {graph: batched.precompute_features(data) for graph, data in inputs.items()}
    jobs = []
    for architecture, graph, index in roster(k):
        spec = {
            **config,
            "architecture": architecture,
            "condition": graph,
            "model_index": index,
            "graph_sha256": sha256(ROOT / graph_paths[graph]),
            "bundle_sha256": bundle["bundle_sha256"],
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
        model = models.make_model(architecture, model_cfg, batched.PILOT_SEED, "cuda")
        require(
            models.parameter_count(model) == bundle["parameter_budget"]["counts"][architecture],
            "batch model parameter count differs from frozen budget",
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_cfg["optimizer"]["learning_rate"],
            weight_decay=model_cfg["optimizer"]["weight_decay"],
        )
        jobs.append(
            batched.BatchJob(
                model,
                optimizer,
                inputs[graph],
                ROOT
                / "checkpoints"
                / "molab_batch_rate"
                / bundle["bundle_sha256"][:12]
                / f"K{k}"
                / f"{architecture}-{graph}-{index:02d}",
                fingerprint(spec),
                torch.cuda.Stream(),
                features[graph],
            )
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_started = monotonic()
    batched.train_batch(jobs, task, BENCH_TRIALS, deadline, stop_at_criterion=False)
    training_seconds = monotonic() - train_started
    total_seconds = monotonic() - started
    require(all(len(job.history) == BENCH_TRIALS for job in jobs), "incomplete batch benchmark")
    projected = (total_seconds - training_seconds + 10 * training_seconds) * 3660 / (k * 3600)
    report = {
        **config,
        "scope": "EXPLORATORY_BATCH_RATE",
        "status": "PASS",
        "architecture_counts": {"GCN": k // 2, "GAT": k // 2},
        "completed_jobs": len(jobs),
        "total_completed_trials": sum(len(job.history) for job in jobs),
        "training_and_checkpoint_seconds": training_seconds,
        "wall_seconds": total_seconds,
        "jobs_per_hour_200_trials": 3600 * k / total_seconds,
        "active_jobs_per_hour_200_trials": 3600 * k / training_seconds,
        "projected_3660_job_hours_at_2000_trials": projected,
        "peak_vram_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "git_commit": meta["git_commit"],
        "config_sha256": fingerprint(config),
        "bundle_sha256": bundle["bundle_sha256"],
        "registration": evidence,
    }
    atomic_json(report_path, report)
    print("BATCH_RESULT", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--rate-grid", action="store_true", help="measure K=8,16,32 once")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    from mbtopology import runner
    from mbtopology.registration import require_registered
    from mbtopology.runtime import Exclusive, Run, require

    evidence = require_registered(ROOT)
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "attach exactly one Molab CUDA GPU",
    )
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    require(bundle["models"]["dropout"] == 0, "batch stream scheduling requires zero dropout")
    require(bundle["task"]["task"]["trials_max"] == 2000, "projection expects the frozen cap")
    widths = (8, 16, 32) if args.rate_grid else (args.jobs,)
    deadline = monotonic() + bundle["experiment"]["session_hours"] * 3600
    with (
        Exclusive(ROOT),
        Run(ROOT, bundle["experiment"], "molab-batch-rate") as progress,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="null-cpu") as cpu,
    ):
        started = monotonic()
        ready = cpu.submit(prepare_graphs, bundle, widths[0], progress, deadline)
        for i, k in enumerate(widths):
            print(f"BATCH_START K={k}", flush=True)
            graphs = ready.result()
            if i + 1 < len(widths):
                ready = cpu.submit(prepare_graphs, bundle, widths[i + 1], progress, deadline)
            run_wave(bundle, graphs, k, k, started, deadline, evidence)
            torch.cuda.empty_cache()
            started = monotonic()
    print("BATCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
