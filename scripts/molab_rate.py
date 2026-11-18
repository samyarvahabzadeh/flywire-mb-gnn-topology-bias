"""Measure exactly three complete GCN seed-0 jobs on a single Molab CUDA GPU."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("real", "A", "B")


def sample_gpu(stop, samples, errors):
    while not stop.is_set():
        try:
            row = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            util, memory = map(int, row.split(","))
            samples.append((util, memory))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(str(exc))
            return
        stop.wait(1)


def run_job(condition):
    started = time.monotonic()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import polars as pl
    import torch

    from mbtopology import gnn_training, runner
    from mbtopology.experiment_io import write_result
    from mbtopology.registration import require_registered
    from mbtopology.runtime import Exclusive, Run, atomic_json, fingerprint, provenance, require

    require(sys.platform.startswith("linux"), "Molab rate measurement needs its Linux worker")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "attach one Molab CUDA GPU before starting the rate jobs",
    )
    registration = require_registered(ROOT)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    experiment = bundle["experiment"]
    require(bundle["task"]["task"]["trials_max"] == 2000, "trial cap differs from frozen config")
    torch.use_deterministic_algorithms(experiment["deterministic_algorithms"])
    torch.set_num_threads(4)
    spec = next(
        spec
        for spec in runner.run_specifications(bundle)
        if spec["architecture"] == "GCN"
        and spec["seed"] == 0
        and spec["condition"] == condition
        and spec["null_instance"] in (None, 0)
    )
    output = ROOT / "results" / "molab_rate" / bundle["bundle_sha256"][:12]
    report_file = output / f"{condition}.resources.json"
    stop, samples, errors = threading.Event(), [], []
    sampler = threading.Thread(target=sample_gpu, args=(stop, samples, errors), daemon=True)
    deadline = started + experiment["session_hours"] * 3600
    with Exclusive(ROOT), Run(ROOT, experiment, f"molab-rate-{condition}") as progress:
        if report_file.exists():
            from mbtopology.experiment_io import cached_result

            require(
                cached_result(
                    output / f"{spec['run_id']}.parquet",
                    fingerprint(spec),
                    bundle["bundle_sha256"],
                ),
                "resource report exists without matching run result",
            )
            print(report_file.read_text(encoding="utf-8"))
            return
        sampler.start()
        try:
            graph = (
                bundle["task"]["source_manifest"]
                if condition == "real"
                else runner.materialize_null(
                    ROOT, bundle, condition, 0, progress, deadline, scope="molab_rate_nulls"
                )
            )
            torch.cuda.reset_peak_memory_stats()
            outcome = gnn_training.run_registered(
                ROOT, bundle, spec, graph, "cuda", progress, deadline, rate_measurement=True
            )
        finally:
            stop.set()
            sampler.join(timeout=6)
        require(samples and not errors, f"GPU sampling failed: {errors}")
        history = outcome.pop("history")
        timing = [row["trial_seconds"] for row in history]
        count = outcome["trials_completed"]
        reached = outcome["trials_to_criterion"]
        require(count == len(timing) and (reached == count or count == 2000), "unfinished rate job")
        require(
            outcome["parameter_count"] == bundle["parameter_budget"]["counts"]["GCN"],
            "GCN parameter budget changed",
        )
        report = {
            "status": "PASS",
            "scope": "EXPLORATORY_MOLAB_RATE_ONLY",
            "condition": condition,
            "seed": 0,
            "architecture": "GCN",
            "trials": count,
            "criterion_trial": reached,
            "termination": "criterion" if reached is not None else "cap_2000",
            "trial_mean_seconds": statistics.mean(timing),
            "trial_sample_variance_seconds_squared": statistics.variance(timing)
            if len(timing) > 1
            else None,
            "job_wall_seconds": time.monotonic() - started,
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_vram_occupied_sampled_mib": max(memory for _, memory in samples),
            "peak_pytorch_reserved_bytes": torch.cuda.max_memory_reserved(),
            "mean_gpu_utilization_percent": statistics.mean(util for util, _ in samples),
            "gpu_samples": len(samples),
            "gpu_sample_interval_seconds": 1,
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "config_sha256": fingerprint(spec),
            "bundle_sha256": bundle["bundle_sha256"],
        }
        trace = provenance(ROOT, spec)
        report["git_commit"] = trace["git_commit"]
        write_result(
            ROOT,
            output / f"{spec['run_id']}.parquet",
            pl.DataFrame(history),
            {
                **trace,
                **outcome,
                **report,
                "run_config_sha256": fingerprint(spec),
                "registration": registration,
                "graph_manifest": graph,
                "run_specification": spec,
            },
        )
        atomic_json(report_file, report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if args.condition:
        run_job(args.condition)
        return
    for condition in CONDITIONS:
        print(f"Starting {condition}", flush=True)
        subprocess.run(
            [sys.executable, "-u", __file__, "--condition", condition], cwd=ROOT, check=True
        )
    print("ALL_THREE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
