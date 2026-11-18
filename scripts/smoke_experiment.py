"""Run a two-trial CPU smoke job using seed 0 and the frozen GCN configuration."""

import argparse
import ctypes
import json
import os
import sys
from ctypes import wintypes
from pathlib import Path
from time import perf_counter

TRIALS = 2


def peak_ram_bytes():
    if sys.platform != "win32":
        import resource

        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ram if sys.platform == "darwin" else ram * 1024

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    api = ctypes.WinDLL("psapi", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not api.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.peak_working_set_size


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=["real", "A", "B"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("POLARS_MAX_THREADS", "4")

    from mbtopology.registration import require_registered
    from mbtopology.runtime import (
        Exclusive,
        Run,
        atomic_json,
        fingerprint,
        now,
        provenance,
        require,
    )

    evidence = require_registered(root)

    import torch

    from mbtopology import gnn_training, runner

    torch.set_num_threads(4)
    bundle = runner.load_bundle(root, root / "configs/experiment.yaml")
    torch.use_deterministic_algorithms(bundle["experiment"]["deterministic_algorithms"])
    spec = next(
        spec
        for spec in runner.run_specifications(bundle)
        if spec["seed"] == 0
        and spec["architecture"] == "GCN"
        and spec["condition"] == args.condition
        and spec["null_instance"] in (None, 0)
    )
    prefix = bundle["bundle_sha256"][:12]
    stage = f"smoke-{spec['run_id']}"
    report_path = root / "reports" / "smoke" / prefix / f"{spec['run_id']}.manifest.json"
    signature = {"run": spec, "scope": "engineering_smoke", "trial_limit": TRIALS, "device": "cpu"}
    with Exclusive(root), Run(root, bundle["experiment"], stage) as progress:
        require(not report_path.exists(), f"completed smoke job already exists: {report_path}")
        graph_started = perf_counter()
        graph = (
            bundle["task"]["source_manifest"]
            if args.condition == "real"
            else runner.materialize_null(
                root, bundle, args.condition, 0, progress, None, scope="smoke_nulls"
            )
        )
        graph_seconds = perf_counter() - graph_started
        outcome = gnn_training.run_registered(
            root, bundle, spec, graph, "cpu", progress, None, smoke_trials=TRIALS
        )
        require(outcome["trials_completed"] == TRIALS, "smoke job did not complete two trials")
        require(
            outcome["parameter_count"] == bundle["parameter_budget"]["counts"]["GCN"],
            "smoke GCN parameter budget changed",
        )
        trace = provenance(root, signature)
        report = {
            "status": "PASS",
            "scope": "ENGINEERING_SMOKE",
            "completed_at": now(),
            "git_commit": trace["git_commit"],
            "config_sha256": fingerprint(signature),
            "bundle_sha256": bundle["bundle_sha256"],
            "registration": evidence,
            "run_id": spec["run_id"],
            "graph_manifest": graph,
            "seed": 0,
            "condition": args.condition,
            "architecture": "GCN",
            "trials_completed": outcome["trials_completed"],
            "graph_preparation_seconds": graph_seconds,
            "wall_seconds": perf_counter() - started,
            "peak_ram_bytes": peak_ram_bytes(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        }
        atomic_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
