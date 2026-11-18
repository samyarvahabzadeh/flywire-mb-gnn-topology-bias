"""Run only the frozen 1,830-job v1.2 matrix with verified external checkpoints."""

import argparse
import json
import signal
import subprocess
from collections import defaultdict
from functools import lru_cache, partial
from pathlib import Path
from time import monotonic, sleep, time_ns

import numpy as np
import polars as pl
import torch
from final_binary_pilot import ROOT, model_for, setup
from phase4_pilot import binary_step, load_real_cues

from mbtopology import gnn_training, models
from mbtopology.binary_protocol import add_observation, endpoints, reached
from mbtopology.experiment_io import SessionYield, write_result
from mbtopology.real_cues import robustness_drives
from mbtopology.runner import materialize_null
from mbtopology.runtime import Exclusive, Run, fingerprint, now, read_json, require, sha256
from mbtopology.sweep_archive import ArchiveStore
from mbtopology.task_inputs import load_source

PLAN_PATH = ROOT / "configs/final_binary_sweep.json"
REAL_GRAPH = "data/graphs/mb_bilateral_v783/MANIFEST.json"
RETRY_LIMIT = 3


def read_plan():
    plan = read_json(PLAN_PATH)
    exclusions = plan["excluded_marked"]
    require(
        plan["scope"] == "CONFIRMATORY_V1_2_FRESH_ONLY"
        and plan["seeds"] == plan["null_instance_ids"] == list(range(30))
        and plan["architecture"] == "GraphSAGE-max"
        and plan["graph_counts"] == {"real": 1, "nullA": 30, "nullB": 30}
        and plan["planned_jobs"] == 1830
        and plan["batch_width"] == 6
        and 0 < plan["checkpoint_seconds"] <= 900
        and len(exclusions) == 6
        and {(j["seed"], j["condition"]) for j in exclusions}
        == {(s, c) for s in (1337, 2026) for c in ("real", "A", "B")}
        and all(j["source_run"] == "c3df5b2c7dfc" and j["excluded_marked"] for j in exclusions),
        "frozen sweep plan or exclusions differ",
    )
    seeds = [
        int(s)
        for s in (ROOT / "configs/seeds.txt").read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]
    require(seeds == plan["seeds"], "seed file and committed sweep plan differ")
    return plan


def specifications(plan):
    return [
        {
            "run_id": f"seed{seed:02d}-{condition}{'' if instance is None else f'-{instance:02d}'}",
            "seed": seed,
            "condition": condition,
            "null_instance": instance,
        }
        for seed in plan["seeds"]
        for condition in ("real", "A", "B")
        for instance in ([None] if condition == "real" else plan["null_instance_ids"])
    ]


def event(name, **fields):
    print(
        "SWEEP",
        json.dumps({"event": name, "at": now(), **fields}, separators=(",", ":")),
        flush=True,
    )


def retry(operation, stage, record):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return operation()
        except SessionYield:
            raise
        except Exception as exc:  # noqa: BLE001 - external storage and CUDA have distinct failure types.
            record(stage=stage, attempt=attempt, error_type=type(exc).__name__)
            if attempt == RETRY_LIMIT:
                raise RuntimeError(
                    f"repeated engineering failure in {stage}: {type(exc).__name__}"
                ) from None
            sleep(2**attempt)


def graph_files(manifest_path):
    document = read_json(ROOT / manifest_path)
    require(document and document.get("status") == "PASS", "graph manifest did not pass")
    return [ROOT / manifest_path, *(ROOT / a["path"] for a in document["artifacts"].values())]


def graphs_for(store, bundle, plan, progress, deadline, failure):
    index = {"real": REAL_GRAPH}
    desired = [(c, i) for c in ("A", "B") for i in plan["null_instance_ids"]]
    keys = store.list_names()
    for offset in range(0, len(desired), 6):
        label = f"graph-bank-{offset // 6:03d}"
        if f"{label}.receipt.json" in keys:
            receipt, _ = store.verify(label, restore_root=ROOT)
            index.update(receipt["metadata"]["graphs"])
            continue
        paths = []
        added = {}
        for condition, instance in desired[offset : offset + 6]:
            path = retry(
                partial(
                    materialize_null,
                    ROOT,
                    bundle,
                    condition,
                    instance,
                    progress,
                    deadline,
                    scope="final_binary_sweep_nulls",
                ),
                f"graph-{condition}-{instance:02d}",
                failure,
            )
            added[f"{condition}-{instance:02d}"] = path
            paths.extend(graph_files(path))
        retry(
            partial(store.commit, label, ROOT, paths, {"graphs": added}), f"store-{label}", failure
        )
        index.update(added)
        event("graph_ready", count=len(index) - 1, of=60)
    require(len(index) == 61, "sweep graph inventory mismatch")
    return index


def safe_files(jobs):
    paths = []
    for job in jobs:
        pointer = job.checkpoint_dir / "checkpoint.json"
        state = read_json(pointer)
        require(state["signature"] == job.signature, "sweep checkpoint signature mismatch")
        file = job.checkpoint_dir / state["file"]
        require(sha256(file) == state["sha256"], "sweep checkpoint hash mismatch")
        paths.extend((pointer, file))
    return paths


def advance(jobs, config, stopped, make_durable):
    default = torch.cuda.current_stream()
    labels = torch.tensor([1.0, 0.0] * config["pairs_per_trial"], device="cuda")
    for job in jobs:
        job.restore()
        require(len(job.history) <= config["trials"], "sweep checkpoint exceeds trial cap")
        job.stream.wait_stream(default)
    last = monotonic()
    while pending := [
        j
        for j in jobs
        if len(j.history) < config["trials"] and not reached(j.history, config["threshold"])
    ]:
        if stopped():
            make_durable(jobs)
            raise SessionYield("sweep session yielded with external checkpoint")
        queued = []
        for job in pending:
            with torch.cuda.stream(job.stream):
                queued.append(
                    binary_step(
                        job.model, job.optimizer, job.inputs, job.drives, len(job.history), labels
                    )
                )
        for job in pending:
            default.wait_stream(job.stream)
        values = torch.stack(queued).cpu().tolist()
        require(all(value[2] for value in values), "nonfinite update in sweep wave")
        for job, (plus, minus, _) in zip(pending, values, strict=True):
            add_observation(job.history, (plus, minus), config["rolling_window"])
        if (monotonic() - last) >= config["checkpoint_seconds"]:
            make_durable(jobs)
            last = monotonic()
    for job in jobs:
        pointer = read_json(job.checkpoint_dir / "checkpoint.json")
        if pointer is None or pointer["cursor"] != len(job.history):
            job.checkpoint()


def gpu_health():
    try:
        output = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=8,
                text=True,
            )
            .strip()
            .split(",")
        )
        return dict(
            zip(
                ("utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c"),
                (int(v) for v in output),
                strict=True,
            )
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"telemetry_available": False}


def audit(store, launches, expected, stopped):
    checked = set()
    censored = 0
    elapsed_floor = defaultdict(float)
    for index, wave in enumerate(launches):
        if stopped():
            raise SessionYield("sweep ended at an external batch boundary before audit finished")
        label = f"batch-{index:04d}"
        receipt, docs = store.verify(label, json_suffix=".manifest.json")
        meta = receipt["metadata"]
        elapsed_floor[meta["session"]] = max(
            elapsed_floor[meta["session"]], meta["elapsed_seconds"]
        )
        ids = {j["run_id"] for j in wave}
        require(
            len(docs) == len(ids) and ids == set(receipt["metadata"]["jobs"]),
            "batch audit inventory mismatch",
        )
        for path, doc in docs.items():
            run_id = doc["run_id"]
            require(
                run_id in ids - checked and doc["scope"] == "CONFIRMATORY_V1_2_FRESH_ONLY",
                "wrong sweep result",
            )
            require(
                doc["parquet_path"] in receipt["files"]
                and receipt["files"][doc["parquet_path"]]["sha256"] == doc["parquet_sha256"]
                and receipt["files"][doc["checkpoint_path"]]["sha256"] == doc["checkpoint_sha256"],
                f"output manifest mismatch: {path}",
            )
            checked.add(run_id)
            censored += int(doc["outcome"]["right_censored"])
        if (index + 1) % 25 == 0:
            event("audit_progress", batches=index + 1, of=len(launches))
    require(checked == expected, "final sweep does not contain exactly the planned job IDs")
    return len(checked), censored, elapsed_floor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe-storage", action="store_true")
    parser.add_argument("--drive-prefix")
    parser.add_argument("--session-hours", type=float, default=11.5)
    args = parser.parse_args()
    plan = read_plan()
    specs = specifications(plan)
    plan_hash = fingerprint(plan)
    require(
        len(specs) == plan["planned_jobs"] and len({j["run_id"] for j in specs}) == len(specs),
        "wrong sweep matrix",
    )
    if not args.execute and not args.probe_storage:
        event(
            "plan_ready",
            plan_sha256=plan_hash,
            seeds=plan["seeds"],
            jobs=len(specs),
            excluded_marked=plan["excluded_marked"],
        )
        return
    require(
        args.drive_prefix and 0 < args.session_hours <= 11.5,
        "execution requires explicit durable drive storage and a valid session limit",
    )
    from final_binary_rate import Job, evaluate_robustness, require_amendment
    from gdrive_fsspec import GoogleDriveFileSystem

    registration = require_amendment(ROOT)
    require(
        registration["tag_object"] == plan["amendment_tag_object"]
        and registration["tagged_commit"] == plan["amendment_git_commit"]
        and registration["tagged_file_sha256"][plan["protocol_path"]]
        == plan["protocol_file_sha256"],
        "sweep plan does not match the public amendment",
    )
    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True)
    store = ArchiveStore(fs, f"{args.drive_prefix.rstrip('/')}/{plan_hash[:12]}", plan_hash)
    store.record("frozen-plan.json", {"plan_sha256": plan_hash, **plan})
    store.put("readback-preflight.txt", b"v1.2 confirmatory sweep external read-back\n")
    require(
        store.read("readback-preflight.txt").startswith(b"v1.2 confirmatory"),
        "external store preflight failed",
    )
    event("external_readback_verified", plan_sha256=plan_hash)
    if not args.execute:
        return
    if "verified-complete.json" in store.list_names():
        done = json.loads(store.read("verified-complete.json"))
        require(
            done["plan_sha256"] == plan_hash and done["verified_jobs"] == len(specs),
            "completed sweep record differs from plan",
        )
        event("already_verified_complete", **done)
        return
    started = monotonic()
    session_id = f"{time_ns()}"
    deadline = started + args.session_hours * 3600
    stop_requested = False
    failures = 0
    pending_errors = {}

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def stopped():
        return stop_requested or monotonic() >= deadline

    def flush_errors():
        for name, record in list(pending_errors.items()):
            try:
                store.record(name, record)
            except Exception:  # noqa: BLE001 - a failing store must not suppress its own retry.
                return
            del pending_errors[name]

    def failure(**fields):
        nonlocal failures
        failures += 1
        event("retry", **fields)
        name = f"retry-{session_id}-{failures:04d}.json"
        pending_errors[name] = {"at": now(), "session": session_id, **fields}
        flush_errors()

    width = plan["batch_width"]
    launches = [specs[i : i + width] for i in range(0, len(specs), width)]
    run_dir = Path("results/final_binary_sweep") / plan_hash[:12]
    check_dir = ROOT / "checkpoints/final_binary_sweep" / plan_hash[:12]
    with Exclusive(ROOT), Run(ROOT, plan, "final-binary-sweep") as progress:
        if "real-source.receipt.json" in store.list_names():
            store.verify("real-source", restore_root=ROOT)
        else:
            load_source(ROOT, REAL_GRAPH)
            store.commit("real-source", ROOT, graph_files(REAL_GRAPH), {"scope": "bootstrap"})
        config, bundle, trace, common, _ = setup()
        require(
            bundle["seeds"] == plan["seeds"] and bundle["task"]["source_manifest"] == REAL_GRAPH,
            "sweep seeds/data differ",
        )
        science = {
            **common,
            "scope": plan["scope"],
            "plan_sha256": plan_hash,
            "sweep_script_sha256": sha256(__file__),
            "amendment": registration,
        }
        store.record("execution-lock.json", science)
        keys = store.list_names()
        completed = set()
        for i, wave in enumerate(launches):
            label = f"batch-{i:04d}"
            if f"{label}.receipt.json" in keys:
                receipt = store.receipt(label)
                require(
                    receipt["metadata"]["jobs"] == [j["run_id"] for j in wave],
                    "committed output receipt does not match its planned batch",
                )
                completed.add(i)
        initial = completed_jobs = sum(len(launches[i]) for i in completed)
        event(
            "session_start",
            session=session_id,
            completed=completed_jobs,
            total=len(specs),
            gpu=gpu_health(),
        )
        for excluded in plan["excluded_marked"]:
            event("excluded_marked", **excluded)
        try:
            graphs = graphs_for(store, bundle, plan, progress, deadline, failure)
            _, nodes, matrix = load_source(ROOT, REAL_GRAPH)
            role = nodes["role"].to_numpy()
            pn_kc = matrix[np.flatnonzero(role == "PN")][:, np.flatnonzero(role == "KC")].tocsr()
            pn_kc.sort_indices()

            @lru_cache(maxsize=2)
            def cues(seed):
                drives, _, _ = load_real_cues(bundle["task"], config, seed)
                noise, dose = robustness_drives(pn_kc, seed, config)
                return drives, noise, dose

            def build_job(spec):
                instance, condition = spec["null_instance"], spec["condition"]
                graph_key = "real" if instance is None else f"{condition}-{instance:02d}"
                path = graphs[graph_key]
                seed = spec["seed"]
                model, optimizer = model_for(bundle, seed)
                inputs = gnn_training.frozen_inputs(
                    ROOT, bundle["task"], path, seed, bundle["nulls"], bundle["models"], "cuda"
                )
                inputs["base"] = models.node_features(
                    inputs["roles"], inputs["kc"], inputs["kc"][:0]
                )
                job_spec = {
                    **science,
                    **spec,
                    "graph_sha256": sha256(ROOT / path),
                    "architecture": plan["architecture"],
                }
                drives, noise, dose = cues(seed)
                return Job(
                    job_spec,
                    model,
                    optimizer,
                    inputs,
                    drives,
                    noise,
                    dose,
                    torch.cuda.Stream(),
                    check_dir / spec["run_id"],
                    fingerprint(job_spec),
                )

            for i, wave in enumerate(launches):
                if i in completed:
                    continue
                if stopped():
                    raise SessionYield("session budget reached at a durable batch boundary")
                label = f"batch-{i:04d}"
                ids = [j["run_id"] for j in wave]
                partial_receipts = sorted(
                    k
                    for k in keys
                    if k.startswith(f"partial-{i:04d}-") and k.endswith(".receipt.json")
                )
                if partial_receipts:
                    store.verify(
                        partial_receipts[-1].removesuffix(".receipt.json"), restore_root=ROOT
                    )
                progress.save(batch=i, completed_jobs=completed_jobs)

                def durable_checkpoint(jobs, batch=i, jobs_in_batch=ids, batch_label=label):
                    for job in jobs:
                        job.checkpoint()
                    retry(
                        partial(
                            store.commit,
                            f"partial-{batch:04d}-{time_ns()}",
                            ROOT,
                            safe_files(jobs),
                            {"jobs": jobs_in_batch},
                        ),
                        f"checkpoint-{batch_label}",
                        failure,
                    )

                def run_wave(wave_specs=wave, save=durable_checkpoint):
                    jobs = [build_job(spec) for spec in wave_specs]
                    advance(jobs, config, stopped, save)
                    paths = safe_files(jobs)
                    for job in jobs:
                        path = ROOT / run_dir / f"{job.spec['run_id']}.parquet"
                        state = read_json(job.checkpoint_dir / "checkpoint.json")
                        file = job.checkpoint_dir / state["file"]
                        report = evaluate_robustness(job, config)
                        manifest = write_result(
                            ROOT,
                            path,
                            pl.DataFrame(job.history),
                            {
                                **trace,
                                **job.spec,
                                "run_config_sha256": job.signature,
                                "checkpoint_path": file.relative_to(ROOT).as_posix(),
                                "checkpoint_sha256": state["sha256"],
                                "outcome": endpoints(job.history, config),
                                "robustness": report,
                                "noise_injection_audit": job.noise_audit,
                            },
                        )
                        paths.extend(
                            (
                                path,
                                path.with_suffix(".manifest.json"),
                                ROOT / manifest["checkpoint_path"],
                            )
                        )
                    return paths

                paths = retry(run_wave, label, partial(failure, jobs=ids))
                archive_metadata = {
                    "jobs": ids,
                    "session": session_id,
                    "elapsed_seconds": monotonic() - started,
                }
                retry(
                    partial(store.commit, label, ROOT, paths, archive_metadata),
                    f"store-{label}",
                    failure,
                )
                flush_errors()
                completed_jobs += len(wave)
                elapsed = monotonic() - started
                rate = (completed_jobs - initial) * 3600 / elapsed
                event(
                    "batch_durable",
                    batch=i,
                    completed=completed_jobs,
                    total=len(specs),
                    jobs_per_hour=round(rate, 2),
                    reference_jobs_per_hour=plan["reference_jobs_per_hour"],
                    eta_hours=round((len(specs) - completed_jobs) / rate, 2),
                    failures=failures,
                    gpu=gpu_health(),
                )
            count, censored, prior_floors = audit(
                store, launches, {j["run_id"] for j in specs}, stopped
            )
            elapsed = monotonic() - started
            previous = [
                json.loads(store.read(k))
                for k in store.list_names()
                if k.startswith("session-") and k.endswith(".json")
            ]
            known = {s["session"] for s in previous} | {session_id}
            missing = set(prior_floors) - known
            total = (
                elapsed
                + sum(s["seconds"] for s in previous)
                + sum(prior_floors[s] for s in missing)
            )
            all_failures = failures + sum(s["failures"] for s in previous)
            final = {
                "plan_sha256": plan_hash,
                "git_commit": trace["git_commit"],
                "config_sha256": trace["config_sha256"],
                "verified_jobs": count,
                "right_censored": censored,
                "total_session_seconds": total,
                "time_is_lower_bound": bool(missing),
                "recorded_failures": all_failures,
            }
            store.record("verified-complete.json", final)
            event("verified_complete", **final)
        except SessionYield:
            event("session_yielded", completed=completed_jobs, total=len(specs), failures=failures)
        finally:
            flush_errors()
            elapsed = monotonic() - started
            hours = elapsed / 3600
            session = {
                "session": session_id,
                "initial": initial,
                "completed": completed_jobs,
                "seconds": elapsed,
                "jobs_per_hour": (completed_jobs - initial) / hours,
                "benchmark_jobs_per_hour": plan["reference_jobs_per_hour"],
                "failures": failures,
            }
            store.record(f"session-{session_id}.json", session)
            event("session_summary", **session)


if __name__ == "__main__":
    main()
