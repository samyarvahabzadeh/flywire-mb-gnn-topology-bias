"""Run the fixed-budget exploratory topology × learning rule screen on CUDA."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from time import monotonic

import polars as pl
import torch
from exploratory_rule_models import FrozenReadouts, adam, mb_embeddings, probe_row, same_structure
from exploratory_rule_report import report_screen
from final_binary_pilot import protocol
from phase4_pilot import binary_step, load_real_cues, valences

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.binary_protocol import add_observation
from mbtopology.experiment_io import (
    load_training_checkpoint,
    save_training_checkpoint,
    write_result,
)
from mbtopology.runtime import fingerprint, now, read_json, require, sha256
from mbtopology.sweep_archive import ArchiveStore

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/exploratory_learning_rules.json"
WAVE_WIDTH = 6


def load_design():
    plan = read_json(PLAN)
    shared = protocol()
    require(
        sha256(ROOT / plan["protocol"]) == plan["protocol_sha256"], "binary protocol bytes changed"
    )
    for field in ("trials", "threshold", "pairs_per_trial", "rolling_window", "checkpoint_seconds"):
        require(plan[field] == shared[field], f"fixed binary task field differs: {field}")
    expected = ["real"]
    for variant in "AB":
        labels = [f"{variant}-{i:02d}" for i in range(30)]
        selected = sorted(
            labels,
            key=lambda label: hashlib.sha256(
                f"lr-screen-v1:{label.replace('-', ':')}".encode()
            ).digest(),
        )[:3]
        expected.extend(sorted(selected))
    require(
        plan["graphs"] == expected and plan["arms"] == ["A", "B", "C"], "wrong exploratory roster"
    )
    fresh = set(plan["seeds"]) | {plan["smoke_seed"]}
    require(
        len(fresh) == 6 and not fresh & (set(range(30)) | {1337, 2026}),
        "reused or missing task seed",
    )
    return plan, shared


def get_graphs(store, plan, smoke):
    desired = plan["graphs"][:1] if smoke else plan["graphs"]
    real, _ = store.verify("real-source", restore_root=ROOT)
    match = [name for name in real["files"] if name.endswith("/MANIFEST.json")]
    require(len(match) == 1, "ambiguous real graph manifest")
    paths = {"real": match[0]}
    labels = {
        f"graph-bank-{(int(label[2:]) + (30 if label[0] == 'B' else 0)) // 6:03d}"
        for label in desired[1:]
    }
    for label in sorted(labels):
        receipt, _ = store.verify(label, restore_root=ROOT)
        paths.update({k: v for k, v in receipt["metadata"]["graphs"].items() if k in desired})
    require(set(paths) == set(desired), "selected graph bank is incomplete")
    return {label: paths[label] for label in desired}


def graph_inputs(bundle, manifest_path):
    # Seed zero is used only to satisfy the legacy builder's checks of the old frozen membership.
    data = gnn_training.frozen_inputs(
        ROOT, bundle["task"], manifest_path, 0, bundle["nulls"], bundle["models"], "cuda"
    )
    data["base"] = models.node_features(data["roles"], data["kc"], data["kc"][:0])
    source, target = data["graph"].edge_index
    data["neighbors"] = data["graph"].edge_index[:, source != target]
    return {
        key: data[key]
        for key in ("roles", "kc", "mbon", "avoidance", "approach", "graph", "base", "neighbors")
    }


def make_cues(bundle, config, seed):
    drives, _, metadata = load_real_cues(bundle["task"], config, seed)
    hashes = []
    for drive in drives:
        hasher = hashlib.sha256()
        for array in (drive.indptr, drive.indices, drive.data):
            hasher.update(array.tobytes())
        hashes.append(hasher.hexdigest())
    require(hashes[0] != hashes[1], "training and held-out drive tables share a checksum")
    metadata["drive_sha256"] = dict(zip(("train", "heldout"), hashes, strict=True))
    return drives, metadata


class Condition:
    def __init__(self, spec, inputs, drives, config, optimizer, model_config, checkpoint):
        self.spec, self.inputs, self.drives, self.config = spec, inputs, drives, config
        self.name = f"seed{spec['seed']:04d}-{spec['graph']}"
        self.signature = fingerprint(spec)
        self.checkpoint = checkpoint / self.name
        self.model = make_benchmark_model(
            "GraphSAGE", model_config, spec["seed"], "cuda", aggregation="max"
        )
        self.reference = {
            key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
        }
        self.adam_a = adam(self.model.parameters(), optimizer)
        self.frozen = FrozenReadouts(self.model, inputs, optimizer, config["local_readout"])
        self.history = {arm: [] for arm in config["arms"]}
        self.gpu_seconds = {arm: 0.0 for arm in config["arms"]}
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        self.stream = torch.cuda.Stream()
        saved = load_training_checkpoint(self.checkpoint, self.signature)
        if saved is not None:
            self.model.load_state_dict(saved["model_a"])
            self.adam_a.load_state_dict(saved["adam_a"])
            self.frozen.restore(saved["frozen"])
            self.history = {arm: [row[arm] for row in saved["history"]] for arm in config["arms"]}
            self.gpu_seconds = saved["gpu_seconds"]

    def enqueue(self, trial, labels):
        self.events[0].record()
        result_a = binary_step(self.model, self.adam_a, self.inputs, self.drives, trial, labels)
        self.events[1].record()
        frozen_train, frozen_eval = (
            mb_embeddings(self.frozen.body, self.inputs, drive, trial, len(labels))
            for drive in self.drives
        )
        self.events[2].record()
        valid_b = self.frozen.train_b(frozen_train, labels)
        with torch.no_grad():
            result_b = probe_row(self.frozen.predict_b(frozen_eval), valid_b)
        self.events[3].record()
        valid_c = self.frozen.train_c(frozen_train, labels)
        result_c = probe_row(self.frozen.predict_c(frozen_eval), valid_c)
        self.events[4].record()
        return torch.stack((result_a, result_b, result_c))

    def observe(self, rows):
        a, b, c, d, e = self.events
        feature_time = b.elapsed_time(c)
        times = (
            a.elapsed_time(b),
            feature_time + c.elapsed_time(d),
            feature_time + d.elapsed_time(e),
        )
        for arm, (plus, minus, valid), milliseconds in zip(
            self.config["arms"], rows, times, strict=True
        ):
            require(bool(valid), f"non-finite training or held-out values in {self.name}/{arm}")
            add_observation(self.history[arm], (plus, minus), self.config["rolling_window"])
            self.gpu_seconds[arm] += milliseconds / 1000

    def save(self):
        records = [
            dict(zip(self.config["arms"], rows, strict=True))
            for rows in zip(*self.history.values(), strict=True)
        ]
        save_training_checkpoint(
            self.checkpoint,
            self.signature,
            {
                "cursor": len(records),
                "history": records,
                "model_a": self.model.state_dict(),
                "adam_a": self.adam_a.state_dict(),
                "frozen": self.frozen.export(),
                "gpu_seconds": self.gpu_seconds,
            },
        )
        path = self.checkpoint / "checkpoint.json"
        return [path, path.parent / read_json(path)["file"]]

    def audit(self):
        b = same_structure(self.frozen.body, self.reference)
        a = same_structure(self.model, self.reference)
        c_head = self.frozen.initial_head.state_dict()
        unchanged = all(
            torch.equal(c_head[key].detach().cpu(), self.reference[f"head.{key}"]) for key in c_head
        )
        c = {"backbone": b["backbone"], "head": float(self.frozen.delta.double().norm())}
        require(b["backbone"] == 0 and unchanged, "frozen parameters changed in arm B or C")
        return {"A": a, "B": b, "C": c}

    def smoke_audit(self, labels, *, preflight):
        trainable = (*self.model.parameters(), *self.frozen.head_b.parameters(), self.frozen.delta)
        before = tuple(p.detach().clone() for p in trainable)
        with torch.no_grad():
            features = mb_embeddings(self.frozen.body, self.inputs, self.drives[1], 0, len(labels))
            a = valences(self.model, self.inputs, self.drives[1], 0, len(labels))
            b, c = self.frozen.predict_b(features), self.frozen.predict_c(features)
            require(all(torch.isfinite(y).all() for y in (a, b, c)), "readout smoke nonfinite")
        require(
            all(torch.equal(p, old) for p, old in zip(trainable, before, strict=True)),
            "held-out probe mutated a parameter",
        )
        if preflight:
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
            torch.testing.assert_close(a, c, rtol=0, atol=1e-6)
            delta = self.frozen.delta.clone()
            features = mb_embeddings(self.frozen.body, self.inputs, self.drives[0], 0, len(labels))
            self.frozen.train_c(features, torch.zeros_like(labels))
            require(
                torch.equal(delta, self.frozen.delta),
                "arm C updated in the absence of reinforcement",
            )
        else:
            drift = self.audit()
            require(
                drift["A"]["backbone"] > 0 and all(d["head"] > 0 for d in drift.values()),
                "expected smoke weight update missing",
            )


def restore_latest(store, label):
    pattern = re.compile(rf"checkpoint-{label}-(\d{{4}})\.receipt\.json")
    names = store.list_names()
    if f"{label}.receipt.json" in names:
        store.verify(label, restore_root=ROOT)
        return True
    usable = sorted(name for name in names if pattern.fullmatch(name))
    if usable:
        store.verify(usable[-1].removesuffix(".receipt.json"), restore_root=ROOT)
    return False


def train_wave(jobs, end, store, label, smoke):
    default = torch.cuda.current_stream()
    plan = jobs[0].config
    labels = torch.tensor([1.0, 0.0] * plan["pairs_per_trial"], device="cuda")
    counts = {len(history) for job in jobs for history in job.history.values()}
    require(len(counts) == 1, "wave checkpoint cursors disagree")
    start = counts.pop()
    require(0 <= start <= end, "wave checkpoint exceeds fixed horizon")
    if smoke and start == 0:
        for job in jobs:
            job.smoke_audit(labels, preflight=True)
    for job in jobs:
        job.stream.wait_stream(default)
    started = last = monotonic()
    for trial in range(start, end):
        outputs = []
        for job in jobs:
            with torch.cuda.stream(job.stream):
                outputs.append(job.enqueue(trial, labels))
        for job in jobs:
            default.wait_stream(job.stream)
        for job, rows in zip(jobs, torch.stack(outputs).cpu().tolist(), strict=True):
            job.observe(rows)
        if (trial + 1) % 250 == 0:
            print(
                f"RULE_PROGRESS {label} {trial + 1}/{end} {monotonic() - started:.1f}s", flush=True
            )
        if trial + 1 < end and monotonic() - last >= plan["checkpoint_seconds"]:
            paths = [item for job in jobs for item in job.save()]
            store.commit(f"checkpoint-{label}-{trial + 1:04d}", ROOT, paths, {"trial": trial + 1})
            last = monotonic()
    if smoke:
        for job in jobs:
            job.smoke_audit(labels, preflight=False)
    return monotonic() - started


def run_endpoints(rows, plan):
    stop = next(
        (
            row["trial"]
            for row in rows
            if row["rolling_discrimination"] is not None
            and row["rolling_discrimination"] >= plan["threshold"]
        ),
        None,
    )
    return {
        "trials": len(rows),
        "ttc": stop,
        "right_censored": stop is None,
        "auc_common_horizon": sum(row["discrimination"] for row in rows) / len(rows),
        "terminal_rolling_discrimination": rows[-1]["rolling_discrimination"],
    }


def write_wave(jobs, output, store, label, wall):
    rows, paths = [], []
    for job in jobs:
        weight_audit = job.audit()
        for arm, history in job.history.items():
            run_id = f"{job.name}-arm{arm}"
            path = output / f"{run_id}.parquet"
            meta = {
                **job.spec,
                "run_id": run_id,
                "arm": arm,
                "outcome": run_endpoints(history, job.config),
                "weight_drift_l2": weight_audit[arm],
                "gpu_stream_seconds": job.gpu_seconds[arm],
                "wave_training_wall_seconds_this_session": wall,
                "wave_peak_gpu_bytes_this_session": torch.cuda.max_memory_allocated(),
                "run_config_sha256": fingerprint((job.spec, arm)),
            }
            if arm == "C":
                cap = job.config["local_readout"]["maximum_depression"]
                meta["fraction_readout_at_depression_bound"] = float(
                    (job.frozen.delta == -cap).float().mean()
                )
            write_result(ROOT, path, pl.DataFrame(history), meta)
            path_manifest = path.with_suffix(".manifest.json")
            paths.extend((path, path_manifest))
            rows.append(read_json(path_manifest))
        paths.extend(job.save())
    store.commit(
        label,
        ROOT,
        paths,
        {"jobs": [row["run_id"] for row in rows], "wave_wall_seconds_this_session": wall},
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wave-width", type=int, default=WAVE_WIDTH, choices=range(1, WAVE_WIDTH + 1)
    )
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from gdrive_fsspec import GoogleDriveFileSystem

    plan, shared = load_design()
    plan_hash = sha256(PLAN)
    stage = "smoke" if args.smoke else "screen"
    config = {**shared, **plan, "trials": plan["smoke_trials" if args.smoke else "trials"]}
    seeds = [plan["smoke_seed"]] if args.smoke else plan["seeds"]
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "need one CUDA GPU for the exploratory screen",
    )
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fs = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )
    prefix = args.output_prefix.rstrip("/")
    require(
        prefix != args.source_prefix.rstrip("/"),
        "write prefix must be separate from graph-bank prefix",
    )
    if not args.smoke:
        smoke_store = ArchiveStore(fs, f"{prefix}/smoke", plan_hash)
        smoke = json.loads(smoke_store.read("smoke-pass.json"))
        require(
            smoke["status"] == "PASS"
            and smoke["git_commit"] == commit
            and smoke["config_sha256"] == plan_hash,
            "CUDA smoke must pass on this exact code and configuration before the screen",
        )
    output = ROOT / "results" / "exploratory" / "learning_rules" / plan_hash[:12] / stage
    output.mkdir(parents=True, exist_ok=True)
    destination = ArchiveStore(fs, f"{prefix}/{stage}", plan_hash)
    source = ArchiveStore(fs, args.source_prefix, plan["graph_bank_plan_sha256"])
    paths = get_graphs(source, plan, args.smoke)
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    optimizer = bundle["models"]["optimizer"]
    require(
        optimizer["learning_rate"] == 0.001 and optimizer["weight_decay"] == 0,
        "baseline Adam config changed",
    )
    base = {
        "scope": plan["scope"],
        "stage": stage,
        "git_commit": commit,
        "config_sha256": plan_hash,
        "frozen_binary_protocol_sha256": plan["protocol_sha256"],
        "bundle_sha256": fingerprint([bundle["bundle_sha256"], plan_hash, commit]),
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "wave_width": args.wave_width,
        "training_cues": config["trials"] * 2 * config["pairs_per_trial"],
        "heldout_cues": config["trials"] * 2 * config["pairs_per_trial"],
    }
    roster = [(seed, graph) for seed in seeds for graph in paths]
    trace = {
        **base,
        "config": plan,
        "roster": [{"seed": s, "graph": g} for s, g in roster],
        "graphs": {k: sha256(ROOT / v) for k, v in paths.items()},
    }
    destination.record("execution-plan.json", trace)
    print(
        f"RULE_START {stage} {len(roster) * len(plan['arms'])} {base['device']} {commit}",
        flush=True,
    )
    graph_data = {label: graph_inputs(bundle, path) for label, path in paths.items()}
    expected = []
    cue_cache = {}
    checkpoint = ROOT / "checkpoints" / "exploratory" / "learning_rules" / plan_hash[:12] / stage
    for offset in range(0, len(roster), args.wave_width):
        label = f"wave-{offset // args.wave_width:03d}"
        wave = roster[offset : offset + args.wave_width]
        ids = [f"seed{seed:04d}-{graph}-arm{arm}" for seed, graph in wave for arm in plan["arms"]]
        expected.extend(ids)
        if restore_latest(destination, label):
            print(f"RULE_WAVE_ALREADY_DURABLE {label} {len(expected)}", flush=True)
            continue
        jobs = []
        torch.cuda.reset_peak_memory_stats()
        for seed, graph in wave:
            if seed not in cue_cache:
                cue_cache[seed] = make_cues(bundle, config, seed)
            drives, cue_meta = cue_cache[seed]
            spec = {
                **base,
                "seed": seed,
                "graph": graph,
                "graph_sha256": trace["graphs"][graph],
                "cue_audit": cue_meta,
                "wave": label,
            }
            jobs.append(
                Condition(
                    spec, graph_data[graph], drives, config, optimizer, bundle["models"], checkpoint
                )
            )
        wall = train_wave(jobs, config["trials"], destination, label, args.smoke)
        rows = write_wave(jobs, output, destination, label, wall)
        require({row["run_id"] for row in rows} == set(ids), "wave output inventory differs")
        print(
            f"RULE_WAVE_DURABLE {label} {len(expected)}/{len(roster) * len(plan['arms'])} {wall:.1f}s",
            flush=True,
        )
        del jobs
    manifests = [read_json(output / f"{job}.manifest.json") for job in expected]
    require(
        all(row is not None and row["complete"] for row in manifests),
        "missing complete job manifest",
    )
    summary_names = destination.list_names()
    if "summary.receipt.json" in summary_names:
        destination.verify("summary", restore_root=ROOT)
    else:
        report_paths = report_screen(output, manifests, trace, config, args.smoke)
        destination.commit("summary", ROOT, report_paths, {"jobs": len(expected)})
    if args.smoke and "smoke-pass.json" not in summary_names:
        destination.record(
            "smoke-pass.json",
            {
                "status": "PASS",
                "git_commit": commit,
                "config_sha256": plan_hash,
                "completed_utc": now(),
                "jobs": len(expected),
            },
        )
    print(f"RULE_COMPLETE {stage} {len(expected)} {plan_hash}", flush=True)


if __name__ == "__main__":
    main()
