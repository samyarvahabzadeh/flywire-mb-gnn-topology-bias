"""Measure one six-job CUDA batch after public registration of AMENDMENT-02."""

import hashlib
import json
import resource
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen

import numpy as np
import polars as pl
import torch
from final_binary_pilot import ROOT, inputs_for, model_for, setup
from phase4_pilot import binary_step, load_real_cues, valences

from mbtopology.binary_protocol import add_observation, endpoints, reached
from mbtopology.experiment_io import (
    SessionYield,
    load_training_checkpoint,
    save_training_checkpoint,
    write_result,
)
from mbtopology.real_cues import robustness_drives
from mbtopology.registration import REPOSITORY, git_bytes
from mbtopology.runner import materialize_null
from mbtopology.runtime import Exclusive, Run, atomic_json, fingerprint, read_json, require, sha256
from mbtopology.task_inputs import load_source

TAG = "v1.2-prereg-amendment"
RATE_JOBS = 6
SWEEP_JOBS = 1830


def require_amendment(root):
    ref = f"refs/tags/{TAG}"
    tagged = git_bytes(root, "rev-parse", ref).decode().strip()
    commit = git_bytes(root, "rev-parse", f"{ref}^{{commit}}").decode().strip()
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}/git/ref/tags/{TAG}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "mbtopology"},
    )
    with urlopen(request, timeout=20) as response:
        public = json.load(response)
    require(
        public.get("ref") == ref
        and public.get("object", {}).get("type") == "tag"
        and public["object"]["sha"] == tagged,
        "public final amendment tag does not match the local annotated tag",
    )
    files = (
        "AMENDMENT-02.md",
        "configs/final_binary.yaml",
        "scripts/final_binary_pilot.py",
        "scripts/final_binary_rate.py",
        "scripts/phase4_pilot.py",
        "src/mbtopology/binary_protocol.py",
        "src/mbtopology/real_cues.py",
        "src/mbtopology/robustness_pilot.py",
    )
    hashes = {}
    for file in files:
        blob = git_bytes(root, "show", f"{commit}:{file}")
        checkout = (root / file).read_bytes()
        require(
            checkout in (blob, blob.replace(b"\n", b"\r\n")), f"final tagged file changed: {file}"
        )
        hashes[file] = hashlib.sha256(blob).hexdigest()
    return {"tag": TAG, "tag_object": tagged, "tagged_commit": commit, "tagged_file_sha256": hashes}


@dataclass
class Job:
    spec: dict
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    inputs: dict
    drives: list
    noise: dict
    noise_audit: dict
    stream: torch.cuda.Stream
    checkpoint_dir: Path
    signature: str
    history: list = field(default_factory=list)

    def restore(self):
        saved = load_training_checkpoint(self.checkpoint_dir, self.signature)
        if saved is None:
            return False
        self.history = saved["history"]
        require(saved["cursor"] == len(self.history), "invalid binary batch checkpoint cursor")
        self.model.load_state_dict(saved["model"])
        self.optimizer.load_state_dict(saved["optimizer"])
        return True

    def checkpoint(self):
        save_training_checkpoint(
            self.checkpoint_dir,
            self.signature,
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cursor": len(self.history),
                "history": self.history,
            },
        )


def train_wave(jobs, config, progress, deadline):
    default = torch.cuda.current_stream()
    labels = torch.tensor([1.0, 0.0] * config["pairs_per_trial"], device="cuda")
    restored = [job.restore() for job in jobs]
    for job in jobs:
        require(len(job.history) <= config["trials"], "binary batch checkpoint exceeds cap")
        job.stream.wait_stream(default)
    started = last_save = monotonic()

    def save():
        for job in jobs:
            job.checkpoint()
        progress.save(trials=[len(job.history) for job in jobs])

    def active(job):
        return len(job.history) < config["trials"] and not reached(job.history, config["threshold"])

    resumed = any(restored) and any(job.history for job in jobs)
    rounds = 0
    full_batch_rounds = 0
    full_batch_seconds = 0.0
    while pending := [job for job in jobs if active(job)]:
        if monotonic() >= deadline:
            save()
            raise SessionYield("final binary rate batch checkpointed at session deadline")
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
        require(
            all(v[2] for v in values), "nonfinite binary batch loss, gradients or held-out probe"
        )
        for job, (a_plus, a_minus, _) in zip(pending, values, strict=True):
            add_observation(job.history, (a_plus, a_minus), config["rolling_window"])
        rounds += 1
        if len(pending) == len(jobs):
            full_batch_rounds += 1
            full_batch_seconds = monotonic() - started
        if rounds % 250 == 0:
            print("FINAL_RATE_PROGRESS", rounds, [len(j.history) for j in jobs], flush=True)
        if monotonic() - last_save >= config["checkpoint_seconds"]:
            save()
            last_save = monotonic()
    torch.cuda.synchronize()
    save()
    return monotonic() - started, rounds, resumed, full_batch_rounds, full_batch_seconds


def prepare_nulls(bundle, seed_list, progress, deadline):
    results = {}
    timings = []
    for seed in seed_list:
        for variant in ("A", "B"):
            started = monotonic()
            path = materialize_null(
                ROOT, bundle, variant, seed, progress, deadline, scope="final_binary_rate_nulls"
            )
            manifest = read_json(ROOT / path)
            timings.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "seconds": monotonic() - started,
                    "created_at": manifest["artifacts"]["csr"]["recorded_at"],
                }
            )
            results[seed, variant] = path
    return results, timings


def evaluate_robustness(job, config):
    scores = []
    presentations = 2 * config["pairs_per_trial"]
    job.model.eval()
    with torch.no_grad():
        for level, drive in job.noise.items():
            probs = torch.stack(
                [
                    valences(job.model, job.inputs, drive, trial, presentations).double()
                    for trial in range(config["robustness"]["probe_trials"])
                ]
            )
            average = (probs[:, ::2].mean() + 1 - probs[:, 1::2].mean()) / 2
            scores.append((level, float(average)))
    levels, averages = zip(*scores, strict=True)
    return {
        "discrimination_by_level": dict(scores),
        "slope": float(np.polyfit(levels, averages, 1)[0]),
    }


def projection(train_seconds, nontraining_wave_seconds, cue_seconds, null_seconds):
    waves = ceil(SWEEP_JOBS / RATE_JOBS)
    setup = (30 / 2) * cue_seconds + (60 / 4) * null_seconds
    hours = (waves * (train_seconds + nontraining_wave_seconds) + setup) / 3600
    return {
        "jobs": SWEEP_JOBS,
        "batch_width": RATE_JOBS,
        "waves": waves,
        "jobs_per_hour_training": RATE_JOBS * 3600 / train_seconds,
        "hours_including_once_per_seed_and_graph_prep": hours,
        "molab_12h_sessions": ceil(hours / 12),
        "molab_11_5h_budget_sessions": ceil(hours / 11.5),
    }


def main():
    registration = require_amendment(ROOT)
    config, bundle, trace, common, original_registration = setup()
    config_id = fingerprint(
        {"protocol": trace["config_sha256"], "width": RATE_JOBS, "seeds": config["pilot_seeds"]}
    )
    launch = {
        **common,
        "scope": "FINAL_BINARY_POST_TAG_RATE_ONLY",
        "rate_config_sha256": config_id,
        "amendment": registration,
        "original_registration": original_registration,
    }
    run_id = fingerprint(launch)[:12]
    out = ROOT / "results" / "final_binary_rate" / run_id
    check = ROOT / "checkpoints" / "final_binary_rate" / run_id
    summary_path = out / "summary.manifest.json"
    if (previous := read_json(summary_path)) is not None:
        require(
            previous["rate_config_sha256"] == config_id and previous["completed_jobs"] == RATE_JOBS,
            "existing final rate result differs",
        )
        print("FINAL_RATE_COMPLETE", run_id, json.dumps(previous["performance"]), flush=True)
        return
    started = monotonic()
    deadline = started + bundle["experiment"]["session_hours"] * 3600
    seeds = tuple(config["pilot_seeds"])
    print("FINAL_RATE_START", run_id, trace["git_commit"], config_id, flush=True)
    graph = bundle["task"]["source_manifest"]
    with (
        Exclusive(ROOT),
        Run(ROOT, config, "final-binary-rate") as progress,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="null-cpu") as cpu,
    ):
        future = cpu.submit(prepare_nulls, bundle, seeds, progress, deadline)
        cue_started = monotonic()
        _, nodes, matrix = load_source(ROOT, graph)
        roles = nodes["role"].to_numpy()
        pn_kc = matrix[np.flatnonzero(roles == "PN")][:, np.flatnonzero(roles == "KC")].tocsr()
        pn_kc.sort_indices()
        prepared = {}
        for seed in seeds:
            drives, _, _ = load_real_cues(bundle["task"], config, seed)
            noise, noise_audit = robustness_drives(pn_kc, seed, config)
            prepared[seed] = drives, noise, noise_audit
        cue_seconds = monotonic() - cue_started
        paths, timings = future.result()
        setup_started = monotonic()
        jobs = []
        for seed in seeds:
            for condition in ("real", "A", "B"):
                path = graph if condition == "real" else paths[seed, condition]
                spec = {
                    **launch,
                    "seed": seed,
                    "condition": condition,
                    "model": "GraphSAGE-max",
                    "null_instance": None if condition == "real" else seed,
                    "graph_sha256": sha256(ROOT / path),
                }
                signature = fingerprint(spec)
                model, optimizer = model_for(bundle, seed)
                jobs.append(
                    Job(
                        spec,
                        model,
                        optimizer,
                        inputs_for(bundle, seed, path),
                        *prepared[seed],
                        torch.cuda.Stream(),
                        check / f"{condition}-seed{seed}",
                        signature,
                    )
                )
        require(len(jobs) == RATE_JOBS, "rate runner must enqueue exactly six pilot jobs")
        torch.cuda.synchronize()
        setup_seconds = monotonic() - setup_started
        torch.cuda.reset_peak_memory_stats()
        train_seconds, rounds, resumed, full_rounds, full_seconds = train_wave(
            jobs, config, progress, deadline
        )
        post_started = monotonic()
        results = []
        for job in jobs:
            seed, condition = job.spec["seed"], job.spec["condition"]
            result = {"seed": seed, "condition": condition, **endpoints(job.history, config)}
            noise_summary = evaluate_robustness(job, config)
            write_result(
                ROOT,
                out / f"{condition}-seed{seed}.parquet",
                pl.DataFrame(job.history),
                {
                    **trace,
                    **job.spec,
                    "run_config_sha256": job.signature,
                    "outcome": result,
                    "robustness": noise_summary,
                    "noise_injection_audit": job.noise_audit,
                },
            )
            results.append(result)
        torch.cuda.synchronize()
        post_seconds = monotonic() - post_started
        projection_args = (
            setup_seconds + post_seconds,
            cue_seconds,
            sum(t["seconds"] for t in timings),
        )
        train_project = projection(train_seconds, *projection_args) if not resumed else None
        at_cap = (
            projection(full_seconds * config["trials"] / full_rounds, *projection_args)
            if full_rounds and not resumed
            else None
        )
        performance = {
            "total_wall_seconds": monotonic() - started,
            "train_wall_seconds": train_seconds,
            "cue_setup_seconds_for_two_seeds": cue_seconds,
            "model_setup_seconds": setup_seconds,
            "post_stop_noise_and_io_seconds": post_seconds,
            "null_builds": timings,
            "batch_rounds": rounds,
            "resumed": resumed,
            "full_width_rounds": full_rounds,
            "full_width_seconds": full_seconds,
            "jobs_per_hour_end_to_end": RATE_JOBS * 3600 / (monotonic() - started),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_process_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "projection_observed": train_project,
            "projection_all_at_cap": at_cap,
        }
        atomic_json(
            summary_path,
            {
                **trace,
                **launch,
                "completed_jobs": len(jobs),
                "runs": results,
                "performance": performance,
            },
        )
        print("FINAL_RATE_COMPLETE", run_id, json.dumps(performance), flush=True)


if __name__ == "__main__":
    main()
