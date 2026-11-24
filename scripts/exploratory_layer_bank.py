"""Replicate the two-layer intervention across a frozen bank of new null graphs."""

import argparse
import contextlib
import io
import os
import subprocess
from itertools import islice, product
from pathlib import Path
from time import monotonic, sleep

import matplotlib.pyplot as plt
import polars as pl
import torch
from exploratory_decoder_topology import graph_inputs, graphs
from exploratory_layer_swap import HybridSAGE, Job, conditions, train_wave, verify
from exploratory_readability import ROOT, read_plan
from phase4_pilot import load_real_cues

from mbtopology import runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.experiment_io import SessionYield
from mbtopology.runtime import (
    atomic_file,
    atomic_json,
    fingerprint,
    now,
    read_json,
    require,
    sha256,
)
from mbtopology.sweep_archive import ArchiveStore

PLAN = ROOT / "configs/exploratory_layer_bank.json"
PAIRS = ("NN", "NR", "RN")
EFFECTS = ("both", "first", "second", "interaction")


def archive_store(prefix, plan):
    if not prefix:
        return None
    try:
        # The notebook has already authorized the account. Fail fast if that
        # credential cache cannot be opened by this noninteractive subprocess.
        from gdrive_fsspec import GoogleDriveFileSystem

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fs = GoogleDriveFileSystem(
                use_listings_cache=False,
                skip_instance_cache=True,
                auth_kwargs={"use_local_webserver": False},
            )
            store = ArchiveStore(fs, f"{prefix.rstrip('/')}/{sha256(PLAN)[:12]}", sha256(PLAN))
            store.record("frozen-plan.json", plan)
            store.put("write-readback.txt", b"FlyWire exploratory crossed-null bank\n")
        print("BANK_DRIVE_READBACK_OK", flush=True)
        return store
    except Exception as exc:  # noqa: BLE001 - external storage is optional for local checkpoints
        print(f"BANK_DRIVE_UNAVAILABLE {type(exc).__name__}; retaining local shards", flush=True)
        return None


def preserve(store, label, root, files, metadata):
    if store is None:
        return False
    for pause in (0, 2, 8):
        try:
            if pause:
                sleep(pause)
            store.commit(label, root, files, metadata)
            return True
        except Exception as exc:  # noqa: BLE001 - the local shard survives any remote failure
            print(f"BANK_ARCHIVE_RETRY {label} {type(exc).__name__}", flush=True)
    return False


def verify_shard(path, metadata, signature, expected, cap):
    spec = read_json(metadata)
    require(
        spec["signature"] == signature
        and spec["files"][path.name]["sha256"] == sha256(path)
        and spec["files"][path.name]["bytes"] == path.stat().st_size,
        "wave shard manifest/hash differs",
    )
    frame = pl.read_parquet(path)
    groups = frame.group_by("seed", "condition").agg(
        pl.len().alias("rows"),
        pl.col("trial").n_unique().alias("trials"),
        pl.col("trial").min().alias("first"),
        pl.col("trial").max().alias("last"),
    )
    require(
        set(groups.select("seed", "condition").iter_rows()) == set(expected)
        and all(row[2:] == (cap, cap, 1, cap) for row in groups.iter_rows()),
        "wave job/trial inventory differs",
    )
    return frame, spec


def run_panel(number, seeds, state, deadline, store):
    plan, config, graphs_on_device, arms, bundle, run, output = state
    folder = output / f"panel-{number:02d}"
    receipt = folder / "MANIFEST.json"
    if receipt.exists():
        spec = read_json(receipt)
        require(
            spec["run_signature"] == run
            and spec["seeds"] == seeds
            and spec["files"]["traces.parquet"]["sha256"] == sha256(folder / "traces.parquet"),
            "cached panel provenance differs",
        )
        return spec
    folder.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    cues = {s: load_real_cues(bundle["task"], config, s)[0] for s in seeds}
    remaining = iter(product(seeds, arms))
    frames, wave_files, archive_count = [], [], 0
    old_archive_names = store.list_names() if store is not None else set()
    while wave := list(islice(remaining, plan["wave_width"])):
        wave_number = len(frames) + 1
        label = f"{folder.name}-wave-{wave_number:02d}"
        shard, meta = (folder / f"{label}{ext}" for ext in (".parquet", ".json"))
        wave_key = fingerprint({"run": run, "panel": number, "jobs": wave})
        if not meta.exists() and f"{label}.receipt.json" in old_archive_names:
            store.verify(label, restore_root=output)
        if not meta.exists():
            jobs = []
            opt = bundle["models"]["optimizer"]
            torch.cuda.reset_peak_memory_stats()
            for seed, condition in wave:
                inputs = arms[condition]
                model = HybridSAGE(
                    make_benchmark_model(
                        "GraphSAGE", bundle["models"], seed, "cuda", aggregation="max"
                    ),
                    inputs,
                )
                if number == 1 and seed == seeds[0]:
                    donor, code = condition.rsplit("-", 1) if condition != "RR" else ("real", "RR")
                    first, final = [graphs_on_device["real" if ch == "R" else donor] for ch in code]
                    verify(
                        model, inputs, first, final, cues[seed][0], 2 * config["pairs_per_trial"]
                    )
                jobs.append(
                    Job(
                        seed,
                        condition,
                        model,
                        torch.optim.Adam(
                            model.parameters(),
                            lr=opt["learning_rate"],
                            weight_decay=opt["weight_decay"],
                        ),
                        inputs,
                        cues[seed],
                        folder / "checkpoints" / f"{seed}-{condition}",
                        fingerprint({"wave": wave_key, "seed": seed, "condition": condition}),
                        torch.cuda.Stream(),
                    )
                )
            try:
                seconds = train_wave(
                    jobs, config, plan["checkpoint_every_trials"], "cuda", deadline=deadline
                )
            except SessionYield:
                pointers = [j.directory / "checkpoint.json" for j in jobs]
                if all(x.exists() for x in pointers):
                    partial = pointers + [p.parent / read_json(p)["file"] for p in pointers]
                    suffix = fingerprint({p.name: sha256(p) for p in partial})[:16]
                    preserve(store, f"{label}-partial-{suffix}", output, partial, {"jobs": wave})
                raise
            frame = pl.from_dicts(
                {"seed": j.seed, "condition": j.label, **row} for j in jobs for row in j.history
            )
            atomic_file(shard, frame.write_parquet)
            atomic_json(
                meta,
                {
                    "signature": wave_key,
                    "git_commit": run["git_commit"],
                    "config_sha256": run["config_sha256"],
                    "seconds": seconds,
                    "jobs": wave,
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                    "files": {shard.name: {"sha256": sha256(shard), "bytes": shard.stat().st_size}},
                },
            )
        frame, spec = verify_shard(shard, meta, wave_key, wave, config["trials"])
        copied = f"{label}.receipt.json" in old_archive_names or preserve(
            store, label, output, (shard, meta), {"panel": number, "jobs": wave}
        )
        archive_count += copied
        frames.append(frame)
        wave_files.extend((shard, meta))
        print(
            f"BANK_WAVE panel={number} wave={wave_number} jobs={len(wave)} "
            f"seconds={spec['seconds']:.1f} archived={int(copied)}",
            flush=True,
        )
    table = pl.concat(frames)
    require(table.height == len(arms) * len(seeds) * config["trials"], "panel trace size differs")
    dest = folder / "traces.parquet"
    atomic_file(dest, table.write_parquet)
    spec = {
        "run_signature": run,
        "seeds": seeds,
        "jobs": len(arms) * len(seeds),
        "git_commit": run["git_commit"],
        "config_sha256": run["config_sha256"],
        "seconds": monotonic() - started,
        "completed_utc": now(),
        "wave_count": len(frames),
        "waves_verified_on_drive": archive_count,
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (dest, *wave_files)
        },
    }
    atomic_json(receipt, spec)
    preserve(store, folder.name, output, (dest, receipt), {"panel": number, "seeds": seeds})
    print(f"BANK_PANEL {number} jobs={spec['jobs']} seconds={spec['seconds']:.1f}", flush=True)
    return spec


def summarize(output, panels, donors, cfg, plan):
    tables = [
        pl.read_parquet(output / f"panel-{i + 1:02d}" / "traces.parquet")
        for i in range(len(panels))
    ]
    results = pl.concat(tables).sort("seed", "condition", "trial")
    cap, threshold, window = cfg["trials"], cfg["threshold"], plan["persistence_window"]
    ends = []
    for (seed, condition), job in results.group_by("seed", "condition", maintain_order=True):
        scores = job["discrimination"].to_list()
        require(len(scores) == cap, "endpoint has an incorrect horizon")
        first = next((i for i, score in enumerate(scores, 1) if score >= threshold), None)
        rolling = next(
            (
                i
                for i, score in enumerate(job["rolling_discrimination"], 1)
                if score is not None and score >= threshold
            ),
            None,
        )
        after = (
            sum(x >= threshold for x in scores[first : first + window]) / window
            if first is not None and first <= cap - window
            else None
        )
        ends.append(
            {
                "seed": seed,
                "condition": condition,
                "auc": job["discrimination"].mean(),
                "first_crossing": first,
                "rolling_crossing": rolling,
                "persistence_next100": after,
                "rolling_right_censored": rolling is None,
            }
        )
    endpoints = pl.DataFrame(
        ends,
        schema_overrides={
            "first_crossing": pl.Int64,
            "rolling_crossing": pl.Int64,
            "persistence_next100": pl.Float64,
        },
    )
    auc = {(q["seed"], q["condition"]): q["auc"] for q in ends}
    contrast = []
    for seed, donor in product(endpoints["seed"].unique().sort(), donors):
        rr = auc[seed, "RR"]
        nn, nr, rn = (auc[seed, f"{donor}-{code}"] for code in PAIRS)
        contrast.append(
            {
                "seed": seed,
                "donor": donor,
                "variant": donor[0],
                "both": nn - rr,
                "first": nr - rr,
                "second": rn - rr,
                "interaction": nn - nr - rn + rr,
            }
        )
    contrasts = pl.DataFrame(contrast)
    by_donor = contrasts.group_by("donor", "variant").agg(pl.col(*EFFECTS).median()).sort("donor")
    conv = (
        endpoints.with_columns(
            pl.when(pl.col("condition") == "RR")
            .then(pl.lit("R"))
            .otherwise(pl.col("condition").str.slice(0, 1))
            .alias("variant"),
            pl.col("condition").str.slice(-2).alias("pair"),
        )
        .group_by("variant", "pair")
        .agg(
            pl.len().alias("n"),
            pl.col("first_crossing").count().alias("first_hits"),
            pl.col("rolling_crossing").count().alias("rolling_hits"),
            pl.col("persistence_next100").count().alias("eligible"),
            pl.col("persistence_next100").median().alias("median_persistence_eligible"),
        )
        .sort("variant", "pair")
    )
    paths = []
    for name, table in (
        ("endpoints", endpoints),
        ("contrasts", contrasts),
        ("donor-medians", by_donor),
        ("persistence-summary", conv),
    ):
        path = output / f"{name}.parquet"
        atomic_file(path, table.write_parquet)
        paths.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained", sharey=True)
    for variant, ax in zip("AB", axes, strict=True):
        sub = by_donor.filter(pl.col("variant") == variant)
        for row in sub.iter_rows(named=True):
            ax.plot([100 * row[k] for k in EFFECTS], "o-", alpha=0.55, label=row["donor"])
        ax.axhline(0, color="#777777", linewidth=1)
        ax.set(
            xticks=range(4),
            xticklabels=("Both", "Layer 1", "Layer 2", "Interaction"),
            title=f"Null {variant}",
        )
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_ylabel("Median seed-paired AUC difference (pp)")
    figure = output / "donor-contrasts.png"
    fig.savefig(figure, dpi=185, facecolor="white")
    plt.close(fig)
    report = output / "REPORT.md"
    lines = [
        "# Exploratory crossed-null layer intervention",
        "",
        f"{len(panels)} complete panels and {len(ends)} fixed-horizon jobs.",
        "R and N refer to real and a donor rewire in layer order. The sensory projection stays real.",
        "Each row below summarizes paired seed differences for one donor. Jobs share seed and donor.",
        "",
        "| Donor | NN−RR (pp) | NR−RR (pp) | RN−RR (pp) | Interaction (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in by_donor.iter_rows(named=True):
        differences = " | ".join(f"{100 * row[k]:+.2f}" for k in EFFECTS)
        lines.append(f"| {row['donor']} | {differences} |")
    lines += [
        "",
        "![Paired donor contrasts](donor-contrasts.png)",
        "",
        "Persistence measures the fraction of the next 100 held-out trials at or above 0.75.",
        "Eligibility requires a first crossing by trial 668. Rolling non-events are right-censored.",
        "",
        "| Variant | Layers | Jobs | First hits | Rolling hits | Eligible | Median persistence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in conv.iter_rows():
        summary = " | ".join(str(v) for v in row[:6])
        value = "—" if row[6] is None else f"{row[6]:.3f}"
        lines.append(f"| {summary} | {value} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths + [figure, report]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--await-study", type=Path)
    parser.add_argument("--drive-prefix")
    args = parser.parse_args()
    started = monotonic()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(torch.cuda.is_available(), "crossed bank requires CUDA")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    plan = read_json(PLAN)
    deadline = started + plan["session_minutes"] * 60 - plan["archive_reserve_seconds"]
    _, config = read_plan()
    config = {**config, "trials": plan["trials"]}
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    require(
        not set(plan["seeds"]) & (set(bundle["seeds"]) | set(config["pilot_seeds"]))
        and all(len(plan["null_instances_by_variant"][v]) == 6 for v in "AB")
        and config["trials"] == 768
        and config["rolling_window"] == 20
        and config["threshold"] == 0.75
        and config["pairs_per_trial"] == 4
        and bundle["models"]["message_passing_depth"] == 2,
        "crossed bank design differs from freeze",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    require(not (args.output / "MANIFEST.json").exists(), "session already finalized")
    store = archive_store(args.drive_prefix, plan)
    if store is not None:
        for name in store.list_names():
            if name.startswith("panel-") and len(name) == len("panel-01.receipt.json"):
                index = name.removesuffix(".receipt.json")
                if not (args.output / index / "MANIFEST.json").exists():
                    store.verify(index, restore_root=args.output)
    # Prepare CPU-only graph artifacts while the small intervention uses the GPU.
    paths = graphs(bundle, plan)
    if args.await_study:
        print("BANK_WAITING_FOR_SMALL_STUDY", flush=True)
        while not args.await_study.exists():
            require(monotonic() < deadline, "small study exceeded the bank session budget")
            sleep(8)
        print(f"BANK_SMALL_STUDY_EXIT {args.await_study.read_text().strip()}", flush=True)
    data = {key: graph_inputs(bundle, path, "cuda") for key, path in paths.items()}
    arms = conditions(data, retain_null_id=True)
    require(len(arms) == 37, "crossed bank arm inventory differs")
    run = {
        "scope": plan["scope"],
        "config_sha256": sha256(PLAN),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "graph_manifests_sha256": {k: sha256(ROOT / path) for k, path in paths.items()},
    }
    state = plan, config, data, arms, bundle, run, args.output
    panels = []
    width = plan["seeds_per_panel"]
    reason = "planned inventory complete"
    for start in range(0, len(plan["seeds"]), width):
        number = start // width + 1
        if not (args.output / f"panel-{number:02d}" / "MANIFEST.json").exists():
            expected = max((p["seconds"] for p in panels), default=0.0) * 1.10
            if monotonic() + expected >= deadline:
                reason = "insufficient engineering time for another complete panel"
                break
        try:
            panels.append(
                run_panel(number, plan["seeds"][start : start + width], state, deadline, store)
            )
        except SessionYield:
            reason = "time cap; next panel has restartable partial checkpoints"
            break
    summary = (
        summarize(args.output, panels, [k for k in paths if k != "real"], config, plan)
        if panels
        else []
    )
    counts = {
        "completed_panels": len(panels),
        "jobs_in_complete_panels": sum(p["jobs"] for p in panels),
        "wave_receipts_verified": sum(p["waves_verified_on_drive"] for p in panels),
        "complete_seeds": [s for p in panels for s in p["seeds"]],
        "stop_reason": reason,
        "wall_seconds": monotonic() - started,
        "completed_utc": now(),
    }
    manifest = args.output / "MANIFEST.json"
    atomic_json(
        manifest,
        {
            **run,
            **counts,
            "files": {
                p.relative_to(args.output).as_posix(): {
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                }
                for p in summary
                + [args.output / f"panel-{i + 1:02d}" / "MANIFEST.json" for i in range(len(panels))]
            },
        },
    )
    # Separate output names are retained across time-bounded attempts.
    saved = preserve(
        store, f"summary-{fingerprint(counts)[:16]}", args.output, [manifest, *summary], counts
    )
    print(
        f"BANK_FINISHED panels={len(panels)} jobs={counts['jobs_in_complete_panels']} "
        f"seconds={counts['wall_seconds']:.1f} archived={int(saved)} {reason}",
        flush=True,
    )


if __name__ == "__main__":
    main()
