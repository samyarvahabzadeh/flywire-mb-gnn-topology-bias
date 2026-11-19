"""Descriptive report for the topology × learning rule screen."""

from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mbtopology.runtime import atomic_json, now, require, sha256

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("real", "A", "B")
TITLES = {
    "A": "A · End-to-end Adam/BCE",
    "B": "B · Frozen body + Adam/BCE",
    "C": "C · Reward-modulated local readout",
}
COLORS = dict(zip(CONDITIONS, ("#922f22", "#266d89", "#67792b"), strict=True))


def plot_curves(values, plan, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True, layout="constrained")
    x = np.arange(1, plan["trials"] + 1)
    for arm, ax in zip(plan["arms"], axes, strict=True):
        for topology in CONDITIONS:
            seed_curves = np.stack(
                [np.median(values[arm, seed, topology], axis=0) for seed in plan["seeds"]]
            )
            lower, median, upper = np.quantile(seed_curves, [0.25, 0.5, 0.75], axis=0)
            label = "Real" if topology == "real" else f"Null-{topology}"
            ax.plot(x, median, color=COLORS[topology], label=label, linewidth=1)
            ax.fill_between(x, lower, upper, color=COLORS[topology], alpha=0.15, linewidth=0)
        ax.axhline(0.75, color="#777777", linestyle=":", linewidth=1)
        ax.set(title=TITLES[arm], xlabel="Trial", xlim=(1, plan["trials"]))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out discrimination")
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle("Exploratory topology × learning rule screen")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def plot_auc(contrasts, output):
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    offsets = np.linspace(-0.10, 0.10, 5)
    for arm_index, arm in enumerate("ABC"):
        for offset, condition in ((-0.20, "A"), (0.20, "B")):
            points = [
                r["auc_real_minus_null"]
                for r in contrasts
                if r["arm"] == arm and r["null"] == condition
            ]
            ax.scatter(
                arm_index + offset + offsets,
                points,
                color=COLORS[condition],
                alpha=0.8,
                label=f"Real − Null-{condition}" if arm_index == 0 else None,
            )
            ax.hlines(
                np.median(points),
                arm_index + offset - 0.13,
                arm_index + offset + 0.13,
                color=COLORS[condition],
                linewidth=2,
            )
    ax.axhline(0, color="#888888", linewidth=1, linestyle=":")
    ax.set_xticks(range(3), ("A · End-to-end", "B · Frozen body", "C · Local readout"))
    ax.set(ylabel="AUC difference", title="Exploratory AUC contrasts across five matched seeds")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def report_screen(output, manifests, trace, plan, smoke):
    runs, frames, values = [], [], defaultdict(list)
    for meta in manifests:
        path = ROOT / meta["parquet_path"]
        require(sha256(path) == meta["parquet_sha256"], "run Parquet hash mismatch in report")
        df = pl.read_parquet(path)
        require(
            df.height == plan["trials"]
            and df["trial"].to_list() == list(range(1, plan["trials"] + 1)),
            "report trial coverage differs",
        )
        condition = meta["graph"].split("-")[0]
        base = {key: meta[key] for key in ("run_id", "seed", "graph", "arm")}
        base["condition"] = condition
        audit = meta["cue_audit"]
        run = {
            **base,
            **meta["outcome"],
            **{f"{key}_drift_l2": value for key, value in meta["weight_drift_l2"].items()},
            "gpu_stream_seconds": meta["gpu_stream_seconds"],
            "wave": meta["wave"],
            "graph_sha256": meta["graph_sha256"],
            "wave_wall_seconds": meta["wave_training_wall_seconds_this_session"],
            "wave_peak_gpu_bytes": meta["wave_peak_gpu_bytes_this_session"],
            "train_sha256": audit["drive_sha256"]["train"],
            "heldout_sha256": audit["drive_sha256"]["heldout"],
            "fraction_c_at_bound": meta.get("fraction_readout_at_depression_bound"),
        }
        require(
            audit["exact_train_heldout_pn_overlap"]
            == audit["exact_train_heldout_kc_drive_overlap"]
            == 0,
            "train/eval overlap in run audit",
        )
        runs.append(run)
        frames.append(df.with_columns(**{key: pl.lit(value) for key, value in base.items()}))
        values[meta["arm"], meta["seed"], condition].append(df["discrimination"].to_numpy())
    for seed in {row["seed"] for row in runs}:
        require(
            len(
                {
                    (row["train_sha256"], row["heldout_sha256"])
                    for row in runs
                    if row["seed"] == seed
                }
            )
            == 1,
            "cues differ by graph or arm inside seed",
        )
    total_wall = sum({row["wave"]: row["wave_wall_seconds"] for row in runs}.values())
    peak_gpu = max(row["wave_peak_gpu_bytes"] for row in runs) / 2**30
    files = [output / name for name in ("job-endpoints.parquet", "curves.parquet", "REPORT.md")]
    run_frame = pl.DataFrame(
        runs, schema_overrides={"ttc": pl.Int64, "fraction_c_at_bound": pl.Float64}
    )
    run_frame.write_parquet(files[0])
    pl.concat(frames).write_parquet(files[1])
    lines = [
        "# Exploratory topology × learning rule screen",
        "",
        (
            f"Device {trace['device']}. Jobs {len(runs)}. Fixed horizon {plan['trials']} trials per job. "
            f"Each job received {trace['training_cues']:,} training cues and {trace['heldout_cues']:,} held-out cues. "
            f"Sum of recorded wave training wall time {total_wall:.1f} s. Largest recorded wave peak VRAM {peak_gpu:.2f} GiB."
        ),
        "",
        (
            "The paired train and held-out drive checksums matched across the conditions and arms of each seed. "
            "The two cue splits have zero exact overlap in PN patterns or projected KC drive. "
            "Per-job GPU stream time includes shared frozen feature computation in both B and C. "
            "Concurrent CUDA intervals overlap in wall time. If resumed, wave wall time covers the final segment."
        ),
        "",
    ]
    if smoke:
        lines += [
            "## CUDA smoke",
            "",
            (
                "Initial predictions matched across arms. A passed the backbone update check, "
                "B and C kept the body fixed, and C's update was zero on an all-unreinforced block. "
                "Each of the three readouts changed after training."
            ),
            "",
            "| Arm | Backbone L2 change | Readout L2 change | GPU stream seconds |",
            "|---|---:|---:|---:|",
        ]
        for r in runs:
            lines.append(
                f"| {r['arm']} | {r['backbone_drift_l2']:.7f} | {r['head_drift_l2']:.7f} | {r['gpu_stream_seconds']:.3f} |"
            )
    else:
        contrasts = []
        for arm in plan["arms"]:
            for seed in plan["seeds"]:
                match = run_frame.filter((pl.col("arm") == arm) & (pl.col("seed") == seed))
                real = match.filter(pl.col("condition") == "real").row(0, named=True)
                for topo in "AB":
                    null = match.filter(pl.col("condition") == topo)
                    cap_ttc = null["ttc"].fill_null(plan["trials"]).median()
                    contrasts.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "null": topo,
                            "auc_real_minus_null": real["auc_common_horizon"]
                            - null["auc_common_horizon"].median(),
                            "display_ttc_real_minus_null": (
                                real["ttc"] if real["ttc"] is not None else plan["trials"]
                            )
                            - cap_ttc,
                        }
                    )
        comp = pl.DataFrame(contrasts)
        base_a = comp.filter(pl.col("arm") == "A").select(
            "seed", "null", pl.col("auc_real_minus_null").alias("end_to_end_auc_contrast")
        )
        comp = comp.join(base_a, on=["seed", "null"], validate="m:1").with_columns(
            auc_contrast_change_vs_A=pl.col("auc_real_minus_null")
            - pl.col("end_to_end_auc_contrast")
        )
        files.extend(
            output / name
            for name in ("paired-contrasts.parquet", "curves.png", "auc-contrasts.png")
        )
        comp.write_parquet(files[3])
        plot_curves(values, plan, files[4])
        plot_auc(contrasts, files[5])
        lines += [
            "## Descriptive endpoints",
            "",
            (
                "TTC records the first full rolling 20-trial mean ≥0.75. "
                "Each job ran to the common cap. Capped misses are shown at 2,000 for the median TTC column. "
                "AUC averages the observed held-out discrimination through the common cap."
            ),
            "",
            "| Arm | Condition | Jobs | Criterion hits | Median displayed TTC | Median AUC | Median GPU stream seconds |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for arm in plan["arms"]:
            for condition in CONDITIONS:
                rows = run_frame.filter((pl.col("arm") == arm) & (pl.col("condition") == condition))
                lines.append(
                    f"| {arm} | {condition} | {rows.height} | {rows['ttc'].len() - rows['ttc'].null_count()} | "
                    f"{rows['ttc'].fill_null(plan['trials']).median():.1f} | {rows['auc_common_horizon'].median():.6f} | "
                    f"{rows['gpu_stream_seconds'].median():.2f} |"
                )
        lines += [
            "",
            "| Arm | Comparator | Median paired real − null AUC | Median change in that contrast vs arm A |",
            "|---|---|---:|---:|",
        ]
        for arm in plan["arms"]:
            for condition in "AB":
                rows = comp.filter((pl.col("arm") == arm) & (pl.col("null") == condition))
                lines.append(
                    f"| {arm} | Null-{condition} | {rows['auc_real_minus_null'].median():+.6f} | "
                    f"{rows['auc_contrast_change_vs_A'].median():+.6f} |"
                )
        lines += [
            "",
            (
                "Each seed contributes one real score and the median over three matching nulls per variant. "
                "The change against arm A is calculated within each seed before summarizing. "
                "These are descriptive contrasts conditional on the sampled bank and seeds."
            ),
            "",
            "![Held-out discrimination through the common horizon](curves.png)",
            "",
            (
                "Curves show the median across five seed summaries and the across-seed IQR. "
                "Null seed trajectories use the pointwise median across three graphs."
            ),
            "",
            "![Paired real minus null AUC by arm](auc-contrasts.png)",
            "",
        ]
    files[2].write_text(
        "\n".join([*lines, "", "[Manifest](MANIFEST.json) links inputs and output checksums.", ""]),
        encoding="utf-8",
        newline="\n",
    )
    manifest = output / "MANIFEST.json"
    atomic_json(
        manifest,
        {
            **trace,
            "completed_utc": now(),
            "jobs": len(runs),
            "files": {
                file.name: {"sha256": sha256(file), "bytes": file.stat().st_size} for file in files
            },
        },
    )
    return [*files, manifest]
