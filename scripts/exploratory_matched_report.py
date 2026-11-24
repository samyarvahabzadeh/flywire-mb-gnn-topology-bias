"""Summarize the fixed-horizon exploratory edge-rule comparison."""

from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from exploratory_matched_edges import METHODS, verified_stage, write_manifest

from mbtopology.runtime import require, sha256


def plot_cycles(frame, graphs, plan, path):
    fig, axes = plt.subplots(
        2, 4, figsize=(12, 6.3), sharex=True, sharey="row", layout="constrained"
    )
    checkpoints = plan["checkpoints"]
    steps = np.arange(len(checkpoints))
    fields = ("heldout_score_gap", "heldout_balanced_accuracy")
    for i, graph in enumerate(graphs):
        axes[0, i].set_title("Real" if graph == "real" else graph)
        for j, field in enumerate(fields):
            ax = axes[j, i]
            ax.axhline((0, 0.5)[j], color="#888888", linestyle=":", linewidth=1)
            for rule, color in zip(METHODS, ("#216880", "#9f4026"), strict=True):
                data = frame.filter((pl.col("graph") == graph) & (pl.col("rule") == rule))
                values = np.stack(
                    [
                        data.filter(pl.col("seed") == s).sort("pairs")[field].to_numpy()
                        for s in plan["seeds"]
                    ]
                )
                low, middle, high = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
                ax.plot(steps, middle, color=color, marker="o", markersize=3, label=rule)
                ax.fill_between(steps, low, high, color=color, alpha=0.15, linewidth=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(steps, checkpoints)
    axes[0, 0].set_ylabel("Mean held-out CS+ − CS− score")
    axes[1, 0].set_ylabel("Held-out balanced accuracy")
    for ax in axes[1]:
        ax.set_xlabel("Training cue pairs")
    axes[0, -1].legend(frameon=False)
    fig.suptitle("Exploratory · matched KCγ→MBON11 learning rules")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def report(root, plan):
    started = monotonic()
    stages = {stage: verified_stage(root / stage) for stage in ("real", "controls")}
    frame = pl.concat([pl.read_parquet(root / stage / "checkpoints.parquet") for stage in stages])
    graphs = ["real", *(f"input-{seed}" for seed in plan["control_seeds"])]
    expected = {
        (g, s, r, t)
        for g in graphs
        for s in plan["seeds"]
        for r in METHODS
        for t in plan["checkpoints"]
    }
    require(
        set(frame.select("graph", "seed", "rule", "pairs").iter_rows()) == expected
        and len(frame) == len(expected),
        "run inventory differs",
    )
    require(
        frame.filter(pl.col("train_presentations") != 2 * pl.col("pairs")).is_empty(),
        "cue counts differ",
    )
    require(
        frame.group_by("graph", "seed", "pairs")
        .agg(pl.col("heldout_threshold").n_unique())["heldout_threshold"]
        .max()
        == 1,
        "compared arms used different thresholds",
    )
    for col in ("train_cue_sha256", "heldout_cue_sha256"):
        require(
            frame.group_by("seed").agg(pl.col(col).n_unique())[col].max() == 1,
            "cue schedule differs across graphs or arms",
        )
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=False)
    rows = output / "all-checkpoints.parquet"
    frame.sort("graph", "seed", "rule", "pairs").write_parquet(rows)
    group = (
        frame.group_by("graph", "rule", "pairs")
        .agg(
            pl.len().alias("seeds"),
            pl.col(
                "heldout_balanced_accuracy",
                "heldout_score_gap",
                "heldout_auroc",
                "heldout_threshold",
                "cumulative_train_seconds",
                "cumulative_heldout_probe_seconds",
            ).median(),
        )
        .sort("graph", "rule", "pairs")
    )
    summary = output / "median-checkpoints.csv"
    group.write_csv(summary)
    figure = output / "learning-curves.png"
    plot_cycles(frame, graphs, plan, figure)
    lines = [
        "# Exploratory matched edge rules",
        "",
        (
            "40 jobs. Each job saw 96 training cue presentations in 48 fixed pairs and 256 held-out cues at each of eight probes. "
            "The same held-out panel was evaluated repeatedly without parameter updates. The two laws share the numeric threshold within each graph, seed and checkpoint. "
            "It is the mean of their four fixed training-pair scores. AUROC and the signed score gap do not depend on this threshold."
        ),
        "",
        "## After 48 pairs",
        "",
        "Values are medians over the five paired seeds. Time columns cover 48 training pairs and all eight held-out probes per job, respectively.",
        "",
        "| Input set | Rule | Balanced accuracy | A+ minus A− score | AUROC | Train ms | Probe ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for graph in graphs:
        for row in group.filter((pl.col("graph") == graph) & (pl.col("pairs") == 48)).iter_rows(
            named=True
        ):
            lines.append(
                f"| {graph} | {row['rule']} | {row['heldout_balanced_accuracy']:.4f} | {row['heldout_score_gap']:+.5f} | "
                f"{row['heldout_auroc']:.4f} | {1000 * row['cumulative_train_seconds']:.2f} | "
                f"{1000 * row['cumulative_heldout_probe_seconds']:.2f} |"
            )
    lines += [
        "",
        (
            f"Measured CPU wall time was {stages['real']['wall_seconds']:.3f} s for the real stage and "
            f"{stages['controls']['wall_seconds']:.3f} s for the 30 input-control jobs, including input preparation."
        ),
        "",
        "![Held-out learning curves](learning-curves.png)",
        "",
        (
            "The figure displays medians and interquartile bands over seeds. The three input sets sample eligible KCs and preserve the modeled plastic capacity. "
            "Many source inputs overlap with real. Synthetic contacts have unknown fine compartment identity. Adam uses both labels and global BCE gradients. "
            "The local update uses the presynaptic activity and the A+ reinforcement pulse. Their learning rates also differ. "
            "The evidence is descriptive and conditional on this cue family, these seeds and these input sets."
        ),
        "",
        "[All checkpoint rows](all-checkpoints.parquet) · [Medians at every checkpoint](median-checkpoints.csv) · [Manifest](MANIFEST.json)",
        "",
    ]
    md = output / "REPORT.md"
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_manifest(
        output,
        plan,
        {
            "jobs": 40,
            "rows": frame.height,
            "wall_seconds": monotonic() - started,
            "stage_manifests_sha256": {key: sha256(root / key / "MANIFEST.json") for key in stages},
        },
        (rows, summary, figure, md),
    )
