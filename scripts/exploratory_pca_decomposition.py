"""Separate graph-mean variance, feature scale and decoder performance."""

import argparse
import subprocess
from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from exploratory_decoder_topology import drives_for, extract, graph_inputs, graphs
from exploratory_gnn_readout import fit
from exploratory_pooling_control import samples_for
from exploratory_readability import ROOT, SPLITS, read_plan, real_projection
from torch import nn

from mbtopology import runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.runtime import atomic_json, now, read_json, require, sha256

PLAN = ROOT / "configs/exploratory_pca_decomposition.json"
ARMS = ("raw_shared", "raw_shared_local_scale", "centered_shared", "per_graph_pca")
LABELS = (
    "Pooled PCA\npooled scale",
    "Pooled PCA\nlocal scale",
    "Within-graph PCA\nlocal scale",
    "Per-graph PCA\nlocal scale",
)


def basis(centered):
    _, d, axes = np.linalg.svd(centered, full_matrices=False)
    require(d[1] > 0, "training data rank below two")
    return axes[:2].T


def project(panels, train_mean, axes):
    return {k: (panels[k] - train_mean) @ axes for k in SPLITS}


def decompose(native, members):
    selected = np.r_[members["avoidance_indices"], members["approach_indices"]]
    require(len(np.unique(selected)) == 91, "MBON cell inventory changed")
    values = {g: {k: v[:, selected] for k, v in row.items()} for g, row in native.items()}
    train = np.stack([panels["train"] for panels in values.values()])
    centers = train.mean(1)
    grand = train.mean((0, 1))
    raw_delta = train - grand
    within = train - centers[:, None, :]
    offsets = np.broadcast_to(centers[:, None, :] - grand, train.shape)
    require(
        np.isclose(
            np.square(raw_delta).sum(),
            np.square(within).sum() + np.square(offsets).sum(),
            rtol=1e-12,
        ),
        "within- and between-graph variance do not reconcile",
    )
    raw_pcs = basis(raw_delta.reshape(-1, 91))
    within_pcs = basis(within.reshape(-1, 91))
    axis_fraction = np.square(offsets @ raw_pcs).sum((0, 1)) / np.square(raw_delta @ raw_pcs).sum(
        (0, 1)
    )
    fraction = float(np.square(offsets).sum() / np.square(raw_delta).sum())
    features = {}
    for graph, center, centered_train in zip(values, centers, within, strict=True):
        raw = project(values[graph], grand, raw_pcs)
        features[graph] = {
            "raw_shared": raw,
            "raw_shared_local_scale": raw,
            "centered_shared": project(values[graph], center, within_pcs),
            "per_graph_pca": project(values[graph], center, basis(centered_train)),
        }
    return features, {
        "total_between_fraction": fraction,
        "between_fraction_raw_pc1_pc2": axis_fraction.tolist(),
    }


def pooled_scaled(panels, training):
    center, scale = training.mean(0), training.std(0)
    require(np.all(scale > 1e-8), "pooled PCA scale is degenerate")
    result = {}
    for key, array in panels.items():
        standardized = torch.from_numpy(((array - center) / scale).astype(np.float32))
        result[key] = standardized, torch.tensor([1.0, 0.0] * (len(array) // 2))
    return result


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = monotonic()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    plan, (old, cfg) = read_json(PLAN), read_plan()
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    paths = graphs(bundle, plan)
    data = {g: graph_inputs(bundle, path) for g, path in paths.items()}
    _, _, _, projection = real_projection(old["source"])
    members = read_json(ROOT / "configs/mbon_readout.json")
    args.output.mkdir(parents=True, exist_ok=False)
    records, decompositions, outputs = [], [], []
    for index, seed in enumerate(plan["seeds"]):
        t = monotonic()
        drives = drives_for(projection, cfg, plan, seed)
        model = make_benchmark_model("GraphSAGE", bundle["models"], seed, aggregation="max").eval()
        native = {
            g: extract(model, inputs, drives, 2 * cfg["pairs_per_trial"], index == 0)
            for g, inputs in data.items()
        }
        formats, audit = decompose(native, members)
        decompositions.append({"seed": seed, **audit})
        pooled = np.concatenate([panels["raw_shared"]["train"] for panels in formats.values()])
        rows = []
        for graph, arms in formats.items():
            for arm, panels in arms.items():
                samples = (
                    pooled_scaled(panels, pooled)
                    if arm == "raw_shared"
                    else samples_for(panels, 1e-8)[0]
                )
                decoder = nn.Sequential(nn.Linear(2, 1), nn.Flatten(0))
                result = fit(decoder, samples, plan)
                require(result["parameters"] == 3, "matched decoder capacity changed")
                rows.append(
                    {
                        "seed": seed,
                        "graph": graph,
                        "arm": arm,
                        "selected_epoch": result["selected_epoch"],
                        **result["selected"]["heldout"],
                    }
                )
        path = args.output / f"seed-{seed}.json"
        atomic_json(path, {"seed": seed, "variance": audit, "rows": rows})
        outputs.append(path)
        records.extend(rows)
        print(f"PCA_DECOMPOSITION_SEED {seed} {len(rows)} fits {monotonic() - t:.1f}s", flush=True)
    df = pl.DataFrame(records)
    require(df.height == 36, "decomposition fit inventory differs")
    dest = args.output / "heldout.parquet"
    df.write_parquet(dest)
    outputs.append(dest)
    figure = args.output / "pca-decomposition.png"
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    for n, q in enumerate(decompositions):
        fraction = [q["total_between_fraction"], *q["between_fraction_raw_pc1_pc2"]]
        axes[0].scatter(
            np.arange(3) + (n - 1) * 0.08, fraction, s=27, color="#286578", label=str(q["seed"])
        )
    axes[0].set(
        xticks=range(3),
        xticklabels=("All features", "Pooled PC1", "Pooled PC2"),
        ylim=(0, 1.03),
        ylabel="Between-graph fraction",
        title="Clean train variance",
    )
    real = df.filter(pl.col("graph") == "real")
    for seed in plan["seeds"]:
        values = {
            q["arm"]: q["balanced_accuracy"]
            for q in real.filter(pl.col("seed") == seed).iter_rows(named=True)
        }
        axes[1].plot(range(4), [values[a] for a in ARMS], "o-", alpha=0.65, color="#91462e")
    axes[1].axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    axes[1].set(
        xticks=range(4),
        xticklabels=LABELS,
        ylabel="Held-out balanced accuracy",
        title="Real graph · paired seeds",
        ylim=(0.35, 1.03),
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure, dpi=190, facecolor="white")
    plt.close(fig)
    outputs.append(figure)
    report = args.output / "REPORT.md"
    lines = [
        "# Exploratory PCA decomposition on three fresh seeds",
        "",
        (
            "The raw pooled basis uses the three graphs' clean training features. The within-graph basis uses the same features after subtracting each graph's training mean, then shares the resulting axes across graphs. "
            "The scale control keeps the raw pooled axes and uses per-graph training centering and scaling. The per-graph positive control fits independent bases. Every classifier has two slopes and a bias."
        ),
        "",
        "| Seed | Between-graph share of total variance | Of PC1 | Of PC2 |",
        "|---:|---:|---:|---:|",
    ]
    for q in decompositions:
        f, g = q["between_fraction_raw_pc1_pc2"]
        lines.append(f"| {q['seed']} | {q['total_between_fraction']:.2%} | {f:.2%} | {g:.2%} |")
    lines += [
        "",
        "| Graph | Basis and scale | Median held-out BA [range] | Median AUROC |",
        "|---|---|---:|---:|",
    ]
    for graph in paths:
        for arm in ARMS:
            q = df.filter((pl.col("graph") == graph) & (pl.col("arm") == arm))
            a = q["balanced_accuracy"]
            lines.append(
                f"| {graph} | {arm} | {a.median():.3f} [{a.min():.3f}, {a.max():.3f}] | {q['auroc'].median():.3f} |"
            )
    lines += [
        "",
        "![Variance decomposition and held-out performance](pca-decomposition.png)",
        "",
        (
            "This is a follow-up motivated by the first interaction experiment. All three new seeds were held out from that experiment. "
            "Only the benchmark's clean cue instances and the frozen real and two sampled null graphs enter this test. Graph-specific training means and variances are available at fitting time. "
            "The outcome concerns a mechanism in the pooled decoder, conditional on these graphs and cue families."
        ),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    outputs.append(report)
    atomic_json(
        args.output / "MANIFEST.json",
        {
            "scope": plan["scope"],
            "config_sha256": sha256(PLAN),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "completed_utc": now(),
            "wall_seconds": monotonic() - start,
            "graph_manifests_sha256": {g: sha256(ROOT / p) for g, p in paths.items()},
            "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs},
        },
    )
    print(f"PCA_DECOMPOSITION_COMPLETE {len(records)} fits {monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    run()
