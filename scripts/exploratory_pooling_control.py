"""Compare grouped, PCA-2 and per-neuron readouts on five frozen real backbones."""

import argparse
import subprocess
from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from exploratory_gnn_readout import fit
from exploratory_readability import ROOT, SPLITS, make_splits, read_plan, real_projection
from exploratory_rule_models import mb_embeddings
from torch import nn

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.runtime import atomic_json, now, read_json, require, sha256

PLAN = ROOT / "configs/exploratory_pooling_control.json"
ARMS = ("group_means", "pca_2", "per_mbon")
LABELS = ("Group means\n2 features", "Train-only PCA\n2 features", "Per MBON\n91 features")


@torch.inference_mode()
def extract_scores(model, inputs, drives, presentations):
    result = {}
    for key, drive in drives.items():
        batches = []
        for t in range(drive.shape[0] // presentations):
            h = mb_embeddings(model, inputs, drive, t, presentations)
            y = torch.sigmoid(model.head(h)).squeeze(-1)
            batches.append(y.cpu().numpy())
        result[key] = np.concatenate(batches).astype(np.float64)
    return result


def representations(scores, membership):
    a, b = (np.array(membership[key]) for key in ("avoidance_indices", "approach_indices"))
    selected = np.r_[a, b]
    require(len(np.unique(selected)) == len(selected) == 91, "MBON group inventory differs")
    cells = {key: scores[key][:, selected] for key in SPLITS}
    centre = cells["train"].mean(0)
    _, singular, directions = np.linalg.svd(cells["train"] - centre, full_matrices=False)
    require(singular[1] > 0, "two-dimensional train PCA is rank deficient")
    return {
        "group_means": {
            key: np.column_stack((scores[key][:, a].mean(1), scores[key][:, b].mean(1)))
            for key in SPLITS
        },
        "pca_2": {key: (cells[key] - centre) @ directions[:2].T for key in SPLITS},
        "per_mbon": cells,
    }, float((singular[:2] ** 2).sum() / (singular**2).sum())


def samples_for(x, floor):
    centre = x["train"].mean(0)
    std = x["train"].std(0)
    active = std > floor
    scale = np.where(active, std, 1.0)
    samples = {}
    for key in SPLITS:
        z = (x[key] - centre) / scale
        z[:, ~active] = 0.0
        y = torch.tensor([1.0, 0.0] * (len(z) // 2))
        samples[key] = torch.from_numpy(z.astype(np.float32)), y
    return samples, int(active.sum())


def plot(frame, seeds, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True, layout="constrained")
    for ax, field, title in zip(
        axes, ("balanced_accuracy", "auroc"), ("Balanced accuracy", "AUROC"), strict=True
    ):
        for seed in seeds:
            by_arm = {
                r["arm"]: r[field]
                for r in frame.filter(pl.col("seed") == seed).iter_rows(named=True)
            }
            ax.plot(
                range(3),
                [by_arm[arm] for arm in ARMS],
                "o-",
                color="#607889",
                alpha=0.48,
                markersize=4,
            )
        summary = {
            r["arm"]: r[field]
            for r in frame.group_by("arm").agg(pl.col(field).median()).iter_rows(named=True)
        }
        ax.scatter(
            range(3), [summary[arm] for arm in ARMS], color="#923f25", marker="D", s=30, zorder=3
        )
        ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
        ax.set(ylim=(0.43, 1.025), xticks=range(3), xticklabels=LABELS, title=title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def interval(values):
    return f"{values.median():.4f} [{values.min():.4f}, {values.max():.4f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    started = monotonic()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    plan = read_json(PLAN)
    old, config = read_plan()
    seeds = plan["seeds"]
    require(
        len(seeds) == len(set(seeds)) == 5 and min(seeds) > 4412, "fresh seed inventory differs"
    )
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    _, _, _, projection = real_projection(old["source"])
    inputs = gnn_training.frozen_inputs(
        ROOT, bundle["task"], old["source"], 0, bundle["nulls"], bundle["models"], args.device
    )
    inputs["base"] = models.node_features(inputs["roles"], inputs["kc"], inputs["kc"][:0])
    edges = inputs["graph"].edge_index
    inputs["neighbors"] = edges[:, edges[0] != edges[1]]
    member_path = ROOT / "configs/mbon_readout.json"
    membership = read_json(member_path)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, files = [], []
    for seed in seeds:
        bounded = monotonic()
        drives = make_splits(projection, config, {**plan, "seed": seed})
        backbone = make_benchmark_model(
            "GraphSAGE", bundle["models"], seed, device=args.device, aggregation="max"
        ).eval()
        native = extract_scores(backbone, inputs, drives, 2 * config["pairs_per_trial"])
        formats, variance = representations(native, membership)
        extracted = monotonic() - bounded
        seed_report = {
            "seed": seed,
            "samples": {k: len(v) for k, v in native.items()},
            "exact_disjoint": True,
            "extract_seconds": extracted,
            "pca_2_train_variance_fraction": variance,
            "arms": {},
        }
        for arm in ARMS:
            data, active = samples_for(formats[arm], plan["standardization_floor"])
            dim = data["train"][0].shape[1]
            model = nn.Sequential(nn.Linear(dim, 1), nn.Flatten(0))
            result = fit(model, data, plan)
            seed_report["arms"][arm] = {"active_features": active, **result}
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "features": dim,
                    "active_features": active,
                    "parameters": result["parameters"],
                    "selected_epoch": result["selected_epoch"],
                    **result["selected"]["heldout"],
                }
            )
        path = args.output / f"seed-{seed}.json"
        atomic_json(path, seed_report)
        files.append(path)
        print(f"POOLING_SEED {seed} 3 arms {monotonic() - bounded:.1f}s", flush=True)
    frame = pl.DataFrame(rows)
    require(frame.height == 15, "pooling run inventory incomplete")
    out, fig, md = (args.output / p for p in ("heldout.parquet", "pooling.png", "REPORT.md"))
    frame.write_parquet(out)
    plot(frame, seeds, fig)
    lines = [
        "# Exploratory frozen-MBON pooling control",
        "",
        (
            "Five fresh, paired initialization and cue seeds on one real graph. Both two-coordinate representations and the per-neuron representation use exactly the same 91 MBONs. "
            "PCA and scale estimates use only training samples. All heads use Adam/BCE with the same budget and threshold. "
            "Validation picks an epoch and held-out is scored once at that epoch."
        ),
        "",
        "| Head | Parameters | Median held-out balanced accuracy [range] | Median held-out AUROC [range] |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        d = frame.filter(pl.col("arm") == arm)
        lines.append(
            f"| {arm} | {d['parameters'][0]} | {interval(d['balanced_accuracy'])} | {interval(d['auroc'])} |"
        )
    lines += [
        "",
        "![Paired results by seed](pooling.png)",
        "",
        (
            "Thin lines pair seeds. Diamonds show medians. "
            "Each head uses 128 train, 64 validation and 128 held-out cues. Fresh cue instances share the previously frozen family layout."
        ),
        "",
        "[Manifest](MANIFEST.json)",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    files.extend((out, fig, md))
    atomic_json(
        args.output / "MANIFEST.json",
        {
            "scope": plan["scope"],
            "config_sha256": sha256(PLAN),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_manifest_sha256": sha256(ROOT / old["source"]),
            "membership_sha256": sha256(member_path),
            "binary_config_sha256": old["binary_config_sha256"],
            "jobs": frame.height,
            "device": args.device,
            "torch": str(torch.__version__),
            "wall_seconds": monotonic() - started,
            "completed_utc": now(),
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
        },
    )
    print(f"POOLING_COMPLETE {frame.height} fits {monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
