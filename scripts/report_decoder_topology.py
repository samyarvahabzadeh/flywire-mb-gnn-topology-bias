"""Audit and summarize the frozen readout comparison without refitting."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from exploratory_decoder_topology import PLAN

from mbtopology.runtime import atomic_json, read_json, require, sha256

GRAPHS = ("real", "A-00", "A-01", "B-00", "B-01")
ARMS = ("group", "pca")
MEASURES = ("balanced_accuracy", "auroc")


def summarize(table, plan):
    records = []
    for seed in plan["seeds"]:
        for noise in plan["noise_levels"]:
            current = table.filter((pl.col("seed") == seed) & (pl.col("noise") == noise))
            for field in MEASURES:
                values = {(r["graph"], r["arm"]): r[field] for r in current.iter_rows(named=True)}
                row = {"seed": seed, "noise": noise, "measure": field}
                for arm in ARMS:
                    real = values["real", arm]
                    null_a = np.mean([values[g, arm] for g in GRAPHS if g.startswith("A-")])
                    null_b = np.mean([values[g, arm] for g in GRAPHS if g.startswith("B-")])
                    row.update(
                        {
                            f"{arm}_real": real,
                            f"{arm}_null_a": float(null_a),
                            f"{arm}_null_b": float(null_b),
                            f"{arm}_null_pooled": float((null_a + null_b) / 2),
                        }
                    )
                for dest in ("null_a", "null_b", "null_pooled"):
                    row[f"interaction_{dest}"] = (row["pca_real"] - row[f"pca_{dest}"]) - (
                        row["group_real"] - row[f"group_{dest}"]
                    )
                records.append(row)
    return pl.DataFrame(records)


def plot(frame, plan, dest):
    material = frame.filter(pl.col("measure") == "balanced_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), layout="constrained")
    doses = np.array(plan["noise_levels"]) * 100
    floor = min(
        0.45,
        *(
            material[f"{arm}_{source}"].min() - 0.02
            for arm in ARMS
            for source in ("real", "null_a", "null_b")
        ),
    )
    for ax, arm, title in zip(axes[:2], ARMS, ("Group means", "Shared training PCA"), strict=True):
        for source, label, color in (
            ("real", "Real", "#963d24"),
            ("null_a", "Null-A", "#19617f"),
            ("null_b", "Null-B", "#697629"),
        ):
            column = f"{arm}_{source}"
            arrays = np.stack(
                [
                    material.filter(pl.col("seed") == seed).sort("noise")[column].to_numpy()
                    for seed in plan["seeds"]
                ]
            )
            q1, med, q3 = np.quantile(arrays, [0.25, 0.5, 0.75], axis=0)
            ax.plot(doses, med, "o-", color=color, label=label)
            ax.fill_between(doses, q1, q3, color=color, alpha=0.14)
        ax.set(ylim=(floor, 1.03), title=title, xticks=doses, xlabel="PN replacements (%)")
        ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Held-out balanced accuracy")
    axes[1].legend(frameon=False, loc="best", fontsize=8)
    selected = material.filter(pl.col("noise") == plan["primary_noise"]).sort("seed")
    x = np.arange(selected.height)
    axes[2].scatter(x, selected["interaction_null_pooled"].to_numpy() * 100, color="#245d70", s=40)
    axes[2].axhline(0, color="#888888", linestyle=":", linewidth=1)
    axes[2].set(
        title="Paired interaction at 30%",
        xticks=x,
        xticklabels=selected["seed"].to_list(),
        xlabel="Seed",
        ylabel="Change in real − null gap (pp)",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory · frozen decoders across topologies")
    fig.savefig(dest, dpi=190, facecolor="white")
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    base = args.input
    manifest = read_json(base / "MANIFEST.json")
    require(sha256(PLAN) == manifest["config_sha256"], "frozen decoder plan differs")
    for name, f in manifest["files"].items():
        require(
            sha256(base / name) == f["sha256"] and (base / name).stat().st_size == f["bytes"],
            f"file differs: {name}",
        )
    plan = read_json(PLAN)
    table = pl.read_parquet(base / "heldout.parquet")
    expected = {
        (s, g, a, n)
        for s in plan["seeds"]
        for g in GRAPHS
        for a in ARMS
        for n in plan["noise_levels"]
    }
    require(
        len(table) == len(expected)
        and set(table.select("seed", "graph", "arm", "noise").iter_rows()) == expected,
        "run inventory is incomplete",
    )
    results = summarize(table, plan)
    folder = base / "report"
    folder.mkdir(exist_ok=False)
    fig, data, md = (
        folder / n for n in ("decoder-topology.png", "paired-contrasts.parquet", "REPORT.md")
    )
    results.write_parquet(data)
    plot(results, plan, fig)
    lines = [
        "# Exploratory test of decoder-by-topology dependence",
        "",
        (
            "Four paired seeds, one real graph and two instances per degree-preserving variant. The two-column decoders see the same 91 MBONs and fit three classifier parameters. "
            "PCA and standardization are shared across topologies within a seed and use only clean training inputs. Validation selects the epoch. No noisy panel is used for fitting or selection."
        ),
        "",
        (
            "The interaction is (real minus the mean of four nulls under PCA) minus (real minus that mean under group pooling). "
            "Positive means that switching to PCA moved the real-relative-to-null contrast upward. The predeclared primary contrast uses balanced accuracy at 30% replacement."
        ),
        "",
        "| PN replacement | Group real | Group mean of nulls | PCA real | PCA mean of nulls | Interaction (pp) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for noise in plan["noise_levels"]:
        group = results.filter(
            (pl.col("measure") == "balanced_accuracy") & (pl.col("noise") == noise)
        )
        medians = [
            group[k].median()
            for k in (
                "group_real",
                "group_null_pooled",
                "pca_real",
                "pca_null_pooled",
                "interaction_null_pooled",
            )
        ]
        a, b, c, d, interaction = medians
        lines.append(
            f"| {noise:.0%} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {100 * interaction:+.2f} |"
        )
    lines += [
        "",
        (
            "Cells show medians over paired seeds. Each null value first averages its four null graph runs within a seed. "
            "The interaction column first computes the contrast within seed."
        ),
        "",
        "| Seed | 30% interaction vs Null-A (pp) | vs Null-B (pp) | vs all nulls (pp) |",
        "|---:|---:|---:|---:|",
    ]
    primary = results.filter(
        (pl.col("measure") == "balanced_accuracy") & (pl.col("noise") == plan["primary_noise"])
    )
    for row in primary.iter_rows(named=True):
        lines.append(
            f"| {row['seed']} | {100 * row['interaction_null_a']:+.2f} | {100 * row['interaction_null_b']:+.2f} | {100 * row['interaction_null_pooled']:+.2f} |"
        )
    lines += [
        "",
        "![Decoding under PN corruption and matched interaction](decoder-topology.png)",
        "",
        (
            "This result is conditional on one biological graph, four null graphs and a fixed cue-family layout. The corrupted test panels share the same held-out base cues. "
            "The analysis concerns frozen feature accessibility. It supplies no trajectory of end-to-end training speed."
        ),
        "",
        f"Execution wall time was {manifest['wall_seconds']:.1f} s. The paired AUROC and both variant contrasts are in [paired-contrasts.parquet](paired-contrasts.parquet).",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    atomic_json(
        folder / "MANIFEST.json",
        {
            "scope": manifest["scope"],
            "run_manifest_sha256": sha256(base / "MANIFEST.json"),
            "config_sha256": manifest["config_sha256"],
            "git_commit": manifest["git_commit"],
            "report_script_sha256": sha256(Path(__file__)),
            "files": {
                p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (md, data, fig)
            },
        },
    )


if __name__ == "__main__":
    run()
