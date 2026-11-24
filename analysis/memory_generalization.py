"""Audit the frozen aversive-memory experiment and produce its generalization figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mbtopology.runtime import atomic_json, now, read_json, require, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/exploratory_memory_generalization.json"


def analyze(output):
    manifest = read_json(output / "MANIFEST.json")
    plan = read_json(PLAN)
    require(manifest["config_sha256"] == sha256(PLAN), "analysis plan hash differs")
    for name, item in manifest["files"].items():
        require(sha256(output / name) == item["sha256"], "output hash differs")
    t = pl.read_parquet(output / "probes.parquet")
    expected = plan["n_seeds"] * (1 + plan["n_controls"]) * 2
    keys = ["seed", "mask", "rule"]
    require(t.select(*keys).unique().height == expected == manifest["jobs"], "job count differs")
    counts = t.group_by(*keys, "shared_PNs").agg(pl.len(), pl.col("probe").n_unique())
    require(
        counts.height == expected * len(plan["shared_pns"])
        and (counts["len"] == plan["probes_per_level"]).all()
        and (counts["probe"] == plan["probes_per_level"]).all(),
        "probe count differs",
    )
    require(
        t["fixed_budget_change"].is_finite().all()
        and t["matched_transfer"].drop_nulls().is_finite().all(),
        "nonfinite responses",
    )
    matches = t.select(*keys, "matched_by_pair").unique()
    misses = matches.filter(pl.col("matched_by_pair").is_null()).sort(*keys)
    misses.write_csv(output / "unmatched-jobs.csv")
    per_level = t.group_by(*keys, "shared_PNs").agg(
        pl.col("matched_transfer", "fixed_budget_change").mean()
    )
    by_seed = (
        per_level.with_columns(
            pl.when(pl.col("mask") == "real")
            .then(pl.lit("Real"))
            .otherwise(pl.lit("Input control"))
            .alias("wiring")
        )
        .group_by("seed", "rule", "shared_PNs", "wiring")
        .agg(pl.col("matched_transfer", "fixed_budget_change").median())
    )
    by_seed.write_parquet(output / "seed-gradients.parquet")
    report = [
        "# Aversive memory generalization in the KCγ→MBON11 model",
        "",
        (
            f"Exploratory. Verified {expected} jobs and {t.height:,} novel-probe rows. "
            f"{misses.height} jobs did not reach the training-only CS+ match by 48 pairs."
        ),
        "",
        (
            "The probes varied in the number of modeled active PNs shared with the reinforced cue. "
            "Each stimulus is a simulated activation pattern over anatomically identified PNs. "
            "The model includes the bilateral KCγ-to-MBON11 circuit."
        ),
        "",
        "| Measure | Complete seeds | Local real − control | Adam real − control | Interaction | Positive seed interactions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for metric, label, ax in zip(
        ("matched_transfer", "fixed_budget_change"),
        ("Matched CS+ strength", "Fixed budget, 48 pairs"),
        axes,
        strict=True,
    ):
        # A matched seed must supply all 13 masks under both rules.
        complete = (
            matches.filter(pl.col("matched_by_pair").is_not_null())
            .group_by("seed")
            .len()
            .filter(pl.col("len") == (1 + plan["n_controls"]) * 2)["seed"]
        )
        table = (
            by_seed
            if metric == "fixed_budget_change"
            else by_seed.filter(pl.col("seed").is_in(complete.to_list()))
        )
        if not table.height:
            report.append(f"| {label} | 0 | — | — | — | — |")
            ax.set_title(label)
            continue
        # Pair at the level of each control mask. The null median is taken after near-minus-far.
        raw = per_level.filter(pl.col("seed").is_in(table["seed"].unique().to_list()))
        # Both endpoint differences and the selectivity contrast are paired by seed.
        ends = (
            raw.filter(pl.col("shared_PNs").is_in([0, 36]))
            .pivot(on="shared_PNs", index=keys, values=metric)
            .with_columns((pl.col("36") - pl.col("0")).alias("selectivity"))
        )
        outcomes = {}
        for k in ("0", "36", "selectivity"):
            real = ends.filter(pl.col("mask") == "real").select(
                "seed", "rule", pl.col(k).alias("real")
            )
            null = (
                ends.filter(pl.col("mask") != "real")
                .group_by("seed", "rule")
                .agg(pl.col(k).median().alias("null"))
            )
            deltas = real.join(null, on=["seed", "rule"]).with_columns(
                (pl.col("real") - pl.col("null")).alias("delta")
            )
            paired = deltas.pivot(on="rule", index="seed", values="delta").sort("seed")
            outcomes[k] = paired.with_columns(
                (pl.col("Local") - pl.col("Adam")).alias("interaction")
            )
        for k, name in (("selectivity", label), ("36", label + " · near"), ("0", label + " · far")):
            d = outcomes[k]
            report.append(
                f"| {name} | {len(d)} | {d['Local'].median():+.5f} | {d['Adam'].median():+.5f} | "
                f"{d['interaction'].median():+.5f} | {(d['interaction'] > 0).sum()}/{len(d)} |"
            )
        outcomes["selectivity"].write_parquet(output / f"{metric}-interaction.parquet")
        for rule, color in (("Local", "#0e7167"), ("Adam", "#9a4725")):
            for wiring, line in (("Real", "-"), ("Input control", "--")):
                rows = table.filter((pl.col("rule") == rule) & (pl.col("wiring") == wiring))
                vals = np.array(
                    [
                        rows.filter(pl.col("shared_PNs") == x)[metric].to_numpy()
                        for x in plan["shared_pns"]
                    ]
                )
                lo, mid, hi = np.quantile(vals, [0.25, 0.5, 0.75], axis=1)
                ax.plot(
                    plan["shared_pns"],
                    mid,
                    line,
                    color=color,
                    marker="o",
                    label=f"{rule} · {wiring}",
                )
                ax.fill_between(plan["shared_pns"], lo, hi, color=color, alpha=0.09)
        ax.axhline(0, color=".65", lw=0.7)
        ax.set(
            title=label,
            xlabel="Shared active PNs (out of 40)",
            ylabel="Novel-cue response change"
            if metric == "fixed_budget_change"
            else "Fraction of CS+ response change",
        )
    axes[1].legend(fontsize=8)
    fig.savefig(output / "generalization.png", dpi=180)
    plt.close(fig)
    controls = [
        m for k, m in read_json(output / "input-masks.json")["masks"].items() if k != "real"
    ]
    overlap = [100 * m["overlap_with_real"] / m["total"] for m in controls]
    endpoints = (
        (
            by_seed.filter(pl.col("seed").is_in(complete.to_list()))
            .filter(pl.col("shared_PNs").is_in([0, 36]))
            .group_by("rule", "wiring", "shared_PNs")
            .agg(pl.col("matched_transfer").median())
            .pivot(on="shared_PNs", index=["rule", "wiring"], values="matched_transfer")
            .sort("rule", "wiring")
        )
        if len(complete)
        else None
    )
    if endpoints is not None:
        report += [
            "",
            "Matched-strength transfer as a percentage of the response change to the conditioned cue.",
            "",
            "| Rule | Wiring | Far probe (0/40) | Near probe (36/40) |",
            "|---|---|---:|---:|",
        ]
        report.extend(
            f"| {rule} | {wiring} | {100 * far:.2f}% | {100 * near:.2f}% |"
            for rule, wiring, far, near in endpoints.select("rule", "wiring", "0", "36").iter_rows()
        )
    report += [
        "",
        f"Input controls retained {min(overlap):.1f}–{max(overlap):.1f}% of real pairs.",
        (
            "Median contrasts are descriptive. The tests hold the PN→KC projection and readout fixed. "
            "Each seed shares the same 12 sampled controls and all jobs share one connectome."
        ),
        "",
        "![Held-out aversive memory generalization](generalization.png)",
        "",
        (
            "Hige and colleagues measured odor-specific depression and generalization in this circuit. "
            "The comparison here tests the contribution of the identity of the presynaptic partners "
            "across two learning rules. The normalized local curve is predicted by conditioned "
            "KC activity weighted by the probe's relative input to each MBON at the first update. "
            "[Hige et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/)"
        ),
        "",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    files = (
        path,
        *output.glob("*.png"),
        *output.glob("*interaction.parquet"),
        output / "seed-gradients.parquet",
        output / "unmatched-jobs.csv",
    )
    atomic_json(
        output / "ANALYSIS_MANIFEST.json",
        {
            "analyzed_utc": now(),
            "git_commit": manifest["git_commit"],
            "config_sha256": sha256(PLAN),
            "analysis_sha256": sha256(__file__),
            "source_manifest_sha256": sha256(output / "MANIFEST.json"),
            "files": {f.name: sha256(f) for f in files},
        },
    )
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
