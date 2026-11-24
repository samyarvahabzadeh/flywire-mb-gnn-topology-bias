"""Verify and summarize the frozen 65-job role-stratified Molab experiment."""

import argparse
import io
import json
from itertools import product
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import polars as pl
from audit_crossed_layer_bank import compare_frame, digest, endpoints_for, read_archive

from mbtopology.runtime import atomic_file, atomic_json, now, read_json, require, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/exploratory_role_layers.json"


def audit(archive, output):
    plan = read_json(PLAN)
    files = read_archive(
        archive, ("traces.parquet", "paired.parquet", "donor_medians.parquet", "MANIFEST.json")
    )
    run = json.loads(files["MANIFEST.json"])
    require(
        run["scope"] == plan["scope"]
        and run["config_sha256"] == sha256(PLAN)
        and run["git_commit"] == "d160b3a62bd7ce1b482feba393a57ce91f42c9d1"
        and run["jobs"] == 65,
        "unexpected run provenance",
    )
    for name in ("traces.parquet", "paired.parquet", "donor_medians.parquet"):
        require(digest(files[name]) == run["files"][name], f"hash mismatch for {name}")
    donors, seeds = plan["donors"], plan["seeds"]
    arms = {"RR"} | {f"{k}-{d}-{p}" for k, d, p in product("BS", donors, ("NR", "RN"))}
    traces = pl.read_parquet(io.BytesIO(files["traces.parquet"]))
    scores = endpoints_for(traces, set(product(seeds, arms)), plan["trials"], 0.75, 100)
    paired = {}
    for seed, donor in product(seeds, donors):
        row = {"rr": scores[seed, "RR"]["auc"]}
        for k in "BS":
            for p in ("NR", "RN"):
                row[f"{k}_{p}"] = scores[seed, f"{k}-{donor}-{p}"]["auc"]
            row[f"{k}_gap_pp"] = 100 * (row[f"{k}_NR"] - row[f"{k}_RN"])
        row["attenuation_pp"] = row["B_gap_pp"] - row["S_gap_pp"]
        paired[seed, donor] = row
    compare_frame(paired, pl.read_parquet(io.BytesIO(files["paired.parquet"])), ["seed", "donor"])
    fields = ("B_gap_pp", "S_gap_pp", "attenuation_pp")
    donors_table = {
        (d,): {f: median(paired[s, d][f] for s in seeds) for f in fields} for d in donors
    }
    compare_frame(
        donors_table, pl.read_parquet(io.BytesIO(files["donor_medians.parquet"])), ["donor"]
    )
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for d in donors:
        row = donors_table[d,]
        ax.plot([0, 1], [row["B_gap_pp"], row["S_gap_pp"]], marker="o", label=str(d))
    ax.axhline(0, ls="--", lw=1, color="grey")
    ax.set_xticks([0, 1], ["Signed degrees", "Signed degrees + cell roles"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("First-layer minus second-layer gap (pp)")
    ax.set_title("Held-out discrimination over 768 trials")
    ax.legend(title="Rewiring seed")
    fig.tight_layout()
    chart = output / "role-layer-gaps.png"
    atomic_file(chart, lambda p: fig.savefig(p, dpi=180))
    plt.close(fig)
    values = [median(r[f] for r in donors_table.values()) for f in fields]
    lines = [
        "# Cell-role constraints and the GraphSAGE layer effect",
        "",
        f"Exploratory Molab run on {run['gpu']}. The independent replay verified {len(scores)} jobs and {traces.height:,} held-out trial rows.",
        "",
        "Positive gaps favor rewiring the first layer. Attenuation is the signed-degree gap minus the role-constrained gap, computed within each seed and donor before taking medians. All contrasts use percentage points.",
        "",
        "| Rewiring seed | Signed-degree gap | Role-constrained gap | Paired attenuation |",
        "|---|---:|---:|---:|",
    ]
    for (donor,), r in donors_table.items():
        lines.append(
            f"| {donor} | {r['B_gap_pp']:+.2f} | {r['S_gap_pp']:+.2f} | {r['attenuation_pp']:+.2f} |"
        )
    lines += [
        f"| Median | {values[0]:+.2f} | {values[1]:+.2f} | {values[2]:+.2f} |",
        "",
        (
            f"The paired gap decreased in {sum(r['attenuation_pp'] > 0 for r in paired.values())} "
            f"of {len(paired)} seed-by-donor pairs. The donor-level paired median decreased in "
            f"{sum(r['attenuation_pp'] > 0 for r in donors_table.values())} of {len(donors)} donors."
        ),
        "",
        "The table below reports differences from the real/real model, paired within seed and then summarized by donor median.",
        "",
        "| Rewiring seed | Signed first | Signed second | Role-constrained first | Role-constrained second |",
        "|---|---:|---:|---:|---:|",
    ]
    for donor in donors:
        deltas = [
            median(100 * (paired[seed, donor][arm] - paired[seed, donor]["rr"]) for seed in seeds)
            for arm in ("B_NR", "B_RN", "S_NR", "S_RN")
        ]
        formatted = " | ".join(f"{delta:+.2f}" for delta in deltas)
        lines.append(f"| {donor} | {formatted} |")
    lines += [
        "",
        "Each donor shares the same five cue and initialization seeds. GraphSAGE-max uses unsigned neighbor identities for message passing. Signs constrain the rewiring. PN-to-KC input is fixed. The 15 comparisons share one connectome.",
        "",
        "The restricted chains also changed fewer original edges. The experiment estimates the contrast between these two control ensembles, including this difference in edge overlap.",
        "",
        "| Rewiring seed | Edges replaced, signed-degree | Edges replaced, role-constrained |",
        "|---|---:|---:|",
    ]
    for d in donors:
        pct = [100 * (1 - run["certificates"][f"{k}-{d}"]["edge_overlap_fraction"]) for k in "BS"]
        lines.append(f"| {d} | {pct[0]:.2f}% | {pct[1]:.2f}% |")
    lines += ["", "![Layer placement gaps across paired donors](role-layer-gaps.png)", ""]
    report = output / "REPORT.md"
    atomic_file(report, lambda p: p.write_text("\n".join(lines), encoding="utf-8"))
    atomic_json(
        output / "AUDIT_MANIFEST.json",
        {
            "completed_utc": now(),
            "archive_sha256": sha256(archive),
            "git_commit": run["git_commit"],
            "config_sha256": sha256(PLAN),
            "audit_sha256": sha256(__file__),
            "files": {p.name: sha256(p) for p in (report, chart)},
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.archive, args.output)
