"""Figures and descriptive tables for the exploratory mechanism analysis."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

CONDITIONS = (("real", "Real", "#8a321e"), ("A", "Null-A", "#1a6785"), ("B", "Null-B", "#58612a"))
METRICS = (
    ("spectral_gap", "Normalized Laplacian λ₂ · full graph"),
    ("largest_component_gap", "Normalized Laplacian λ₂ · largest component"),
    ("mean_directed_path", "Mean reachable directed path · hops"),
    ("clustering", "Mean local clustering"),
    ("modularity", "Louvain modularity"),
    ("reciprocity", "Directed arc reciprocity"),
)


def quantiles(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return np.quantile(array, [0.5, 0.25, 0.75]).tolist() if array.size else None


def cell(values, digits=2):
    if values is None:
        return "—"
    median, q1, q3 = values
    return f"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def rho(x, y):
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def metric_plot(grouped, centres, output):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.4), sharey=True, layout="constrained")
    for index, (ax, (key, name)) in enumerate(zip(axes.flat, METRICS, strict=True)):
        for condition, label, color in CONDITIONS:
            data, dots = grouped[condition], centres[condition]
            ax.scatter(data[key], data["ttc_display"], s=5, alpha=0.14, color=color, linewidths=0)
            ax.scatter(
                dots[key],
                dots["ttc_display"],
                s=24,
                color=color,
                alpha=0.95,
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
            censored = data.filter(pl.col("ttc_censored"))
            if censored.height:
                ax.scatter(
                    censored[key],
                    censored["ttc_display"],
                    s=34,
                    marker="^",
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.1,
                )
        ax.set_xlabel(name, fontsize=9)
        ax.grid(axis="y", alpha=0.24)
        ax.spines[["top", "right"]].set_visible(False)
        if index % 3 == 0:
            ax.set_ylabel("Trials to criterion")
        if (
            index == 0
            and grouped["real"][key][0] == 0
            and all(d[key].max() == 0 for d in grouped.values())
        ):
            ax.set_xticks([0])
            ax.set_xlim(-0.005, 0.005)
            ax.text(
                0.04,
                0.95,
                "All full graphs disconnected",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#474747",
            )
    axes.flat[-1].legend(loc="upper right", frameon=False)
    figure.suptitle("Exploratory · topology and learning time", fontsize=15)
    figure.savefig(output / "graph-metrics-vs-ttc.png", dpi=210, facecolor="white")
    plt.close(figure)


def dynamics_plot(grouped, output):
    panels = (
        ("final_window_sd", "Final-window discrimination SD"),
        ("first_crossing", "First trial at or above 0.75"),
        ("crossing_to_stop_or_cap", "Trials from first crossing to stable criterion"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), layout="constrained")
    rng = np.random.default_rng(20261119)
    for ax, (key, name) in zip(axes, panels, strict=True):
        for i, (condition, _, color) in enumerate(CONDITIONS):
            data = grouped[condition]
            if key == "crossing_to_stop_or_cap":
                data = data.filter(~pl.col("ttc_censored"))
            seeds = data.group_by("seed").agg(pl.col(key).median()).sort("seed")[key].to_numpy()
            values = data[key].drop_nulls().to_numpy()
            ax.scatter(
                i + rng.uniform(-0.26, 0.26, values.size),
                values,
                color=color,
                s=6,
                alpha=0.13,
                linewidths=0,
            )
            ax.scatter(
                i + rng.uniform(-0.13, 0.13, seeds.size),
                seeds,
                color=color,
                s=14,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
            mid, q1, q3 = quantiles(seeds)
            ax.vlines(i, q1, q3, color="black", linewidth=1.4, zorder=4)
            ax.hlines(mid, i - 0.095, i + 0.095, color="black", linewidth=2, zorder=5)
        ax.set_xticks(range(len(CONDITIONS)), [c[1] for c in CONDITIONS])
        ax.set_title(name, fontsize=9)
        ax.grid(axis="y", alpha=0.20)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Exploratory · threshold crossing, stabilization and final volatility", fontsize=13
    )
    figure.savefig(output / "discrimination-dynamics.png", dpi=210, facecolor="white")
    plt.close(figure)


def summarize(group):
    endpoints = ("ttc_display", "first_crossing", "crossing_to_stop_or_cap", "final_window_sd")
    seeds = group.group_by("seed").agg(pl.col(*endpoints).median()).sort("seed")
    observed_lags = group.filter(~pl.col("ttc_censored"))
    seeds_hit = observed_lags.group_by("seed").agg(pl.col("crossing_to_stop_or_cap").median())
    summary = {
        "jobs": group.height,
        "censored": int(group["ttc_censored"].sum()),
        "first_crossing_missing": group["first_crossing"].null_count(),
        "short_tails": group.filter(pl.col("tail_trials") < 100).height,
        "minimum_tail_trials": int(group["tail_trials"].min()),
    }
    for endpoint in endpoints:
        if endpoint == "crossing_to_stop_or_cap":
            summary["observed_lag_jobs"] = observed_lags.height
            summary["observed_lag_seed"] = quantiles(seeds_hit[endpoint])
            summary["observed_lag_job"] = quantiles(observed_lags[endpoint])
        else:
            summary[f"{endpoint}_seed"] = quantiles(seeds[endpoint])
            summary[f"{endpoint}_job"] = quantiles(group[endpoint])
    return summary, seeds


def write_report(topology, jobs, confirmatory, output):
    groups = {
        condition: jobs.filter(pl.col("condition") == condition) for condition, _, _ in CONDITIONS
    }
    centres, associations, descriptors = {}, {}, {}
    for condition, group in groups.items():
        by_graph = topology.filter(
            pl.col("graph_id") == "real"
            if condition == "real"
            else pl.col("graph_id").str.starts_with(condition + "-")
        )
        centres[condition] = by_graph.join(
            group.group_by("graph_id").agg(pl.col("ttc_display").median()),
            on="graph_id",
            validate="1:1",
        )
        associations[condition] = {
            key: rho(centres[condition][key], centres[condition]["ttc_display"])
            for key, _ in METRICS
        }
        descriptors[condition] = {key: quantiles(by_graph[key]) for key, _ in METRICS}
        descriptors[condition]["directed_reachability"] = quantiles(
            by_graph["directed_reachability"]
        )
        descriptors[condition]["weak_components"] = quantiles(by_graph["weak_components"])
        descriptors[condition]["largest_component_vertices"] = quantiles(
            by_graph["largest_component_vertices"]
        )
    metric_plot(groups, centres, output)
    dynamics_plot(groups, output)

    groups["pooled"] = jobs.filter(pl.col("condition") != "real")
    summaries, seeds = {}, {}
    for condition, group in groups.items():
        summaries[condition], seeds[condition] = summarize(group)
    a, b = seeds["A"]["ttc_display"], seeds["B"]["ttc_display"]
    null_b_minus_a = b - a
    paired = {
        "B_minus_A_seed_median_iqr": quantiles(null_b_minus_a),
        "B_gt_A": int((null_b_minus_a > 0).sum()),
        "B_eq_A": int((null_b_minus_a == 0).sum()),
        "B_lt_A": int((null_b_minus_a < 0).sum()),
    }
    payload = {
        "topology": descriptors,
        "metric_ttc_graph_spearman": associations,
        "dynamics": summaries,
        "null_B_minus_A": paired,
    }
    (output / "exploratory-summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Exploratory topology and learning dynamics",
        "",
        (
            "This post hoc analysis uses the 1,830 jobs already frozen in `v2.0-confirmatory-results`. "
            "Each of 61 distinct graph artifacts was linked by SHA-256 to the 30 jobs that used it. "
            "No additional training or confirmatory testing was performed."
        ),
        "",
        "## Topology and trials to criterion",
        "",
        (
            "All measures use the unweighted edge support of the archived sparse message-passing adjacency. "
            "Self-loops, if present, are excluded. For spectral gap, clustering and modularity, an undirected edge exists when either arc exists. "
            "The full-graph gap is λ₂ of the normalized Laplacian, including disconnected and isolated nodes. "
            "We also give λ₂ of the largest weak component. Clustering averages the local coefficient over all vertices. "
            "Modularity uses an unweighted Louvain partition (resolution 1, seed 20261119). "
            "The mean path and reachability use distinct ordered pairs and directed arcs. Mean path length averages over reachable pairs only. "
            "Reciprocity is the fraction of directed arcs with an arc in the opposite direction."
        ),
        "",
        "| Metric | Real, one topology | Null-A, 30 topologies | Null-B, 30 topologies | ρ in A | ρ in B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, name in METRICS + (
        ("directed_reachability", "Directed reachability"),
        ("weak_components", "Weak components"),
        ("largest_component_vertices", "Largest weak component size"),
    ):
        real = descriptors["real"][key][0]
        digits = 0 if key in ("weak_components", "largest_component_vertices") else 5
        r = [associations[k].get(key) for k in ("A", "B")]
        rank_a, rank_b = ("—" if v is None else f"{v:+.3f}" for v in r)
        lines.append(
            f"| {name} | {real:.{digits}f} | {cell(descriptors['A'][key], digits)} | {cell(descriptors['B'][key], digits)} | {rank_a} | {rank_b} |"
        )
    lines += [
        "",
        (
            "Null cells are median [Q1, Q3] over topologies. ρ is descriptive Spearman correlation with each topology's median TTC across seeds, within the indicated null variant. "
            "Real has one topology and no within-condition ρ. Constant metrics have undefined ρ. TTC displays the censored job at 2,000."
        ),
        "",
        "![Five topology descriptors and component-level gap versus TTC](graph-metrics-vs-ttc.png)",
        "",
        (
            "Small marks are jobs. Large marks represent topology-level medians over 30 seeds. A hollow triangle marks right censoring. "
            "The spectral gap of a symmetrized graph describes undirected connectivity. It does not estimate mixing time of the directed neural circuit."
        ),
        "",
        "## Trial-to-trial discrimination and stabilization",
        "",
        (
            "Final-window volatility is the sample SD (ddof 1) of the last 100 observed held-out trial scores before stopping or the cap. "
            "Any shorter trace contributes all its observed trials. The window is aligned to the stop event, so it samples different calendar trials across jobs. "
            "Stabilization lag is observed TTC minus first crossing. The censored run is retained in the TTC and volatility tables and in the job data, with a lower bound for its lag when a first crossing is observed."
        ),
        "",
        "| Condition | Jobs | TTC censored | No first crossing | Tails <100 | First crossing | TTC | Observed stabilization lag | Final SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, label, _ in (*CONDITIONS, ("pooled", "Pooled nulls", "")):
        d = summaries[condition]
        lines.append(
            f"| {label} | {d['jobs']:,} | {d['censored']} | {d['first_crossing_missing']} | {d['short_tails']} | "
            f"{cell(d['first_crossing_seed'])} | {cell(d['ttc_display_seed'])} | "
            f"{cell(d['observed_lag_seed'])} | {cell(d['final_window_sd_seed'], 5)} |"
        )
    lines += [
        "",
        (
            "Cells use the median [Q1, Q3] across 30 seed summaries. Each null seed summary is the median of the relevant jobs. "
            "Lag summaries use the criterion hits, and are conditional on observing stabilization within follow-up."
        ),
        "",
        "![Held-out discrimination volatility, first crossing, and stabilization lag](discrimination-dynamics.png)",
        "",
        "Small points are jobs. Larger points are seed medians. Black marks show the across-seed median and IQR.",
        "",
        "## Null-A and Null-B",
        "",
        "| Tagged comparison | Real median TTC | Matched null median TTC | Rank-biserial r [95% CI] | Registered one-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("Null-A", "Null-B"):
        d = confirmatory["ttc"][label]
        lo, hi = d["rank_biserial_ci95"]
        lines.append(
            f"| Real vs {label} | {d['real_median']:.2f} | {d['matched_null_median']:.2f} | "
            f"{d['rank_biserial']:+.4f} [{lo:+.4f}, {hi:+.4f}] | {d['p_raw']:.10f} |"
        )
    lines += [
        "",
        (
            f"The paired B−A difference of within-seed median TTC is **{cell(paired['B_minus_A_seed_median_iqr'])} trials**. "
            f"B is higher in {paired['B_gt_A']}/30 seeds, equal in {paired['B_eq_A']}, and lower in {paired['B_lt_A']}. "
            "These are descriptive differences between rewiring ensembles. Null-B additionally preserves per-node directed degree within the excitatory, inhibitory and unassigned strata. "
            "The model receives the rewired edge support without direct signed message weights. The contrast is compatible with a role for sign-constrained topology. "
            "It cannot isolate an effect of synaptic sign from changes in the topology sampled by that constraint."
        ),
        "",
        "## Reproducibility",
        "",
        (
            "[Manifest](MANIFEST.json) records the git commit, config, input receipts, source hashes and output hashes. "
            "[Topology metrics](topology-metrics.parquet) gives the 61 graph rows. [Job metrics](job-mechanism.parquet) gives the 1,830-job join. "
            "[Descriptive JSON](exploratory-summary.json) provides the numerical summaries, including raw job-level quantiles."
        ),
        "",
    ]
    (output / "EXPLORATORY.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
