"""Descriptive analysis for the Molab graded-similarity exploratory run."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_load(input_dir):
    input_dir = Path(input_dir)
    manifest_path = input_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    shards = manifest["shards"]
    if len(shards) != 65 or manifest["jobs"] != 249600:
        raise ValueError("run manifest does not describe the full 65-shard assay")
    frames = []
    for name, receipt in sorted(shards.items()):
        path = input_dir / name
        if sha256(path) != receipt["sha256"]:
            raise ValueError(f"shard SHA-256 mismatch: {name}")
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if sidecar != receipt:
            raise ValueError(f"receipt mismatch: {name}")
        frame = pl.read_parquet(path)
        if frame.height != receipt["rows"]:
            raise ValueError(f"shard row-count mismatch: {name}")
        frames.append(frame)
    data = pl.concat(frames, how="vertical_relaxed")
    if data.height != manifest["jobs"]:
        raise ValueError("combined row count differs from run manifest")
    return data, manifest


def bootstrap_mean_ci(values, rng, n_boot=5000):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite value in bootstrap input")
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.mean(values)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize(data):
    keys = ["seed", "mask", "train_shared_pns", "probe_shared_pns", "rule"]
    means = data.group_by(keys + ["schedule"]).agg(
        pl.col("heldout_discrimination").mean().alias("heldout_discrimination"),
        pl.col("probe_csplus_response").mean().alias("probe_csplus_response"),
        pl.col("probe_csminus_response").mean().alias("probe_csminus_response"),
    )
    differential = means.filter(pl.col("schedule") == "differential").drop("schedule").rename(
        {
            "heldout_discrimination": "differential_discrimination",
            "probe_csplus_response": "differential_csplus_response",
            "probe_csminus_response": "differential_csminus_response",
        }
    )
    absolute = means.filter(pl.col("schedule") == "absolute").drop("schedule").rename(
        {
            "heldout_discrimination": "absolute_discrimination",
            "probe_csplus_response": "absolute_csplus_response",
            "probe_csminus_response": "absolute_csminus_response",
        }
    )
    paired = differential.join(absolute, on=keys, how="inner").with_columns(
        (pl.col("differential_discrimination") - pl.col("absolute_discrimination")).alias("schedule_effect")
    )
    control_keys = ["seed", "train_shared_pns", "probe_shared_pns", "rule"]
    control = (
        paired.filter(pl.col("mask") != "real")
        .group_by(control_keys)
        .agg(pl.col("schedule_effect").median().alias("control_median_effect"))
    )
    real = (
        paired.filter(pl.col("mask") == "real")
        .select(control_keys + ["schedule_effect", "differential_discrimination", "absolute_discrimination"])
        .rename({"schedule_effect": "real_effect"})
    )
    contrasts = real.join(control, on=control_keys, how="inner").with_columns(
        (pl.col("real_effect") - pl.col("control_median_effect")).alias("real_minus_control")
    )
    rng = np.random.default_rng(20261124)
    specificity_rows = []
    for train_level in sorted(contrasts["train_shared_pns"].unique().to_list()):
        q36 = (
            contrasts.filter(
                (pl.col("train_shared_pns") == train_level) & (pl.col("probe_shared_pns") == 36)
            )
            .select(["seed", "real_effect", "control_median_effect", "real_minus_control"])
            .rename({
                "real_effect": "real_q36",
                "control_median_effect": "control_q36",
                "real_minus_control": "topology_q36",
            })
        )
        q0 = (
            contrasts.filter(
                (pl.col("train_shared_pns") == train_level) & (pl.col("probe_shared_pns") == 0)
            )
            .select(["seed", "real_effect", "control_median_effect", "real_minus_control"])
            .rename({
                "real_effect": "real_q0",
                "control_median_effect": "control_q0",
                "real_minus_control": "topology_q0",
            })
        )
        paired_probe = q36.join(q0, on="seed", how="inner").with_columns(
            (pl.col("real_q36") - pl.col("real_q0")).alias("real_specificity_gradient"),
            (pl.col("control_q36") - pl.col("control_q0")).alias("control_specificity_gradient"),
            (pl.col("topology_q36") - pl.col("topology_q0")).alias("topology_specificity_interaction"),
        )
        row = {"train_shared_pns": train_level, "n_layouts": paired_probe.height}
        for col in (
            "real_specificity_gradient",
            "control_specificity_gradient",
            "topology_specificity_interaction",
        ):
            mean, low, high = bootstrap_mean_ci(paired_probe[col].to_numpy(), rng)
            row.update({f"{col}_mean": mean, f"{col}_ci_low": low, f"{col}_ci_high": high})
        specificity_rows.append(row)
    probe_specificity = pl.DataFrame(specificity_rows).sort("train_shared_pns")
    marginal = paired.group_by(["seed", "mask", "train_shared_pns", "rule"]).agg(
        pl.col("schedule_effect").mean().alias("probe_averaged_effect")
    )
    marginal_real = marginal.filter(pl.col("mask") == "real").select(
        ["seed", "train_shared_pns", "rule", "probe_averaged_effect"]
    )
    marginal_control = (
        marginal.filter(pl.col("mask") != "real")
        .group_by(["seed", "train_shared_pns", "rule"])
        .agg(pl.col("probe_averaged_effect").median().alias("probe_averaged_effect"))
    )
    summaries = []
    for row in contrasts.group_by(["rule", "train_shared_pns", "probe_shared_pns"], maintain_order=True):
        group_key, frame = row
        values = frame.sort("seed")
        result = {"rule": group_key[0], "train_shared_pns": group_key[1], "probe_shared_pns": group_key[2], "n_layouts": values.height}
        for col in ("real_effect", "control_median_effect", "real_minus_control"):
            mean, low, high = bootstrap_mean_ci(values[col].to_numpy(), rng)
            result.update({f"{col}_mean": mean, f"{col}_ci_low": low, f"{col}_ci_high": high})
        summaries.append(result)
    surface = pl.DataFrame(summaries).sort(["rule", "train_shared_pns", "probe_shared_pns"])
    marginal_rows = []
    for population, frame in (("real", marginal_real), ("control_median", marginal_control)):
        for group_key, values in frame.group_by(["rule", "train_shared_pns"], maintain_order=True):
            mean, low, high = bootstrap_mean_ci(values["probe_averaged_effect"].to_numpy(), rng)
            marginal_rows.append({
                "population": population,
                "rule": group_key[0],
                "train_shared_pns": group_key[1],
                "n_layouts": values.height,
                "mean_probe_averaged_effect": mean,
                "ci_low": low,
                "ci_high": high,
            })
    marginal_summary = pl.DataFrame(marginal_rows).sort(["population", "rule", "train_shared_pns"])
    rule_diff_rows = []
    for population, frame in (("real", marginal_real), ("control_median", marginal_control)):
        for first, second in (("prediction_error", "pulse"), ("prediction_error", "adam_bce")):
            a = frame.filter(pl.col("rule") == first).select(
                ["seed", "train_shared_pns", "probe_averaged_effect"]
            ).rename({"probe_averaged_effect": "first_effect"})
            b = frame.filter(pl.col("rule") == second).select(
                ["seed", "train_shared_pns", "probe_averaged_effect"]
            ).rename({"probe_averaged_effect": "second_effect"})
            difference = a.join(b, on=["seed", "train_shared_pns"], how="inner").with_columns(
                (pl.col("first_effect") - pl.col("second_effect")).alias("difference")
            )
            for train_level, values in difference.group_by("train_shared_pns", maintain_order=True):
                mean, low, high = bootstrap_mean_ci(values["difference"].to_numpy(), rng)
                rule_diff_rows.append({
                    "population": population,
                    "rule_contrast": f"{first} minus {second}",
                    "train_shared_pns": train_level[0],
                    "n_layouts": values.height,
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                })
    rule_differences = pl.DataFrame(rule_diff_rows).sort(["population", "rule_contrast", "train_shared_pns"])
    pulse = paired.filter(pl.col("rule") == "pulse").select(pl.col("schedule_effect").abs().max()).item()
    if pulse != 0.0:
        raise ValueError(f"pulse schedule invariance failed: max absolute difference={pulse}")
    return paired, contrasts, surface, probe_specificity, marginal_summary, rule_differences, pulse


def write_report(data, manifest, paired, contrasts, surface, probe_specificity, marginal_summary, rule_differences, pulse, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.write_parquet(output_dir / "paired_schedule_effects.parquet")
    contrasts.write_parquet(output_dir / "seed_level_topology_contrasts.parquet")
    surface.write_csv(output_dir / "similarity_surface_summary.csv")
    probe_specificity.write_csv(output_dir / "paired_probe_specificity.csv")
    marginal_summary.write_csv(output_dir / "probe_averaged_rule_summary.csv")
    rule_differences.write_csv(output_dir / "paired_rule_differences.csv")

    rules = surface["rule"].unique(maintain_order=True).to_list()
    train_levels = sorted(surface["train_shared_pns"].unique().to_list())
    probe_levels = sorted(surface["probe_shared_pns"].unique().to_list())
    vmax = float(surface["real_minus_control_mean"].abs().max()) or 1e-9
    fig, axes = plt.subplots(1, len(rules), figsize=(5.3 * len(rules), 4.8), constrained_layout=True)
    for ax, rule in zip(np.atleast_1d(axes), rules):
        subset = surface.filter(pl.col("rule") == rule).sort(["train_shared_pns", "probe_shared_pns"])
        matrix = subset["real_minus_control_mean"].to_numpy().reshape(len(train_levels), len(probe_levels))
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(rule.replace("_", " "))
        ax.set_xticks(range(len(probe_levels)), probe_levels)
        ax.set_yticks(range(len(train_levels)), train_levels)
        ax.set_xlabel("Held-out probe overlap with trained cue (PNs)")
        ax.set_ylabel("Training CS+/CS− overlap (PNs)")
    fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label="Real schedule effect − control median")
    fig.savefig(output_dir / "similarity_surface.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = {"pulse": "#666666", "prediction_error": "#1769aa", "adam_bce": "#e07a19"}
    for population, linestyle in (("real", "-"), ("control_median", "--")):
        for rule in rules:
            subset = marginal_summary.filter(
                (pl.col("population") == population) & (pl.col("rule") == rule)
            ).sort("train_shared_pns")
            x = subset["train_shared_pns"].to_numpy()
            mean = subset["mean_probe_averaged_effect"].to_numpy()
            low = subset["ci_low"].to_numpy()
            high = subset["ci_high"].to_numpy()
            label = f"{rule.replace('_', ' ')} · {population.replace('_', ' ')}"
            ax.plot(x, mean, linestyle=linestyle, marker="o", color=colors[rule], label=label)
            ax.fill_between(x, low, high, color=colors[rule], alpha=0.10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Training CS+/CS− overlap (active PN identities out of 40)")
    ax.set_ylabel("Differential minus absolute held-out discrimination")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_dir / "rule_similarity_gradient.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = probe_specificity["train_shared_pns"].to_numpy()
    for population, color, prefix in (
        ("real partner mask", "#1769aa", "real_specificity_gradient"),
        ("median input control", "#e07a19", "control_specificity_gradient"),
    ):
        mean = probe_specificity[f"{prefix}_mean"].to_numpy()
        low = probe_specificity[f"{prefix}_ci_low"].to_numpy()
        high = probe_specificity[f"{prefix}_ci_high"].to_numpy()
        ax.plot(x, mean, marker="o", color=color, label=population)
        ax.fill_between(x, low, high, color=color, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Training CS+/CS− overlap (PN identities out of 40)")
    ax.set_ylabel("Safe-cue schedule effect at probe overlap 36 − 0")
    ax.legend(frameon=False)
    fig.savefig(output_dir / "probe_specificity_gradient.png", dpi=180)
    plt.close(fig)

    global_summary = data.group_by(["mask", "rule", "schedule"]).agg(
        pl.col("heldout_discrimination").mean().alias("mean_discrimination"),
        pl.col("heldout_discrimination").std().alias("sd_row_discrimination"),
        pl.len().alias("rows"),
    )
    global_summary.write_csv(output_dir / "condition_summary.csv")
    record = {
        "scope": "exploratory_only",
        "timestamp_unix": time.time(),
        "input_manifest_sha256": sha256(Path(manifest["_manifest_path"])),
        "run_git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_sha256"],
        "runner_sha256": manifest["runner_sha256"],
        "analysis_script_sha256": sha256(Path(manifest["_analysis_script_path"])),
        "rows": data.height,
        "seeds": sorted(data["seed"].unique().to_list()),
        "masks": sorted(data["mask"].unique().to_list()),
        "rules": sorted(data["rule"].unique().to_list()),
        "max_abs_pulse_schedule_effect": pulse,
        "layout_bootstrap_draws": 5000,
        "layout_bootstrap_seed": 20261124,
        "biological_replicates": 1,
        "notes": [
            "Input-mask controls are not full degree-preserving graph nulls.",
            "Bootstrap intervals describe cue-layout variability on one connectome, not biological uncertainty.",
            "PN identity overlap is a modeled cue-similarity manipulation, not measured odor chemistry.",
            "No confirmatory analysis or claim about fly behavior is made.",
        ],
    }
    record["files"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "MANIFEST.json"
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    outliers = surface.sort("real_minus_control_mean")
    low = outliers.row(0, named=True)
    high = outliers.row(-1, named=True)
    pe = marginal_summary.filter(
        (pl.col("population") == "real") & (pl.col("rule") == "prediction_error")
    ).sort("train_shared_pns")
    pe_start = pe.row(0, named=True)
    pe_end = pe.row(-1, named=True)
    adam_max = marginal_summary.filter(
        (pl.col("population") == "real") & (pl.col("rule") == "adam_bce")
    )["mean_probe_averaged_effect"].abs().max()
    control_pe = marginal_summary.filter(
        (pl.col("population") == "control_median") & (pl.col("rule") == "prediction_error")
    ).sort("train_shared_pns")
    pe_vs_pulse = rule_differences.filter(
        (pl.col("population") == "real") & (pl.col("rule_contrast") == "prediction_error minus pulse")
    ).sort("train_shared_pns")
    pe_vs_adam = rule_differences.filter(
        (pl.col("population") == "real") & (pl.col("rule_contrast") == "prediction_error minus adam_bce")
    ).sort("train_shared_pns")
    pattern_rows = []
    for index in range(pe.height):
        r = pe.row(index, named=True)
        c = control_pe.row(index, named=True)
        p = pe_vs_pulse.row(index, named=True)
        a = pe_vs_adam.row(index, named=True)
        pattern_rows.append(
            f"| {r['train_shared_pns']}/40 | {r['mean_probe_averaged_effect']:.4f} "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {c['mean_probe_averaged_effect']:.4f} "
            f"| {p['mean_difference']:.4f} [{p['ci_low']:.4f}, {p['ci_high']:.4f}] "
            f"| {a['mean_difference']:.4f} [{a['ci_low']:.4f}, {a['ci_high']:.4f}] |"
        )
    pattern_table = "\n".join(pattern_rows)
    specificity_rows = []
    for row in probe_specificity.iter_rows(named=True):
        specificity_rows.append(
            f"| {row['train_shared_pns']}/40 | {row['real_specificity_gradient_mean']:.4f} "
            f"[{row['real_specificity_gradient_ci_low']:.4f}, {row['real_specificity_gradient_ci_high']:.4f}] "
            f"| {row['control_specificity_gradient_mean']:.4f} "
            f"[{row['control_specificity_gradient_ci_low']:.4f}, {row['control_specificity_gradient_ci_high']:.4f}] "
            f"| {row['topology_specificity_interaction_mean']:.4f} "
            f"[{row['topology_specificity_interaction_ci_low']:.4f}, {row['topology_specificity_interaction_ci_high']:.4f}] |"
        )
    specificity_table = "\n".join(specificity_rows)
    report = f"""# Exploratory graded-similarity assay

This report describes a new exploratory computation only. It does not reopen the confirmatory dataset or analysis, and it does not test a behavioral effect in flies.

## Data and controls

- Molab runner commit: `{manifest['git_commit']}`
- Config SHA-256: `{manifest['config_sha256']}`; runner SHA-256: `{manifest['runner_s