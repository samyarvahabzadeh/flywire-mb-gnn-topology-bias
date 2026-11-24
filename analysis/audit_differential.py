"""Audit the Molab differential-conditioning archive and reproduce its contrasts."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata

from mbtopology.runtime import now, require, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/exploratory_differential.json"
RULES = ("pulse", "prediction_error", "adam")
KEYS = ("seed", "mask", "direction", "rule", "differential")
BUNDLE_ID = "0d2562989d88adfafaddbe3bae1b3289fd861dd19696e0a0f9040b18937bb4aa"


def load_run(output, archive, bundle):
    require(sha256(bundle) == BUNDLE_ID, "original input bundle hash differs")
    with ZipFile(bundle) as source:
        plan_bytes = source.read("plan.json")
        require(plan_bytes == PLAN.read_bytes(), "local frozen plan differs")
        plan = json.loads(plan_bytes)
        provenance_bytes = source.read("bundle.json")
        provenance = json.loads(provenance_bytes)
        run_id = hashlib.sha256(provenance_bytes).hexdigest()
        with (
            source.open("masks.npz") as masks_file,
            np.load(masks_file, allow_pickle=False) as data,
        ):
            masks = sorted(k.removesuffix("_pre") for k in data.files if k.endswith("_pre"))
    seeds = plan["seeds"]
    shards = {f"seed-{s}.parquet" for s in seeds}
    items = shards | {f"seed-{s}.json" for s in seeds} | {"seed-contrasts.parquet", "MANIFEST.json"}
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as remote:
        require(sorted(remote.namelist()) == sorted(items), "remote archive inventory differs")
        for name in sorted(items):
            (output / name).write_bytes(remote.read(name))
    manifest = json.loads((output / "MANIFEST.json").read_text())
    expected_jobs = len(seeds) * len(masks) * 12
    require(
        all(manifest[k] == v for k, v in provenance.items())
        and manifest["jobs"] == expected_jobs
        and manifest["smoke"] is False
        and set(manifest["shards"]) == shards
        and sha256(output / "seed-contrasts.parquet") == manifest["result_sha256"],
        "final manifest fails provenance or inventory check",
    )
    frames = []
    for seed in seeds:
        filename = f"seed-{seed}.parquet"
        receipt = json.loads((output / f"seed-{seed}.json").read_text())
        require(
            receipt["run_id"] == run_id
            and receipt["config_sha256"] == sha256(PLAN)
            and receipt["git_commit"] == provenance["git_commit"]
            and sha256(output / filename) == receipt["sha256"] == manifest["shards"][filename],
            f"receipt verification failed for {seed}",
        )
        frame = pl.read_parquet(output / filename)
        expected_keys = set(itertools.product([seed], masks, (0, 1), RULES, (False, True)))
        require(
            frame.height == len(expected_keys)
            and set(frame.select(KEYS).iter_rows()) == expected_keys,
            f"crossed design inventory differs for {seed}",
        )
        frames.append(frame)
    return pl.concat(frames), manifest


def summarize(frame, output):
    # Units are fractions of the fractional MBON11 depression scale.
    metrics = ("gap", "normalized_gap", "cs_plus", "cs_minus", "near_plus", "near_minus")
    for m in metrics:
        require(frame[m].is_finite().all(), f"invalid values in {m}")
    require(
        np.allclose(frame["gap"], frame["near_plus"] - frame["near_minus"], atol=1e-14),
        "gap and novel family components disagree",
    )
    require(
        np.allclose(frame["normalized_gap"], frame["gap"] / frame["cs_plus"], atol=1e-14),
        "normalized gap and CS+ strength disagree",
    )
    keys = ["seed", "mask", "rule", "direction"]
    absent = frame.filter(~pl.col("differential")).select(*keys, *metrics)
    present = frame.filter(pl.col("differential")).select(*keys, *metrics)
    paired = present.join(absent, on=keys, suffix="_absolute", validate="1:1")
    changes = paired.group_by("seed", "mask", "rule").agg(
        *((pl.col(m) - pl.col(f"{m}_absolute")).mean().alias(m) for m in metrics)
    )
    require(
        (changes.filter(pl.col("rule") == "pulse").select(metrics).to_numpy() == 0).all(),
        "pulse control changed under explicit nonreinforcement",
    )

    keys = ["seed", "rule"]
    real = changes.filter(pl.col("mask") == "real").drop("mask")
    controls = changes.filter(pl.col("mask") != "real").group_by(keys).agg(pl.col(metrics).median())
    contrasts = real.join(controls, on=keys, suffix="_control", validate="1:1").sort(keys)
    remote = pl.read_parquet(output / "seed-contrasts.parquet").sort(keys)
    require(remote.select(keys).equals(contrasts.select(keys)), "Molab summary keys differ")
    for name, column in {
        "real_gain": contrasts["gap"],
        "null_gain": contrasts["gap_control"],
        "topology_effect": contrasts["gap"] - contrasts["gap_control"],
    }.items():
        require(
            np.allclose(remote[name], column, atol=1e-14, rtol=0), f"Molab summary {name} differs"
        )
    return contrasts, changes


def estimate(values):
    lo, mid, hi = np.quantile(values, [0.25, 0.5, 0.75])
    positives = np.count_nonzero(values > 0)
    return f"{100 * mid:+.3f} [{100 * lo:+.3f}, {100 * hi:+.3f}] ({positives}/{len(values)})"


def report(output, contrasts, changes, manifest, archive):
    rules = {
        "pulse": "Punishment pulse",
        "prediction_error": "Prediction error",
        "adam": "Adam/BCE",
    }
    lines = [
        "# Molab differential conditioning",
        "",
        (
            f"Exploratory results from {manifest['jobs']:,} conditions on 20 odor layouts. The audit "
            "verified every shard, the frozen inputs and the independently recomputed paired contrasts."
        ),
        "",
        (
            "Training alternated the punished cue with an explicitly safe similar cue or a blank. "
            "Each contrast uses novel odor variants and averages both counterbalanced assignments."
        ),
        "",
        (
            "Cells give median [IQR] in percentage points, followed by positive layouts out of 20. "
            "They summarize simulation layouts on one connectome. The normalized outcome divides the "
            "novel discrimination score by that condition's learned response to CS+."
        ),
        "",
        "| Rule | Wiring contrast | Raw discrimination gain | Normalized discrimination gain |",
        "|---|---|---:|---:|",
    ]
    components = [
        "",
        "| Rule | Real-circuit response change | Median [IQR] (positive layouts) |",
        "|---|---|---:|",
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
    for i, (rule, label) in enumerate(rules.items()):
        rows = contrasts.filter(pl.col("rule") == rule)
        for name, suffix in (
            ("Real", ""),
            ("Control median", "_control"),
            ("Real minus control", None),
        ):
            scores = [
                rows[m + suffix] if suffix is not None else rows[m] - rows[m + "_control"]
                for m in ("gap", "normalized_gap")
            ]
            lines.append(
                f"| {label} | {name} | {estimate(scores[0].to_numpy())} | "
                f"{estimate(scores[1].to_numpy())} |"
            )
        for name, field in (
            ("Trained punished cue", "cs_plus"),
            ("Trained safe cue", "cs_minus"),
            ("Novel punished family", "near_plus"),
            ("Novel safe family", "near_minus"),
        ):
            components.append(f"| {label} | {name} | {estimate(rows[field].to_numpy())} |")
        for shift, col, color, graph in (
            (-0.15, "gap", "#1f6f7d", "Real"),
            (0.15, "gap_control", "#b85c00", "Control median"),
        ):
            vals = 100 * rows[col].to_numpy()
            x = i + shift + (rankdata(vals) / len(vals) - 0.5) * 0.1
            ax.scatter(x, vals, s=20, color=color, alpha=0.65, label=graph if i == 0 else None)
            ax.hlines(np.median(vals), i + shift - 0.1, i + shift + 0.1, lw=2.5, color=color)
    lines += components
    lines += [
        "",
        (
            "All pulse-rule contrasts were exactly zero. The prediction-error versus pulse "
            "difference of topology effects is therefore numerically the prediction-error "
            "real-minus-control contrast. These are reduced-model responses on simulated PN "
            "patterns and a fixed set of 12 similar replacement masks."
        ),
        "",
    ]
    ax.axhline(0, color="#555555", lw=0.8, ls="--")
    ax.set(
        xticks=range(len(rules)),
        xticklabels=list(rules.values()),
        ylabel="Differential minus absolute discrimination (pp)",
    )
    ax.legend(frameon=False)
    figure = output / "differential-conditioning.png"
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    contrasts.write_csv(output / "audited-seed-contrasts.csv")
    changes.write_parquet(output / "mask-effects.parquet")
    result = output / "REPORT.md"
    result.write_text("\n".join(lines), encoding="utf-8")
    artifacts = [
        result,
        figure,
        output / "audited-seed-contrasts.csv",
        output / "mask-effects.parquet",
    ]
    (output / "AUDIT_MANIFEST.json").write_text(
        json.dumps(
            {
                "timestamp": now(),
                "archive_sha256": sha256(archive),
                "analysis_sha256": sha256(__file__),
                "git_commit": manifest["git_commit"],
                "config_sha256": sha256(PLAN),
                "files": {p.name: sha256(p) for p in artifacts},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist/differential-molab.zip")
    args = parser.parse_args()
    rows, manifest = load_run(args.output, args.archive, args.bundle)
    contrasts, changes = summarize(rows, args.output)
    report(args.output, contrasts, changes, manifest, args.archive)
