"""Run the locked v1.2 analysis once, independently of the sweep writer."""

import argparse
import io
import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import sleep
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo

import numpy as np
from confirmatory_data import (
    AMENDMENT,
    CAP,
    CONDITIONS,
    CONFIG_SHA,
    PLAN_SHA,
    SEEDS,
    WRITER_COMMIT,
    digest,
    directory_reader,
    read_dataset,
    require,
)
from confirmatory_plots import LABELS, box_strip, curve, noise_panel, padded_curve
from confirmatory_stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, analysis_rows, seed_values

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "analysis/confirmatory_plan.json"
TAG = "v1.2-prereg-amendment"
AMENDMENT_SHA = "045f34049ff9b6f747c054f325360bf0af88ea26f94f2da2ccfb6517349b2e9b"
TAG_URL = (
    f"https://github.com/samyarvahabzadeh/flywire-mb-gnn-topology-bias/blob/{TAG}/AMENDMENT-02.md"
)


def json_bytes(payload):
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def put_immutable(fs, prefix, name, payload):
    dest = f"{prefix}/{name}"
    if fs.exists(dest):
        require(fs.cat_file(dest) == payload, f"different artifact already exists at {name}")
        return
    fs.pipe_file(dest, payload)
    fs.invalidate_cache(prefix)
    require(fs.cat_file(dest) == payload, f"publication read-back differs for {name}")


def transport_retry(operation):
    from googleapiclient.errors import HttpError

    for attempt in range(3):
        try:
            return operation()
        except HttpError as exc:
            if exc.resp.status not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise
        sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable transport retry state")


def ttc_at_cap(job):
    return CAP if job.ttc is None else job.ttc


def summary_cell(values):
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return f"{median:.2f} [{q1:.2f}, {q3:.2f}]"


def describe(jobs):
    rows = []
    for condition, label in zip(CONDITIONS, LABELS, strict=True):
        selected = [job for job in jobs if job.condition == condition]
        rows.append(
            {
                "condition": label,
                "jobs": len(selected),
                "seed_groups": len(SEEDS),
                "ttc_by_job": summary_cell([ttc_at_cap(job) for job in selected]),
                "ttc_by_seed": summary_cell(seed_values(jobs, ttc_at_cap, (condition,))),
                "censored": sum(job.ttc is None for job in selected),
                "uncensored_at_cap": sum(job.ttc == CAP for job in selected),
            }
        )
    text = [
        "## Descriptive checks",
        "",
        (
            "The data passed the frozen matrix, receipt, archive, Parquet and registered-endpoint checks. "
            "The six rate-benchmark jobs were outside the eligible inventory. "
            "For display, a capped miss is placed at trial 2,000. "
            "The seed summaries use one real run or the within-seed median for the relevant null variant."
        ),
        "",
        "| Condition | Jobs | Seed groups | Job TTC median [Q1, Q3] | Seed TTC median [Q1, Q3] | Right-censored | Observed hits at cap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        text.append(
            f"| {row['condition']} | {row['jobs']} | {row['seed_groups']} | {row['ttc_by_job']} | "
            f"{row['ttc_by_seed']} | {row['censored']} | {row['uncensored_at_cap']} |"
        )
    text += [
        "",
        (
            "All observed TTC values were within trials 20–2,000. The table preserves every cap event. "
            "No data were removed or modified."
        ),
        "",
    ]
    return rows, text


def make_figures(jobs, output):
    """Render the descriptive figures in their locked order."""
    files = []
    name = "auc-descriptive.png"
    box_strip(
        jobs,
        lambda j: float(j.observed.mean()),
        "Normalized observed AUC",
        "Observed discrimination AUC through stopping or cap",
        output / name,
    )
    files.append(
        (
            name,
            "Discrete clean AUC is normalized by each job's observed trial count through stopping or cap. Each point is a seed summary.",
        )
    )
    name = "ttc-strip-box.png"
    box_strip(
        jobs, ttc_at_cap, "Trials to criterion", "Trials to criterion by topology", output / name
    )
    files.append(
        (
            name,
            "Each point is one of 30 seed summaries per condition. A censored job is displayed at its observed cap.",
        )
    )
    name = "discrimination-curve.png"
    curve(
        jobs,
        padded_curve,
        np.arange(1, CAP + 1),
        "Trial",
        "Held-out discrimination",
        "Discrimination over training",
        output / name,
    )
    files.append(
        (
            name,
            "Lines average 30 seed summaries and bands show the inter-seed IQR. Curves after early stop carry forward each job's terminal 20-trial mean.",
        )
    )
    name = "noise-and-ace.png"
    noise_panel(jobs, output / name)
    files.append(
        (
            name,
            (
                "Points, medians and IQRs are based on 30 seed summaries per condition. "
                "Null values are within-seed medians. Panel B summarizes job ACE values at the seed level."
            ),
        )
    )
    return files


def format_inference(primary, secondary):
    focal = primary["Pooled nulls"]
    lines = [
        "## Primary TTC analysis",
        "",
        (
            "The analysis pairs each real TTC with the median TTC of its 60 null jobs. "
            "A capped miss receives rank value 2,001. A positive difference favors the real graph. "
            "The one-sided Wilcoxon upper tail uses the exact sign distribution of midranks "
            "with zero pairs dropped. The 95% rank-biserial interval is a percentile bootstrap over "
            f"30 paired seeds with {BOOTSTRAP_DRAWS:,} draws and seed {BOOTSTRAP_SEED}."
        ),
        "",
        "| Null comparator | Paired seeds | Nonzero | Real median | Median of seed null medians | Median null − real | W+ | p | Rank-biserial r [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ttc_row("Pooled nulls", focal),
        "",
        "## Requested stratified secondary comparisons",
        "",
        (
            "The v1.2 plan specifies descriptive reporting by null variant. The researcher-requested "
            "Wilcoxon rows below were locked before inspecting outputs and are supplementary to the "
            "registered primary test and three-outcome BH family."
        ),
        "",
        "| Null comparator | Paired seeds | Nonzero | Real median | Median of seed null medians | Median null − real | W+ | p | Rank-biserial r [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ttc_row("Null-A", primary["Null-A"]),
        ttc_row("Null-B", primary["Null-B"]),
        "",
        "## Secondary outcomes and BH correction",
        "",
        (
            "Each row pairs the real outcome with the median over all 60 matched nulls. "
            "Smaller first-crossing values favor real and missing first crossings receive rank value 2,001. "
            "Larger AUC and ACE favor real in the effect-size convention. "
            "The first two tests follow the one-sided v1.2 directions. ACE uses a two-sided exact signed-rank test. "
            "The researcher replaced the registered noise slope with ACE after data generation and before outcome inspection. "
            "ACE is the normalized trapezoid `(d0 + 2*d0.1 + 2*d0.2 + d0.3)/6`. "
            "Benjamini–Hochberg correction is applied to these three tests."
        ),
        "",
        "«عملیاتی‌سازی ACE پیش از اجرای آمار استنباطی فیکس شد.»",
        "",
        "| Outcome | Tail | Paired seeds | Nonzero | Real median | Median of seed null medians | W+ | Raw p | BH-adjusted p | r |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in secondary.items():
        lines.append(
            f"| {name} | {row['alternative']} | {row['n_pairs']} | {row['n_nonzero']} | {row['real_median']:.6g} | "
            f"{row['matched_null_median']:.6g} | {row['w_plus']:.1f} | {row['p_raw']:.6g} | "
            f"{row['p_bh']:.6g} | {row['rank_biserial']:.4f} |"
        )
    lines += [
        "",
        (
            "Each bootstrap draw resamples whole matched seeds and recomputes signed ranks. "
            "The interval conditions on the generated bank of null graphs. "
            "[Wilcoxon reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) · "
            "[BH reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)"
        ),
        "",
    ]
    return lines


def ttc_row(name, row):
    low, high = row["rank_biserial_ci95"]
    return (
        f"| {name} | {row['n_pairs']} | {row['n_nonzero']} | {row['real_median']:.2f} | "
        f"{row['matched_null_median']:.2f} | {row['median_difference_favor_real']:.2f} | "
        f"{row['w_plus']:.1f} | {row['p_raw']:.6g} | {row['rank_biserial']:.4f} "
        f"[{low:.4f}, {high:.4f}] |"
    )


def progress(batch, count):
    if batch % 25 == 0 or batch == 305:
        print(f"ANALYSIS_INPUT_VERIFIED {batch}/305 {count}/1830", flush=True)


def current_inputs():
    plan = json.loads(PLAN.read_bytes())
    require(
        plan["status"] == "LOCKED_BEFORE_OUTCOME_ACCESS"
        and plan["plan_sha256"] == PLAN_SHA
        and plan["amendment_tag_object"] == AMENDMENT
        and plan["primary_alternative"] == "greater"
        and plan["rank_code_for_capped_miss"] == 2001,
        "analysis decision record does not match the approved v1.2 inferential plan",
    )
    tag = (
        subprocess.check_output(["git", "rev-parse", f"refs/tags/{TAG}"], cwd=ROOT).decode().strip()
    )
    amendment = subprocess.check_output(["git", "show", f"{TAG}:AMENDMENT-02.md"], cwd=ROOT)
    require(
        tag == AMENDMENT and digest(amendment) == AMENDMENT_SHA, "frozen amendment evidence changed"
    )
    sources = {
        f.relative_to(ROOT).as_posix(): digest(f.read_bytes())
        for f in sorted(PLAN.parent.glob("confirmatory*"))
        if f.is_file()
    }
    require(
        plan["ace"]["weights"] == [1, 2, 2, 1]
        and plan["ace"]["normalizing_denominator"] == 6
        and plan["secondary_bh_family"]
        == [
            {"outcome": "first_single_crossing", "alternative": "greater"},
            {"outcome": "normalized_auc_to_cap", "alternative": "greater"},
            {"outcome": "ace", "alternative": "two-sided"},
        ]
        and plan["descriptive_figure"] == "observed_clean_auc_until_stop_or_cap",
        "analysis plan does not match the locked secondary outcomes or figures",
    )
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive-dir", type=Path)
    inputs.add_argument("--drive-prefix")
    parser.add_argument("--export-prefix")
    parser.add_argument("--output", type=Path, default=ROOT / "results/confirmatory")
    args = parser.parse_args(argv)
    require(
        bool(args.drive_prefix) == bool(args.export_prefix),
        "Drive reads must use a separate export prefix",
    )
    sources = current_inputs()
    fs = None
    if args.drive_prefix:
        from gdrive_fsspec import GoogleDriveFileSystem

        fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True)
        require(
            args.export_prefix.rstrip("/") != args.drive_prefix.rstrip("/"),
            "analysis and source archives must use separate prefixes",
        )
        fs.makedirs(args.export_prefix, exist_ok=True)
        require(
            not fs.exists(f"{args.export_prefix}/inference-started.json"),
            "one-time inference already started",
        )

        def read(filename):
            return transport_retry(lambda: fs.cat_file(f"{args.drive_prefix}/{filename}"))
    else:
        read = directory_reader(args.archive_dir)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(UTC)
    jobs, provenance = read_dataset(read, progress)
    descriptions, descriptive_text = describe(jobs)
    figures = make_figures(jobs, output)
    (output / "DESCRIPTIVE.md").write_text("\n".join(descriptive_text), encoding="utf-8")
    trace = {
        "timestamp_utc": now.isoformat(),
        "timestamp_tehran": now.astimezone(ZoneInfo("Asia/Tehran")).isoformat(),
        "inference_started_at_utc": datetime.now(UTC).isoformat(),
        "data_plan_sha256": PLAN_SHA,
        "protocol_config_sha256": CONFIG_SHA,
        "sweep_git_commit": WRITER_COMMIT,
        "amendment_tag": TAG,
        "analysis_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "analysis_source_sha256": sources,
        "package_versions": {p: version(p) for p in ("numpy", "scipy", "polars", "matplotlib")},
        **provenance,
    }
    if fs is not None:
        transport_retry(
            lambda: put_immutable(
                fs, args.export_prefix, "inference-started.json", json_bytes(trace)
            )
        )
    (output / "inference-started.json").write_bytes(json_bytes(trace))
    primary, secondary = analysis_rows(jobs)
    report = {
        **trace,
        "job_count": len(jobs),
        "descriptive": descriptions,
        "ttc": primary,
        "secondary_bh_family": secondary,
        "figures": [item[0] for item in figures],
    }
    summary = json_bytes(report)
    (output / "analysis-summary.json").write_bytes(summary)
    lines = [
        "# Confirmatory results",
        "",
        (
            f"Analysis started at **{trace['timestamp_tehran']}** (Tehran). "
            f"Verified {len(jobs):,} fresh jobs in {trace['archive_count']} archives. "
            f"The registered primary used [AMENDMENT-02]({TAG_URL})."
        ),
        "",
        *descriptive_text,
    ]
    for name, caption in figures:
        lines += [f"![{caption}]({name})", "", caption, ""]
    lines += format_inference(primary, secondary)
    lines += [
        "## Reproducibility",
        "",
        f"- Analysis commit `{trace['analysis_git_commit']}`",
        f"- Analysis script SHA-256 `{sources['analysis/confirmatory.py']}`",
        f"- Frozen data plan SHA-256 `{PLAN_SHA}`",
        f"- Drive completion marker SHA-256 `{trace['completion_marker_sha256']}`",
        f"- Verified receipt sequence SHA-256 `{trace['receipt_sequence_sha256']}`",
        "- Full helper hashes, versions and tables in [analysis-summary.json](analysis-summary.json).",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if fs is not None:
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=3) as archive:
            for result in sorted(output.iterdir()):
                archive.write(result, result.name)
        payload = buffer.getvalue()
        output.with_suffix(".zip").write_bytes(payload)
        transport_retry(
            lambda: put_immutable(fs, args.export_prefix, "analysis-summary.json", summary)
        )
        transport_retry(
            lambda: put_immutable(fs, args.export_prefix, "confirmatory-results.zip", payload)
        )
        print(
            f"ANALYSIS_COMPLETE confirmatory-results.zip {digest(payload)} {len(payload)}",
            flush=True,
        )
    else:
        print(f"ANALYSIS_COMPLETE {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
