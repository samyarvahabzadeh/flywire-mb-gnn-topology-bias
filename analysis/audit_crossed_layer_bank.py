"""Verify the completed Molab layer-bank archives and recompute exploratory endpoints."""

import argparse
import io
import json
import re
from collections import defaultdict
from hashlib import sha256
from itertools import product
from math import fsum, isclose
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np
import polars as pl

from mbtopology.runtime import atomic_file, atomic_json, now, require

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/exploratory_layer_bank.json"
EFFECTS = ("both", "first", "second", "interaction")


def digest(data):
    return sha256(data).hexdigest()


def read_archive(path, expected_members):
    blob = path.read_bytes()
    expected_hash = re.search(r"-([0-9a-f]{64})\.zip$", path.name)
    require(
        expected_hash and digest(blob) == expected_hash[1], f"archive hash differs: {path.name}"
    )
    with ZipFile(io.BytesIO(blob)) as archive:
        require(
            sorted(archive.namelist()) == sorted(expected_members),
            f"archive member inventory differs: {path.name}",
        )
        return {name: archive.read(name) for name in expected_members}


def check_files(entries, specs):
    require(set(entries) == set(specs), "archived artifact inventory differs from manifest")
    for name, meta in specs.items():
        require(
            len(entries[name]) == meta["bytes"] and digest(entries[name]) == meta["sha256"],
            f"artifact digest or length differs: {name}",
        )


def endpoints_for(traces, jobs, cap, threshold, persistence_window):
    result = {}
    for key, job in traces.sort("seed", "condition", "trial").group_by("seed", "condition"):
        require(
            len(job) == cap and np.array_equal(job["trial"].to_numpy(), np.arange(1, cap + 1)),
            f"duplicate or missing trials: {key}",
        )
        values = job["discrimination"].to_numpy()
        a, b = job["heldout_a_plus"].to_numpy(), job["heldout_a_minus"].to_numpy()
        require(
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values <= 1).all()
            and np.allclose(values, (a + 1 - b) / 2, rtol=0, atol=1e-12),
            f"invalid held-out discrimination: {key}",
        )
        window = 20
        rolling = np.array([fsum(values[i - window : i]) / window for i in range(window, cap + 1)])
        require(
            job["rolling_discrimination"].head(window - 1).null_count() == window - 1
            and np.allclose(
                rolling,
                job["rolling_discrimination"].slice(window - 1).to_numpy(),
                rtol=0,
                atol=1e-12,
            ),
            f"invalid rolling discrimination: {key}",
        )
        first = next((i for i, v in enumerate(values, 1) if v >= threshold), None)
        stable = next((i for i, v in enumerate(rolling, window) if v >= threshold), None)
        recent = (
            float(np.mean(values[first : first + persistence_window] >= threshold))
            if first is not None and first <= cap - persistence_window
            else None
        )
        result[key] = {
            "auc": fsum(values) / cap,
            "first_crossing": first,
            "rolling_crossing": stable,
            "persistence_next100": recent,
            "rolling_right_censored": stable is None,
        }
    require(set(result) == jobs, "unexpected, missing, or duplicate jobs")
    return result


def compare_frame(expected, frame, key_fields):
    require(frame.height == len(expected), "published endpoint row count differs")
    rows = {tuple(row[k] for k in key_fields): row for row in frame.iter_rows(named=True)}
    require(set(rows) == set(expected), "published endpoint keys differ")
    for key, metrics in expected.items():
        for field, value in metrics.items():
            found = rows[key][field]
            equal = (
                found is not None and isclose(value, found, abs_tol=1e-12, rel_tol=0)
                if isinstance(value, float)
                else found == value
            )
            require(equal, f"published {field} disagrees with raw traces at {key}")


def summarize_donors(contrasts, donors):
    per_donor = {
        (donor,): {
            field: median(row[field] for (_, d), row in contrasts.items() if d == donor)
            for field in EFFECTS
        }
        for donor in donors
    }
    by_variant = {}
    for variant in ("A", "B"):
        by_variant[variant] = {}
        for field in EFFECTS:
            vals = [row[field] for (donor,), row in per_donor.items() if donor[0] == variant]
            by_variant[variant][field] = {
                "median_pp": 100 * median(vals),
                "range_pp": [100 * min(vals), 100 * max(vals)],
                "positive_donors": sum(v > 0 for v in vals),
                "donors": len(vals),
            }
    return per_donor, by_variant


def check_persistence(expected, table):
    groups = defaultdict(list)
    for (_, condition), metrics in expected.items():
        groups[("R" if condition == "RR" else condition[0], condition[-2:])].append(metrics)
    computed = {}
    for key, values in groups.items():
        eligible = [
            v["persistence_next100"] for v in values if v["persistence_next100"] is not None
        ]
        computed[key] = {
            "n": len(values),
            "first_hits": sum(v["first_crossing"] is not None for v in values),
            "rolling_hits": sum(v["rolling_crossing"] is not None for v in values),
            "eligible": len(eligible),
            "median_persistence_eligible": median(eligible) if eligible else None,
        }
    compare_frame(computed, table, ("variant", "pair"))


def write_report(path, report):
    lines = [
        "# Exploratory crossed-null layer bank · independent audit",
        "",
        (
            f"The completed Molab session yielded {report['jobs']} jobs and {report['trials']} held-out "
            f"trial rows over {report['seeds']} initialization/cue seeds and {report['donor_graphs']} donor rewires."
        ),
        "",
        (
            "For each condition, training used 768 trials and a fixed real PN→KC input. Differences below "
            "are relative to the real/real baseline within seed. The unit summarized in the table is one donor's "
            "median across the same seeds. The values are descriptive."
        ),
        "",
        "| Donor variant | Contrast | Median of donor medians (pp) | Donor range (pp) | Positive donors |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, contrasts in report["auc_by_variant"].items():
        for field, row in contrasts.items():
            lo, hi = row["range_pp"]
            lines.append(
                f"| {variant} | {field} | {row['median_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] "
                f"| {row['positive_donors']}/{row['donors']} |"
            )
    lines.extend(
        [
            "",
            (
                "Censoring is applied at trial 768. The post-crossing fraction is conditional on a first crossing "
                "by trial 668. The groups share seeds across donors."
            ),
            "",
            "| Variant | Layers | Jobs | First hits | Rolling hits | Post-crossing eligible | Median next-100 proportion |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["persistence"]:
        value = row["median_persistence_eligible"]
        m = "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['variant']} | {row['pair']} | {row['n']} | {row['first_hits']} | "
            f"{row['rolling_hits']} | {row['eligible']} | {m} |"
        )
    lines.extend(
        [
            "",
            (
                "The scope of the intervention is computational routing in one connectome, one network "
                "architecture, one cue-family layout, and the fixed donor bank."
            ),
            "",
            "[Molab report and donor plot](REPORT.md)",
            "",
        ]
    )
    atomic_file(path, lambda dest: dest.write_text("\n".join(lines), encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_bytes())
    (summary_path,) = args.archives.glob("summary-*.zip")
    summary_members = (
        "MANIFEST.json",
        "REPORT.md",
        "endpoints.parquet",
        "contrasts.parquet",
        "donor-medians.parquet",
        "persistence-summary.parquet",
        "donor-contrasts.png",
    )
    summary = read_archive(summary_path, summary_members)
    meta = json.loads(summary["MANIFEST.json"])
    require(meta["config_sha256"] == digest(PLAN.read_bytes()), "config differs from the Molab run")
    require(meta["scope"] == plan["scope"], "scope differs from the frozen plan")
    seeds, count = meta["complete_seeds"], meta["completed_panels"]
    require(seeds == plan["seeds"][: count * plan["seeds_per_panel"]], "seed panel order differs")
    tables, committed_files = [], {k: v for k, v in summary.items() if k != "MANIFEST.json"}
    archives = [summary_path]
    wave_count = drive_waves = 0
    for i in range(1, count + 1):
        prefix = f"panel-{i:02d}"
        (archive,) = args.archives.glob(f"{prefix}-*.zip")
        members = read_archive(archive, (f"{prefix}/MANIFEST.json", f"{prefix}/traces.parquet"))
        panel = json.loads(members[f"{prefix}/MANIFEST.json"])
        require(panel["seeds"] == seeds[(i - 1) * 2 : i * 2], f"panel {i} seeds differ")
        require(
            panel["run_signature"]["git_commit"] == meta["git_commit"]
            and panel["run_signature"]["graph_manifests_sha256"] == meta["graph_manifests_sha256"],
            f"panel {i} execution provenance differs",
        )
        trace = members[f"{prefix}/traces.parquet"]
        check_files({"traces.parquet": trace}, {"traces.parquet": panel["files"]["traces.parquet"]})
        committed_files[f"{prefix}/MANIFEST.json"] = members[f"{prefix}/MANIFEST.json"]
        table = pl.read_parquet(io.BytesIO(trace))
        require(
            set(table["seed"]) == set(panel["seeds"]), f"panel {i} trace seed inventory differs"
        )
        require(len(table) == panel["jobs"] * plan["trials"], f"panel {i} trace count differs")
        tables.append(table)
        wave_count += panel["wave_count"]
        drive_waves += panel["waves_verified_on_drive"]
        archives.append(archive)
    check_files(committed_files, meta["files"])
    require(drive_waves == wave_count == meta["wave_receipts_verified"], "wave receipts disagree")
    donors = [f"{v}-{i}" for v, ids in plan["null_instances_by_variant"].items() for i in ids]
    require(set(meta["graph_manifests_sha256"]) == {"real", *donors}, "graph roster differs")
    arms = ["RR", *(f"{donor}-{code}" for donor, code in product(donors, ("NN", "NR", "RN")))]
    jobs, cap = set(product(seeds, arms)), plan["trials"]
    require(len(jobs) == meta["jobs_in_complete_panels"], "completed job inventory differs")
    traces = pl.concat(tables)
    expected = endpoints_for(traces, jobs, cap, 0.75, plan["persistence_window"])
    compare_frame(
        expected, pl.read_parquet(io.BytesIO(summary["endpoints.parquet"])), ("seed", "condition")
    )
    contrasts = {}
    for seed, donor in product(seeds, donors):
        rr = expected[seed, "RR"]["auc"]
        nn, nr, rn = (expected[seed, f"{donor}-{code}"]["auc"] for code in ("NN", "NR", "RN"))
        contrasts[seed, donor] = dict(
            zip(EFFECTS, (nn - rr, nr - rr, rn - rr, nn - nr - rn + rr), strict=True)
        )
    compare_frame(
        contrasts, pl.read_parquet(io.BytesIO(summary["contrasts.parquet"])), ("seed", "donor")
    )
    per_donor, by_variant = summarize_donors(contrasts, donors)
    compare_frame(
        per_donor, pl.read_parquet(io.BytesIO(summary["donor-medians.parquet"])), ("donor",)
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    verified_members = {"MANIFEST.json": summary["MANIFEST.json"], **committed_files}
    for name, data in verified_members.items():
        atomic_file(output / name, lambda dest, data=data: dest.write_bytes(data))
    history = output / "verified-traces.parquet"
    atomic_file(history, traces.write_parquet)
    persistence = pl.read_parquet(io.BytesIO(summary["persistence-summary.parquet"]))
    check_persistence(expected, persistence)
    report = {
        "jobs": len(jobs),
        "trials": len(traces),
        "seeds": len(seeds),
        "donor_graphs": len(donors),
        "total_seconds": meta["wall_seconds"] + meta["elapsed_before_seconds"],
        "panels": count,
        "waves": drive_waves,
        "auc_by_variant": by_variant,
        "persistence": persistence.to_dicts(),
    }
    note = output / "AUDIT.md"
    write_report(note, report)
    products = [output / name for name in verified_members] + [history, note]
    from mbtopology.runtime import sha256 as hash_file

    atomic_json(
        output / "AUDIT_MANIFEST.json",
        {
            **report,
            "source_git_commit": meta["git_commit"],
            "config_sha256": meta["config_sha256"],
            "analysis_sha256": hash_file(__file__),
            "timestamp_utc": now(),
            "archives_sha256": {p.name: hash_file(p) for p in archives},
            "files": {
                p.relative_to(output).as_posix(): {"sha256": hash_file(p), "bytes": p.stat().st_size}
                for p in products
            },
        },
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
