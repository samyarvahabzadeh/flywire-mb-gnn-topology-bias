"""Read and independently check exactly the frozen v1.2 confirmatory data."""

import hashlib
import io
import json
from dataclasses import dataclass
from math import fsum, isclose
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl

PLAN_SHA = "149776bcf90f9d2a9ce5455d1c6e9b2cf2b97f8f9e1d56d76b7828121f6ee23f"
CONFIG_SHA = "baf78c324c31eaf9375af24a76094ec6a1e391d81b83824f0a637eb264d976ec"
WRITER_COMMIT = "da7b9cb802f78b4eee5c598023c8dc350fd41250"
AMENDMENT = "50fcd6000781c27cdb96ce7f6258b75193f62c22"
SCOPE = "CONFIRMATORY_V1_2_FRESH_ONLY"
CAP = 2000
WINDOW = 20
CRITERION = 0.75
LEVELS = (0.0, 0.1, 0.2, 0.3)
SEEDS = tuple(range(30))
CONDITIONS = ("real", "A", "B")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(check, message):
    if not check:
        raise ValueError(message)


@dataclass(frozen=True)
class Job:
    seed: int
    condition: str
    ttc: int | None
    single_crossing: int | None
    auc: float
    noise_scores: tuple[float, ...]
    slope: float
    observed: np.ndarray
    terminal_mean: float


def frozen_jobs():
    return [
        (
            f"seed{seed:02d}-{condition}" + ("" if instance is None else f"-{instance:02d}"),
            seed,
            condition,
            instance,
        )
        for seed in SEEDS
        for condition in CONDITIONS
        for instance in ((None,) if condition == "real" else SEEDS)
    ]


def check_frozen_source(read):
    plan = json.loads(read("frozen-plan.json"))
    require(plan.pop("plan_sha256") == PLAN_SHA, "Drive plan identifier changed")
    require(
        digest(json.dumps(plan, sort_keys=True).encode()) == PLAN_SHA, "frozen plan bytes differ"
    )
    require(
        plan["scope"] == SCOPE
        and plan["amendment_tag_object"] == AMENDMENT
        and tuple(plan["seeds"]) == SEEDS
        and tuple(plan["null_instance_ids"]) == SEEDS
        and plan["planned_jobs"] == 1830
        and plan["batch_width"] == 6
        and plan["architecture"] == "GraphSAGE-max",
        "unexpected frozen experimental matrix",
    )
    excluded = plan["excluded_marked"]
    require(
        len(excluded) == 6
        and all(row["source_run"] == "c3df5b2c7dfc" and row["excluded_marked"] for row in excluded)
        and {(row["seed"], row["condition"]) for row in excluded}
        == {(seed, cond) for seed in (1337, 2026) for cond in CONDITIONS},
        "benchmark exclusions changed",
    )
    complete_raw = read("verified-complete.json")
    complete = json.loads(complete_raw)
    require(
        complete["plan_sha256"] == PLAN_SHA
        and complete["config_sha256"] == CONFIG_SHA
        and complete["git_commit"] == WRITER_COMMIT
        and complete["verified_jobs"] == 1830,
        "external completion marker does not certify this analysis",
    )
    lock = json.loads(read("execution-lock.json"))
    require(
        lock["plan_sha256"] == PLAN_SHA
        and lock["scope"] == SCOPE
        and lock["amendment"]["tag_object"] == AMENDMENT,
        "execution lock does not identify the registered analysis source",
    )
    return complete, digest(complete_raw), lock["bundle_sha256"]


def parse_job(parquet, doc, spec):
    run_id, seed, condition, _instance = spec
    require(
        doc["complete"] is True
        and (doc["run_id"], doc["seed"], doc["condition"], doc["null_instance"]) == spec
        and doc["git_commit"] == WRITER_COMMIT
        and doc["config_sha256"] == CONFIG_SHA
        and doc["scope"] == SCOPE
        and doc["plan_sha256"] == PLAN_SHA
        and doc["architecture"] == "GraphSAGE-max"
        and doc["amendment"]["tag_object"] == AMENDMENT
        and doc["parquet_sha256"] == digest(parquet),
        f"manifest provenance differs for {run_id}",
    )
    table = pl.read_parquet(io.BytesIO(parquet))
    expected = {
        "trial",
        "discrimination",
        "heldout_a_plus",
        "heldout_a_minus",
        "rolling_discrimination",
    }
    require(set(table.columns) == expected, f"unexpected Parquet schema for {run_id}")
    outcome = doc["outcome"]
    trials = len(table)
    require(
        WINDOW <= trials <= CAP
        and doc["rows"] == trials == outcome["trials"]
        and np.array_equal(table["trial"].to_numpy(), np.arange(1, trials + 1)),
        f"invalid trial count or sequence for {run_id}",
    )
    observed = table["discrimination"].to_numpy().astype(np.float64)
    plus = table["heldout_a_plus"].to_numpy()
    minus = table["heldout_a_minus"].to_numpy()
    require(
        np.all(np.isfinite(observed))
        and np.all((observed >= 0) & (observed <= 1))
        and np.array_equal(observed, (plus + 1 - minus) / 2),
        f"invalid held-out scores for {run_id}",
    )
    rolling = np.array([fsum(observed[i - WINDOW : i]) / WINDOW for i in range(WINDOW, trials + 1)])
    saved = table["rolling_discrimination"]
    require(
        saved[: WINDOW - 1].null_count() == WINDOW - 1
        and np.array_equal(rolling, saved[WINDOW - 1 :].to_numpy()),
        f"reported rolling means differ for {run_id}",
    )
    crossing = np.flatnonzero(rolling >= CRITERION)
    ttc = int(crossing[0]) + WINDOW if crossing.size else None
    singles = np.flatnonzero(observed >= CRITERION)
    first = int(singles[0]) + 1 if singles.size else None
    terminal = float(rolling[-1])
    auc = (fsum(observed) + (CAP - trials) * terminal) / CAP
    require(
        trials == (CAP if ttc is None else ttc)
        and outcome["ttc"] == ttc
        and outcome["right_censored"] is (ttc is None)
        and outcome["first_single_crossing"] == first
        and outcome["terminal_rolling_discrimination"] == terminal
        and isclose(outcome["auc_to_cap_normalized"], auc, rel_tol=0, abs_tol=1e-13),
        f"registered endpoint or early stopping mismatch for {run_id}",
    )
    robust = doc["robustness"]
    dose = robust["discrimination_by_level"]
    require(set(dose) == {str(x) for x in LEVELS}, f"unexpected noise doses for {run_id}")
    scores = tuple(float(dose[str(level)]) for level in LEVELS)
    slope = float(robust["slope"])
    require(
        all(0 <= s <= 1 for s in scores)
        and isclose(slope, float(np.polyfit(LEVELS, scores, 1)[0]), rel_tol=1e-10, abs_tol=1e-13),
        f"noise robustness slope differs for {run_id}",
    )
    return Job(seed, condition, ttc, first, auc, scores, slope, observed, terminal)


def read_dataset(read, progress):
    complete, completion_sha, bundle_sha = check_frozen_source(read)
    specifications = frozen_jobs()
    jobs = []
    receipt_hashes = []
    for start in range(0, len(specifications), 6):
        label = f"batch-{start // 6:04d}"
        batch = specifications[start : start + 6]
        raw = read(f"{label}.receipt.json")
        receipt_hashes.append(digest(raw))
        receipt = json.loads(raw)
        require(
            receipt["plan_sha256"] == PLAN_SHA
            and receipt["label"] == label
            and receipt["metadata"]["jobs"] == [spec[0] for spec in batch],
            f"receipt job inventory differs for {label}",
        )
        filename = f"{label}-{receipt['archive_sha256']}.zip"
        require(receipt["archive"] == filename, f"unexpected archive path for {label}")
        blob = read(filename)
        require(
            len(blob) == receipt["archive_bytes"] and digest(blob) == receipt["archive_sha256"],
            f"archive hash differs for {label}",
        )
        with ZipFile(io.BytesIO(blob)) as archive:
            require(
                sorted(archive.namelist()) == sorted(receipt["files"]),
                f"unexpected members in {label}",
            )
            for spec in batch:
                base = f"results/final_binary_sweep/{PLAN_SHA[:12]}/{spec[0]}"
                outputs = {}
                for ext in (".parquet", ".manifest.json"):
                    name = base + ext
                    data = archive.read(name)
                    record = receipt["files"][name]
                    require(
                        len(data) == record["bytes"] and digest(data) == record["sha256"],
                        f"member hash differs for {name}",
                    )
                    outputs[ext] = data
                doc = json.loads(outputs[".manifest.json"])
                require(
                    doc["bundle_sha256"] == bundle_sha and doc["parquet_path"] == base + ".parquet",
                    f"bundle or path mismatch for {spec[0]}",
                )
                jobs.append(parse_job(outputs[".parquet"], doc, spec))
        progress(start // 6 + 1, len(jobs))
    require(
        len(jobs) == complete["verified_jobs"]
        and sum(j.ttc is None for j in jobs) == complete["right_censored"],
        "verified outcomes disagree with the final sweep marker",
    )
    provenance = {
        "completion_marker_sha256": completion_sha,
        "receipt_sequence_sha256": digest("".join(receipt_hashes).encode()),
        "archive_count": len(receipt_hashes),
        "bundle_sha256": bundle_sha,
    }
    return jobs, provenance


def directory_reader(directory):
    return lambda filename: (Path(directory) / filename).read_bytes()
