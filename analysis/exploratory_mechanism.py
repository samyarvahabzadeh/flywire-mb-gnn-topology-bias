"""Describe topology and discrimination dynamics from the completed v1.2 sweep."""

import argparse
import io
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import polars as pl
from confirmatory import put_immutable, transport_retry
from confirmatory_data import (
    CAP,
    CONFIG_SHA,
    PLAN_SHA,
    SEEDS,
    digest,
    directory_reader,
    frozen_jobs,
    read_dataset,
    require,
)
from exploratory_graph_metrics import graph_metrics
from exploratory_mechanism_report import write_report
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
REAL_GRAPH = "data/graphs/mb_bilateral_v783/MANIFEST.json"
RESULT_TAG = "v2.0-confirmatory-results"
RESULT_SUMMARY = "results/confirmatory/analysis-summary.json"
WINDOW = 100


def verified_member(archive, receipt, name):
    data = archive.read(name)
    record = receipt["files"][name]
    require(
        len(data) == record["bytes"] and digest(data) == record["sha256"],
        f"graph member mismatch: {name}",
    )
    return data


def read_graphs(read):
    graphs, receipt_hashes = {}, []
    expected = [f"{condition}-{i:02d}" for condition in ("A", "B") for i in SEEDS]
    labels = ["real-source", *(f"graph-bank-{i:03d}" for i in range(10))]
    for index, label in enumerate(labels):
        receipt_raw = read(f"{label}.receipt.json")
        receipt = json.loads(receipt_raw)
        receipt_hashes.append(digest(receipt_raw))
        archive_name = f"{label}-{receipt['archive_sha256']}.zip"
        require(
            receipt["label"] == label
            and receipt["plan_sha256"] == PLAN_SHA
            and receipt["archive"] == archive_name,
            f"invalid topology receipt: {label}",
        )
        blob = read(archive_name)
        require(
            len(blob) == receipt["archive_bytes"] and digest(blob) == receipt["archive_sha256"],
            f"invalid archive: {label}",
        )
        entries = {"real": REAL_GRAPH} if index == 0 else receipt["metadata"]["graphs"]
        require(
            index == 0 or set(entries) == set(expected[(index - 1) * 6 : index * 6]),
            f"invalid graph inventory: {label}",
        )
        with ZipFile(io.BytesIO(blob)) as archive:
            require(
                sorted(archive.namelist()) == sorted(receipt["files"]),
                f"topology archive members changed: {label}",
            )
            for graph_id, manifest_path in entries.items():
                manifest = verified_member(archive, receipt, manifest_path)
                doc = json.loads(manifest)
                require(doc["status"] == "PASS", f"missing PASS graph status for {graph_id}")
                csr_spec = doc["artifacts"]["csr"]
                csr = verified_member(archive, receipt, csr_spec["path"])
                require(
                    digest(csr) == csr_spec["sha256"] and len(csr) == csr_spec["bytes"],
                    f"CSR integrity failed: {graph_id}",
                )
                graphs[graph_id] = (digest(manifest), digest(csr), csr)
        print(f"EXPLORATORY_GRAPH_ARCHIVE {index + 1}/{len(labels)} {len(graphs)}/61", flush=True)
    require(set(graphs) == {"real", *expected}, "incomplete graph bank")
    require(len({g[1] for g in graphs.values()}) == 61, "duplicated CSR graph artifacts")
    return graphs, digest("".join(receipt_hashes).encode())


def read_jobs(read, graphs):
    hashes = {}

    def capture_graph_hashes(filename):
        blob = read(filename)
        if re.fullmatch(r"batch-\d{4}-[0-9a-f]{64}\.zip", filename):
            with ZipFile(io.BytesIO(blob)) as archive:
                for name in archive.namelist():
                    if name.endswith(".manifest.json"):
                        job_id = Path(name).name.removesuffix(".manifest.json")
                        require(job_id not in hashes, f"duplicate manifest for {job_id}")
                        hashes[job_id] = json.loads(archive.read(name))["graph_sha256"]
        return blob

    def progress(batches, jobs):
        if batches % 20 == 0 or jobs == 1830:
            print(f"EXPLORATORY_JOB_VERIFIED {batches}/305 {jobs}/1830", flush=True)

    jobs, provenance = read_dataset(capture_graph_hashes, progress)
    specifications = frozen_jobs()
    require(set(hashes) == {spec[0] for spec in specifications}, "job graph-hash inventory differs")
    rows = []
    for job, (job_id, seed, condition, instance) in zip(jobs, specifications, strict=True):
        graph_id = "real" if condition == "real" else f"{condition}-{instance:02d}"
        require(
            hashes[job_id] == graphs[graph_id][0], f"job/graph manifest hash mismatch for {job_id}"
        )
        require(
            job.seed == seed and job.condition == condition, f"job metadata differs for {job_id}"
        )
        tail = job.observed[-WINDOW:]
        ttc = CAP if job.ttc is None else job.ttc
        rows.append(
            {
                "job_id": job_id,
                "seed": seed,
                "condition": condition,
                "instance": instance,
                "graph_id": graph_id,
                "graph_manifest_sha256": hashes[job_id],
                "ttc_display": ttc,
                "ttc_observed": job.ttc,
                "ttc_censored": job.ttc is None,
                "first_crossing": job.single_crossing,
                "crossing_to_stop_or_cap": None
                if job.single_crossing is None
                else ttc - job.single_crossing,
                "observed_trials": int(job.observed.size),
                "tail_trials": int(tail.size),
                "final_window_sd": float(tail.std(ddof=1)),
            }
        )
    require(
        all(row["tail_trials"] >= 2 and np.isfinite(row["final_window_sd"]) for row in rows),
        "invalid volatility tail",
    )
    require(
        Counter(row["graph_id"] for row in rows) == Counter({graph_id: 30 for graph_id in graphs}),
        "unequal seeds per topology",
    )
    return rows, provenance


def metrics_from_bytes(data):
    return graph_metrics(sparse.load_npz(io.BytesIO(data)))


def write_atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def git(*command):
    return subprocess.check_output(["git", *command], cwd=ROOT)


def run(read, output, workers):
    started = datetime.now(UTC)
    commit = git("rev-parse", "HEAD").decode().strip()
    sources = {}
    for name in (
        "exploratory_mechanism",
        "exploratory_graph_metrics",
        "exploratory_mechanism_report",
        "confirmatory_data",
    ):
        path = f"analysis/{name}.py"
        data = (ROOT / path).read_bytes()
        require(
            digest(data.replace(b"\r\n", b"\n")) == digest(git("show", f"{commit}:{path}")),
            f"working source differs from commit: {path}",
        )
        sources[path] = digest(data)
    frozen = git("show", f"{RESULT_TAG}:{RESULT_SUMMARY}")
    confirmatory = json.loads(frozen)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "MANIFEST.json").exists(), "exploratory output is already complete")
    metrics_cache = output.parent / (output.name + "-checkpoint")
    metrics_cache.mkdir(exist_ok=True)
    versions = {p: version(p) for p in ("numpy", "scipy", "networkx", "polars", "matplotlib")}
    signature = digest(json.dumps([sources, versions], sort_keys=True).encode())
    graphs, graph_sequence = read_graphs(read)
    graph_rows = {}
    with (
        ProcessPoolExecutor(max_workers=workers) as pool,
        ThreadPoolExecutor(max_workers=1) as input_reader,
    ):
        pending = {}
        for graph_id, (manifest_sha, csr_sha, csr) in graphs.items():
            key = digest(f"{signature}:{manifest_sha}:{csr_sha}".encode())
            checkpoint = metrics_cache / f"{key}.json"
            metadata = {
                "graph_id": graph_id,
                "graph_manifest_sha256": manifest_sha,
                "csr_sha256": csr_sha,
            }
            if checkpoint.exists():
                row = json.loads(checkpoint.read_bytes())
                require(
                    all(row[k] == v for k, v in metadata.items()),
                    f"checkpoint identity changed for {graph_id}",
                )
                graph_rows[graph_id] = row
            else:
                pending[pool.submit(metrics_from_bytes, csr)] = (graph_id, metadata, checkpoint)
        job_input = input_reader.submit(read_jobs, read, graphs)
        for task in as_completed(pending):
            graph_id, metadata, checkpoint = pending[task]
            row = metadata | task.result()
            write_atomic(checkpoint, json.dumps(row, allow_nan=False).encode())
            graph_rows[graph_id] = row
            print(f"EXPLORATORY_GRAPH_METRICS {len(graph_rows)}/61 {graph_id}", flush=True)
        jobs, provenance = job_input.result()
    for field, value in provenance.items():
        require(confirmatory[field] == value, f"data differs from tagged results: {field}")
    topology = pl.DataFrame(list(graph_rows.values())).sort("graph_id")
    joined = pl.DataFrame(jobs).join(
        topology, on=["graph_id", "graph_manifest_sha256"], validate="m:1"
    )
    require(joined.height == 1830 and topology.height == 61, "final exploratory join incomplete")
    topology.write_parquet(output / "topology-metrics.parquet")
    joined.write_parquet(output / "job-mechanism.parquet")
    write_report(topology, joined, confirmatory, output)
    manifest = {
        "scope": "EXPLORATORY_ONLY",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "config_sha256": CONFIG_SHA,
        "plan_sha256": PLAN_SHA,
        "confirmatory_results_tag": RESULT_TAG,
        "tagged_summary_sha256": digest(frozen),
        "graph_receipt_sequence_sha256": graph_sequence,
        "source_sha256": sources,
        "versions": versions,
        "jobs": joined.height,
        "distinct_graphs": topology.height,
        **provenance,
        "files": {
            p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
            for p in sorted(output.iterdir())
        },
    }
    write_atomic(
        output / "MANIFEST.json", (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    )


def publish(fs, prefix, output):
    fs.makedirs(prefix, exist_ok=True)
    bundle = io.BytesIO()
    with ZipFile(bundle, "w", ZIP_DEFLATED, compresslevel=3) as zf:
        for file in sorted(output.iterdir()):
            zf.write(file, file.name)
    payload = bundle.getvalue()
    filename = "exploratory-mechanism.zip"
    transport_retry(lambda: put_immutable(fs, prefix, filename, payload))
    print(f"EXPLORATORY_COMPLETE {filename} {digest(payload)} {len(payload)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archives", type=Path)
    source.add_argument("--drive-prefix")
    parser.add_argument("--export-prefix")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.archives:
        run(directory_reader(args.archives), args.output, args.workers)
        return
    from gdrive_fsspec import GoogleDriveFileSystem

    require(
        args.export_prefix and args.drive_prefix.rstrip("/") != args.export_prefix.rstrip("/"),
        "choose a separate Drive export prefix",
    )
    fs = GoogleDriveFileSystem(
        use_listings_cache=False,
        skip_instance_cache=True,
        auth_kwargs={"use_local_webserver": False},
    )
    source_prefix = args.drive_prefix.rstrip("/")
    if not (args.output / "MANIFEST.json").exists():
        run(
            lambda filename: transport_retry(lambda: fs.cat_file(f"{source_prefix}/{filename}")),
            args.output,
            args.workers,
        )
    publish(fs, args.export_prefix, args.output)


if __name__ == "__main__":
    main()
