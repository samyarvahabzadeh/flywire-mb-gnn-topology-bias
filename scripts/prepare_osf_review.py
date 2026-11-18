"""Create a local DRAFT review archive; never register, upload or create live OSF evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

import polars as pl

from mbtopology.runner import load_bundle
from mbtopology.runtime import atomic_file, atomic_json, now, read_json, require, sha256

ROOT_FILES = (
    "README.md",
    "PREREGISTRATION.md",
    "EXPERIMENT_SPEC.md",
    "DECISIONS.md",
    "RUN_OVERNIGHT.md",
    "RUN_EXPERIMENT.cmd",
    "pyproject.toml",
    "uv.lock",
    ".gitattributes",
    "configs/week1.yaml",
    "configs/mb_bilateral.yaml",
    "configs/task_engine.yaml",
    "configs/task_toy.yaml",
    "configs/models.yaml",
    "configs/nulls.yaml",
    "configs/experiment.yaml",
    "configs/seeds.txt",
    "configs/odor_sets.json",
    "configs/mbon_readout.json",
    "data/MANIFEST.json",
    "reports/mb_sanity.json",
    "reports/task_inputs.json",
    "reports/task_toy_sanity.json",
    "reports/connection_sanity.json",
    "reports/source_graph_sanity.json",
    "reports/schema_inventory.json",
    "reports/experiment_plan.json",
    "reports/validation.json",
    "reports/engineering_tests.xml",
)


def safe_artifacts(root, candidates):
    root = Path(root).resolve()
    selected = {}
    for candidate in candidates:
        path = (root / candidate).resolve()
        require(
            path.is_relative_to(root) and path.is_file(),
            f"missing/outside review artifact: {candidate}",
        )
        relative = path.relative_to(root)
        require(
            relative.as_posix() == ".gitattributes"
            or not any(part.startswith(".") for part in relative.parts),
            "hidden/private files are excluded from the review package",
        )
        require(
            "token" not in path.name.lower() and path.name != "osf_registration.json",
            "credentials and live registration evidence are excluded",
        )
        require(
            relative.parts[0] != "data" or path.name == "MANIFEST.json",
            "raw and processed dataset payloads are excluded; only data manifests are included",
        )
        require(
            relative.parts[0] not in ("results", "checkpoints", "logs"),
            "learning outputs and checkpoints are excluded",
        )
        selected[relative.as_posix()] = path
    return selected


def write_archive(root, destination, selected, extra_documents, metadata):
    root = Path(root)
    records = {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in sorted(selected.items())
    }
    package_manifest = {
        **metadata,
        "files": records,
        "contains_dataset_payloads": False,
        "contains_credentials": False,
        "uploaded": False,
        "registered": False,
    }

    def write(path):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, source in sorted(selected.items()):
                archive.writestr("project/" + name, source.read_bytes())
            for name, value in extra_documents.items():
                archive.writestr("review/" + name, value)
            archive.writestr(
                "review/PACKAGE_MANIFEST.json",
                json.dumps(package_manifest, indent=2, ensure_ascii=False) + "\n",
            )

    atomic_file(destination, write)
    return {
        **package_manifest,
        "archive": Path(destination).relative_to(root).as_posix(),
        "archive_bytes": Path(destination).stat().st_size,
        "archive_sha256": sha256(destination),
    }


def prepare(root):
    root = Path(root).resolve()
    protected_hash = sha256(root / "PREREGISTRATION.md")
    bundle = load_bundle(root, root / "configs/experiment.yaml")
    plan = read_json(root / "reports/experiment_plan.json")
    require(
        plan and plan["bundle_sha256"] == bundle["bundle_sha256"],
        "regenerate --plan for the current code before packaging",
    )
    require(
        "DRAFT" in (root / "PREREGISTRATION.md").read_text(encoding="utf-8"),
        "this command packages DRAFT review material only",
    )
    prefix = bundle["bundle_sha256"][:12]
    candidates = list(ROOT_FILES)
    candidates += [
        path.relative_to(root).as_posix()
        for name in ("src", "tests", "scripts")
        for path in sorted((root / name).rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    source = read_json(root / bundle["task"]["source_manifest"])
    candidates += [
        bundle["task"]["source_manifest"],
        source["source_manifest"],
        plan["plan_path"],
        str(Path(plan["plan_path"]).with_suffix(".manifest.json")).replace("\\", "/"),
        f"plans/{prefix}/bundle.json",
    ]
    table = pl.read_parquet(root / plan["plan_path"])
    require(len(table) == 3660, "incomplete review execution plan")
    for row in table.iter_rows(named=True):
        require(
            sha256(root / row["run_config_path"]) == row["yaml_sha256"],
            "run YAML changed before packaging",
        )
        candidates.append(row["run_config_path"])
    selected = safe_artifacts(root, candidates)
    template = {
        "status": "TEMPLATE_ONLY_NOT_VALID_EVIDENCE",
        "osf_doi": None,
        "registered_at": None,
        "preregistration_sha256": None,
        "draft_preregistration_sha256": protected_hash,
        "task_config_sha256": bundle["inputs"]["task_config_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "note": "Researcher must register the exact final plan. Never copy this draft as live evidence.",
    }
    notes = f"""# DRAFT OSF review package — not a registration

This local archive contains code, configuration, fixed task inputs, source checksums and
the 3660-job execution plan. It contains no raw/processed dataset payloads, credentials,
learning results or real null graphs. Nothing was uploaded and no account was accessed.

Scientific bundle: {bundle["bundle_sha256"]}
Current DRAFT preregistration SHA-256: {protected_hash}

Before registration the researcher must reconcile the existing DRAFT with accepted
DECISIONS, including the current connections_princeton source name, bilateral KC scope,
global-scalar DAN simplification, the baseline-vs-gradient model integration, PROPOSED
readout/E-I surrogate maps, discrimination definition and trial-cap/right-censoring versus
the separate 10x-median-null exclusion clause. Hypotheses and analysis are not rewritten
by this packaging command. PREREGISTRATION.md is copied unchanged.

REGISTRATION_EVIDENCE_TEMPLATE.json is intentionally invalid and must not be treated as
proof of registration. The live runner requires genuine registration metadata binding the
exact final preregistration hash, task configuration and scientific/code bundle.
See project/RUN_OVERNIGHT.md. Implementation tests use explicitly synthetic fixtures;
their placeholder metadata does not represent an actual OSF registration.
"""
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    destination = root / "dist/osf_review" / prefix / f"osf_review_{prefix}.zip"
    report = write_archive(
        root,
        destination,
        selected,
        {
            "REVIEW_NOTES.md": notes,
            "REGISTRATION_EVIDENCE_TEMPLATE.json": json.dumps(template, indent=2) + "\n",
        },
        {
            "status": "DRAFT_REVIEW_ONLY",
            "created_at": now(),
            "git_commit": git_commit,
            "bundle_sha256": bundle["bundle_sha256"],
            "planned_runs": len(table),
            "draft_preregistration_sha256": protected_hash,
        },
    )
    require(
        sha256(root / "PREREGISTRATION.md") == protected_hash,
        "packaging changed the protected preregistration",
    )
    atomic_json(destination.with_suffix(".manifest.json"), report)
    atomic_json(
        root / "reports/osf_review_package.json",
        {key: value for key, value in report.items() if key != "files"},
    )
    return {key: value for key, value in report.items() if key != "files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=2))
