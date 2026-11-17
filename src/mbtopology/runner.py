from __future__ import annotations

import argparse
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from scipy import sparse

from . import gnn_training, models, nulls
from .experiment_io import SessionYield, cached_result, save_yaml, write_result
from .ingest import record_file
from .runtime import (
    Exclusive,
    Fail,
    Run,
    atomic_file,
    atomic_json,
    fingerprint,
    now,
    provenance,
    read_json,
    require,
    sha256,
)
from .task_inputs import load_source, load_task, require_registered


def load_bundle(root, experiment_path):
    root = Path(root)
    experiment = yaml.safe_load(Path(experiment_path).read_text(encoding="utf-8"))
    require(
        experiment["dataset"] == "fafb" and str(experiment["snapshot"]) == "783",
        "runner source must be fafb/v783",
    )
    require(
        experiment["require_osf"] is True
        and experiment["precision"] == "float32"
        and 0 < experiment["checkpoint_seconds"] <= 900,
        "unsupported execution policy or disabled OSF gate",
    )
    task = load_task(root / experiment["task_config"])
    model_config = yaml.safe_load((root / experiment["models_config"]).read_text(encoding="utf-8"))
    null_config = yaml.safe_load((root / experiment["nulls_config"]).read_text(encoding="utf-8"))
    seeds = [
        int(line)
        for line in (root / task["seeds_file"]).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    require(
        seeds == list(range(30)) and null_config["instances_per_variant"] == 30,
        "scientific plan must use 30 seeds and 30 instances per null variant",
    )
    require(
        model_config["architectures"] == ["GCN", "GAT"] and null_config["variants"] == ["A", "B"],
        "unexpected architecture/null plan",
    )
    require(
        null_config["seed_list"] == task["seeds_file"]
        and not null_config["allow_self_loops"]
        and not null_config["allow_parallel_edges"]
        and null_config["accepted_swaps_per_edge"] > 0,
        "unsupported null topology/seed policy",
    )
    require(
        model_config["input_projection"]
        and model_config["us_as_input_feature"] is False
        and model_config["loss"] == "binary_cross_entropy_on_fixed_valence_readout"
        and model_config["edge_weight_policy"]
        == "task_initialized_KC_MBON_else_source_synapse_count"
        and model_config["probe_all_odors_after_each_trial"],
        "unsupported frozen GNN integration policy",
    )
    require(
        null_config["unknown_sign_label"] == 0
        and null_config["unknown_sign_message_multiplier"] == 1,
        "unsupported unknown-sign modeling policy",
    )
    inputs = read_json(root / "reports/task_inputs.json")
    require(
        inputs and inputs["task_config_sha256"] == fingerprint(task),
        "prepare frozen task inputs before planning",
    )
    for path, key in (
        (task["cs_encoding"]["odor_sets"], "odor_sets_sha256"),
        (task["readout"]["frozen_membership"], "readout_sha256"),
        (task["seeds_file"], "seeds_sha256"),
    ):
        require(sha256(root / path) == inputs[key], "frozen artifact checksum mismatch")
    load_source(root, task["source_manifest"])
    implementation = {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted((root / "src").rglob("*.py"))
        if "__pycache__" not in p.parts
    }
    bundle = {
        "experiment": experiment,
        "task": task,
        "models": model_config,
        "nulls": null_config,
        "seeds": seeds,
        "inputs": inputs,
        "implementation": implementation,
        "source_manifest_sha256": sha256(root / task["source_manifest"]),
        "parameter_budget": models.parameter_budget(model_config),
    }
    bundle["environment_lock"] = {
        name: sha256(root / name) for name in ("uv.lock", "pyproject.toml")
    }
    bundle["bundle_sha256"] = fingerprint(bundle)
    return bundle


def run_specifications(bundle):
    prefix = bundle["bundle_sha256"][:12]
    for architecture in bundle["models"]["architectures"]:
        for seed in bundle["seeds"]:
            for condition in ("real", "A", "B"):
                instances = (
                    [None]
                    if condition == "real"
                    else range(bundle["nulls"]["instances_per_variant"])
                )
                for instance in instances:
                    graph_id = "real" if instance is None else f"null{condition}-{instance:02d}"
                    yield {
                        "run_id": f"{architecture}-{graph_id}-seed{seed:02d}",
                        "architecture": architecture,
                        "seed": seed,
                        "condition": condition,
                        "null_instance": instance,
                        "bundle_sha256": bundle["bundle_sha256"],
                        "bundle_path": f"{bundle['experiment']['plans_dir']}/{prefix}/bundle.json",
                    }


def make_plan(root, bundle, progress=None):
    root = Path(root)
    prefix, experiment = bundle["bundle_sha256"][:12], bundle["experiment"]
    plan_dir = root / experiment["plans_dir"] / prefix
    plan_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = plan_dir / "bundle.json"
    previous = read_json(bundle_path)
    require(
        previous is None or fingerprint(previous) == fingerprint(bundle),
        "immutable plan bundle changed",
    )
    if previous is None:
        atomic_json(bundle_path, bundle)
    rows = []
    for index, spec in enumerate(run_specifications(bundle)):
        path = root / experiment["run_configs_dir"] / prefix / f"{spec['run_id']}.yaml"
        if path.exists():
            require(
                yaml.safe_load(path.read_text(encoding="utf-8")) == spec,
                "run YAML differs from immutable plan",
            )
        else:
            save_yaml(path, spec)
        rows.append(
            {
                **spec,
                "run_config_path": path.relative_to(root).as_posix(),
                "run_config_sha256": fingerprint(spec),
                "yaml_sha256": sha256(path),
            }
        )
        if progress and index % 100 == 0:
            progress.save(planned_runs=index + 1)
    frame = pl.DataFrame(rows)
    require(
        len(frame) == 3660 and frame["run_id"].n_unique() == 3660,
        "run matrix incomplete or duplicated",
    )
    meta = provenance(root, experiment)
    write_result(
        root,
        plan_dir / "plan.parquet",
        frame,
        {
            "git_commit": meta["git_commit"],
            "run_config_sha256": fingerprint(experiment),
            "bundle_sha256": bundle["bundle_sha256"],
            "scope": "EXECUTION_PLAN_ONLY_NO_MODEL_METRICS",
            "source_sha256": meta["source_sha256"],
            "model_metrics_computed": False,
        },
    )
    try:
        require_registered(root, bundle["task"], bundle["bundle_sha256"])
        gate = "registration_evidence_present"
    except Fail as exc:
        gate = str(exc)
    report = {
        "status": "PLAN_READY",
        "planned_runs": len(frame),
        "seeds": 30,
        "real_graphs": 1,
        "null_graphs_per_variant": 30,
        "architectures": ["GCN", "GAT"],
        "parameter_budget": bundle["parameter_budget"],
        "bundle_sha256": bundle["bundle_sha256"],
        "plan_path": (plan_dir / "plan.parquet").relative_to(root).as_posix(),
        "osf_gate": gate,
        "executes_models": False,
        "model_metrics_computed": False,
    }
    atomic_json(root / "reports/experiment_plan.json", report)
    return frame, report


def materialize_null(root, bundle, variant, instance, progress, deadline):
    root = Path(root)
    prefix = bundle["bundle_sha256"][:12]
    output = root / "data" / "nulls" / prefix / f"{variant}-{instance:02d}"
    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        require(saved["bundle_sha256"] == bundle["bundle_sha256"], "null bundle changed")
        load_source(root, manifest_path.relative_to(root))
        return manifest_path.relative_to(root).as_posix()
    source, nodes, matrix = load_source(root, bundle["task"]["source_manifest"])
    coo = matrix.tocoo(copy=False)
    node_signs = nulls.signs_from_nt(nodes["nt_type"].to_list(), bundle["nulls"]["nt_signs"])
    chain = nulls.DirectedSwapChain(matrix, node_signs[coo.row], variant, instance)
    checkpoint = root / "checkpoints" / "nulls" / prefix / f"{variant}-{instance:02d}"
    chain.restore(checkpoint)
    target = int(matrix.nnz * bundle["nulls"]["accepted_swaps_per_edge"])

    def update(**values):
        progress.save(null_variant=variant, null_instance=instance, **values)
        if deadline is not None and time.monotonic() >= deadline:
            raise SessionYield("null state checkpointed at session deadline")

    report = chain.run(
        target,
        target * bundle["nulls"]["max_attempts_per_accepted_swap"],
        checkpoint,
        bundle["nulls"]["checkpoint_seconds"],
        update,
    )
    output.mkdir(parents=True, exist_ok=True)
    csr_path = output / "null_csr.npz"
    atomic_file(csr_path, lambda p: sparse.save_npz(p, chain.as_csr()))
    manifest = {
        "status": "PASS",
        "bundle_sha256": bundle["bundle_sha256"],
        "variant": variant,
        "instance": instance,
        "report": report,
        "source_manifest": bundle["task"]["source_manifest"],
        "artifacts": {"nodes": source["artifacts"]["nodes"], "csr": record_file(root, csr_path)},
    }
    atomic_json(manifest_path, manifest)
    return manifest_path.relative_to(root).as_posix()


def execution_environment(root, bundle):
    configured = bundle["models"]["device"]
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if configured == "auto" else configured
    )
    require(
        device in ("cpu", "cuda") and (device != "cuda" or torch.cuda.is_available()),
        "requested device unavailable",
    )
    environment = {
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyg": importlib.metadata.version("torch-geometric"),
        "bundle_sha256": bundle["bundle_sha256"],
    }
    path = (
        Path(root)
        / bundle["experiment"]["plans_dir"]
        / bundle["bundle_sha256"][:12]
        / "execution_environment.json"
    )
    previous = read_json(path)
    require(
        previous is None or previous == environment,
        "execution device/library versions changed; do not mix conditions",
    )
    if previous is None:
        atomic_json(path, environment)
    return device, environment


def execute(root, bundle, frame, progress):
    root = Path(root)
    registration = require_registered(root, bundle["task"], bundle["bundle_sha256"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bundle["experiment"]["deterministic_algorithms"])
    device, environment = execution_environment(root, bundle)
    deadline = time.monotonic() + bundle["experiment"]["session_hours"] * 3600
    prefix = bundle["bundle_sha256"][:12]
    completed = 0
    for row in frame.iter_rows(named=True):
        if time.monotonic() >= deadline:
            raise SessionYield("execution ledger checkpointed; resume the same command")
        spec = yaml.safe_load((root / row["run_config_path"]).read_text(encoding="utf-8"))
        require(
            fingerprint(spec) == row["run_config_sha256"]
            and sha256(root / row["run_config_path"]) == row["yaml_sha256"],
            "run configuration changed after planning",
        )
        result_path = (
            root / bundle["experiment"]["results_dir"] / prefix / f"{spec['run_id']}.parquet"
        )
        if cached_result(result_path, row["run_config_sha256"], bundle["bundle_sha256"]):
            completed += 1
            continue
        graph_manifest = (
            bundle["task"]["source_manifest"]
            if spec["condition"] == "real"
            else materialize_null(
                root, bundle, spec["condition"], spec["null_instance"], progress, deadline
            )
        )
        progress.save(current_run=spec["run_id"], completed_runs=completed)
        outcome = gnn_training.run_registered(
            root, bundle, spec, graph_manifest, device, progress, deadline
        )
        require(
            outcome["parameter_count"]
            == bundle["parameter_budget"]["counts"][spec["architecture"]],
            "parameter budget changed by condition",
        )
        table = pl.DataFrame(outcome.pop("history")).with_columns(
            pl.lit(spec["run_id"]).alias("run_id"),
            pl.lit(spec["seed"]).alias("seed"),
            pl.lit(spec["architecture"]).alias("architecture"),
            pl.lit(spec["condition"]).alias("condition"),
            pl.lit(spec["null_instance"], dtype=pl.Int64).alias("null_instance"),
        )
        trace = provenance(root, spec)
        write_result(
            root,
            result_path,
            table,
            {
                **trace,
                **outcome,
                "git_commit": trace["git_commit"],
                "run_config_sha256": row["run_config_sha256"],
                "config_file_sha256": row["yaml_sha256"],
                "bundle_sha256": bundle["bundle_sha256"],
                "graph_manifest": graph_manifest,
                "graph_manifest_sha256": sha256(root / graph_manifest),
                "task_inputs": bundle["inputs"],
                "environment": environment,
                "registration": registration,
                "run_specification": spec,
                "model_metrics_computed": True,
            },
        )
        completed += 1
        atomic_json(
            root / "reports/experiment_execution.json",
            {
                "status": "RUNNING",
                "completed_runs": completed,
                "planned_runs": len(frame),
                "updated_at": now(),
                "bundle_sha256": bundle["bundle_sha256"],
            },
        )
    report = {
        "status": "COMPLETE",
        "completed_runs": completed,
        "planned_runs": len(frame),
        "bundle_sha256": bundle["bundle_sha256"],
    }
    atomic_json(root / "reports/experiment_execution.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plan safely; execute only after verified local OSF registration evidence"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        path = args.config or root / "configs/experiment.yaml"
        if args.execute:
            experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
            require_registered(root, load_task(root / experiment["task_config"]))
        bundle = load_bundle(root, path)
        with Exclusive(root), Run(root, bundle["experiment"], "experiment") as progress:
            frame, report = make_plan(root, bundle, progress)
            if args.execute:
                try:
                    report = execute(root, bundle, frame, progress)
                except SessionYield as exc:
                    report = {
                        "status": "SESSION_CHECKPOINTED",
                        "reason": str(exc),
                        "resume": "python src/run_experiment.py --execute",
                    }
                    atomic_json(root / "reports/experiment_session.json", report)
        import json

        print(json.dumps(report, indent=2))
        return 0
    except Fail as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
