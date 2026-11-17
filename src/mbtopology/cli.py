from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

from . import graph, ingest, mb_subset, task_engine, task_inputs, task_toy, toy
from .runtime import (
    Exclusive,
    Fail,
    Run,
    atomic_json,
    fingerprint,
    load_config,
    now,
    read_json,
    require,
    sha256,
)


def environment(root):
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "polars",
            "pyarrow",
            "scipy",
            "torch",
            "torch-geometric",
            "networkx",
            "matplotlib",
        )
    }
    report = {
        "created_at": now(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "data_root": str(Path(root).resolve()),
        "free_disk_bytes": shutil.disk_usage(root).free,
        "model_metrics_computed": False,
    }
    require(sys.version_info >= (3, 11), "Python >=3.11 required")
    atomic_json(Path(root) / "reports" / "environment.json", report)
    return report


def status_report(root, config):
    root = Path(root)
    manifest = read_json(root / "data" / "MANIFEST.json", {"artifacts": {}})
    sanity = read_json(root / "reports" / "sanity.json")
    toy_result = read_json(root / "reports" / "toy_sanity.json")
    graph_manifest = read_json(root / "data" / "graphs" / "MANIFEST.json")
    blockers = []
    for product in config["products"]:
        if f"raw/{product}" not in manifest["artifacts"]:
            blockers.append(f"raw/{product} is missing")
        if f"parquet/{product}" not in manifest["artifacts"]:
            blockers.append(f"parquet/{product} is missing")
    for key, item in manifest["artifacts"].items():
        for artifact in item.get("shards", [item]):
            path = root / artifact["path"]
            if not path.exists() or sha256(path) != artifact["sha256"]:
                blockers.append(f"checksum/missing-file failure for {key}")
    if not config["approval"]["reviewed_by"]:
        blockers.append("human schema and semantic review pending")
    if not sanity or sanity["status"] != "PASS":
        blockers.append("real graph sanity checks have not passed")
    if not toy_result or toy_result["status"] != "PASS":
        blockers.append("synthetic toy mechanics check has not passed")
    if not graph_manifest or graph_manifest.get("status") != "PASS":
        blockers.append("completed graph manifest is missing")
    else:
        if graph_manifest["provenance"]["config_sha256"] != fingerprint(config):
            blockers.append("graph config differs from current config")
        for key, item in graph_manifest["artifacts"].items():
            path = root / item["path"]
            if not path.exists() or sha256(path) != item["sha256"]:
                blockers.append(f"graph checksum/missing-file failure for {key}")
    complete = not blockers
    result = {
        "created_at": now(),
        "status": "PASS" if complete else "INCOMPLETE",
        "blockers": blockers,
        "data_artifacts": manifest["artifacts"],
        "sanity": sanity,
        "toy_sanity": toy_result,
        "downloaded_raw_bytes": sum(
            item["bytes"] for key, item in manifest["artifacts"].items() if key.startswith("raw/")
        ),
        "parquet_bytes": sum(
            item["bytes"]
            for key, item in manifest["artifacts"].items()
            if key.startswith("parquet/")
        ),
        "decisions_sha256": sha256(root / "DECISIONS.md"),
        "model_metrics_computed": False,
    }
    atomic_json(root / "reports" / "week1_status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Week-one ingestion and sanity checks only")
    parser.add_argument(
        "command",
        choices=(
            "env",
            "catalogue",
            "download",
            "inspect",
            "convert",
            "values",
            "graph",
            "toy",
            "report",
            "profile",
            "full-graph",
            "mb-subset",
            "task-prepare",
            "task-toy",
            "task-run",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--product", action="append")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = args.config or root / "configs" / "week1.yaml"
    try:
        config = load_config(config_path)
        with Exclusive(root), Run(root, config, args.command) as run:
            if args.command == "env":
                result = environment(root)
            elif args.command == "catalogue":
                result = ingest.catalogue(
                    root,
                    config,
                    ingest.read_token(args.token_file or root / ".local" / "codex_token.txt"),
                )
            elif args.command == "download":
                products = args.product or list(config["products"])
                require(set(products) <= set(config["products"]), "unknown product key")
                token = ingest.read_token(args.token_file or root / ".local" / "codex_token.txt")
                result = {
                    product: ingest.download(root, config, product, token, run)
                    for product in products
                }
            elif args.command == "inspect":
                result = ingest.inspect_raw(root, config)
                result = {
                    **result,
                    "inventory_sha256": sha256(root / "reports" / "schema_inventory.json"),
                    "next_action": "STOP: human schema/product review before conversion",
                }
            elif args.command == "convert":
                inventory = ingest.approved(root, config)
                products = args.product or list(config["products"])
                require(set(products) <= set(config["products"]), "unknown product key")
                result = {
                    product: ingest.convert_product(root, config, product, inventory, run)
                    for product in products
                }
            elif args.command == "values":
                ingest.approved(root, config)
                result = ingest.value_inventory(root, config)
            elif args.command == "profile":
                result = ingest.connection_profile(root, config, run)
            elif args.command == "graph":
                result = graph.build_graph(root, config, run)
            elif args.command == "full-graph":
                result = graph.build_full_source(root, config, run)
            elif args.command == "mb-subset":
                result = mb_subset.build(
                    root, args.selection_config or root / "configs/mb_bilateral.yaml", run
                )
            elif args.command == "task-prepare":
                task_config = task_inputs.load_task(
                    args.task_config or root / "configs/task_engine.yaml"
                )
                result = task_inputs.prepare(root, task_config, run)
            elif args.command == "task-toy":
                task_config = task_inputs.load_task(
                    args.task_config or root / "configs/task_engine.yaml"
                )
                result = task_toy.run_toy(root, task_config, run)
            elif args.command == "task-run":
                task_config = task_inputs.load_task(
                    args.task_config or root / "configs/task_engine.yaml"
                )
                engine = task_engine.run_biological(root, task_config, args.seed, run)
                result = {
                    "status": "COMPLETE",
                    "trials": engine.state.cursor,
                    "trials_to_criterion": engine.state.reached_at,
                    "right_censored": engine.state.reached_at is None,
                }
            elif args.command == "toy":
                result = toy.toy_check(root, run)
            else:
                result = status_report(root, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Fail as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — redact credentials from unexpected exceptions.
        print(
            f"FAIL: {type(exc).__name__}; inspect local inputs/column mappings. "
            "Raw exception text withheld to protect authenticated URLs.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
