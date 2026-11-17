from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from scipy import sparse
from torch_geometric.data import Data

from .ingest import record_file
from .runtime import atomic_file, atomic_json, fingerprint, read_json, require, sha256


def select_nodes(nodes, config):
    require(
        config["scope"] == "bilateral", "MB configuration must explicitly select bilateral scope"
    )
    rules = config["roles"]
    masks = {
        role: nodes["class"].fill_null("") == rules[role]["class_equals"]
        for role in ("KC", "PN", "MBON")
    }
    dan_rule = rules["DAN"]
    labels = nodes[dan_rule["type_column"]].fill_null("")
    prefix = pl.Series("prefix", [False] * len(nodes))
    for value in dan_rule["starts_with"]:
        prefix = prefix | labels.str.starts_with(value)
    masks["DAN"] = prefix & (nodes[dan_rule["nt_column"]].fill_null("") == dan_rule["nt_equals"])
    membership = np.stack([mask.to_numpy() for mask in masks.values()])
    require(np.all(membership.sum(axis=0) <= 1), "a source neuron matches multiple MB roles")
    roles = np.full(len(nodes), None, dtype=object)
    for role, mask in masks.items():
        roles[mask.to_numpy()] = role
    selected = np.flatnonzero(membership.any(axis=0))
    mb_nodes = nodes.with_columns(pl.Series("role", roles.tolist(), dtype=pl.String))[selected]
    mb_nodes = mb_nodes.rename({"node_index": "whole_brain_index"}).with_row_index("node_index")
    candidates = nodes.filter(prefix)
    excluded = candidates.filter(
        pl.col(dan_rule["nt_column"]).fill_null("") != dan_rule["nt_equals"]
    )
    diagnostics = {
        "dan_prefix_candidates": len(candidates),
        "dan_excluded_by_nt": len(excluded),
        "dan_excluded_types": excluded.group_by(dan_rule["type_column"], dan_rule["nt_column"])
        .len()
        .sort(dan_rule["type_column"], dan_rule["nt_column"])
        .to_dicts(),
        "dan_selected_types": mb_nodes.filter(pl.col("role") == "DAN")
        .group_by(dan_rule["type_column"])
        .len()
        .sort(dan_rule["type_column"])
        .rename({"len": "count"})
        .to_dicts(),
    }
    return mb_nodes, selected, diagnostics


def sanity_report(parent, matrix, nodes, config, diagnostics):
    counts = {
        role: nodes.filter(pl.col("role") == role).height for role in ("KC", "PN", "MBON", "DAN")
    }
    roles = nodes["role"].to_numpy()
    pn, kc = np.flatnonzero(roles == "PN"), np.flatnonzero(roles == "KC")
    convergence = matrix[pn][:, kc].getnnz(axis=0)
    values, frequencies = np.unique(convergence, return_counts=True)
    checks = {
        "whole_brain_count": parent.shape[0] == config["sanity"]["whole_brain_expected"],
        "bilateral_kc_count": config["sanity"]["kc_minimum"]
        <= counts["KC"]
        <= config["sanity"]["kc_maximum"],
        "all_groups_nonempty": all(counts.values()),
        "sparse_csr": sparse.isspmatrix_csr(matrix),
        "threshold_ge_5": matrix.nnz > 0 and bool(np.all(matrix.data >= config["threshold"])),
        "all_selected_nodes_retained": matrix.shape == (sum(counts.values()),) * 2,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "bilateral",
        "threshold": config["threshold"],
        "checks": checks,
        "whole_brain_neurons": parent.shape[0],
        "mb_neurons": matrix.shape[0],
        "mb_roles": counts,
        "mb_edges": matrix.nnz,
        "side_counts": nodes.group_by("role", "side").len().sort("role", "side").to_dicts(),
        "csr_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
        "pn_kc_convergence": {
            "definition": "unique ALPN sources per KC at pair-total synapse threshold 5",
            "minimum": int(convergence.min()) if kc.size else None,
            "maximum": int(convergence.max()) if kc.size else None,
            "mean": float(convergence.mean()) if kc.size else None,
            "median": float(np.median(convergence)) if kc.size else None,
            "zero_input_kcs": int(np.count_nonzero(convergence == 0)),
            "histogram": [
                {"pn_sources": int(v), "kc_count": int(f)} for v, f in zip(values, frequencies)
            ],
            "biological_acceptance": "numerical reference bounds not supplied; report for human verification",
        },
        "nt_sign_policy": "not assigned; raw predicted NT retained",
        "model_metrics_computed": False,
        **diagnostics,
    }


def make_data(matrix, nodes):
    coo = matrix.tocoo(copy=False)
    root_ids = nodes["root_id"].to_numpy()
    require(int(root_ids.max()) <= np.iinfo(np.int64).max, "root IDs exceed torch int64")
    data = Data(
        num_nodes=len(nodes),
        edge_index=torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
        edge_weight=torch.from_numpy(coo.data.copy()),
        root_id=torch.from_numpy(root_ids.astype(np.int64)),
        whole_brain_index=torch.tensor(nodes["whole_brain_index"].to_list(), dtype=torch.int64),
        cell_type=nodes["primary_type"].to_list(),
    )
    for column in nodes.columns:
        if column not in ("root_id", "node_index", "whole_brain_index"):
            data[column] = nodes[column].to_list()
    data.validate(raise_on_error=True)
    return data


def build(root, selection_path, run):
    root = Path(root)
    config = yaml.safe_load(Path(selection_path).read_text(encoding="utf-8"))
    require(
        config["dataset"] == "fafb" and str(config["snapshot"]) == "783",
        "MB source must be fafb/v783",
    )
    require(config["threshold"] == 5, "MB primary threshold must be 5")
    decisions = (root / "DECISIONS.md").read_text(encoding="utf-8")
    require(
        config["approval"]["reviewed_by"]
        and all(item in decisions for item in config["approval"]["decision_ids"]),
        "MB selection decisions have not been recorded",
    )
    source = read_json(root / config["source_manifest"])
    for item in source["artifacts"].values():
        require(
            sha256(root / item["path"]) == item["sha256"], "full-source artifact checksum mismatch"
        )
    source_nodes = pl.read_parquet(root / source["artifacts"]["nodes"]["path"])
    parent = sparse.load_npz(root / source["artifacts"]["csr"]["path"])
    require(sparse.isspmatrix_csr(parent), "source adjacency must be sparse CSR")
    require(
        np.array_equal(source_nodes["node_index"].to_numpy(), np.arange(parent.shape[0])),
        "source node-index order differs from CSR",
    )
    signature = fingerprint(
        {"selection": config, "source": source["artifacts"], "builder_sha256": sha256(__file__)}
    )
    output = root / "data" / "graphs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    state_path = root / "checkpoints" / f"{config['run_id']}_stages.json"
    state = read_json(state_path, {"signature": signature, "artifacts": {}})
    require(state["signature"] == signature, "MB selection/input/code changed; use a new run_id")
    nodes, selected, diagnostics = select_nodes(source_nodes, config)
    matrix = parent[selected][:, selected].tocsr()
    report = sanity_report(parent, matrix, nodes, config, diagnostics)
    report["selection_config"] = config
    report["source_manifest_sha256"] = sha256(root / config["source_manifest"])
    report["builder_sha256"] = sha256(__file__)
    atomic_json(root / "reports" / "mb_sanity.json", report)
    require(
        report["status"] == "PASS",
        "MB sanity checks failed: "
        + ", ".join(name for name, passed in report["checks"].items() if not passed),
    )
    for name, path, writer in (
        ("nodes", output / "nodes.parquet", lambda p: nodes.write_parquet(p)),
        ("csr", output / "mb_csr.npz", lambda p: sparse.save_npz(p, matrix)),
        ("pyg", output / "mb_pyg.pt", lambda p: torch.save(make_data(matrix, nodes), p)),
    ):
        if name in state["artifacts"]:
            require(
                sha256(path) == state["artifacts"][name]["sha256"], "MB artifact checksum mismatch"
            )
        else:
            atomic_file(path, writer)
            state["artifacts"][name] = record_file(root, path)
            atomic_json(state_path, state)
        run.save(mb_completed_stage=name, mb_nodes=len(nodes), mb_edges=matrix.nnz)
    manifest = {
        **state,
        "status": "PASS",
        "config": config,
        "report": report,
        "provenance": run.state["provenance"],
        "source_manifest": config["source_manifest"],
    }
    atomic_json(output / "MANIFEST.json", manifest)
    return report
