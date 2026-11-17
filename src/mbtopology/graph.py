from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from scipy import sparse
from torch_geometric.data import Data

from .ingest import approved, record_file, scan_product
from .runtime import atomic_file, atomic_json, fingerprint, read_json, require, sha256

ROLES = ("KC", "MBON", "DAN", "PN")


def collect(frame):
    return frame.collect(engine="streaming")


def check_range(value, settings, name):
    low, high = settings.get("minimum"), settings.get("maximum")
    require(
        low is not None and high is not None and 0 <= low <= high,
        f"human must set accepted bounds for {name}",
    )
    require(low <= value <= high, f"{name}={value} outside approved [{low}, {high}]")


def validate_semantics(config):
    require(
        config["graph"]["aggregation"] == "sum_by_directed_pair",
        "aggregation must be explicitly reviewed as sum_by_directed_pair",
    )
    require(
        config["graph"]["self_loops"] in ("retain", "drop"), "self-loop policy needs human review"
    )
    require(
        set(config["roles"]) == set(ROLES) and all(config["roles"].values()),
        "explicit nonempty cell-type lists required for KC/MBON/DAN/olfactory PN",
    )
    labels = [label for role in ROLES for label in config["roles"][role]]
    require(len(labels) == len(set(labels)), "cell type appears in multiple roles or twice")
    require(
        config["nt_signs"] and set(config["nt_signs"].values()) <= {-1, 0, 1},
        "neurotransmitter signs need human review, including modulators/unknowns",
    )
    require(
        {"cell_type", "nt_type"} <= set(config["node_attributes"]),
        "cell_type and predicted nt_type source columns must be reviewed",
    )


def normalize_nodes(root, config):
    universe = config["node_universe"]
    require(universe["product"] and universe["id_column"], "node universe source is undecided")
    nodes = collect(
        scan_product(root, universe["product"]).select(
            pl.col(universe["id_column"]).cast(pl.UInt64, strict=True).alias("root_id")
        )
    )
    require(
        not nodes["root_id"].has_nulls() and nodes["root_id"].n_unique() == len(nodes),
        "node universe IDs must be nonnull and unique",
    )
    for target, settings in config["node_attributes"].items():
        require(target in ("cell_type", "nt_type", "super_class"), "unknown node attribute")
        values = collect(
            scan_product(root, settings["product"]).select(
                pl.col(settings["id_column"]).cast(pl.UInt64, strict=True).alias("root_id"),
                pl.col(settings["value_column"]).alias(target),
            )
        )
        require(
            values["root_id"].n_unique() == len(values) and not values["root_id"].has_nulls(),
            f"duplicate/null IDs in {target} source",
        )
        require(
            values.join(nodes, on="root_id", how="anti").is_empty(),
            f"{target} source contains IDs outside the approved node universe",
        )
        nodes = nodes.join(values, on="root_id", how="left", validate="1:1")
    nodes = nodes.sort("root_id").with_row_index("node_index")
    mapping = {label: role for role, labels in config["roles"].items() for label in labels}
    absent = set(mapping) - set(nodes["cell_type"].drop_nulls().to_list())
    require(not absent, f"approved cell-type labels absent from data: {sorted(absent)}")
    nodes = nodes.with_columns(
        pl.col("cell_type")
        .replace_strict(mapping, default=None, return_dtype=pl.String)
        .alias("role")
    )
    nodes = nodes.with_columns(
        pl.col("nt_type")
        .replace_strict(config["nt_signs"], default=None, return_dtype=pl.Int8)
        .alias("nt_sign")
    )
    return nodes


def normalize_edges(root, config, nodes, destination):
    columns = config["edge_columns"]
    require(all(columns.values()), "edge column mapping needs human review")
    edges = scan_product(root, "connections").select(
        pl.col(columns["pre"]).cast(pl.UInt64, strict=True).alias("pre_root_id"),
        pl.col(columns["post"]).cast(pl.UInt64, strict=True).alias("post_root_id"),
        pl.col(columns["weight"]).cast(pl.Int64, strict=True).alias("weight"),
    )
    invalid = collect(
        edges.filter(pl.any_horizontal(pl.all().is_null()) | (pl.col("weight") <= 0)).limit(1)
    )
    require(invalid.is_empty(), "null endpoint/weight or nonpositive synapse count")
    ids = nodes.select("root_id").lazy()
    for endpoint in ("pre_root_id", "post_root_id"):
        missing = collect(
            edges.select(pl.col(endpoint).alias("root_id"))
            .unique()
            .join(ids, on="root_id", how="anti")
            .limit(1)
        )
        require(missing.is_empty(), f"{endpoint} contains an ID outside approved universe")
    paired = edges.group_by("pre_root_id", "post_root_id").agg(pl.col("weight").sum())
    paired = paired.filter(pl.col("weight") >= config["graph"]["threshold"])
    if config["graph"]["self_loops"] == "drop":
        paired = paired.filter(pl.col("pre_root_id") != pl.col("post_root_id"))
    lookup = nodes.select("root_id", "node_index")
    paired = paired.join(
        lookup.lazy().rename({"root_id": "pre_root_id", "node_index": "pre"}), on="pre_root_id"
    )
    paired = paired.join(
        lookup.lazy().rename({"root_id": "post_root_id", "node_index": "post"}), on="post_root_id"
    )
    atomic_file(
        destination,
        lambda p: (
            paired.select("pre", "post", "weight")
            .sort("pre", "post")
            .sink_parquet(p, compression="zstd", engine="streaming")
        ),
    )


def csr_from_edges(path, n_nodes, checkpoint_dir, signature, run, interval=900):
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "csr_resume.json"
    state = read_json(state_path, {})
    require(not state or state["signature"] == signature, "CSR resume inputs changed")
    if state:
        matrix_path = checkpoint_dir / state["matrix_file"]
        require(sha256(matrix_path) == state["matrix_sha256"], "CSR checkpoint checksum mismatch")
        matrix = sparse.load_npz(matrix_path)
        completed = state["completed_batches"]
    else:
        matrix = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.int64)
        completed = 0
    last_save = time.monotonic()
    seen = 0
    for index, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=262144)):
        seen = index + 1
        if index < completed:
            continue
        frame = pl.from_arrow(batch)
        block = sparse.coo_matrix(
            (frame["weight"].to_numpy(), (frame["pre"].to_numpy(), frame["post"].to_numpy())),
            shape=(n_nodes, n_nodes),
            dtype=np.int64,
        ).tocsr()
        matrix = matrix + block
        if time.monotonic() - last_save >= interval:
            matrix_path = checkpoint_dir / f"csr_resume_{seen:06d}.npz"
            atomic_file(matrix_path, lambda p, matrix=matrix: sparse.save_npz(p, matrix))
            atomic_json(
                state_path,
                {
                    "signature": signature,
                    "completed_batches": seen,
                    "matrix_sha256": sha256(matrix_path),
                    "matrix_file": matrix_path.name,
                },
            )
            last_save = time.monotonic()
        run.save(csr_completed_batches=seen, nnz=matrix.nnz)
    require(seen >= completed, "CSR checkpoint has more batches than input")
    matrix.sort_indices()
    require(
        sparse.isspmatrix_csr(matrix) and matrix.shape == (n_nodes, n_nodes),
        "graph must be square sparse CSR",
    )
    require(
        matrix.nnz == pq.ParquetFile(path).metadata.num_rows,
        "CSR lost/coalesced unexpected edge pairs",
    )
    return matrix


def make_pyg(matrix, nodes):
    require(sparse.issparse(matrix), "only sparse input accepted")
    coo = matrix.tocoo(copy=False)
    require(
        not nodes["nt_sign"].has_nulls(),
        "unknown/missing predicted NT sign; request human mapping, never guess",
    )
    root_ids = nodes["root_id"].to_numpy()
    require(
        root_ids.size == 0 or int(root_ids.max()) <= np.iinfo(np.int64).max,
        "root IDs exceed torch int64; preserve parquet and request implementation review",
    )
    signs = nodes["nt_sign"].to_numpy()
    result = Data(
        num_nodes=matrix.shape[0],
        edge_index=torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
        edge_weight=torch.from_numpy(coo.data.copy()),
        edge_sign=torch.from_numpy(signs[coo.row].copy()),
        root_id=torch.from_numpy(root_ids.astype(np.int64)),
        cell_type=nodes["cell_type"].to_list(),
        nt_type=nodes["nt_type"].to_list(),
        role=nodes["role"].to_list(),
    )
    if "super_class" in nodes.columns:
        result.super_class = nodes["super_class"].to_list()
    result.validate(raise_on_error=True)
    return result


def graph_sanity(full, mb, nodes, mb_nodes, settings):
    require(sparse.issparse(full) and sparse.issparse(mb), "adjacency must remain sparse")
    counts = {role: mb_nodes.filter(pl.col("role") == role).height for role in ROLES}
    report = {
        "status": "checking",
        "whole_brain_neurons": full.shape[0],
        "mb_neurons": mb.shape[0],
        "mb_roles": counts,
        "full_edges": full.nnz,
        "mb_edges": mb.nnz,
        "full_csr_bytes": full.data.nbytes + full.indices.nbytes + full.indptr.nbytes,
        "mb_csr_bytes": mb.data.nbytes + mb.indices.nbytes + mb.indptr.nbytes,
    }
    pn = np.flatnonzero(np.array(mb_nodes["role"].to_list(), dtype=object) == "PN")
    kc = np.flatnonzero(np.array(mb_nodes["role"].to_list(), dtype=object) == "KC")
    convergence = mb[pn][:, kc].getnnz(axis=0)
    report["pn_kc_convergence"] = {
        "minimum": int(convergence.min()) if kc.size else None,
        "maximum": int(convergence.max()) if kc.size else None,
        "mean": float(convergence.mean()) if kc.size else None,
        "median": float(np.median(convergence)) if kc.size else None,
        "zero_input_kcs": int(np.count_nonzero(convergence == 0)),
    }
    errors = []
    for value, bounds, name in (
        (full.shape[0], settings["whole_brain_neurons"], "whole-brain neurons"),
        (counts["KC"], settings["kc_neurons"], "KC neurons"),
    ):
        try:
            check_range(value, bounds, name)
        except RuntimeError as exc:
            errors.append(str(exc))
    for role, count in counts.items():
        if count == 0:
            errors.append(f"FAIL: no {role} nodes")
    convergence_settings = settings["pn_kc_convergence"]
    statistic = convergence_settings.get("statistic")
    try:
        require(
            statistic in ("mean", "median", "minimum", "maximum"),
            "PN->KC sanity statistic needs human review",
        )
        value = report["pn_kc_convergence"][statistic]
        require(value is not None, "no KCs for PN->KC convergence")
        check_range(value, convergence_settings, "PN->KC convergence")
    except RuntimeError as exc:
        errors.append(str(exc))
    report.update(status="FAIL" if errors else "PASS", failures=errors)
    return report


def build_graph(root, config, run):
    root = Path(root)
    approved(root, config)
    validate_semantics(config)
    output = root / "data" / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "data" / "MANIFEST.json")
    identities = {
        key: {
            field: item[field]
            for field in (
                "sha256",
                "raw_sha256",
                "remote_product",
                "resolved_resource",
                "bytes",
                "rows",
            )
            if field in item
        }
        for key, item in manifest["artifacts"].items()
    }
    signature = fingerprint(
        {
            "inputs": identities,
            "config": config,
            "source": run.state["provenance"]["source_sha256"],
        }
    )
    stages_path = root / "checkpoints" / "graph_stages.json"
    stages = read_json(stages_path, {"signature": signature, "artifacts": {}})
    require(
        stages["signature"] == signature, "graph config/inputs/code changed; use a new run root"
    )

    def cached(name):
        item = stages["artifacts"].get(name)
        if item:
            require(sha256(root / item["path"]) == item["sha256"], f"corrupt artifact: {name}")
        return bool(item)

    def done(name, path):
        stages["artifacts"][name] = record_file(root, path)
        atomic_json(stages_path, stages)
        run.save(completed_stage=name)

    node_path = output / "nodes.parquet"
    if not cached("nodes"):
        nodes = normalize_nodes(root, config)
        atomic_file(node_path, lambda p: nodes.write_parquet(p))
        done("nodes", node_path)
    nodes = pl.read_parquet(node_path)
    edge_path = output / "edges.parquet"
    if not cached("edges"):
        normalize_edges(root, config, nodes, edge_path)
        done("edges", edge_path)
    full_path = output / "full_csr.npz"
    if not cached("full_csr"):
        full = csr_from_edges(
            edge_path,
            len(nodes),
            root / "checkpoints",
            signature,
            run,
            config["checkpoint_seconds"],
        )
        atomic_file(full_path, lambda p: sparse.save_npz(p, full))
        done("full_csr", full_path)
    full = sparse.load_npz(full_path)
    selected = np.flatnonzero(nodes["role"].is_not_null().to_numpy())
    mb_nodes = (
        nodes[selected].rename({"node_index": "whole_brain_index"}).with_row_index("node_index")
    )
    mb = full[selected][:, selected].tocsr()
    sanity = graph_sanity(full, mb, nodes, mb_nodes, config["sanity"])
    sanity["provenance"] = run.state["provenance"]
    atomic_json(root / "reports" / "sanity.json", sanity)
    require(sanity["status"] == "PASS", "; ".join(sanity["failures"]))
    for name, path, writer in (
        ("mb_nodes", output / "mb_nodes.parquet", lambda p: mb_nodes.write_parquet(p)),
        ("mb_csr", output / "mb_csr.npz", lambda p: sparse.save_npz(p, mb)),
        ("full_pyg", output / "full_pyg.pt", lambda p: torch.save(make_pyg(full, nodes), p)),
        ("mb_pyg", output / "mb_pyg.pt", lambda p: torch.save(make_pyg(mb, mb_nodes), p)),
    ):
        if not cached(name):
            atomic_file(path, writer)
            done(name, path)
    atomic_json(
        output / "MANIFEST.json",
        {**stages, "status": "PASS", "provenance": run.state["provenance"]},
    )
    return sanity


def build_full_source(root, config, run):
    root = Path(root)
    approved(root, config)
    profile = read_json(root / "reports" / "connection_sanity.json")
    require(
        profile and profile["rows_syn_count_ge_5"] > 0,
        "report the raw connection sanity count before source graph construction",
    )
    output = root / "data" / "graphs" / "source_full"
    output.mkdir(parents=True, exist_ok=True)
    inputs = read_json(root / "data" / "MANIFEST.json")["artifacts"]
    signature = fingerprint(
        {
            "config": config,
            "inputs": {key: item["sha256"] for key, item in inputs.items()},
            "code": run.state["provenance"]["source_sha256"],
        }
    )
    state_path = root / "checkpoints" / "source_full_stages.json"
    state = read_json(state_path, {"signature": signature, "artifacts": {}})
    require(state["signature"] == signature, "source graph inputs/config/code changed")

    def cached(name):
        item = state["artifacts"].get(name)
        if item:
            require(sha256(root / item["path"]) == item["sha256"], "source graph artifact corrupt")
        return bool(item)

    def done(name, path):
        state["artifacts"][name] = record_file(root, path)
        atomic_json(state_path, state)
        run.save(source_stage=name)

    node_path = output / "nodes.parquet"
    if not cached("nodes"):
        nodes = collect(scan_product(root, "neurons")).with_columns(
            pl.col("root_id").cast(pl.UInt64, strict=True)
        )
        require(
            not nodes["root_id"].has_nulls() and nodes["root_id"].n_unique() == len(nodes),
            "neuron table has duplicate/null root IDs",
        )
        for product in ("consolidated_cell_types", "classification"):
            values = collect(scan_product(root, product)).with_columns(
                pl.col("root_id").cast(pl.UInt64, strict=True)
            )
            require(
                not values["root_id"].has_nulls() and values["root_id"].n_unique() == len(values),
                f"duplicate/null annotation IDs in {product}",
            )
            require(
                values.select("root_id")
                .join(nodes.select("root_id"), on="root_id", how="anti")
                .is_empty(),
                f"{product} IDs outside source neuron table",
            )
            nodes = nodes.join(values, on="root_id", how="left", validate="1:1")
        nodes = nodes.sort("root_id").with_row_index("node_index")
        atomic_file(node_path, lambda p: nodes.write_parquet(p))
        done("nodes", node_path)
    nodes = pl.read_parquet(node_path)
    edge_path = output / "edges.parquet"
    if not cached("edges"):
        normalize_edges(root, config, nodes, edge_path)
        done("edges", edge_path)
    matrix_path = output / "full_csr.npz"
    if not cached("csr"):
        matrix = csr_from_edges(
            edge_path,
            len(nodes),
            root / "checkpoints" / "source_full",
            signature,
            run,
            config["checkpoint_seconds"],
        )
        atomic_file(matrix_path, lambda p: sparse.save_npz(p, matrix))
        done("csr", matrix_path)
    matrix = sparse.load_npz(matrix_path)
    require(
        sparse.isspmatrix_csr(matrix) and matrix.data.min() >= 5,
        "source graph must remain CSR with pair-total weights >=5",
    )
    pyg_path = output / "full_pyg.pt"
    if not cached("pyg"):
        coo = matrix.tocoo(copy=False)
        root_ids = nodes["root_id"].to_numpy()
        require(int(root_ids.max()) <= np.iinfo(np.int64).max, "IDs exceed torch int64")
        data = Data(
            num_nodes=len(nodes),
            edge_index=torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
            edge_weight=torch.from_numpy(coo.data.copy()),
            root_id=torch.from_numpy(root_ids.astype(np.int64)),
        )
        for column in nodes.columns:
            if column not in ("root_id", "node_index"):
                data[column] = nodes[column].to_list()
        data.validate(raise_on_error=True)
        atomic_file(pyg_path, lambda p: torch.save(data, p))
        done("pyg", pyg_path)
    report = {
        "status": "SOURCE_GRAPH_BUILT_MB_SEMANTICS_PENDING",
        "neurons": len(nodes),
        "directed_pairs_syn_count_ge_5": matrix.nnz,
        "raw_rows_syn_count_ge_5": profile["rows_syn_count_ge_5"],
        "pair_count_difference_from_v630_anchor": matrix.nnz - profile["v630_reference"],
        "csr_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
        "self_loops": int(np.count_nonzero(matrix.diagonal())),
        "isolated_neurons": int(
            np.count_nonzero((matrix.getnnz(axis=0) + matrix.getnnz(axis=1)) == 0)
        ),
        "direction": "row=pre, column=post",
        "threshold": 5,
        "node_annotations": nodes.columns,
        "edge_sign": "PENDING_HUMAN_NT_SIGN_POLICY",
        "whole_brain_anchor": "approximately 139000; actual count reported without resizing",
        "model_metrics_computed": False,
        "provenance": run.state["provenance"],
    }
    atomic_json(root / "reports" / "source_graph_sanity.json", report)
    atomic_json(output / "MANIFEST.json", {**state, "report": report})
    return report
