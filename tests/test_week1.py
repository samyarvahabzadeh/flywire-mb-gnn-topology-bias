"""All data in this file are deliberately synthetic engineering fixtures."""

from __future__ import annotations

import csv
import gzip
import json
import threading
import time
from pathlib import Path

import polars as pl
import pytest
import torch
import yaml
from scipy import sparse

from mbtopology import graph, ingest, toy
from mbtopology.cli import main
from mbtopology.runtime import Fail, Run, atomic_json, read_json, sha256

REPO = Path(__file__).resolve().parents[1]
HIGH_ID = 720575940000000001


class Progress:
    def save(self, **kwargs):
        pass


def synthetic_csv(root, product, header, rows):
    path = root / "data" / "raw" / f"{product}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    ingest.manifest_update(
        root,
        f"raw/{product}",
        ingest.record_file(root, path, remote_product=product, fixture="SYNTHETIC_TEST_ONLY"),
    )
    return path


@pytest.fixture
def fixture_root(tmp_path):
    config = yaml.safe_load((REPO / "configs" / "week1.yaml").read_text())
    config["minimum_free_bytes"] = 0
    config["arrow_block_bytes"] = 256
    config["products"] = {
        "connections": {"remote_product": "connections"},
        "consolidated_cell_types": {"remote_product": "consolidated_cell_types"},
    }
    ids = [HIGH_ID + i for i in range(5)]
    synthetic_csv(
        tmp_path,
        "consolidated_cell_types",
        ["ID", "TYPE", "NT", "SUPER"],
        [
            [ids[0], "TEST_KC", "TEST_EXC", "TEST_INTERNAL"],
            [ids[1], "TEST_KC", "TEST_EXC", "TEST_INTERNAL"],
            [ids[2], "TEST_PN", "TEST_EXC", "TEST_INPUT"],
            [ids[3], "TEST_MBON", "TEST_INH", "TEST_OUTPUT"],
            [ids[4], "TEST_DAN", "TEST_MOD", "TEST_INTERNAL"],
        ],
    )
    synthetic_csv(
        tmp_path,
        "connections",
        ["SOURCE", "TARGET", "COUNT", "REGION"],
        [
            [ids[2], ids[0], 3, "TEST_REGION_A"],
            [ids[2], ids[0], 3, "TEST_REGION_B"],
            [ids[0], ids[2], 5, "TEST_REGION_A"],
            [ids[0], ids[3], 4, "TEST_REGION_A"],
            [ids[4], ids[3], 5, "TEST_REGION_A"],
            [ids[3], ids[3], 5, "TEST_REGION_A"],
        ],
    )
    inventory = ingest.inspect_raw(tmp_path, config)
    for product, item in inventory["products"].items():
        config["schema"][product] = {"header": item["header"]}
    config["approval"] = {
        "reviewed_by": "SYNTHETIC TEST FIXTURE - NOT HUMAN APPROVAL",
        "reviewed_at": "2026-11-17",
        "decision_ids": ["TEST-FIXTURE"],
        "inventory_sha256": sha256(tmp_path / "reports/schema_inventory.json"),
    }
    (tmp_path / "DECISIONS.md").write_text("TEST-FIXTURE: synthetic fixture only", encoding="utf-8")
    config["node_universe"] = {"product": "consolidated_cell_types", "id_column": "ID"}
    config["node_attributes"] = {
        name: {"product": "consolidated_cell_types", "id_column": "ID", "value_column": col}
        for name, col in (("cell_type", "TYPE"), ("nt_type", "NT"), ("super_class", "SUPER"))
    }
    config["edge_columns"] = {"pre": "SOURCE", "post": "TARGET", "weight": "COUNT"}
    config["roles"] = {role: ["TEST_" + role] for role in graph.ROLES}
    config["nt_signs"] = {"TEST_EXC": 1, "TEST_INH": -1, "TEST_MOD": 0}
    config["graph"].update(aggregation="sum_by_directed_pair", self_loops="retain")
    config["sanity"] = {
        "whole_brain_neurons": {"minimum": 5, "maximum": 5},
        "kc_neurons": {"minimum": 2, "maximum": 2},
        "pn_kc_convergence": {"minimum": 0.5, "maximum": 0.5, "statistic": "mean"},
    }
    for product in config["products"]:
        ingest.convert_product(tmp_path, config, product, inventory, Progress())
    return tmp_path, config, inventory


def test_sparse_graph_direction_pair_threshold_and_isolates(fixture_root, monkeypatch):
    root, config, inventory = fixture_root

    def forbid_dense(*args, **kwargs):
        pytest.fail("adjacency densification attempted")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_dense)
    monkeypatch.setattr(sparse.csr_matrix, "todense", forbid_dense)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbid_dense)
    with Run(root, config, "synthetic_graph") as run:
        result = graph.build_graph(root, config, run)
    assert result["status"] == "PASS"
    matrix = sparse.load_npz(root / "data/graphs/full_csr.npz")
    assert matrix.shape == (5, 5) and matrix.nnz == 4
    assert matrix[2, 0] == 6 and matrix[0, 2] == 5 and matrix[0, 3] == 0
    assert matrix[3, 3] == 5 and matrix[4, 3] == 5
    assert matrix.getrow(1).nnz == matrix.getcol(1).nnz == 0
    data = torch.load(root / "data/graphs/full_pyg.pt", weights_only=False)
    assert data.root_id.tolist() == [HIGH_ID + i for i in range(5)]
    assert data.edge_weight.dtype == torch.int64
    assert data.edge_sign.tolist() == [1, 1, -1, 0]
    assert data.cell_type[0] == "TEST_KC"
    assert data.super_class[2] == "TEST_INPUT"
    assert data.num_nodes == 5
    before = sha256(root / "data/graphs/full_csr.npz")
    for product in config["products"]:
        ingest.convert_product(root, config, product, inventory, Progress())
    with Run(root, config, "synthetic_graph") as run:
        assert graph.build_graph(root, config, run)["status"] == "PASS"
    assert sha256(root / "data/graphs/full_csr.npz") == before


def test_fail_loudly_and_preserve_failed_counts(fixture_root):
    root, config, _ = fixture_root
    config["sanity"]["kc_neurons"] = {"minimum": 3, "maximum": 3}
    with Run(root, config, "synthetic_bad_count") as run, pytest.raises(Fail, match="KC neurons=2"):
        graph.build_graph(root, config, run)
    result = read_json(root / "reports/sanity.json")
    assert result["status"] == "FAIL" and result["mb_roles"]["KC"] == 2
    assert not (root / "data/graphs/MANIFEST.json").exists()


def test_unreviewed_schema_cannot_convert(fixture_root):
    root, config, _ = fixture_root
    config["approval"]["reviewed_by"] = None
    with pytest.raises(Fail, match="human semantic review"):
        ingest.approved(root, config)
    config["approval"]["reviewed_by"] = "SYNTHETIC TEST FIXTURE"
    config["schema"]["connections"]["header"] = ["WRONG"]
    with pytest.raises(Fail, match="schema mismatch"):
        ingest.approved(root, config)


def test_conversion_precision_blanks_multiline_and_resume(tmp_path):
    config = yaml.safe_load((REPO / "configs/week1.yaml").read_text())
    config.update(
        products={"consolidated_cell_types": {"remote_product": "consolidated_cell_types"}},
        minimum_free_bytes=0,
        arrow_block_bytes=256,
    )
    rows = [[str(HIGH_ID + i), "", "TEST,label\nline" + str(i)] for i in range(80)]
    synthetic_csv(tmp_path, "consolidated_cell_types", ["ID", "EMPTY", "LABEL"], rows)
    inventory = ingest.inspect_raw(tmp_path, config)

    class Interrupted:
        def save(self, **kwargs):
            raise KeyboardInterrupt("SYNTHETIC SESSION DEATH")

    with pytest.raises(KeyboardInterrupt):
        ingest.convert_product(
            tmp_path, config, "consolidated_cell_types", inventory, Interrupted()
        )
    committed = next((tmp_path / "data/parquet/consolidated_cell_types").glob("*.parquet"))
    stamp = committed.stat().st_mtime_ns
    ingest.convert_product(tmp_path, config, "consolidated_cell_types", inventory, Progress())
    frame = ingest.scan_product(tmp_path, "consolidated_cell_types").collect()
    assert frame.rows() == [tuple(row) for row in rows]
    assert frame["ID"].cast(pl.UInt64).to_list() == [HIGH_ID + i for i in range(80)]
    assert committed.stat().st_mtime_ns == stamp
    assert not frame["EMPTY"].has_nulls() and frame["EMPTY"].unique().to_list() == [""]
    committed.write_bytes(b"SYNTHETIC CORRUPTION")
    with pytest.raises(Fail, match="corrupt"):
        ingest.convert_product(tmp_path, config, "consolidated_cell_types", inventory, Progress())


def test_missing_universe_endpoint_fails(fixture_root):
    root, config, _ = fixture_root
    nodes = graph.normalize_nodes(root, config).head(4)
    with pytest.raises(Fail, match="outside approved universe"):
        graph.normalize_edges(root, config, nodes, root / "bad.parquet")


def test_unknown_nt_and_overlapping_roles_fail(fixture_root):
    root, config, _ = fixture_root
    nodes = graph.normalize_nodes(root, config).with_columns(
        pl.lit(None).cast(pl.Int8).alias("nt_sign")
    )
    with pytest.raises(Fail, match="unknown/missing"):
        graph.make_pyg(sparse.eye(5, format="csr"), nodes)
    config["roles"]["DAN"] = ["TEST_KC"]
    with pytest.raises(Fail, match="multiple roles"):
        graph.validate_semantics(config)


def test_changed_graph_configuration_refuses_resume(fixture_root):
    root, config, _ = fixture_root
    with Run(root, config, "synthetic_graph") as run:
        graph.build_graph(root, config, run)
    config["graph"]["self_loops"] = "drop"
    with Run(root, config, "synthetic_changed_graph") as run, pytest.raises(Fail, match="changed"):
        graph.build_graph(root, config, run)


def test_toy_plumbing_only(tmp_path):
    with Run(tmp_path, {"checkpoint_seconds": 900}, "synthetic_toy") as run:
        result = toy.toy_check(tmp_path, run)
    assert result["status"] == "PASS" and result["model_metrics_computed"] is False


def test_heartbeat_during_long_operation(tmp_path):
    with Run(tmp_path, {"checkpoint_seconds": 0.03}, "synthetic_heartbeat") as run:
        initial = read_json(run.path)["checkpoint_at"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if read_json(run.path)["checkpoint_at"] != initial:
                break
            threading.Event().wait(0.01)
        assert read_json(run.path)["checkpoint_at"] != initial


class Response:
    def __init__(self, payload, status=200, start=0, url=None):
        self.status_code = status
        self.url = url or "https://storage.googleapis.com/TEST/fafb/783/TEST.csv.gz"
        self.payload = payload[start:]
        self.headers = {"Content-Length": str(len(self.payload)), "ETag": '"TEST_V1"'}
        if status == 206:
            self.headers["Content-Range"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield self.payload


@pytest.mark.parametrize("ignore_range", [True, False])
def test_download_resume_and_server_range_fallback(tmp_path, monkeypatch, ignore_range):
    payload = gzip.compress(b"TEST_COLUMN\nTEST_VALUE\n")
    partial = tmp_path / "data/raw/connections.csv.gz.partial"
    partial.parent.mkdir(parents=True)
    offset = 10
    partial.write_bytes(payload[:offset])
    atomic_json(
        tmp_path / "checkpoints/download_connections_resume.json",
        {"remote_product": "connections", "etag": '"TEST_V1"'},
    )
    config = {
        "products": {"connections": {"remote_product": "connections"}},
        "minimum_free_bytes": 0,
    }

    def fake_request(url, **kwargs):
        assert kwargs["headers"]["Range"] == f"bytes={offset}-"
        assert kwargs["headers"]["If-Range"] == '"TEST_V1"'
        return Response(payload, 200 if ignore_range else 206, 0 if ignore_range else offset)

    monkeypatch.setattr(ingest, "request", fake_request)
    result = ingest.download(tmp_path, config, "connections", "SYNTHETIC_SECRET", Progress())
    assert (tmp_path / result["path"]).read_bytes() == payload
    assert "SYNTHETIC_SECRET" not in json.dumps(read_json(tmp_path / "data/MANIFEST.json"))


def test_wrong_snapshot_download_fails(tmp_path, monkeypatch):
    config = {
        "products": {"connections": {"remote_product": "connections"}},
        "minimum_free_bytes": 0,
    }
    monkeypatch.setattr(
        ingest,
        "request",
        lambda *a, **k: Response(
            b"TEST", url="https://storage.googleapis.com/TEST/fafb/999/TEST.csv.gz"
        ),
    )
    with pytest.raises(Fail, match="cannot verify"):
        ingest.download(tmp_path, config, "connections", "SYNTHETIC_SECRET", Progress())
    assert not (tmp_path / "data/raw/connections.csv.gz").exists()


def test_cli_missing_token_returns_fail_without_secret(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FLYWIRE_CODEX_TOKEN", raising=False)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/week1.yaml").write_text((REPO / "configs/week1.yaml").read_text())
    assert main(["download", "--root", str(tmp_path)]) == 2
    assert "FAIL:" in capsys.readouterr().err


def test_self_loop_drop_is_explicit(fixture_root):
    root, config, _ = fixture_root
    config["graph"]["self_loops"] = "drop"
    graph.normalize_edges(root, config, graph.normalize_nodes(root, config), root / "drop.parquet")
    edges = pl.read_parquet(root / "drop.parquet")
    assert edges.height == 3 and edges.filter(pl.col("pre") == pl.col("post")).is_empty()


def test_download_recovers_death_between_rename_and_manifest(tmp_path, monkeypatch):
    payload = gzip.compress(b"TEST_COLUMN\nTEST_VALUE\n")
    config = {
        "products": {"connections": {"remote_product": "connections"}},
        "minimum_free_bytes": 0,
    }
    monkeypatch.setattr(ingest, "request", lambda *a, **k: Response(payload))
    original_update = ingest.manifest_update

    def die_before_manifest(*args):
        raise OSError("SYNTHETIC SESSION DEATH")

    monkeypatch.setattr(ingest, "manifest_update", die_before_manifest)
    with pytest.raises(OSError, match="SYNTHETIC SESSION DEATH"):
        ingest.download(tmp_path, config, "connections", "SYNTHETIC_SECRET", Progress())
    assert (tmp_path / "data/raw/connections.csv.gz").exists()
    monkeypatch.setattr(ingest, "manifest_update", original_update)
    monkeypatch.setattr(ingest, "request", lambda *a, **k: pytest.fail("unnecessary redownload"))
    result = ingest.download(tmp_path, config, "connections", "SYNTHETIC_SECRET", Progress())
    assert result["sha256"] == sha256(tmp_path / result["path"])


def test_sparse_csr_resumes_committed_matrix(tmp_path):
    import numpy as np

    size = 262145
    indices = np.arange(size)
    edges = pl.DataFrame(
        {"pre": indices // 600, "post": indices % 600, "weight": np.full(size, 5, dtype=np.int64)}
    )
    path = tmp_path / "synthetic_edges.parquet"
    edges.write_parquet(path)

    class Interrupted:
        def save(self, **kwargs):
            raise KeyboardInterrupt("SYNTHETIC SESSION DEATH")

    with pytest.raises(KeyboardInterrupt):
        graph.csr_from_edges(path, 600, tmp_path, "TEST_SIGNATURE", Interrupted(), interval=0)
    state = read_json(tmp_path / "csr_resume.json")
    assert state["completed_batches"] == 1
    first_hash = sha256(tmp_path / state["matrix_file"])
    matrix = graph.csr_from_edges(path, 600, tmp_path, "TEST_SIGNATURE", Progress(), interval=0)
    assert matrix.nnz == size and matrix[0, 0] == matrix[436, 544] == 5
    assert sha256(tmp_path / state["matrix_file"]) == first_hash


def test_full_source_keeps_unselected_annotations_and_unsigned_graph(tmp_path):
    config = yaml.safe_load((REPO / "configs/week1.yaml").read_text())
    config.update(minimum_free_bytes=0, arrow_block_bytes=256)
    ids = [HIGH_ID + i for i in range(3)]
    synthetic_csv(
        tmp_path,
        "neurons",
        ["root_id", "nt_type", "group"],
        [
            [ids[0], "TEST_UNKNOWN", "TEST_A"],
            [ids[1], "TEST_UNKNOWN", "TEST_B"],
            [ids[2], "TEST_UNKNOWN", "TEST_C"],
        ],
    )
    synthetic_csv(
        tmp_path,
        "consolidated_cell_types",
        ["root_id", "primary_type", "additional_type(s)"],
        [[ids[0], "TEST_PRIMARY", "TEST_ALT"], [ids[1], "TEST_SECOND", ""]],
    )
    synthetic_csv(
        tmp_path,
        "classification",
        ["root_id", "class", "super_class", "side"],
        [[node, "TEST_CLASS", "TEST_SUPER", "TEST_SIDE"] for node in ids],
    )
    synthetic_csv(
        tmp_path,
        "connections",
        ["pre_root_id", "post_root_id", "syn_count", "neuropil", "nt_type"],
        [
            [ids[0], ids[1], 3, "TEST_A", "TEST_UNKNOWN"],
            [ids[0], ids[1], 3, "TEST_B", "TEST_UNKNOWN"],
            [ids[1], ids[0], 5, "TEST_A", "TEST_UNKNOWN"],
        ],
    )
    inventory = ingest.inspect_raw(tmp_path, config)
    config["schema"] = {
        key: {"header": value["header"]} for key, value in inventory["products"].items()
    }
    config["products"]["connections"]["remote_product"] = "connections"
    config["approval"] = {
        "reviewed_by": "SYNTHETIC FIXTURE ONLY",
        "reviewed_at": "2026-11-17",
        "decision_ids": ["TEST-FIXTURE"],
        "inventory_sha256": sha256(tmp_path / "reports/schema_inventory.json"),
    }
    (tmp_path / "DECISIONS.md").write_text("TEST-FIXTURE", encoding="utf-8")
    for product in config["products"]:
        ingest.convert_product(tmp_path, config, product, inventory, Progress())
    with Run(tmp_path, config, "synthetic_source_graph") as run:
        profile = ingest.connection_profile(tmp_path, config, run)
        result = graph.build_full_source(tmp_path, config, run)
    assert profile["rows"] == 3 and profile["rows_syn_count_ge_5"] == 1
    assert profile["unique_pre_root_id"] == profile["unique_post_root_id"] == 2
    assert result["neurons"] == 3 and result["directed_pairs_syn_count_ge_5"] == 2
    assert result["isolated_neurons"] == 1
    data = torch.load(tmp_path / "data/graphs/source_full/full_pyg.pt", weights_only=False)
    assert data.primary_type == ["TEST_PRIMARY", "TEST_SECOND", None]
    assert data["additional_type(s)"] == ["TEST_ALT", "", None]
    assert data.edge_weight.tolist() == [6, 5]
    assert "edge_sign" not in data and "role" not in data
