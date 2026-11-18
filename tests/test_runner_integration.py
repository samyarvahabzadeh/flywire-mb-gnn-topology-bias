"""Temporary synthetic fixtures only: no FlyWire training or real OSF registration."""

import copy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from scipy import sparse

from mbtopology import experiment_io, models, runner, task_inputs
from mbtopology.ingest import record_file
from mbtopology.runtime import Fail, atomic_json, fingerprint, read_json, sha256

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def synthetic_registration(monkeypatch):
    def verify_fixture_root(root):
        assert Path(root) != ROOT
        return {"type": "SYNTHETIC_FIXTURE_ONLY"}

    monkeypatch.setattr(runner, "require_registered", verify_fixture_root)
    monkeypatch.setattr(runner.gnn_training, "require_registered", verify_fixture_root)


class Progress:
    def save(self, **values):
        pass


def fixture_bundle(root, architecture):
    (root / "configs").mkdir()
    (root / "configs/seeds.txt").write_text("\n".join(map(str, range(30))) + "\n")
    task = yaml.safe_load((ROOT / "configs/task_engine.yaml").read_text())
    task.update(
        version="SYNTHETIC_INTEGRATION_ONLY", dataset="SYNTHETIC_FIXTURE", snapshot="FIXTURE"
    )
    task["source_manifest"] = "data/synthetic_fixture/MANIFEST.json"
    task["task"]["trials_max"] = 2
    experiment = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    experiment["dataset"] = "SYNTHETIC_FIXTURE"
    model_config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    model_config["device"] = "cpu"
    null_config = yaml.safe_load((ROOT / "configs/nulls.yaml").read_text())
    directory = root / "data/synthetic_fixture"
    directory.mkdir(parents=True)
    roles = ["KC"] * 4 + ["PN", "MBON", "MBON", "DAN", "DAN"]
    nodes = pl.DataFrame(
        {
            "node_index": range(9),
            "whole_brain_index": range(9),
            "root_id": [1000 + i for i in range(9)],
            "role": roles,
            "primary_type": [f"SYNTHETIC_{value}_{i}" for i, value in enumerate(roles)],
            "nt_type": ["ACH"] * 5 + ["GLUT", "ACH", "DA", "DA"],
        }
    ).with_columns(pl.col("root_id").cast(pl.UInt64))
    nodes_path = directory / "nodes.parquet"
    nodes.write_parquet(nodes_path)
    edges = [(i, (i + offset) % 9) for i in range(9) for offset in (1, 3)]
    pre, post = np.array(edges).T
    matrix = sparse.coo_matrix((np.full(len(edges), 5), (pre, post)), shape=(9, 9)).tocsr()
    csr_path = directory / "fixture_csr.npz"
    sparse.save_npz(csr_path, matrix)
    atomic_json(
        root / task["source_manifest"],
        {
            "status": "PASS",
            "scope": "SYNTHETIC_UNIT_FIXTURE",
            "artifacts": {
                "nodes": record_file(root, nodes_path),
                "csr": record_file(root, csr_path),
            },
        },
    )
    prepared = task_inputs.prepare(root, task, Progress())
    bundle = {
        "experiment": experiment,
        "task": task,
        "models": model_config,
        "nulls": null_config,
        "seeds": [0],
        "inputs": prepared,
        "parameter_budget": models.parameter_budget(model_config),
        "fixture": "SYNTHETIC_UNIT_FIXTURE_NO_EXTERNAL_REGISTRATION",
    }
    bundle["bundle_sha256"] = fingerprint(bundle)
    spec = {
        "run_id": f"SYNTHETIC-{architecture}-seed00",
        "architecture": architecture,
        "seed": 0,
        "condition": "real",
        "null_instance": None,
        "bundle_sha256": bundle["bundle_sha256"],
        "fixture": "SYNTHETIC_UNIT_FIXTURE",
    }
    spec_path = root / "configs/synthetic_run.yaml"
    experiment_io.save_yaml(spec_path, spec)
    frame = pl.DataFrame(
        [
            {
                **spec,
                "run_config_path": "configs/synthetic_run.yaml",
                "run_config_sha256": fingerprint(spec),
                "yaml_sha256": sha256(spec_path),
            }
        ]
    )
    return bundle, frame


@pytest.mark.parametrize("architecture", ["GCN", "GAT"])
def test_temporary_synthetic_runner_result_and_verified_skip(tmp_path, monkeypatch, architecture):
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    protected = sha256(ROOT / "PREREGISTRATION.md")
    torch.set_num_threads(1)
    try:
        bundle, frame = fixture_bundle(tmp_path, architecture)
        report = runner.execute(tmp_path, bundle, frame, Progress())
        assert report["completed_runs"] == 1
        path = (
            tmp_path
            / "results"
            / bundle["bundle_sha256"][:12]
            / f"SYNTHETIC-{architecture}-seed00.parquet"
        )
        manifest = read_json(path.with_suffix(".manifest.json"))
        assert manifest["parquet_sha256"] == sha256(path) and manifest["rows"] == 2
        assert manifest["run_specification"]["fixture"] == "SYNTHETIC_UNIT_FIXTURE"
        assert manifest["parameter_count"] == bundle["parameter_budget"]["counts"][architecture]
        assert manifest["right_censored"] is True and manifest["trials_to_criterion"] is None
        assert manifest["git_commit"] and manifest["bundle_sha256"] == bundle["bundle_sha256"]
        monkeypatch.setattr(
            runner.gnn_training,
            "run_registered",
            lambda *a: pytest.fail("completed fixture was retrained"),
        )
        assert runner.execute(tmp_path, bundle, frame, Progress())["completed_runs"] == 1
        assert sha256(ROOT / "PREREGISTRATION.md") == protected
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def test_changed_bundle_cannot_keep_old_registered_hash(tmp_path, monkeypatch):
    bundle, frame = fixture_bundle(tmp_path, "GCN")
    changed = copy.deepcopy(bundle)
    changed["models"]["optimizer"]["learning_rate"] = 0.5
    monkeypatch.setattr(
        runner, "execution_environment", lambda *a: pytest.fail("tampered bundle reached execution")
    )
    with pytest.raises(Fail, match="bundle contents"):
        runner.execute(tmp_path, changed, frame, Progress())
    assert not (tmp_path / "results").exists()
