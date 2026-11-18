"""Synthetic-only runner, durability and pre-registration execution-gate tests."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from mbtopology import cli, experiment_io, gnn_training, models, runner, task_inputs
from mbtopology.runtime import Fail, fingerprint, read_json, sha256

ROOT = Path(__file__).parents[1]


def synthetic_training_fixture(architecture):
    task = task_inputs.load_task(ROOT / "configs/task_engine.yaml")
    task["version"] = "SYNTHETIC_TRAINING_MECHANICS_ONLY"
    task["task"].update(n_odors=3, trials_max=4, cs_plus=[0], cs_minus=[1, 2])
    model_config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    torch.manual_seed(7)
    model = models.make_model(architecture, model_config, 7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    edges = torch.tensor([[0, 1, 2, 3, 3, 6, 6, 4], [4, 5, 4, 0, 1, 4, 5, 5]])
    graph = models.prepare_graph(edges, torch.ones(8), torch.tensor([1, 1, 1, 1, 1, 0, 0, -1]), 7)
    data = {
        "graph": graph,
        "roles": torch.tensor([0, 0, 0, 1, 2, 2, 3]),
        "kc": torch.tensor([0, 1, 2]),
        "mbon": torch.tensor([4, 5]),
        "avoidance": torch.tensor([0]),
        "approach": torch.tensor([1]),
        "patterns": [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])],
        "schedule": np.array([0, 1, 2, 0]),
    }
    return model, optimizer, data, task


@pytest.mark.parametrize("architecture", ["GCN", "GAT"])
def test_interruption_mid_trial_never_commits_half_update(tmp_path, monkeypatch, architecture):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        complete, optimizer, data, task = synthetic_training_fixture(architecture)
        expected = gnn_training.train_loop(
            complete,
            optimizer,
            data,
            task,
            tmp_path / "complete",
            "SYNTHETIC_ONLY",
            None,
            checkpoint_seconds=0,
        )
        interrupted, optimizer2, data2, task2 = synthetic_training_fixture(architecture)
        original_probe = gnn_training.probe
        calls = [0]

        def interrupt_after_optimizer(*args):
            calls[0] += 1
            if calls[0] == 3:
                raise KeyboardInterrupt("SYNTHETIC SESSION DEATH AFTER OPTIMIZER STEP")
            return original_probe(*args)

        monkeypatch.setattr(gnn_training, "probe", interrupt_after_optimizer)
        with pytest.raises(KeyboardInterrupt):
            gnn_training.train_loop(
                interrupted,
                optimizer2,
                data2,
                task2,
                tmp_path / "resume",
                "SYNTHETIC_ONLY",
                None,
                checkpoint_seconds=0,
            )
        assert read_json(tmp_path / "resume/checkpoint.json")["cursor"] == 2
        monkeypatch.setattr(gnn_training, "probe", original_probe)
        resumed, optimizer3, data3, task3 = synthetic_training_fixture(architecture)
        actual = gnn_training.train_loop(
            resumed,
            optimizer3,
            data3,
            task3,
            tmp_path / "resume",
            "SYNTHETIC_ONLY",
            None,
            checkpoint_seconds=0,
        )
        assert actual["history"] == expected["history"]
        assert all(
            torch.equal(value, complete.state_dict()[key])
            for key, value in resumed.state_dict().items()
        )
    finally:
        torch.set_num_threads(threads)


def test_result_manifest_orphan_recovery_and_mismatch_refusal(tmp_path):
    path = tmp_path / "results/SYNTHETIC_fixture.parquet"
    frame = pl.DataFrame({"trial": [1, 2], "fixture_value": [0.0, 1.0]})
    metadata = {
        "git_commit": "SYNTHETIC_TEST_ONLY",
        "run_config_sha256": "test-config",
        "bundle_sha256": "test-bundle",
        "model_metrics_computed": False,
    }
    manifest = experiment_io.write_result(tmp_path, path, frame, metadata)
    assert manifest["parquet_sha256"] == sha256(path)
    assert experiment_io.cached_result(path, "test-config", "test-bundle")
    stamp = path.stat().st_mtime_ns
    path.with_suffix(".manifest.json").unlink()
    experiment_io.write_result(tmp_path, path, frame, metadata)
    assert path.stat().st_mtime_ns == stamp
    with pytest.raises(Fail, match="completed result differs"):
        experiment_io.write_result(tmp_path, path, frame.head(1), metadata)
    with pytest.raises(Fail, match="different run"):
        experiment_io.cached_result(path, "other-config", "test-bundle")
    path.write_bytes(b"SYNTHETIC PARQUET CORRUPTION")
    with pytest.raises(Fail, match="missing/corrupt"):
        experiment_io.cached_result(path, "test-config", "test-bundle")


def test_full_matrix_yaml_and_plan_manifest_without_execution(tmp_path, monkeypatch):
    task = task_inputs.load_task(ROOT / "configs/task_engine.yaml")
    experiment = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    bundle = {
        "experiment": experiment,
        "task": task,
        "seeds": list(range(30)),
        "models": {"architectures": ["GCN", "GAT"]},
        "nulls": {"instances_per_variant": 30},
        "parameter_budget": {"scope": "SYNTHETIC_PLAN_FIXTURE"},
    }
    bundle["bundle_sha256"] = fingerprint(bundle)
    (tmp_path / "PREREGISTRATION.md").write_text("Status: DRAFT\nOSF DOI: <TODO>\n")
    monkeypatch.setattr(
        runner,
        "provenance",
        lambda *a: {"git_commit": "SYNTHETIC_TEST_ONLY", "source_sha256": "fixture"},
    )
    frame, report = runner.make_plan(tmp_path, bundle)
    assert len(frame) == frame["run_id"].n_unique() == 3660
    assert frame.filter(pl.col("condition") == "real").height == 60
    assert frame.filter(pl.col("condition") == "A").height == 1800
    assert frame.filter(pl.col("condition") == "B").height == 1800
    assert frame.filter((pl.col("seed") == 0) & (pl.col("architecture") == "GCN")).height == 61
    assert "OSF gate" in report["osf_gate"] and report["executes_models"] is False
    assert not (tmp_path / "results").exists()
    assert (tmp_path / report["plan_path"]).with_suffix(".manifest.json").exists()
    again, _ = runner.make_plan(tmp_path, bundle)
    assert again.equals(frame)
    first = tmp_path / frame["run_config_path"][0]
    first.write_text(first.read_text() + "seed: 999\n", encoding="utf-8")
    with pytest.raises(Fail, match="YAML differs"):
        runner.make_plan(tmp_path, bundle)


def test_one_button_execution_gate_cannot_load_graphs_or_train(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    for name in ("task_engine.yaml", "experiment.yaml"):
        (tmp_path / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    prereg = tmp_path / "PREREGISTRATION.md"
    original = b"Status: DRAFT v0.1\nOSF DOI: <TODO>\n"
    prereg.write_bytes(original)
    monkeypatch.setattr(
        runner, "load_bundle", lambda *a: pytest.fail("OSF gate must precede data/model work")
    )
    assert runner.main(["--execute", "--root", str(tmp_path)]) == 2
    assert "OSF gate" in capsys.readouterr().out
    assert prereg.read_bytes() == original and not (tmp_path / "results").exists()


def test_week1_report_reads_approved_bilateral_artifacts(tmp_path):
    from mbtopology.runtime import atomic_json

    config = {"products": {}, "approval": {"reviewed_by": "SYNTHETIC_TEST_ONLY"}}
    selection = {"run_id": "SYNTHETIC_MB", "scope": "bilateral"}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/mb_bilateral.yaml").write_text(yaml.safe_dump(selection))
    (tmp_path / "DECISIONS.md").write_text("SYNTHETIC_TEST_ONLY")
    artifact = tmp_path / "data/fixture.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"SYNTHETIC_ARTIFACT_ONLY")
    atomic_json(
        tmp_path / "data/graphs/SYNTHETIC_MB/MANIFEST.json",
        {
            "status": "PASS",
            "config": selection,
            "artifacts": {"fixture": {"path": "data/fixture.bin", "sha256": sha256(artifact)}},
        },
    )
    atomic_json(
        tmp_path / "reports/mb_sanity.json", {"status": "PASS", "scope": "SYNTHETIC_TEST_ONLY"}
    )
    atomic_json(
        tmp_path / "reports/task_toy_sanity.json",
        {"status": "PASS", "scope": "SYNTHETIC_TEST_ONLY"},
    )
    assert cli.status_report(tmp_path, config)["status"] == "PASS"
    selection["scope"] = "changed_fixture"
    (tmp_path / "configs/mb_bilateral.yaml").write_text(yaml.safe_dump(selection))
    assert cli.status_report(tmp_path, config)["status"] == "INCOMPLETE"
