"""Synthetic mechanics/invariant tests only; no biological performance runs."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from mbtopology import task_engine, task_inputs, task_toy
from mbtopology.runtime import Fail, atomic_json, fingerprint, read_json, sha256

ROOT = Path(__file__).parents[1]


def configs():
    return (
        task_inputs.load_task(ROOT / "configs/task_engine.yaml"),
        yaml.safe_load((ROOT / "configs/task_toy.yaml").read_text()),
    )


def test_three_odor_rule_and_bitwise_resume(tmp_path):
    config, fixture = configs()
    engine = task_toy.make_engine(config, fixture)
    before = engine.state.weights.copy()
    engine.step()
    assert np.array_equal(engine.state.weights, before)
    pre, post = engine.activity(0)
    engine.step()
    expected = np.clip(before - 0.05 * pre[engine.rows] * post[engine.cols], 0, 1)
    assert np.array_equal(engine.state.weights, expected)
    assert np.all(engine.state.weights[engine.rows == 0] < before[engine.rows == 0])
    assert np.array_equal(engine.state.weights[engine.rows != 0], before[engine.rows != 0])
    engine.checkpoint(tmp_path)
    resumed = task_toy.make_engine(config, fixture)
    assert resumed.restore(tmp_path)
    while engine.state.cursor < len(fixture["schedule"]):
        engine.step()
        resumed.step()
    assert np.array_equal(engine.state.weights, resumed.state.weights)
    assert resumed.state.scores.size == 0
    assert 0 <= resumed.valence(0) <= 1


def test_checkpoint_corruption_and_input_change_fail(tmp_path):
    config, fixture = configs()
    engine = task_toy.make_engine(config, fixture)
    engine.step()
    engine.checkpoint(tmp_path)
    changed = copy.deepcopy(fixture)
    changed["seed"] = 1
    with pytest.raises(Fail, match="resume config"):
        task_toy.make_engine(config, changed).restore(tmp_path)
    pointer = read_json(tmp_path / "checkpoint.json")
    (tmp_path / pointer["file"]).write_bytes(b"SYNTHETIC CHECKPOINT CORRUPTION")
    with pytest.raises(Fail, match="checksum"):
        task_toy.make_engine(config, fixture).restore(tmp_path)


def test_criterion_requires_twenty_consecutive_trials(monkeypatch):
    config, fixture = configs()
    fixture["schedule"] = [0] * 40
    engine = task_toy.make_engine(config, fixture)
    engine.assess_metrics = True
    literal_fixture_scores = iter([0.8] * 19 + [0.1] + [0.8] * 20)
    monkeypatch.setattr(engine, "discrimination", lambda _: next(literal_fixture_scores))
    for _ in range(19):
        engine.step()
    assert engine.state.reached_at is None and engine.state.streak == 19
    engine.step()
    assert engine.state.streak == 0
    for _ in range(20):
        engine.step()
    assert engine.state.reached_at == 40 and engine.state.streak == 20


def test_frozen_odors_schedule_and_shared_edge_initialization(tmp_path):
    config, _ = configs()
    a = task_inputs.make_patterns(5177, 8, 0.05, 0)
    assert a == task_inputs.make_patterns(5177, 8, 0.05, 0)
    assert all(len(value) == len(set(value)) == 259 for value in a)
    assert a != task_inputs.make_patterns(5177, 8, 0.05, 1)
    schedule = task_inputs.schedule(config, 4)
    assert np.array_equal(schedule, task_inputs.schedule(config, 4))
    assert len(schedule) == 2000
    assert all(sorted(block.tolist()) == list(range(8)) for block in schedule.reshape(-1, 8))
    first = task_inputs.initial_weights([1, 2, 3], [4, 5, 6], 3)
    reordered = task_inputs.initial_weights([3, 1, 2], [6, 4, 5], 3)
    assert np.array_equal(first, reordered[[1, 2, 0]])
    assert np.all((first >= 0.8) & (first < 1))
    path = tmp_path / "frozen.json"
    task_inputs.freeze_file(path, {"patterns": a})
    stamp = path.stat().st_mtime_ns
    task_inputs.freeze_file(path, {"patterns": a})
    assert stamp == path.stat().st_mtime_ns
    with pytest.raises(Fail, match="frozen input changed"):
        task_inputs.freeze_file(path, {"patterns": []})


def test_osf_gate_precedes_biological_graph_loading(tmp_path, monkeypatch):
    config, _ = configs()
    (tmp_path / "PREREGISTRATION.md").write_text("Status: DRAFT v0.1\nOSF DOI: <TODO>\n")
    monkeypatch.setattr(
        task_engine, "load_source", lambda *a: pytest.fail("graph loaded before OSF gate")
    )
    with pytest.raises(Fail, match="OSF gate"):
        task_engine.from_frozen(tmp_path, config, 0)
    with pytest.raises(Fail, match="OSF gate"):
        task_engine.run_biological(tmp_path, config, 0, None)


def test_registration_evidence_binds_exact_task_and_bundle(tmp_path):
    config, _ = configs()
    document = tmp_path / "PREREGISTRATION.md"
    document.write_text(
        "Status: REGISTERED — SYNTHETIC UNIT FIXTURE\nOSF DOI: 10.17605/OSF.IO/TESTFIXTURE\n",
        encoding="utf-8",
    )
    evidence = {
        "osf_doi": "10.17605/OSF.IO/TESTFIXTURE",
        "registered_at": "SYNTHETIC_FIXTURE_ONLY",
        "preregistration_sha256": sha256(document),
        "task_config_sha256": fingerprint(config),
        "bundle_sha256": "SYNTHETIC_BUNDLE_ONLY",
    }
    atomic_json(tmp_path / "configs/osf_registration.json", evidence)
    assert task_inputs.require_registered(tmp_path, config, "SYNTHETIC_BUNDLE_ONLY") == evidence
    with pytest.raises(Fail, match="bundle differs"):
        task_inputs.require_registered(tmp_path, config, "CHANGED_FIXTURE_BUNDLE")
    changed = copy.deepcopy(config)
    changed["plasticity"]["eta"] = 0.1
    with pytest.raises(Fail, match="task configuration differs"):
        task_inputs.require_registered(tmp_path, changed, "SYNTHETIC_BUNDLE_ONLY")
