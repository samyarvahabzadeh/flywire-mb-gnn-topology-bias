from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from .runtime import atomic_json, require
from .task_engine import TaskEngine


def make_engine(scientific_config, fixture):
    config = copy.deepcopy(scientific_config)
    config["version"] = "SYNTHETIC_TASK_FIXTURE_ONLY"
    config["task"].update(
        n_odors=fixture["n_odors"],
        trials_max=len(fixture["schedule"]),
        cs_plus=fixture["cs_plus"],
        cs_minus=fixture["cs_minus"],
    )
    config["plasticity"].update(
        eta=fixture["eta"],
        w_bounds=fixture["w_bounds"],
        d_us=fixture["d_us"],
        d_no_us=fixture["d_no_us"],
    )
    config["plasticity"]["w_init"].update(low=fixture["w_init"][0], high=fixture["w_init"][1])
    shape = tuple(fixture["weights_shape"])
    rows = np.repeat(np.arange(shape[0]), shape[1])
    columns = np.tile(np.arange(shape[1]), shape[0])
    pattern = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=shape).tocsr()
    return TaskEngine(
        pattern,
        np.arange(10, 10 + shape[0]),
        np.arange(20, 20 + shape[1]),
        [[index] for index in range(fixture["n_odors"])],
        fixture["schedule"],
        fixture["cs_plus"],
        config,
        fixture["avoidance_mbon_indices"],
        fixture["approach_mbon_indices"],
        fixture["seed"],
        assess_metrics=False,
    )


def run_toy(root, scientific_config, run):
    root = Path(root)
    fixture = yaml.safe_load((root / "configs/task_toy.yaml").read_text(encoding="utf-8"))
    require(
        fixture["fixture"] == "SYNTHETIC_ENGINEERING_TEST_ONLY",
        "toy source must be explicitly synthetic",
    )
    uninterrupted = make_engine(scientific_config, fixture)
    for _ in fixture["schedule"]:
        uninterrupted.step()
    engine = make_engine(scientific_config, fixture)
    initial = engine.state.weights.copy()
    engine.step()
    require(np.array_equal(engine.state.weights, initial), "CS- trial changed plastic weights")
    before = engine.state.weights.copy()
    engine.step()
    active = engine.rows == 0
    require(
        np.all(engine.state.weights[active] < before[active])
        and np.array_equal(engine.state.weights[~active], before[~active]),
        "US update did not target exactly the active KC synapses",
    )
    directory = root / "checkpoints/task_toy"
    engine.checkpoint(directory)
    restored = make_engine(scientific_config, fixture)
    require(restored.restore(directory), "toy checkpoint did not restore")
    while restored.state.cursor < len(fixture["schedule"]):
        restored.step()
    require(
        np.array_equal(restored.state.weights, uninterrupted.state.weights),
        "resumed toy differs from uninterrupted execution",
    )
    require(
        np.all((restored.state.weights >= 0) & (restored.state.weights <= 1)),
        "toy weights outside bounds",
    )
    require(
        restored.state.scores.size == 0, "toy must not generate learning curves or model metrics"
    )
    restored.checkpoint(directory)
    run.save(toy_completed_trials=restored.state.cursor, fixture="SYNTHETIC_ENGINEERING_TEST_ONLY")
    report = {
        "status": "PASS",
        "fixture": fixture["fixture"],
        "n_odors": fixture["n_odors"],
        "checks": [
            "CS- leaves weights unchanged",
            "US depresses only active KC synapses",
            "weights remain in bounds",
            "resumed state equals uninterrupted state",
            "no model metrics or learning curve computed",
        ],
        "model_metrics_computed": False,
    }
    atomic_json(root / "reports/task_toy_sanity.json", report)
    return report
