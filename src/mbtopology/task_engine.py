from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import sparse

from .registration import require_registered
from .runtime import atomic_file, atomic_json, fingerprint, read_json, require, sha256
from .task_inputs import initial_weights, load_source, node_identity, schedule


@dataclass(frozen=True)
class State:
    cursor: int
    weights: np.ndarray
    streak: int
    reached_at: int | None
    scores: np.ndarray


class TaskEngine:
    def __init__(
        self,
        pattern,
        kc_ids,
        mbon_ids,
        patterns,
        trial_order,
        cs_plus,
        config,
        avoidance,
        approach,
        seed,
        assess_metrics=False,
    ):
        require(
            sparse.isspmatrix_csr(pattern) and pattern.has_canonical_format,
            "plastic topology must be canonical sparse CSR",
        )
        require(
            pattern.shape == (len(kc_ids), len(mbon_ids)) and pattern.nnz > 0,
            "plastic topology shape/edge mismatch",
        )
        coo = pattern.tocoo(copy=False)
        self.rows, self.cols = coo.row.astype(np.int64), coo.col.astype(np.int64)
        self.shape = pattern.shape
        self.indegree = np.maximum(1, np.bincount(self.cols, minlength=self.shape[1]))
        self.patterns = [np.asarray(value, dtype=np.int64) for value in patterns]
        require(
            all(
                value.size
                and len(np.unique(value)) == len(value)
                and value.min() >= 0
                and value.max() < self.shape[0]
                for value in self.patterns
            ),
            "invalid frozen odor indices",
        )
        self.order = np.asarray(trial_order, dtype=np.int64)
        require(
            self.order.size and self.order.min() >= 0 and self.order.max() < len(patterns),
            "invalid trial schedule",
        )
        self.cs_plus = set(cs_plus)
        self.config, self.seed, self.assess_metrics = config, seed, assess_metrics
        self.avoidance = np.asarray(avoidance, dtype=np.int64)
        self.approach = np.asarray(approach, dtype=np.int64)
        require(
            self.avoidance.size
            and self.approach.size
            and not np.intersect1d(self.avoidance, self.approach).size,
            "readout requires disjoint nonempty groups",
        )
        require(
            min(self.avoidance.min(), self.approach.min()) >= 0
            and max(self.avoidance.max(), self.approach.max()) < self.shape[1],
            "readout index out of bounds",
        )
        plasticity = config["plasticity"]
        require(plasticity["eligibility"] == "instantaneous", "v1 has no eligibility trace")
        self.low, self.high = plasticity["w_bounds"]
        require(
            0 <= self.low < self.high <= 1 and plasticity["eta"] > 0,
            "invalid plasticity bounds/rate",
        )
        initial = initial_weights(
            np.asarray(kc_ids)[self.rows],
            np.asarray(mbon_ids)[self.cols],
            seed,
            plasticity["w_init"]["low"],
            plasticity["w_init"]["high"],
        )
        self.state = State(0, initial, 0, None, np.empty(0, dtype=np.float64))
        self.signature = fingerprint(
            {
                "config": config,
                "seed": seed,
                "assess_metrics": assess_metrics,
                "kc_ids": [str(x) for x in kc_ids],
                "mbon_ids": [str(x) for x in mbon_ids],
                "edge_rows": self.rows.tolist(),
                "edge_cols": self.cols.tolist(),
                "patterns": [x.tolist() for x in self.patterns],
                "order": self.order.tolist(),
                "avoidance": self.avoidance.tolist(),
                "approach": self.approach.tolist(),
                "implementation_sha256": sha256(__file__),
            }
        )

    def activity(self, odor, weights=None):
        require(0 <= odor < len(self.patterns), "odor ID outside frozen set")
        pre = np.zeros(self.shape[0], dtype=np.float64)
        pre[self.patterns[odor]] = 1.0
        weights = self.state.weights if weights is None else weights
        post = (
            np.bincount(self.cols, weights=weights * pre[self.rows], minlength=self.shape[1])
            / self.indegree
        )
        require(
            np.isfinite(post).all() and np.all((post >= 0) & (post <= 1 + 1e-12)),
            "MBON activity out of bounds",
        )
        return pre, post

    def valence(self, odor, weights=None):
        _, post = self.activity(odor, weights)
        contrast = post[self.avoidance].mean() - post[self.approach].mean()
        return float(np.clip((contrast + 1) / 2, 0, 1))

    def discrimination(self, weights):
        values = np.array([self.valence(odor, weights) for odor in range(len(self.patterns))])
        plus = np.array(sorted(self.cs_plus), dtype=np.int64)
        minus = np.array(
            [odor for odor in range(len(values)) if odor not in self.cs_plus], dtype=np.int64
        )
        require(plus.size and minus.size, "both CS classes required for discrimination")
        return float(0.5 * (values[plus].mean() + (1 - values[minus]).mean()))

    def step(self):
        old = self.state
        require(old.cursor < len(self.order), "task trial cap already reached")
        odor = int(self.order[old.cursor])
        pre, post = self.activity(odor, old.weights)
        plasticity = self.config["plasticity"]
        d = plasticity["d_us"] if odor in self.cs_plus else plasticity["d_no_us"]
        updated = np.clip(
            old.weights - plasticity["eta"] * pre[self.rows] * post[self.cols] * d,
            self.low,
            self.high,
        )
        require(
            np.isfinite(updated).all() and np.all(updated <= old.weights + 1e-12),
            "anti-Hebbian v1 update must be finite and nonincreasing",
        )
        streak, reached, scores = old.streak, old.reached_at, old.scores
        if self.assess_metrics:
            value = self.discrimination(updated)
            scores = np.append(scores, value)
            streak = streak + 1 if value >= self.config["readout"]["criterion"] else 0
            if reached is None and streak >= self.config["readout"]["consecutive_trials"]:
                reached = old.cursor + 1
        self.state = State(old.cursor + 1, updated, streak, reached, scores)

    def checkpoint(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.state
        target = directory / f"state-{state.cursor:06d}-{uuid4().hex}.npz"
        atomic_file(
            target,
            lambda p: np.savez_compressed(
                p,
                weights=state.weights,
                cursor=np.array(state.cursor),
                streak=np.array(state.streak),
                reached_at=np.array(-1 if state.reached_at is None else state.reached_at),
                scores=state.scores,
            ),
        )
        pointer = {
            "signature": self.signature,
            "file": target.name,
            "sha256": sha256(target),
            "completed_trials": state.cursor,
            "assess_metrics": self.assess_metrics,
        }
        atomic_json(directory / "checkpoint.json", pointer)
        return pointer

    def restore(self, directory):
        directory = Path(directory)
        pointer = read_json(directory / "checkpoint.json")
        if pointer is None:
            return False
        require(
            pointer["signature"] == self.signature, "task resume config/graph/inputs/code differ"
        )
        path = directory / pointer["file"]
        require(sha256(path) == pointer["sha256"], "task checkpoint checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            weights = saved["weights"].copy()
            cursor, streak, reached = (
                int(saved["cursor"]),
                int(saved["streak"]),
                int(saved["reached_at"]),
            )
            scores = saved["scores"].copy()
        require(
            weights.shape == self.state.weights.shape
            and np.isfinite(weights).all()
            and np.all((weights >= self.low) & (weights <= self.high)),
            "task checkpoint weights invalid",
        )
        require(
            0 <= cursor <= len(self.order)
            and streak >= 0
            and len(scores) == (cursor if self.assess_metrics else 0),
            "task checkpoint cursor/history invalid",
        )
        self.state = State(cursor, weights, streak, None if reached == -1 else reached, scores)
        return True


def from_frozen(root, config, seed, graph_manifest=None, assess_metrics=True):
    root = Path(root)
    if assess_metrics:
        require_registered(root)
    _, canonical, _ = load_source(root, config["source_manifest"])
    _, nodes, matrix = load_source(root, graph_manifest or config["source_manifest"])
    inputs = read_json(root / "reports" / "task_inputs.json")
    require(
        inputs and inputs["task_config_sha256"] == fingerprint(config),
        "task inputs not prepared for current config",
    )
    require(
        node_identity(nodes)
        == node_identity(canonical)
        == inputs["canonical_node_identity_sha256"],
        "FAIRNESS FAIL: node identities, roles or annotations changed between conditions",
    )
    for path, expected in (
        (config["cs_encoding"]["odor_sets"], inputs["odor_sets_sha256"]),
        (config["readout"]["frozen_membership"], inputs["readout_sha256"]),
        (config["seeds_file"], inputs["seeds_sha256"]),
    ):
        require(sha256(root / path) == expected, "FAIRNESS FAIL: frozen task artifact changed")
    odors = read_json(root / config["cs_encoding"]["odor_sets"])
    readout = read_json(root / config["readout"]["frozen_membership"])
    require(str(seed) in odors["seeds"], "seed is not in the frozen 0..29 seed set")
    roles = nodes["role"].to_numpy()
    kc, mbon = np.flatnonzero(roles == "KC"), np.flatnonzero(roles == "MBON")
    kc_ids, mbon_ids = nodes["root_id"].to_numpy()[kc], nodes["root_id"].to_numpy()[mbon]
    require(
        [str(x) for x in kc_ids] == odors["kc_root_ids"]
        and [str(x) for x in mbon_ids] == readout["mbon_root_ids"],
        "FAIRNESS FAIL: frozen neuron order changed",
    )
    return TaskEngine(
        matrix[kc][:, mbon].tocsr(),
        kc_ids,
        mbon_ids,
        odors["seeds"][str(seed)],
        schedule(config, seed),
        config["task"]["cs_plus"],
        config,
        readout["avoidance_indices"],
        readout["approach_indices"],
        seed,
        assess_metrics=assess_metrics,
    )


def run_biological(root, config, seed, run, graph_manifest=None, run_id=None):
    require_registered(root)
    engine = from_frozen(root, config, seed, graph_manifest, assess_metrics=True)
    directory = Path(root) / "checkpoints" / "task" / (run_id or f"real-seed-{seed:02d}")
    engine.restore(directory)
    last_save = time.monotonic()
    try:
        while (
            engine.state.cursor < config["task"]["trials_max"] and engine.state.reached_at is None
        ):
            engine.step()
            if time.monotonic() - last_save >= config["checkpoint_seconds"]:
                engine.checkpoint(directory)
                run.save(task_completed_trials=engine.state.cursor)
                last_save = time.monotonic()
    finally:
        engine.checkpoint(directory)
    return engine
