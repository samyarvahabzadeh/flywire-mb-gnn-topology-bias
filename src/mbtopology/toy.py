from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from .runtime import atomic_file, atomic_json, require


def three_factor_update(weights, pre, post_eligibility, dopamine, learning_rate):
    require(sparse.isspmatrix_csr(weights), "plastic weights must be sparse CSR")
    require(
        pre.shape == (weights.shape[0],)
        and post_eligibility.shape == dopamine.shape == (weights.shape[1],),
        "three-factor vector shape mismatch",
    )
    coo = weights.tocoo()
    change = learning_rate * pre[coo.row] * post_eligibility[coo.col] * dopamine[coo.col]
    return sparse.coo_matrix((coo.data + change, (coo.row, coo.col)), shape=weights.shape).tocsr()


def toy_check(root, run):
    root = Path(root)
    path = root / "checkpoints" / "toy_weights.npz"
    # Explicitly synthetic algebra fixture; no FlyWire data or fitted model.
    weights = sparse.eye(3, format="csr", dtype=np.float64)
    odors = np.eye(3, dtype=np.float64)
    eligibility = np.ones(3, dtype=np.float64)
    initial = weights.copy()
    unpaired = three_factor_update(weights, odors[0], eligibility, np.zeros(3), 0.25)
    require((unpaired != initial).nnz == 0, "unpaired CS changed weights")
    for trial, odor in enumerate(odors):
        modulator = np.zeros(3, dtype=np.float64)
        modulator[trial] = 1.0
        before = weights.copy()
        weights = three_factor_update(weights, odor, eligibility, modulator, 0.25)
        delta = weights - before
        require(
            delta.nnz == 1 and delta[trial, trial] == 0.25,
            "three-factor update changed an inactive synapse",
        )
        atomic_file(path, lambda p, weights=weights: sparse.save_npz(p, weights))
        restored = sparse.load_npz(path)
        require((restored != weights).nnz == 0, "toy checkpoint restore mismatch")
        weights = restored
        run.save(trial=trial + 1, fixture="synthetic algebra only")
    report = {
        "status": "PASS",
        "fixture": "SYNTHETIC_UNIT_TEST_ONLY",
        "n_odors": 3,
        "checks": [
            "zero US leaves weights unchanged",
            "only active synapse changes",
            "sparse checkpoint round trip",
        ],
        "scope": "algebra and checkpoint plumbing; biological task engine remains unapproved",
        "model_metrics_computed": False,
    }
    atomic_json(root / "reports" / "toy_sanity.json", report)
    return report
