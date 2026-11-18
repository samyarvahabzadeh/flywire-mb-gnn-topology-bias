"""Proposed robustness perturbation, used only by the exploratory pilot."""

import numpy as np

from .runtime import require


def perturb_cue(active, n_kc, level, *, seed, cue):
    active = np.asarray(active, dtype=np.int64)
    require(
        np.isfinite(level)
        and 0 <= level <= 1
        and 0 < active.size < n_kc
        and active.ndim == 1
        and np.unique(active).size == active.size
        and active.min() >= 0
        and active.max() < n_kc,
        "invalid sparse cue or robustness level",
    )
    if level == 0:
        return np.sort(active)
    inactive = np.setdiff1d(np.arange(n_kc), active)
    count = min(active.size, inactive.size, max(1, int(np.floor(active.size * level + 0.5))))
    rng = np.random.default_rng(np.random.SeedSequence([seed, cue]))
    removed = rng.permutation(active)[:count]
    added = rng.permutation(inactive)[:count]
    return np.sort(np.concatenate((np.setdiff1d(active, removed), added)))
