"""Frozen seed-paired estimands and signed-rank inference."""

import numpy as np
from confirmatory_data import CAP, CONDITIONS, SEEDS
from scipy.stats import false_discovery_control, rankdata

BOOTSTRAP_SEED = 20261119
BOOTSTRAP_DRAWS = 10_000


def seed_values(jobs, getter, groups=CONDITIONS):
    cells = {(seed, group): [] for seed in SEEDS for group in CONDITIONS}
    for job in jobs:
        cells[job.seed, job.condition].append(getter(job))
    return np.asarray(
        [
            np.median([value for group in groups for value in cells[seed, group]], axis=0)
            for seed in SEEDS
        ]
    )


def ranked(differences):
    kept = differences[differences != 0]
    return kept, rankdata(np.abs(kept), method="average")


def rank_biserial(differences):
    nonzero, ranks = ranked(differences)
    return float(np.dot(np.sign(nonzero), ranks) / ranks.sum()) if nonzero.size else 0.0


def exact_signed_rank(differences, *, two_sided=False):
    """Compute exact tails over independent signs of the observed midranks."""
    d, ranks = ranked(differences)
    if not d.size:
        return {"w_plus": 0.0, "w_minus": 0.0, "p_raw": 1.0, "rank_biserial": 0.0, "n_nonzero": 0}
    # Twice the average rank is an integer even in the presence of ties.
    scaled = (2 * ranks).astype(np.int64)
    observed = int(scaled[d > 0].sum())
    counts = np.array([1], dtype=np.int64)
    for rank in scaled:
        expanded = np.zeros(len(counts) + rank, dtype=np.int64)
        expanded[: len(counts)] = counts
        expanded[rank:] += counts
        counts = expanded
    upper = int(counts[observed:].sum()) / (1 << len(d))
    lower = int(counts[: observed + 1].sum()) / (1 << len(d))
    return {
        "w_plus": observed / 2,
        "w_minus": (int(scaled.sum()) - observed) / 2,
        "p_raw": min(1.0, 2 * min(upper, lower)) if two_sided else upper,
        "rank_biserial": (2 * observed - int(scaled.sum())) / int(scaled.sum()),
        "n_nonzero": len(d),
    }


def compare(jobs, getter, null_groups, *, smaller_favors_real, interval=False, two_sided=False):
    real = seed_values(jobs, getter, ("real",))
    null = seed_values(jobs, getter, null_groups)
    difference = (null - real) if smaller_favors_real else (real - null)
    report = {
        **exact_signed_rank(difference, two_sided=two_sided),
        "alternative": "two-sided" if two_sided else "greater",
        "n_pairs": len(SEEDS),
        "real_median": float(np.median(real)),
        "matched_null_median": float(np.median(null)),
        "median_difference_favor_real": float(np.median(difference)),
    }
    if interval:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(len(difference), size=(BOOTSTRAP_DRAWS, len(difference)))
        effects = [rank_biserial(difference[draw]) for draw in indices]
        report["rank_biserial_ci95"] = np.quantile(effects, [0.025, 0.975]).tolist()
    return report


def noise_ace(job):
    d0, d01, d02, d03 = job.noise_scores
    return (d0 + 2 * d01 + 2 * d02 + d03) / 6


def analysis_rows(jobs):
    def ttc_for_rank(job):
        return CAP + 1 if job.ttc is None else job.ttc

    def first_for_rank(job):
        return CAP + 1 if job.single_crossing is None else job.single_crossing

    comparisons = {
        label: compare(jobs, ttc_for_rank, groups, smaller_favors_real=True, interval=True)
        for label, groups in (("Pooled nulls", ("A", "B")), ("Null-A", ("A",)), ("Null-B", ("B",)))
    }
    secondaries = {
        "First single crossing ≥0.75": compare(
            jobs, first_for_rank, ("A", "B"), smaller_favors_real=True
        ),
        "Normalized AUC to cap": compare(
            jobs, lambda j: j.auc, ("A", "B"), smaller_favors_real=False
        ),
        "ACE": compare(jobs, noise_ace, ("A", "B"), smaller_favors_real=False, two_sided=True),
    }
    q_values = false_discovery_control([row["p_raw"] for row in secondaries.values()], method="bh")
    for row, q in zip(secondaries.values(), q_values, strict=True):
        row["p_bh"] = float(q)
    return comparisons, secondaries
