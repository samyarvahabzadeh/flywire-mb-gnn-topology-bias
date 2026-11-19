from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import wilcoxon


def test_exact_signed_rank_includes_tied_permutations_and_drops_zeros(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "analysis"))
    from confirmatory_stats import exact_signed_rank

    result = exact_signed_rank(np.array([0, 1, -1, 2]))
    assert result == {
        "w_plus": 4.5,
        "w_minus": 1.5,
        "p_raw": 0.375,
        "rank_biserial": 0.5,
        "n_nonzero": 3,
    }
    all_zero = exact_signed_rank(np.zeros(30))
    assert all_zero["p_raw"] == 1.0 and all_zero["rank_biserial"] == 0.0
    difference = np.array([-2, 3, 4, -1, 6, 5])
    assert exact_signed_rank(difference)["p_raw"] == pytest.approx(
        wilcoxon(difference, method="exact", alternative="greater").pvalue
    )
    assert exact_signed_rank(np.array([0, 1, -1, 2]), two_sided=True)["p_raw"] == 0.75
    assert exact_signed_rank(difference, two_sided=True)["p_raw"] == pytest.approx(
        wilcoxon(difference, method="exact", alternative="two-sided").pvalue
    )


def test_prescribed_inventory_excludes_benchmark_seeds_and_covers_all_nulls(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "analysis"))
    from confirmatory_data import frozen_jobs

    jobs = frozen_jobs()
    assert len(jobs) == len({row[0] for row in jobs}) == 1830
    assert {job[1] for job in jobs} == set(range(30))
    for seed in range(30):
        assert (
            sum((s, group, instance) == (seed, "real", None) for _, s, group, instance in jobs) == 1
        )
        for variant in ("A", "B"):
            assert {
                instance for _, s, group, instance in jobs if (s, group) == (seed, variant)
            } == set(range(30))


def test_ace_uses_the_four_locked_trapezoid_weights(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "analysis"))
    from confirmatory_stats import noise_ace

    job = SimpleNamespace(noise_scores=(0.6, 0.8, 0.6, 0.4))
    assert noise_ace(job) == pytest.approx((0.6 + 1.6 + 1.2 + 0.4) / 6)
