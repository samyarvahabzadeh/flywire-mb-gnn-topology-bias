import pytest

from mbtopology.binary_protocol import add_observation, endpoints, reached

CONFIG = {"threshold": 0.75, "rolling_window": 20, "trials": 40}


def record(history, score):
    add_observation(history, (score, 1 - score), 20)


def test_first_complete_window_can_pass_without_twenty_individual_crossings():
    history = []
    for _ in range(19):
        record(history, 0.75)
        assert not reached(history, 0.75)
    record(history, 0.625)
    assert not reached(history, 0.75)
    record(history, 0.875)
    assert reached(history, 0.75)
    results = endpoints(history, CONFIG)
    assert results["ttc"] == 21 and results["first_single_crossing"] == 1
    assert results["auc_to_cap_normalized"] == pytest.approx(0.75)


def test_time_cap_distinguishes_exact_hit_from_right_censoring():
    history = []
    for _ in range(39):
        record(history, 0.5)
    record(history, 1.0)
    results = endpoints(history, CONFIG)
    assert results["ttc"] is None and results["right_censored"]
    assert results["first_single_crossing"] == 40
    assert results["auc_to_cap_normalized"] == pytest.approx(20.5 / 40)
