from pathlib import Path

import numpy as np
import yaml

from mbtopology.cue_families import choose_cap, make_cues, overlap_statistics, second_order

CONFIG = yaml.safe_load(
    (Path(__file__).parents[1] / "configs/phase3_pilot.yaml").read_text(encoding="utf-8")
)
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_family_splits_preserve_activity_and_measured_overlap():
    cfg = CONFIG["family_design"]
    pn, kc, pool = make_cues(cfg, 1337, 685, 5177, 24, PAIRS)
    assert pn[0].shape == pn[1].shape == (24, 8, 48)
    assert kc[0].shape == kc[1].shape == (24, 8, 259)
    for train, test in (pn, kc):
        assert np.all(np.diff(train, axis=-1) > 0)
        assert {x.tobytes() for x in train.reshape(-1, train.shape[-1])}.isdisjoint(
            x.tobytes() for x in test.reshape(-1, test.shape[-1])
        )
    empirical = overlap_statistics(*pn, PAIRS)
    exact = second_order(cfg, pool)
    assert pool == 493
    for relation, result in empirical.items():
        assert abs(result["mean_shared"] - exact[f"{relation}_expected_shared"]) < 0.5
    assert exact["two_variable_pn_covariance"] < 0


def test_cap_requires_full_qualifying_window_and_padding():
    cfg = CONFIG["cap_selection"]
    values = np.r_[np.full(200, 0.5), np.full(400, 0.81)]
    assert choose_cap(values, 220, 0.8, cfg) == (400, 280)
    assert choose_cap(values, None, 0.8, cfg) == (None, None)
    assert choose_cap(values[:280], 220, 0.8, cfg) == (None, 280)
