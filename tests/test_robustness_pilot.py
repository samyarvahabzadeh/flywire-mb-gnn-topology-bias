from pathlib import Path

import numpy as np
import pytest
import yaml

from mbtopology.robustness_pilot import perturb_cue

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/robustness_pilot.yaml").read_text())


@pytest.mark.parametrize("level", CONFIG["levels"])
def test_configured_robustness_level_has_expected_feature_noise(level):
    active = np.arange(259)
    result = perturb_cue(active, 5177, level, seed=1337, cue=0)
    clean_mask, noisy_mask = np.zeros(5177), np.zeros(5177)
    clean_mask[active] = 1
    noisy_mask[result] = 1
    norm = np.linalg.norm(noisy_mask - clean_mask)
    assert (norm > 0) == (level > 0)
    assert len(np.unique(result)) == len(active)
    assert np.array_equal(result, perturb_cue(active, 5177, level, seed=1337, cue=0))
