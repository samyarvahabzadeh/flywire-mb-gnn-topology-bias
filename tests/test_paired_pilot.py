from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from mbtopology.paired_pilot import PILOT_SEEDS, paired_schedule
from mbtopology.runtime import Fail
from mbtopology.task_inputs import schedule

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed", PILOT_SEEDS)
def test_pair_schedule_preserves_every_seeded_cue_block(seed):
    config = yaml.safe_load((ROOT / "configs/task_engine.yaml").read_text())
    paired = paired_schedule(config, seed)
    assert paired.shape == (2000, 2)
    expanded = deepcopy(config)
    expanded["task"]["trials_max"] = 4000
    replay = schedule(expanded, seed).reshape(-1, 8)
    assert np.array_equal(np.sort(paired.reshape(-1, 8), axis=1), np.sort(replay, axis=1))
    assert set(map(tuple, np.sort(paired, axis=1))) == {(0, 1), (2, 3), (4, 5), (6, 7)}
    assert paired_schedule(config, seed).tolist() == paired.tolist()
    assert config["task"]["trials_max"] == 2000
    with pytest.raises(Fail, match="approved pilot seeds"):
        paired_schedule(config, 0)
