import numpy as np
import polars as pl
import pytest
from scipy import sparse

from mbtopology.real_cues import (
    check_projected_disjoint,
    cue_audit,
    project_cues,
    robustness_drives,
)


def test_real_projection_and_true_indegree_are_distinct_from_synapse_count():
    adjacency = sparse.csr_matrix([[5, 0, 0], [7, 11, 0], [8, 0, 9], [0, 0, 0]])
    cues = np.array([[[0, 1], [1, 2]]])
    activity, degree, weight = project_cues(adjacency, cues)
    np.testing.assert_allclose(activity.data, [0.6, 1.0, 0.75, 1.0, 1.0])
    assert degree.data.tolist() == [2, 1, 2, 1, 1]
    assert weight.data.tolist() == [12, 11, 15, 11, 9]

    ids = np.array([720575940000000000, 720575940000000001, 720575940000000002], dtype=np.uint64)
    frame = pl.DataFrame(cue_audit(cues, ids, degree, weight, "heldout")).to_dicts()[0]
    assert frame["heldout_kc_active"] == [2, 3]
    assert frame["heldout_kc_active_union"] == 3
    assert frame["heldout_pn_kc_edges"] == [3, 4]
    assert frame["heldout_pn_kc_synapses"] == [23, 35]
    assert frame["heldout_kc_indegree_histogram"] == [[1, 1, 1], [0, 2, 1]]
    assert frame["heldout_pn_root_ids"] == [ids[:2].tolist(), ids[1:].tolist()]

    check_projected_disjoint(activity[:1], activity[1:])
    with pytest.raises(Exception, match="held-out KC drive duplicates"):
        check_projected_disjoint(activity, activity[:1])


def test_positive_pn_noise_reaches_kcs_at_all_registered_levels():
    config = {
        "pairs_per_trial": 4,
        "robustness": {"levels": [0.0, 0.1, 0.2, 0.3], "probe_trials": 3, "stream": 406},
        "family_design": {
            "layout_seed": 1337,
            "pair_shared_pns": 16,
            "family_specific_pns": 16,
            "instance_variable_pns": 8,
            "train_stream": 404,
            "heldout_stream": 405,
        },
    }
    adjacency = sparse.eye(96, dtype=np.int64, format="csr")
    curves, audit = robustness_drives(adjacency, 1337, config)
    assert list(curves) == [0.0, 0.1, 0.2, 0.3]
    assert [values["pn_replacements_per_cue"] for values in audit.values()] == [0, 4, 8, 12]
    assert all(np.all(matrix.getnnz(axis=1) == 40) for matrix in curves.values())
    assert [values["projected_kc_rms_l2"] for values in audit.values()] == pytest.approx(
        np.sqrt([0, 8, 16, 24])
    )
