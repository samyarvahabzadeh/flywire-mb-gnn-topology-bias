"""Synthetic tests of researcher-approved bilateral selection rules."""

from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from scipy import sparse

from mbtopology import mb_subset


def test_bilateral_class_rules_dan_prefix_nt_and_induction(monkeypatch):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/mb_bilateral.yaml").read_text())
    config["sanity"].update(whole_brain_expected=8, kc_minimum=2, kc_maximum=2)
    nodes = pl.DataFrame(
        {
            "node_index": range(8),
            "root_id": [720575940000000001 + i for i in range(8)],
            "class": ["Kenyon_Cell", "Kenyon_Cell", "ALPN", "MBON", "DAN", "DAN", "DAN", "visual"],
            "primary_type": [
                "TEST_KCγ",
                "TEST_KCαβ",
                "TEST_ALPN",
                "TEST_MBON",
                "PAM_TEST",
                "PPL1_TEST",
                "PPL2_TEST",
                "TEST_PN",
            ],
            "nt_type": ["ACH", "ACH", "ACH", "GABA", "DA", "", "DA", "ACH"],
            "side": ["left", "right", "left", "right", "left", "right", "left", "left"],
        }
    ).with_columns(pl.col("root_id").cast(pl.UInt64))
    selected_nodes, selected, diagnostics = mb_subset.select_nodes(nodes, config)
    assert selected.tolist() == [0, 1, 2, 3, 4]
    assert selected_nodes["role"].to_list() == ["KC", "KC", "PN", "MBON", "DAN"]
    assert diagnostics["dan_prefix_candidates"] == 2 and diagnostics["dan_excluded_by_nt"] == 1
    assert diagnostics["dan_selected_types"] == [{"primary_type": "PAM_TEST", "count": 1}]
    parent = sparse.coo_matrix(
        (np.full(5, 5), ([2, 2, 0, 4, 6], [0, 1, 3, 3, 3])), shape=(8, 8)
    ).tocsr()
    monkeypatch.setattr(
        sparse.csr_matrix, "toarray", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dense"))
    )
    matrix = parent[selected][:, selected].tocsr()
    report = mb_subset.sanity_report(parent, matrix, selected_nodes, config, diagnostics)
    assert report["status"] == "PASS" and report["mb_edges"] == 4
    assert report["pn_kc_convergence"]["mean"] == 1
    data = mb_subset.make_data(matrix, selected_nodes)
    assert data.num_nodes == 5 and data.edge_weight.dtype == torch.int64
    assert data.root_id.tolist() == nodes["root_id"].to_list()[:5]
    assert "edge_sign" not in data
    config["sanity"]["kc_minimum"] = 3
    assert (
        mb_subset.sanity_report(parent, matrix, selected_nodes, config, diagnostics)["status"]
        == "FAIL"
    )
    leading_unselected = nodes.with_columns(
        pl.when(pl.col("node_index") == 0)
        .then(pl.lit("visual"))
        .otherwise(pl.col("class"))
        .alias("class")
    )
    reduced, indices, _ = mb_subset.select_nodes(leading_unselected, config)
    assert indices.tolist() == [1, 2, 3, 4] and reduced["role"].to_list()[0] == "KC"
