"""Synthetic model plumbing and exact null invariants; no connectome performance run."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy import sparse

from mbtopology import models, nulls
from mbtopology.runtime import Fail

ROOT = Path(__file__).parents[1]


def synthetic_graph():
    n = 12
    edges = [(source, (source + offset) % n) for source in range(n) for offset in (1, 3, 5)]
    pre, post = np.array(edges).T
    matrix = sparse.coo_matrix((np.arange(len(pre)) + 5, (pre, post)), shape=(n, n)).tocsr()
    coo = matrix.tocoo()
    node_signs = np.array([-1, 0, 1] * 4, dtype=np.int8)
    return matrix, node_signs[coo.row], node_signs


@pytest.mark.parametrize("architecture", ["GCN", "GAT"])
def test_sparse_model_forward_gradient_and_identical_seed(architecture, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    matrix, signs, _ = synthetic_graph()
    coo = matrix.tocoo()
    graph = models.prepare_graph(
        torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long),
        torch.tensor(coo.data, dtype=torch.float32),
        torch.tensor(signs),
        matrix.shape[0],
    )
    monkeypatch.setattr(torch.Tensor, "to_dense", lambda *a, **k: pytest.fail("dense adjacency"))
    monkeypatch.setattr(
        sparse.csr_matrix, "toarray", lambda *a, **k: pytest.fail("dense adjacency")
    )
    first = models.make_model(architecture, config, 5)
    second = models.make_model(architecture, config, 5)
    assert all(
        torch.equal(first.state_dict()[key], value) for key, value in second.state_dict().items()
    )
    roles = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 3])
    x = models.node_features(roles, torch.tensor([0, 1, 2, 9]), torch.tensor([0, 2]))
    output = first(x, graph)
    assert output.shape == (12,) and torch.isfinite(output).all()
    assert bool(((output >= 0) & (output <= 1)).all())
    value = models.valence_readout(
        output, torch.tensor([5, 6]), torch.tensor([0]), torch.tensor([1])
    )
    assert torch.isfinite(value) and 0 <= float(value.detach()) <= 1
    output.sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in first.parameters()
    )


def test_parameter_budget_and_magnitude_normalization():
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    budget = models.parameter_budget(config)
    assert budget["status"] == "PASS" and budget["relative_difference"] <= 0.05
    graph = models.prepare_graph(
        torch.tensor([[0, 1], [1, 0]]), torch.tensor([100.0, 1.0]), torch.tensor([-1, 0]), 3
    )
    assert torch.isfinite(graph.gcn_norm).all() and graph.gcn_norm[0] < 0
    assert graph.message_sign[1] == 1
    assert graph.edge_index.shape[1] == 5


@pytest.mark.parametrize("variant", ["A", "B"])
def test_exact_nodewise_degree_weight_and_signed_invariants(variant, monkeypatch):
    matrix, signs, node_signs = synthetic_graph()
    monkeypatch.setattr(
        sparse.csr_matrix, "toarray", lambda *a, **k: pytest.fail("dense adjacency")
    )
    chain = nulls.DirectedSwapChain(matrix, signs, variant, 9)
    report = chain.run(180, 18000)
    result = chain.as_csr()
    assert report["accepted_swaps"] == 180
    assert np.array_equal(matrix.getnnz(axis=0), result.getnnz(axis=0))
    assert np.array_equal(matrix.getnnz(axis=1), result.getnnz(axis=1))
    assert np.array_equal(np.sort(matrix.data), np.sort(result.data))
    assert result.nnz == matrix.nnz and not result.diagonal().any()
    assert not np.array_equal(matrix.indices, result.indices)
    if variant == "B":
        before, after = matrix.tocoo(), result.tocoo()
        for sign in (-1, 0, 1):
            assert np.array_equal(
                np.bincount(before.col[node_signs[before.row] == sign], minlength=12),
                np.bincount(after.col[node_signs[after.row] == sign], minlength=12),
            )


@pytest.mark.parametrize("variant", ["A", "B"])
def test_null_checkpoint_restores_rng_bitwise(tmp_path, variant):
    matrix, signs, _ = synthetic_graph()
    uninterrupted = nulls.DirectedSwapChain(matrix, signs, variant, 3)
    uninterrupted.run(180, 18000)
    interrupted = nulls.DirectedSwapChain(matrix, signs, variant, 3)
    interrupted.run(40, 18000, checkpoint_dir=tmp_path)
    restored = nulls.DirectedSwapChain(matrix, signs, variant, 3)
    assert restored.restore(tmp_path)
    restored.run(180, 18000)
    assert np.array_equal(restored.post, uninterrupted.post)
    assert restored.attempts == uninterrupted.attempts
    assert restored.rng.bit_generator.state == uninterrupted.rng.bit_generator.state


def test_null_attempt_limit_and_unknown_nt_fail_loudly():
    matrix = sparse.coo_matrix(
        (np.ones(6), ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1])), shape=(3, 3)
    ).tocsr()
    with pytest.raises(Fail, match="budget exhausted"):
        nulls.DirectedSwapChain(matrix, np.zeros(6), "A", 0).run(1, 10)
    with pytest.raises(Fail, match="NT labels missing"):
        nulls.signs_from_nt(["TEST_UNKNOWN"], {"": 0})
