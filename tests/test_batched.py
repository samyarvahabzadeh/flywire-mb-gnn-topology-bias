from pathlib import Path

import pytest
import torch
import yaml

from mbtopology import batched, gnn_training, models


@pytest.mark.parametrize("architecture", ["GCN", "GAT"])
def test_batch_probe_matches_sequential_probe(architecture):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/models.yaml").read_text())
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [2, 2, 3, 4, 4, 0]])
    graph = models.prepare_graph(
        edges,
        torch.tensor([1.0, 2.0, 3.0, 0.8, 0.9, 5.0]),
        torch.tensor([1, -1, 1, -1, 0, 1]),
        5,
    )
    data = {
        "graph": graph,
        "roles": torch.tensor([0, 0, 2, 2, 3]),
        "kc": torch.tensor([0, 1]),
        "patterns": [torch.tensor([0]), torch.tensor([1]), torch.tensor([0, 1])],
        "mbon": torch.tensor([2, 3]),
        "avoidance": torch.tensor([0]),
        "approach": torch.tensor([1]),
    }
    model = models.make_model(architecture, config, 1337).eval()
    with torch.no_grad():
        expected = torch.stack([gnn_training.predict(model, data, i) for i in range(3)])
        actual = batched.probe_values(model, data, batched.precompute_features(data))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
