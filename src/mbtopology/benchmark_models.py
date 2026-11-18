"""Architecture variants used by engineering benchmarks only."""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .models import make_model

ARCHITECTURES = ("GCN", "GraphSAGE")


class BenchmarkGraphSAGE(nn.Module):
    architecture = "GraphSAGE"

    def __init__(self, config, aggregation="mean"):
        super().__init__()
        if aggregation not in ("mean", "max"):
            raise ValueError("GraphSAGE aggregation must be mean or max")
        width = config["hidden_width"]
        self.encoder = nn.Linear(5, width)
        self.convs = nn.ModuleList(
            SAGEConv(width, width, aggr=aggregation) for _ in range(config["message_passing_depth"])
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x, graph):
        source, target = graph.edge_index
        # SAGEConv has its own root transform. Use directed, unweighted neighbors.
        x = torch.relu(self.encoder(x))
        neighbors = graph.edge_index[:, source != target]
        for layer in self.convs:
            x = torch.relu(layer(x, neighbors))
        return torch.sigmoid(self.head(x)).squeeze(-1)


def make_benchmark_model(architecture, config, seed, device="cpu", *, aggregation="mean"):
    if architecture == "GCN":
        return make_model(architecture, config, seed, device)
    if architecture != "GraphSAGE":
        raise ValueError(f"unknown benchmark architecture: {architecture}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BenchmarkGraphSAGE(config, aggregation)
    return model.to(device)
