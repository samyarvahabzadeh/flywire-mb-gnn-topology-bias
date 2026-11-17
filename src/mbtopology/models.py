from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

from .runtime import require


@dataclass(frozen=True)
class PreparedGraph:
    edge_index: torch.Tensor
    gcn_norm: torch.Tensor
    gat_attributes: torch.Tensor
    message_sign: torch.Tensor


def prepare_graph(edge_index, magnitudes, signs, n_nodes, self_loop_weight=1.0):
    require(
        edge_index.dtype == torch.long and edge_index.shape == (2, len(magnitudes)),
        "invalid sparse edge index",
    )
    require(
        len(signs) == len(magnitudes)
        and bool(torch.isfinite(magnitudes).all())
        and bool((magnitudes > 0).all()),
        "edge magnitudes must be positive/finite",
    )
    require(bool(torch.isin(signs, signs.new_tensor([-1, 0, 1])).all()), "invalid sign label")
    require(
        edge_index.numel() == 0 or (int(edge_index.min()) >= 0 and int(edge_index.max()) < n_nodes),
        "edge endpoint outside node set",
    )
    source, target = edge_index
    present = torch.zeros(n_nodes, dtype=torch.bool, device=edge_index.device)
    present[source[source == target]] = True
    missing = torch.arange(n_nodes, device=edge_index.device)[~present]
    edge_index = torch.cat((edge_index, torch.stack((missing, missing))), dim=1)
    magnitudes = torch.cat((magnitudes, magnitudes.new_full((len(missing),), self_loop_weight)))
    signs = torch.cat((signs, signs.new_ones(len(missing))))
    source, target = edge_index
    degree = magnitudes.new_zeros(n_nodes).index_add_(0, target, magnitudes)
    inverse = degree.clamp_min(torch.finfo(degree.dtype).tiny).rsqrt()
    effective = torch.where(signs == 0, torch.ones_like(signs), signs).to(magnitudes.dtype)
    norm = magnitudes * inverse[source] * inverse[target] * effective
    attributes = torch.stack((torch.log1p(magnitudes), signs.to(magnitudes.dtype)), dim=1)
    return PreparedGraph(edge_index, norm, attributes, effective)


class SignedGATConv(GATConv):
    def forward(self, x, edge_index, edge_attr, message_sign):
        self._message_sign = message_sign
        return super().forward(x, edge_index, edge_attr=edge_attr)

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j * self._message_sign[:, None, None]


class ConnectomeGNN(nn.Module):
    def __init__(self, architecture, config, input_features=5):
        super().__init__()
        require(architecture in ("GCN", "GAT"), "unknown model architecture")
        width, depth, heads = (
            config["hidden_width"],
            config["message_passing_depth"],
            config["gat_heads"],
        )
        require(
            width == 128 and depth in (2, 3) and width % heads == 0,
            "model must respect frozen width/depth budget",
        )
        self.architecture = architecture
        self.encoder = nn.Linear(input_features, width)
        if architecture == "GCN":
            self.convs = nn.ModuleList(
                [
                    GCNConv(width, width, add_self_loops=False, normalize=False, cached=False)
                    for _ in range(depth)
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    SignedGATConv(
                        width,
                        width // heads,
                        heads=heads,
                        concat=True,
                        edge_dim=2,
                        dropout=config["dropout"],
                        add_self_loops=False,
                        residual=False,
                    )
                    for _ in range(depth)
                ]
            )
        self.dropout = nn.Dropout(config["dropout"])
        self.head = nn.Linear(width, 1)

    def forward(self, x, graph):
        x = torch.relu(self.encoder(x))
        for layer in self.convs:
            if self.architecture == "GCN":
                x = layer(x, graph.edge_index, edge_weight=graph.gcn_norm)
            else:
                x = layer(x, graph.edge_index, graph.gat_attributes, graph.message_sign)
            x = self.dropout(torch.relu(x))
        return torch.sigmoid(self.head(x)).squeeze(-1)


def make_model(architecture, config, seed, device="cpu"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ConnectomeGNN(architecture, config)
    return model.to(device)


def parameter_count(model):
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def parameter_budget(config):
    counts = {name: parameter_count(make_model(name, config, 0)) for name in ("GCN", "GAT")}
    difference = abs(counts["GCN"] - counts["GAT"]) / min(counts.values())
    require(
        difference <= config["parameter_budget_relative_tolerance"],
        "GCN/GAT parameter budgets differ by more than the frozen tolerance",
    )
    return {"counts": counts, "relative_difference": difference, "status": "PASS"}


def node_features(role_ids, kc_indices, active_kc_indices):
    x = torch.zeros((len(role_ids), 5), dtype=torch.float32, device=role_ids.device)
    x[torch.arange(len(role_ids), device=role_ids.device), role_ids] = 1
    x[kc_indices[active_kc_indices], 4] = 1
    return x


def valence_readout(outputs, mbon_indices, avoidance, approach):
    mbon = outputs[mbon_indices]
    return ((mbon[avoidance].mean() - mbon[approach].mean() + 1) / 2).clamp(0, 1)
