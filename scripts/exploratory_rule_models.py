"""Readout learning arms for the separate exploratory screen."""

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from mbtopology.runtime import require


def adam(parameters, config):
    return torch.optim.Adam(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


def pool_valence(scores, inputs):
    avoid = scores[:, inputs["avoidance"]].mean(1)
    approach = scores[:, inputs["approach"]].mean(1)
    return ((avoid - approach + 1) / 2).clamp(0, 1)


def probe_row(probe, valid=None):
    probe = probe.double()
    finite = torch.isfinite(probe).all()
    if valid is not None:
        finite = finite & valid
    return torch.stack((probe[::2].mean(), probe[1::2].mean(), finite.double()))


@torch.no_grad()
def mb_embeddings(model, inputs, drive, trial, presentations):
    first, last = trial * presentations, (trial + 1) * presentations
    pointers = drive.indptr[first : last + 1]
    low, high = pointers[0], pointers[-1]
    device = inputs["base"].device
    row = torch.from_numpy(np.repeat(np.arange(presentations), np.diff(pointers))).to(device)
    col = torch.from_numpy(drive.indices[low:high]).to(device)
    activity = torch.from_numpy(drive.data[low:high]).to(device)
    x = inputs["base"].expand(presentations, -1, -1).clone()
    x[row, inputs["kc"][col], 4] = activity
    x = torch.relu(model.encoder(x))
    for conv in model.convs:
        x = torch.relu(conv(x, inputs["neighbors"]))
    return x[:, inputs["mbon"]]


class FrozenReadouts:
    def __init__(self, model, inputs, optimizer_config, local):
        self.body = deepcopy(model).eval()
        self.body.requires_grad_(False)
        self.head_b = self.body.head
        self.head_b.requires_grad_(True)
        self.optimizer_b = adam(self.head_b.parameters(), optimizer_config)
        self.initial_head = deepcopy(self.head_b).requires_grad_(False)
        self.delta = torch.zeros_like(self.initial_head.weight[0])
        self.inputs = inputs
        self.local = local

    def predict_b(self, features):
        return pool_valence(torch.sigmoid(self.head_b(features)).squeeze(-1), self.inputs)

    @torch.no_grad()
    def predict_c(self, features):
        logits = self.initial_head(features).squeeze(-1)
        approach = self.inputs["approach"]
        logits[:, approach] += features[:, approach] @ self.delta
        return pool_valence(torch.sigmoid(logits), self.inputs)

    def train_b(self, train_features, labels):
        self.optimizer_b.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy(self.predict_b(train_features), labels)
        loss.backward()
        valid = torch.isfinite(loss)
        for parameter in self.head_b.parameters():
            valid = valid & torch.isfinite(parameter.grad).all()
        self.optimizer_b.step()
        return valid

    @torch.no_grad()
    def train_c(self, train_features, reinforcement):
        approach = train_features[:, self.inputs["approach"]]
        eligibility = (approach * reinforcement.view(-1, 1, 1)).mean((0, 1))
        self.delta.add_(eligibility, alpha=-self.local["step_size"])
        self.delta.clamp_(min=-self.local["maximum_depression"], max=0)
        return torch.isfinite(self.delta).all()

    def export(self):
        return {
            "head_b": self.head_b.state_dict(),
            "optimizer_b": self.optimizer_b.state_dict(),
            "delta_c": self.delta,
        }

    def restore(self, state):
        self.head_b.load_state_dict(state["head_b"])
        self.optimizer_b.load_state_dict(state["optimizer_b"])
        self.delta.copy_(state["delta_c"])


def same_structure(model, reference):
    require(set(model.state_dict()) == set(reference), "model parameter inventory changed")
    drift = {"backbone": 0.0, "head": 0.0}
    for key, value in model.state_dict().items():
        name = "head" if key.startswith("head.") else "backbone"
        drift[name] += float((value.detach().cpu() - reference[key]).double().square().sum())
    return {name: squared**0.5 for name, squared in drift.items()}
