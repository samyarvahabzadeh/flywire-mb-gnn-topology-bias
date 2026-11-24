"""One frozen-body Adam/BCE check using the existing seed-4301 cue splits."""

import argparse
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from exploratory_matched_edges import PLAN, write_manifest
from exploratory_readability import ROOT, SPLITS, statistics
from torch import nn
from torch.nn import functional as F

from mbtopology.runtime import atomic_json, read_json, require, sha256


class SharedHead(nn.Module):
    def __init__(self, width, membership):
        super().__init__()
        self.head = nn.Linear(width, 1)
        self.avoidance = membership["avoidance_indices"]
        self.approach = membership["approach_indices"]

    def forward(self, x):
        probabilities = torch.sigmoid(self.head(x).squeeze(-1))
        v = (
            1 + probabilities[:, self.avoidance].mean(1) - probabilities[:, self.approach].mean(1)
        ) / 2
        return torch.logit(v, eps=1e-7)


@torch.no_grad()
def score(model, sample):
    features, labels = sample
    return statistics(labels.numpy(), model(features).numpy())


def fit(model, samples, config):
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    x, y = samples["train"]
    history = []
    best = None
    started = monotonic()
    for epoch in range(max(config["epochs"]) + 1):
        if epoch:
            for start in range(0, len(x), config["batch_size"]):
                block = slice(start, start + config["batch_size"])
                opt.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(model(x[block]), y[block]).backward()
                opt.step()
        if epoch in config["epochs"]:
            scores = {key: score(model, samples[key]) for key in ("train", "validation")}
            history.append({"epoch": epoch, **scores})
            loss = scores["validation"]["log_loss"]
            if best is None or loss < best[0]:
                best = (loss, epoch, deepcopy(model.state_dict()))
    seconds = monotonic() - started
    model.load_state_dict(best[2])
    evaluation_start = monotonic()
    final = {key: score(model, samples[key]) for key in SPLITS}
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "selected_epoch": best[1],
        "training_seconds": seconds,
        "final_evaluation_seconds": monotonic() - evaluation_start,
        "train_presentations_actual": len(x) * max(config["epochs"]),
        "train_presentations_at_selection": len(x) * best[1],
        "history": history,
        "selected": final,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = monotonic()
    torch.set_num_threads(2)
    plan = read_json(PLAN)
    cfg = plan["gnn"]
    root = ROOT / "results/exploratory/readability_positive_control_v1/B"
    source = root / "representations.npz"
    manifest = read_json(root / "MANIFEST.json")
    require(
        sha256(source) == cfg["representations_sha256"] == manifest["files"][source.name]["sha256"],
        "seed-4301 archive differs",
    )
    prior_path = root / "readability.json"
    require(
        sha256(prior_path) == manifest["files"][prior_path.name]["sha256"],
        "seed-4301 metadata differs",
    )
    prior = read_json(prior_path)
    require(
        prior["seed"] == cfg["seed"] == 4301 and prior["paired_exact_disjoint"],
        "invalid seed-4301 split provenance",
    )
    split = {}
    with np.load(source) as archived:
        for key in SPLITS:
            h = archived[f"{key}_mbon_full_prehead"]
            y = archived[f"{key}_label"]
            require(
                h.shape == (prior["sample_counts"][key], 96 * 128)
                and y.shape == (len(h),)
                and np.isfinite(h).all()
                and np.array_equal(y, np.tile([1, 0], len(h) // 2)),
                "invalid raw frozen MBON features",
            )
            split[key] = (
                torch.from_numpy(h.reshape(-1, 96, 128)),
                torch.tensor(y, dtype=torch.float32),
            )
    membership = read_json(ROOT / "configs/mbon_readout.json")
    shared = SharedHead(128, membership)
    independent = nn.Sequential(nn.Flatten(1), nn.Linear(96 * 128, 1), nn.Flatten(0))
    models = {"shared_fixed_pooling": shared, "independent_mbon": independent}
    result = {label: fit(model, split, cfg) for label, model in models.items()}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    md, json_path = output / "REPORT.md", output / "readout.json"
    atomic_json(
        json_path, {"seed": cfg["seed"], "sample_counts": prior["sample_counts"], "results": result}
    )
    lines = [
        "# Exploratory seed-4301 GNN readout check",
        "",
        (
            "Both heads start at zero and use the same raw frozen pre-head MBON tensors and ordered splits. "
            "Validation BCE selects the epoch. Held-out is scored at the selected epoch at the common probability cut 0.5."
        ),
        "",
        "| Head | Parameters | Epoch | Train balanced accuracy | Validation balanced accuracy | Held-out balanced accuracy | Held-out AUROC | Fit s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in result.items():
        s = r["selected"]
        lines.append(
            f"| {name} | {r['parameters']} | {r['selected_epoch']} | {s['train']['balanced_accuracy']:.4f} | "
            f"{s['validation']['balanced_accuracy']:.4f} | {s['heldout']['balanced_accuracy']:.4f} | "
            f"{s['heldout']['auroc']:.4f} | {r['training_seconds']:.3f} |"
        )
    lines += [
        "",
        (
            "Each head processed 8,192 training presentations across 16 passes through the 512 fixed training samples. "
            "Independent coefficients, parameter count and placement of the output nonlinearity all change. "
            "The check uses one graph and seed."
        ),
        "",
        "[Train and validation checkpoints](readout.json) · [Manifest](MANIFEST.json)",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_manifest(
        output,
        plan,
        {
            "seed": cfg["seed"],
            "features_sha256": sha256(source),
            "readout_membership_sha256": sha256(ROOT / "configs/mbon_readout.json"),
            "wall_seconds": monotonic() - started,
        },
        (md, json_path),
    )


if __name__ == "__main__":
    main()
