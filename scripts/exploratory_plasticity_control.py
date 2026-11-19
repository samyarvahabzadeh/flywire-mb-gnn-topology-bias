"""A real-edge fixed-cue plasticity control with an unreinforced family probe gated on PASS."""

import argparse
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from exploratory_readability import ROOT, artifact_manifest, read_plan, real_projection, statistics
from torch import nn
from torch.nn import functional as F

from mbtopology.real_cues import binary_cues, check_projected_disjoint, project_cues
from mbtopology.runtime import atomic_json, read_json, require


class PlasticEdges(nn.Module):
    def __init__(self, pre, post, outputs, initial, bounds):
        super().__init__()
        self.register_buffer("pre", torch.as_tensor(pre, dtype=torch.long))
        self.register_buffer("post", torch.as_tensor(post, dtype=torch.long))
        self.weight = nn.Parameter(torch.full((len(pre),), initial, dtype=torch.float64))
        self.outputs, self.bounds = outputs, bounds

    def forward(self, activity):
        drive = activity[:, self.pre]
        blank = drive.new_zeros((len(drive), self.outputs))
        signal = blank.index_add(1, self.post, drive * self.weight)
        total = blank.index_add(1, self.post, drive)
        require(bool((total > 0).all()), "cue lacks presynaptic drive in a selected MBON11")
        return 0.25 + 0.5 * (1 - (signal / total).mean(1))

    @torch.no_grad()
    def local_step(self, activity, reinforced, step):
        self.weight.add_(activity[self.pre], alpha=-step * reinforced)
        self.weight.clamp_(*self.bounds)


def adam_step(model, optimizer, activity, labels):
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy(model(activity), labels)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.weight.clamp_(*model.bounds)


def edge_mask(nodes, graph, kc, plan):
    roles, kinds, nt, sides = (
        nodes[name].to_numpy() for name in ("role", "primary_type", "nt_type", "side")
    )
    ids = [str(value) for value in nodes["root_id"].to_list()]
    populations = {}
    for label, role, nt_type in (("mbon", "MBON", "GABA"), ("dan", "DAN", "DA")):
        indices = np.array([ids.index(root) for root in plan[f"{label}_ids"]])
        require(
            all(
                (roles[indices] == role)
                & (kinds[indices] == plan[f"{label}_type"])
                & (nt[indices] == nt_type)
            ),
            f"{label} biological identity differs",
        )
        require(set(sides[indices]) == {"left", "right"}, f"{label} lacks bilateral identity")
        populations[label] = indices
    mbon, dan = populations["mbon"], populations["dan"]
    code = read_json(ROOT / "configs/mbon_readout.json")
    require(
        all(code["mbon_root_ids"].index(ids[i]) in code["approach_indices"] for i in mbon),
        "published MBON11 direction disagrees with the sign used by the old code",
    )
    edges = graph[kc][:, mbon].tocoo()
    gamma = np.array(
        [isinstance(t, str) and t.startswith(plan["kc_type_prefix"]) for t in kinds[kc]]
    )
    keep = gamma[edges.row] & (sides[kc[edges.row]] == sides[mbon[edges.col]])
    pre, post, counts = edges.row[keep], edges.col[keep], edges.data[keep]
    require(
        set(post) == {0, 1} and len(set(zip(pre, post, strict=True))) == len(pre),
        "invalid plastic edge support",
    )
    annotations = {
        "mbon_roots": [ids[i] for i in mbon],
        "dan_roots": [ids[i] for i in dan],
        "mbon_side_order": sides[mbon].tolist(),
        "edges_per_mbon": np.bincount(post).tolist(),
        "pairs": len(pre),
        "synapses_on_selected_pairs": int(counts.sum()),
    }
    return pre, post, annotations


@torch.no_grad()
def evaluate(model, x):
    before = model.weight.detach().clone()
    scores = model(x).numpy()
    require(torch.equal(model.weight, before), "probe changed an edge weight")
    return scores


def run_control(model, x, reinforcement, plan):
    initial = model.weight.detach().clone()
    baseline = evaluate(model, x)
    for _ in range(plan["fixed_pair_presentations"]):
        for cue, reward in zip(x, reinforcement, strict=True):
            model.local_step(cue, reward, plan["step"])
    final = evaluate(model, x)
    delta = initial - model.weight.detach()
    x_edges = x[:, model.pre] > 0
    reinforced = (x_edges & torch.tensor(reinforcement, dtype=torch.bool)[:, None]).any(0)
    actual = delta != 0
    require(bool((delta >= 0).all()), "unexpected increase in a depression-only rule")
    partition = {
        "A_plus_only": x_edges[0] & ~x_edges[1],
        "A_minus_only": x_edges[1] & ~x_edges[0],
        "both": x_edges[0] & x_edges[1],
        "neither": ~x_edges.any(0),
    }
    return {
        "baseline_score": baseline.tolist(),
        "post_score": final.tolist(),
        "signed_A_plus_minus_A_minus": float(final[0] - final[1]),
        "changed_edges": int(actual.sum()),
        "expected_changed_edges": int(reinforced.sum()),
        "selective_exact_match": torch.equal(actual, reinforced),
        "max_depression": float(delta.max()),
        "edge_partition": {
            name: {"edges": int(mask.sum()), "changed_edges": int(actual[mask].sum())}
            for name, mask in partition.items()
        },
    }


def make_model(pre, post, plan):
    return PlasticEdges(pre, post, 2, plan["initial_weight"], plan["bounds"])


def verify_adam_inventory(plan):
    local = make_model([0, 1, 2], [0, 0, 1], plan)
    gradient = make_model([0, 1, 2], [0, 0, 1], plan)
    require(
        list(dict(local.named_parameters()))
        == list(dict(gradient.named_parameters()))
        == ["weight"],
        "learnable inventories differ",
    )
    x = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float64)
    before = gradient.weight.detach().clone()
    adam_step(
        gradient,
        torch.optim.Adam(gradient.parameters(), lr=plan["adam_learning_rate"]),
        x,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    require(bool((gradient.weight != before).any()), "toy Adam update had no effect")
    require(
        torch.equal(local.pre, gradient.pre) and torch.equal(local.post, gradient.post),
        "optimizer changed edge mask",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = monotonic()
    torch.set_num_threads(2)
    plan, cfg = read_plan()
    p = plan["plasticity"]
    nodes, csr, kc, projection = real_projection(plan["source"])
    pre, post, data = edge_mask(nodes, csr, kc, p)
    verify_adam_inventory(p)
    cue_pairs, _, _ = binary_cues(
        projection, p["seed"], p["heldout_blocks"], cfg["pairs_per_trial"], cfg["family_design"]
    )
    fixed = project_cues(projection, cue_pairs[0][:1, :2])[0]
    x = torch.tensor(fixed.toarray(), dtype=torch.float64)
    controls, states = {}, {}
    fit_start = monotonic()
    for name, rewards in (("paired", (1, 0)), ("no_reinforcement", (0, 0)), ("swapped", (0, 1))):
        model = make_model(pre, post, p)
        require(
            torch.equal(model.pre, torch.as_tensor(pre, dtype=torch.long))
            and list(dict(model.named_parameters())) == ["weight"],
            "real learnable inventory differs from the shared edge model",
        )
        controls[name] = run_control(model, x, rewards, p)
        states[name] = model
    fit_time = monotonic() - fit_start
    paired, no_us, swapped = (controls[key] for key in ("paired", "no_reinforcement", "swapped"))
    threshold = p["minimum_fixed_avoidance_gap"]
    checks = {
        "paired_gap": paired["signed_A_plus_minus_A_minus"] >= threshold,
        "swapped_gap": -swapped["signed_A_plus_minus_A_minus"] >= threshold,
        "no_reinforcement_invariant": no_us["baseline_score"] == no_us["post_score"]
        and no_us["changed_edges"] == 0,
        "selective_edges": all(c["selective_exact_match"] for c in controls.values()),
    }
    passed, generalization, heldout_time = all(checks.values()), {}, None
    if passed:
        eval_start = monotonic()
        unseen = project_cues(projection, cue_pairs[1])[0]
        check_projected_disjoint(fixed, unseen)
        evaluation = torch.tensor(unseen.toarray(), dtype=torch.float64)
        targets = np.tile([1.0, 0.0], len(evaluation) // 2)
        for name, tags in (("paired", targets), ("swapped", 1 - targets)):
            scores = evaluate(states[name], evaluation)
            midpoint = float(np.mean(controls[name]["post_score"]))
            metric = statistics(tags, scores - midpoint)
            generalization[name] = {key: metric[key] for key in ("balanced_accuracy", "auroc")}
            generalization[name].update(
                {
                    "midpoint_from_fixed_training_pair": midpoint,
                    "reinforced_family_avoidance": float(scores[tags == 1].mean()),
                    "unreinforced_family_avoidance": float(scores[tags == 0].mean()),
                    "raw_gap": float(scores[tags == 1].mean() - scores[tags == 0].mean()),
                    "cues": len(scores),
                }
            )
        heldout_time = monotonic() - eval_start
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report, weights = output / "plasticity.json", output / "weights.npz"
    atomic_json(
        report,
        {
            "seed": p["seed"],
            "status": "PASS" if passed else "FAIL",
            "edge_support": data,
            "checks": checks,
            "fixed": controls,
            "fresh_family_probe": generalization,
            "control_fit_seconds": fit_time,
            "heldout_seconds": heldout_time,
            "adam_inventory_toy_check": "PASS",
        },
    )
    np.savez_compressed(
        weights,
        pre_root=nodes["root_id"].to_numpy()[kc[pre]],
        post_root=np.array(p["mbon_ids"], dtype=np.uint64)[post],
        **{name: model.weight.detach().numpy() for name, model in states.items()},
    )
    artifact_manifest(output, plan, started, (report, weights))
    print(
        f"PLASTICITY {'PASS' if passed else 'FAIL'} {len(pre)} edges {monotonic() - started:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
