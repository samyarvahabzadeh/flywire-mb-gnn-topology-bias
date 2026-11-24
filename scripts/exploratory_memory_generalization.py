"""Test whether wiring and plasticity interact in aversive memory generalization."""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from exploratory_matched_edges import roster, write_manifest
from exploratory_plasticity_control import adam_step, edge_mask, evaluate, make_model
from exploratory_readability import ROOT, read_plan, real_projection

from mbtopology.real_cues import check_projected_disjoint, project_cues
from mbtopology.runtime import atomic_file, atomic_json, fingerprint, read_json, require

PLAN = ROOT / "configs/exploratory_memory_generalization.json"


def make_cues(projection, seed, plan):
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(projection.getnnz(axis=1))
    size = plan["cue_size"]
    order = rng.permutation(active)
    plus, minus = order[:size], order[size : 2 * size]
    require(plan["train_shared_pns"] == 0, "expected disjoint training odors")
    pool = order[size:]
    seen = {tuple(sorted(plus)), tuple(sorted(minus))}
    probes, levels = [], []
    for shared in plan["shared_pns"]:
        for _ in range(plan["probes_per_level"]):
            while True:
                p = tuple(
                    sorted(
                        np.r_[
                            rng.choice(plus, shared, replace=False),
                            rng.choice(pool, size - shared, replace=False),
                        ]
                    )
                )
                if p not in seen:
                    break
            probes.append(p)
            levels.append(shared)
            seen.add(p)
    patterns = np.asarray([sorted(plus), sorted(minus), *probes])
    activity, _, _ = project_cues(projection, patterns)
    check_projected_disjoint(activity[:2], activity[2:])
    return activity, levels, fingerprint(patterns.tolist())


def trajectory(model, rule, train, probes, plasticity, plan):
    base = evaluate(model, train)
    pre = evaluate(model, probes)
    opt = torch.optim.Adam(model.parameters(), lr=plasticity["adam_learning_rate"])
    labels = torch.tensor([1.0, 0.0], dtype=torch.float64)
    level = plan["match_cs_plus_change"]
    prev_gain = 0.0
    previous = model.weight.detach().clone()
    matched, pair = None, None
    for count in range(1, plan["training_pairs"] + 1):
        if rule == "Adam":
            adam_step(model, opt, train, labels)
        else:
            model.local_step(train[0], 1, plasticity["step"])
        gain = float(evaluate(model, train)[0] - base[0])
        if matched is None and gain >= level:
            with torch.no_grad():
                current = model.weight.detach().clone()
                alpha = (level - prev_gain) / (gain - prev_gain)
                model.weight.copy_(torch.lerp(previous, current, alpha))
                require(
                    abs(float(evaluate(model, train)[0] - base[0]) - level) < 1e-10,
                    "CS+ interpolation missed the match",
                )
                matched = (evaluate(model, probes) - pre) / level
                model.weight.copy_(current)
            pair = count
        if matched is None:
            previous = model.weight.detach().clone()
            prev_gain = gain
    fixed = evaluate(model, probes) - pre
    return matched, fixed, pair, gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(PLAN)
    old, _ = read_plan()
    plasticity = old["plasticity"]
    nodes, graph, kc, projection = real_projection(old["source"])
    pre, post, identity = edge_mask(nodes, graph, kc, plasticity)
    controls = list(
        range(plan["control_seed_start"], plan["control_seed_start"] + plan["n_controls"])
    )
    masks, catalog = roster(nodes, projection, kc, pre, post, identity, {"control_seeds": controls})
    records, cue_hashes = [], {}
    for seed in range(plan["seed_start"], plan["seed_start"] + plan["n_seeds"]):
        activity, levels, cue_hash = make_cues(projection, seed, plan)
        cue_hashes[str(seed)] = cue_hash
        for name, (pres, posts) in masks.items():
            # This is a cue-by-feature array, with only the selected KC columns populated.
            local_columns = np.unique(pres)
            dense = torch.from_numpy(activity[:, local_columns].toarray()).double()
            column_indices = np.searchsorted(local_columns, pres)
            train, probes = dense[:2], dense[2:]
            model = make_model(column_indices, posts, plasticity)
            base = evaluate(model, probes)
            model.local_step(train[0], 0, plasticity["step"])
            require(
                np.array_equal(base, evaluate(model, probes)), "unreinforced update changed scores"
            )

            # Exact one-pair prediction for the unsaturated local rule.
            drive = probes[:, column_indices].numpy()
            conditioned = train[0, column_indices].numpy()
            expected = (
                np.mean(
                    [
                        (drive[:, posts == j] @ conditioned[posts == j])
                        / drive[:, posts == j].sum(1)
                        for j in np.unique(posts)
                    ],
                    axis=0,
                )
                * plasticity["step"]
                / 2
            )
            model.local_step(train[0], 1, plasticity["step"])
            require(
                np.allclose(evaluate(model, probes) - base, expected, rtol=0, atol=1e-12),
                "local update and synaptic overlap prediction disagree",
            )

            for rule in ("Local", "Adam"):
                matched, fixed, pair, gain = trajectory(
                    make_model(column_indices, posts, plasticity),
                    rule,
                    train,
                    probes,
                    plasticity,
                    plan,
                )
                records.extend(
                    {
                        "seed": seed,
                        "mask": name,
                        "rule": rule,
                        "probe": i,
                        "shared_PNs": shared,
                        "matched_transfer": None if matched is None else float(matched[i]),
                        "fixed_budget_change": float(fixed[i]),
                        "matched_by_pair": pair,
                        "final_cs_plus_change": gain,
                        "local_one_pair_prediction": float(expected[i]),
                    }
                    for i, shared in enumerate(levels)
                )
        print("MEMORY_SEED_COMPLETE", seed, flush=True)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    probes = out / "probes.parquet"
    atomic_file(probes, pl.DataFrame(records).write_parquet)
    masks_path = out / "input-masks.json"
    atomic_json(masks_path, catalog)
    write_manifest(
        out,
        plan,
        {"jobs": len(masks) * plan["n_seeds"] * 2, "runner_cue_sha256": cue_hashes},
        [probes, masks_path],
        plan_path=PLAN,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
