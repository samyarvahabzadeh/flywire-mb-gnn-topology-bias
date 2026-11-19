"""Run and report the matched real-edge and input-set learning controls."""

import argparse
import subprocess
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl
import torch
from exploratory_plasticity_control import adam_step, edge_mask, evaluate, make_model
from exploratory_readability import ROOT, read_plan, real_projection, statistics

from mbtopology.real_cues import binary_cues, check_projected_disjoint, project_cues
from mbtopology.runtime import atomic_json, fingerprint, now, read_json, require, sha256

PLAN = ROOT / "configs/exploratory_matched_edge_rules.json"
METHODS = ("Adam", "Local")


def write_manifest(output, plan, metadata, paths):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    atomic_json(
        output / "MANIFEST.json",
        {
            "scope": plan["scope"],
            "git_commit": commit,
            "config_sha256": sha256(PLAN),
            "completed_utc": now(),
            **metadata,
            "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths},
        },
    )


def verified_stage(output):
    manifest = read_json(output / "MANIFEST.json")
    require(
        manifest and manifest["config_sha256"] == sha256(PLAN),
        f"missing current stage {output.name}",
    )
    for name, entry in manifest["files"].items():
        require(sha256(output / name) == entry["sha256"], f"stage file checksum differs: {name}")
    return manifest


def connections_source():
    paths = sorted((ROOT / "data/parquet/connections").glob("*.parquet"))
    require(bool(paths), "no existing connection source")
    return pl.scan_parquet(paths).select("pre_root_id", "post_root_id", "neuropil", "syn_count")


def audit_edge_source(raw, roots, recipients, pre, post):
    pairs = pl.DataFrame({"pre_root_id": roots[pre].astype(str), "post_root_id": recipients[post]})
    pieces = raw.filter(pl.col("post_root_id").is_in(recipients.tolist())).join(
        pairs.lazy(), on=["pre_root_id", "post_root_id"]
    )
    grouped = (
        pieces.group_by("pre_root_id", "post_root_id")
        .agg(
            rows=pl.len(),
            synapses=pl.col("syn_count").cast(pl.Int64).sum(),
            ped_synapses=pl.col("syn_count")
            .filter(pl.col("neuropil").is_in(["MB_PED_L", "MB_PED_R"]))
            .cast(pl.Int64)
            .sum(),
        )
        .collect()
    )
    summary = {
        "unique_pairs": grouped.height,
        "source_rows": grouped["rows"].sum(),
        "multirow_pairs": int((grouped["rows"] > 1).sum()),
        "all_synapses": grouped["synapses"].sum(),
        "pedunculus_synapses": grouped["ped_synapses"].sum(),
    }
    require(
        grouped.height == len(pre) == 1638
        and grouped["ped_synapses"].min() > 0
        and summary["all_synapses"] == 12020,
        "real pair or source annotation audit differs",
    )
    return summary


def roster(nodes, projection, kc, pre, post, old, plan):
    roots = nodes["root_id"].to_numpy()[kc]
    kinds = nodes["primary_type"].to_numpy()[kc]
    sides = nodes["side"].to_numpy()[kc]
    reachable = projection.getnnz(axis=0) > 0
    raw = connections_source()
    recipients = np.array(old["mbon_roots"])
    audit = audit_edge_source(raw, roots, recipients, pre, post)
    pool = raw.filter(pl.col("neuropil").is_in(["MB_PED_L", "MB_PED_R"]))
    eligible_fields = (
        pl.DataFrame({"pre_root_id": roots.astype(str), "side": sides, "type": kinds})
        .filter(pl.col("type").str.starts_with("KCg-") & pl.col("side").is_in(["left", "right"]))
        .with_columns(
            neuropil=pl.when(pl.col("side") == "left")
            .then(pl.lit("MB_PED_L"))
            .otherwise(pl.lit("MB_PED_R"))
        )
    )
    ids = frozenset(
        pool.join(eligible_fields.lazy(), on=["pre_root_id", "neuropil"])
        .select("pre_root_id")
        .unique()
        .collect()["pre_root_id"]
    )
    eligibility = np.array([str(i) in ids for i in roots])
    require(eligibility[pre].all(), "a real presynaptic KC falls outside the replacement pool")

    def strata(pres, posts):
        return Counter(
            (int(b), str(kinds[a]), bool(reachable[a])) for a, b in zip(pres, posts, strict=True)
        )

    keys = strata(pre, post)
    masks = {"real": (pre, post)}
    strata_rows = []
    for (b, kind, reached), count in sorted(keys.items()):
        eligible = np.flatnonzero(
            eligibility
            & (sides == old["mbon_side_order"][b])
            & (kinds == kind)
            & (reachable == reached)
        )
        require(len(eligible) >= count, "replacement pool cannot match the real capacity")
        strata_rows.append(
            {
                "recipient": str(recipients[b]),
                "kc_type": kind,
                "pn_reachable": reached,
                "selected": count,
                "eligible": len(eligible),
            }
        )
    for seed in plan["control_seeds"]:
        rng = np.random.default_rng(seed)
        new_pre, new_post = [], []
        for (b, kind, reached), count in sorted(keys.items()):
            candidates = np.flatnonzero(
                eligibility
                & (sides == old["mbon_side_order"][b])
                & (kinds == kind)
                & (reachable == reached)
            )
            new_pre.extend(np.sort(rng.choice(candidates, size=count, replace=False)).tolist())
            new_post.extend([b] * count)
        p, q = np.array(new_pre), np.array(new_post)
        require(
            len(p) == len(np.unique(p)) == len(pre) and strata(p, q) == keys,
            "control capacity differs",
        )
        masks[f"input-{seed}"] = (p, q)
    require(
        len({frozenset(zip(p.tolist(), q.tolist(), strict=True)) for p, q in masks.values()}) == 4,
        "duplicate input sets",
    )
    real_pairs = set(zip(pre.tolist(), post.tolist(), strict=True))
    catalog = {
        "source_audit": audit,
        "strata": strata_rows,
        "compartment_model": "ipsilateral MBON11 / PPL101",
        "masks": {},
    }
    for name, (p, q) in masks.items():
        catalog["masks"][name] = {
            "presynaptic_roots": roots[p].astype(str).tolist(),
            "postsynaptic_roots": recipients[q].tolist(),
            "total": len(p),
            "pn_reachable": int(reachable[p].sum()),
            "overlap_with_real": len(real_pairs & set(zip(p.tolist(), q.tolist(), strict=True))),
        }
    return masks, catalog


def cues(projection, config, plasticity, seed):
    patterns, _, _ = binary_cues(
        projection,
        seed,
        plasticity["heldout_blocks"],
        config["pairs_per_trial"],
        config["family_design"],
    )
    fixed = project_cues(projection, patterns[0][:1, :2])[0]
    fresh = project_cues(projection, patterns[1])[0]
    check_projected_disjoint(fixed, fresh)
    train, heldout = (torch.tensor(m.toarray(), dtype=torch.float64) for m in (fixed, fresh))
    hashes = {
        name: fingerprint({"shape": list(pattern.shape), "pairs": pattern.tolist()})
        for name, pattern in (("train", patterns[0][:1, :2]), ("heldout", patterns[1]))
    }
    return train, heldout, hashes


def run_matched(graph, seed, mask, dataset, p, plan):
    train, heldout, hashes = dataset
    models = {name: make_model(*mask, p) for name in METHODS}
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    held_labels = np.tile([1.0, 0.0], len(heldout) // 2)
    opt = torch.optim.Adam(models["Adam"].parameters(), lr=p["adam_learning_rate"])
    for model in models.values():
        require(
            len(model.weight) == 1638 and bool((model.weight == 1).all()),
            "invalid model parameter inventory",
        )
        evaluate(model, train)
        evaluate(model, heldout)
    training_times = dict.fromkeys(METHODS, 0.0)
    probe_times = dict.fromkeys(METHODS, 0.0)
    records, previous = [], 0
    for count in plan["checkpoints"]:
        for name, model in models.items():
            t = monotonic()
            for _ in range(previous, count):
                if name == "Adam":
                    adam_step(model, opt, train, target)
                else:
                    for x, us in zip(train, (1, 0), strict=True):
                        model.local_step(x, us, p["step"])
            training_times[name] += monotonic() - t
            w = model.weight
            require(
                bool((torch.isfinite(w) & (w >= p["bounds"][0]) & (w <= p["bounds"][1])).all()),
                "weights outside common finite bounds",
            )
        shared_threshold = float(np.stack([evaluate(m, train) for m in models.values()]).mean())
        if count == 0:
            require(shared_threshold == 0.25, "common starting reference changed")
        for name, model in models.items():
            t = monotonic()
            scores = evaluate(model, heldout)
            metric = statistics(held_labels, scores - shared_threshold)
            probe_times[name] += monotonic() - t
            require(np.isfinite(scores).all(), "nonfinite held-out predictions")
            records.append(
                {
                    "graph": graph,
                    "seed": seed,
                    "rule": name,
                    "pairs": count,
                    "train_presentations": count * 2,
                    "heldout_presentations_per_probe": len(scores),
                    "heldout_threshold": shared_threshold,
                    "heldout_balanced_accuracy": metric["balanced_accuracy"],
                    "heldout_auroc": metric["auroc"],
                    "heldout_score_gap": float(scores[::2].mean() - scores[1::2].mean()),
                    "cumulative_train_seconds": training_times[name],
                    "cumulative_heldout_probe_seconds": probe_times[name],
                    "train_cue_sha256": hashes["train"],
                    "heldout_cue_sha256": hashes["heldout"],
                    "weights_at_lower_bound": int((model.weight == p["bounds"][0]).sum()),
                    "weights_at_upper_bound": int((model.weight == p["bounds"][1]).sum()),
                }
            )
        previous = count
    return records


def run(stage, root, plan):
    started = monotonic()
    old, config = read_plan()
    plasticity = old["plasticity"]
    nodes, graph, kc, projection = real_projection(old["source"])
    pre, post, metadata = edge_mask(nodes, graph, kc, plasticity)
    masks, catalog = roster(nodes, projection, kc, pre, post, metadata, plan)
    output = root / stage
    if stage == "controls":
        manifest = verified_stage(root / "real")
        require(
            manifest["jobs"] == 10 and read_json(root / "real" / "input-masks.json") == catalog,
            "real stage or pre-frozen controls do not match",
        )
    output.mkdir(parents=True, exist_ok=False)
    selected = ["real"] if stage == "real" else list(masks)[1:]
    specs = {seed: cues(projection, config, plasticity, seed) for seed in plan["seeds"]}
    rows = [
        row
        for graph_id in selected
        for seed, dataset in specs.items()
        for row in run_matched(graph_id, seed, masks[graph_id], dataset, plasticity, plan)
    ]
    path, masks_path = output / "checkpoints.parquet", output / "input-masks.json"
    pl.DataFrame(rows).write_parquet(path)
    atomic_json(masks_path, catalog)
    jobs = len(rows) // len(plan["checkpoints"])
    write_manifest(
        output,
        plan,
        {
            "jobs": jobs,
            "stage": stage,
            "wall_seconds": monotonic() - started,
            "source_graph_manifest_sha256": sha256(ROOT / old["source"]),
            "prior_design_sha256": sha256(
                ROOT / "configs/exploratory_readability_and_plasticity.json"
            ),
        },
        (path, masks_path),
    )
    print(f"MATCHED_EDGES {stage} {jobs} jobs {monotonic() - started:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("real", "controls", "report"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(PLAN)
    torch.set_num_threads(2)
    if args.stage == "report":
        from exploratory_matched_report import report

        report(args.output, plan)
    else:
        run(args.stage, args.output, plan)


if __name__ == "__main__":
    main()
