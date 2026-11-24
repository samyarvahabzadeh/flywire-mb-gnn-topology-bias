"""Match two-dimensional decoders across real and degree-preserving graphs."""

import argparse
import os
import subprocess
from itertools import combinations
from pathlib import Path
from time import monotonic

import numpy as np
import polars as pl
import torch
from exploratory_gnn_readout import fit, score
from exploratory_readability import ROOT, SPLITS, read_plan, real_projection
from exploratory_rule_models import mb_embeddings
from torch import nn

from mbtopology import gnn_training, models, runner
from mbtopology.benchmark_models import make_benchmark_model
from mbtopology.cue_families import check_disjoint
from mbtopology.real_cues import binary_cues, check_projected_disjoint, project_cues
from mbtopology.robustness_pilot import perturb_cue
from mbtopology.runtime import atomic_json, now, read_json, require, sha256
from mbtopology.task_inputs import load_source

PLAN = ROOT / "configs/exploratory_decoder_topology.json"
ARMS = ("group", "pca")


class Progress:
    def save(self, **fields):
        print(fields, flush=True)


def graphs(bundle, plan):
    real = bundle["task"]["source_manifest"]
    _, _, original = load_source(ROOT, real)
    paths = {"real": real}
    roster = plan.get("null_instances_by_variant")
    for variant in "AB":
        indexes = roster[variant] if roster is not None else plan["null_instances_per_variant"]
        for index in indexes:
            label = f"{variant}-{index:02d}"
            which = (
                f"data/smoke_nulls/4bfaf9e768c6/{label}/MANIFEST.json"
                if index == 0
                else runner.materialize_null(
                    ROOT,
                    bundle,
                    variant,
                    index,
                    Progress(),
                    None,
                    scope=plan.get("graph_cache_scope", "decoder_topology_nulls_v1"),
                )
            )
            info, _, matrix = load_source(ROOT, which)
            require(
                info["source_manifest"] == real
                and info["variant"] == variant
                and info["instance"] == index,
                f"null provenance differs for {label}",
            )
            cert = info["report"]
            require(
                cert["accepted_swaps"] == 10 * original.nnz
                and cert["exact_nodewise_in_out_degree"]
                and (variant != "B" or cert["exact_signed_degree"]),
                f"invalid null certificate: {label}",
            )
            require(
                all(np.array_equal(matrix.getnnz(axis=i), original.getnnz(axis=i)) for i in (0, 1)),
                f"degree mismatch for {label}",
            )
            paths[label] = which
            print(f"DECODER_GRAPH {label}", flush=True)
    return paths


def drives_for(projection, cfg, plan, seed):
    blocks, family, pairs = plan["blocks"], cfg["family_design"], cfg["pairs_per_trial"]
    main, _, _ = binary_cues(
        projection, seed, max(blocks["train"], blocks["heldout"]), pairs, family
    )
    extra, _, _ = binary_cues(
        projection,
        seed,
        blocks["validation"],
        pairs,
        {**family, "heldout_stream": plan["validation_stream"]},
    )
    patterns = {
        "train": main[0][: blocks["train"]],
        "validation": extra[1],
        "heldout": main[1][: blocks["heldout"]],
    }
    drives = {key: project_cues(projection, cues)[0] for key, cues in patterns.items()}
    for a, b in combinations(SPLITS, 2):
        check_disjoint(patterns[a], patterns[b], "PN")
        check_projected_disjoint(drives[a], drives[b])
    connected = np.flatnonzero(projection.getnnz(axis=1))
    base = patterns["heldout"]
    compact = np.searchsorted(connected, base).reshape(-1, base.shape[-1])
    for noise in plan["noise_levels"][1:]:
        changed = np.stack(
            [
                perturb_cue(
                    row,
                    len(connected),
                    noise,
                    seed=seed,
                    cue=plan["noise_stream"] * len(compact) + i,
                )
                for i, row in enumerate(compact)
            ]
        )
        expected = round(base.shape[-1] * noise)
        require(
            all(
                len(a) - np.intersect1d(a, b).size == expected
                for a, b in zip(compact, changed, strict=True)
            ),
            "invalid PN corruption dose",
        )
        actual = connected[changed].reshape(base.shape)
        projected = project_cues(projection, actual)[0]
        for key in ("train", "validation"):
            check_disjoint(patterns[key], actual, "PN")
            check_projected_disjoint(drives[key], projected)
        drives[f"noise-{noise}"] = projected
    return drives


def gather_layers(neighbors, mbon, n, *, final_neighbors=None):
    final = neighbors if final_neighbors is None else final_neighbors
    src, dst = final
    ending = torch.isin(dst, mbon)
    first_nodes = torch.cat((mbon, src[ending])).unique(sorted=True)
    first_src, first_dst = neighbors
    first_edges = torch.isin(first_dst, first_nodes)
    device = neighbors.device
    local = torch.full((n,), -1, dtype=torch.long, device=device)
    local[first_nodes] = torch.arange(len(first_nodes), device=device)
    last = torch.full((n,), -1, dtype=torch.long, device=device)
    last[mbon] = torch.arange(len(mbon), device=device)
    e0 = torch.stack((first_src[first_edges], local[first_dst[first_edges]]))
    e1 = torch.stack((local[src[ending]], last[dst[ending]]))
    return first_nodes, local[mbon], e0, e1


def graph_inputs(bundle, path, device="cpu"):
    inputs = gnn_training.frozen_inputs(
        ROOT, bundle["task"], path, 0, bundle["nulls"], bundle["models"], device
    )
    inputs["base"] = models.node_features(inputs["roles"], inputs["kc"], inputs["kc"][:0])
    edges = inputs["graph"].edge_index
    inputs["neighbors"] = edges[:, edges[0] != edges[1]]
    inputs["layers"] = gather_layers(inputs["neighbors"], inputs["mbon"], len(inputs["roles"]))
    return inputs


def reduced_embeddings(model, inputs, drive, trial, presentations):
    first, last = trial * presentations, (trial + 1) * presentations
    pointers = drive.indptr[first : last + 1]
    low, high = pointers[0], pointers[-1]
    device = inputs["base"].device
    row = torch.from_numpy(np.repeat(np.arange(presentations), np.diff(pointers))).to(device)
    col = torch.from_numpy(drive.indices[low:high]).to(device)
    x = inputs["base"].expand(presentations, -1, -1).clone()
    x[row, inputs["kc"][col], 4] = torch.from_numpy(drive.data[low:high]).to(device)
    initial = torch.relu(model.encoder(x))
    first_nodes, last_nodes, left_edge, right_edge = inputs["layers"]
    hidden = torch.relu(model.convs[0]((initial, initial[:, first_nodes]), left_edge))
    return torch.relu(model.convs[1]((hidden, hidden[:, last_nodes]), right_edge))


@torch.inference_mode()
def extract(model, inputs, drives, presentations, check):
    if check:
        basis = drives["train"]
        torch.testing.assert_close(
            reduced_embeddings(model, inputs, basis, 0, presentations),
            mb_embeddings(model, inputs, basis, 0, presentations),
            atol=1e-6,
            rtol=1e-6,
        )
    results = {}
    for key, drive in drives.items():
        scores = []
        for t in range(drive.shape[0] // presentations):
            h = reduced_embeddings(model, inputs, drive, t, presentations)
            scores.append(torch.sigmoid(model.head(h)).squeeze(-1).cpu().numpy())
        results[key] = np.concatenate(scores).astype(np.float64)
    return results


def feature_formats(scores, members):
    a, b = (np.array(members[key]) for key in ("avoidance_indices", "approach_indices"))
    cells = np.r_[a, b]
    require(len(np.unique(cells)) == len(cells) == 91, "MBON membership differs")
    stacked = np.concatenate([rows["train"][:, cells] for rows in scores.values()])
    centre = stacked.mean(axis=0)
    _, singular, basis = np.linalg.svd(stacked - centre, full_matrices=False)
    require(singular[1] > 0, "training PCA has fewer than two axes")
    features = {graph: {"group": {}, "pca": {}} for graph in scores}
    for graph, panels in scores.items():
        for split, x in panels.items():
            features[graph]["group"][split] = np.column_stack((x[:, a].mean(1), x[:, b].mean(1)))
            features[graph]["pca"][split] = (x[:, cells] - centre) @ basis[:2].T
    for arm in ARMS:
        pooled = np.concatenate([panels[arm]["train"] for panels in features.values()])
        mu, sigma = pooled.mean(0), pooled.std(0)
        require(np.all(sigma > 1e-8), "degenerate common training feature scale")
        for formats in features.values():
            for split, x in formats[arm].items():
                z = torch.from_numpy(((x - mu) / sigma).astype(np.float32))
                y = torch.tensor([1.0, 0.0] * (len(z) // 2))
                formats[arm][split] = (z, y)
    return features


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    started = monotonic()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(args.device == "cpu" or torch.cuda.is_available(), "requested CUDA GPU unavailable")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    plan, (old, cfg) = read_json(PLAN), read_plan()
    bundle = runner.load_bundle(ROOT, ROOT / "configs/experiment.yaml")
    paths = graphs(bundle, plan)
    models_in = {g: graph_inputs(bundle, p, args.device) for g, p in paths.items()}
    _, _, _, projection = real_projection(old["source"])
    members = read_json(ROOT / "configs/mbon_readout.json")
    args.output.mkdir(parents=True, exist_ok=False)
    records, files = [], []
    for index, seed in enumerate(plan["seeds"]):
        clock = monotonic()
        drives = drives_for(projection, cfg, plan, seed)
        backbone = make_benchmark_model(
            "GraphSAGE", bundle["models"], seed, args.device, aggregation="max"
        ).eval()
        native = {
            g: extract(backbone, inputs, drives, 2 * cfg["pairs_per_trial"], index == 0)
            for g, inputs in models_in.items()
        }
        features = feature_formats(native, members)
        rows = []
        for graph, formats in features.items():
            for arm, data in formats.items():
                decoder = nn.Sequential(nn.Linear(2, 1), nn.Flatten(0))
                learned = fit(decoder, data, plan)
                for noise in plan["noise_levels"]:
                    stats = (
                        learned["selected"]["heldout"]
                        if noise == 0
                        else score(decoder, data[f"noise-{noise}"])
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "graph": graph,
                            "arm": arm,
                            "noise": noise,
                            "features": 2,
                            "parameters": learned["parameters"],
                            "selected_epoch": learned["selected_epoch"],
                            **stats,
                        }
                    )
        records.extend(rows)
        path = args.output / f"seed-{seed}.json"
        atomic_json(
            path,
            {
                "seed": seed,
                "samples": {k: x.shape[0] for k, x in drives.items()},
                "seconds": monotonic() - clock,
                "rows": rows,
            },
        )
        files.append(path)
        print(f"DECODER_SEED {seed} 10 fits {monotonic() - clock:.1f}s", flush=True)
    table = pl.DataFrame(records)
    require(table.height == 120 and set(table["parameters"]) == {3}, "decoder matrix is incomplete")
    path = args.output / "heldout.parquet"
    table.write_parquet(path)
    files.append(path)
    atomic_json(
        args.output / "MANIFEST.json",
        {
            "scope": plan["scope"],
            "device": args.device,
            "torch": str(torch.__version__),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "config_sha256": sha256(PLAN),
            "completed_utc": now(),
            "wall_seconds": monotonic() - started,
            "graph_manifests_sha256": {key: sha256(ROOT / value) for key, value in paths.items()},
            "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files},
        },
    )
    print(f"DECODER_DONE {len(records)} rows {monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    run()
