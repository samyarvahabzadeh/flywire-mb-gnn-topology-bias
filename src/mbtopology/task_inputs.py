from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import sparse

from .runtime import atomic_json, fingerprint, read_json, require, sha256


def load_task(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    require(
        config["dataset"] == "fafb" and str(config["snapshot"]) == "783",
        "task source must be fafb/v783",
    )
    require(config["task"]["n_odors"] == 8, "scientific task requires eight frozen odors")
    require(
        config["plasticity"]["rule"] == "anti_hebbian_depression"
        and config["plasticity"]["eligibility"] == "instantaneous",
        "unsupported locked v1 rule",
    )
    require(0 < config["checkpoint_seconds"] <= 900, "task checkpoint interval must be <=900s")
    plus, minus = config["task"]["cs_plus"], config["task"]["cs_minus"]
    require(
        len(plus) == len(minus) == 4 and sorted(plus + minus) == list(range(8)),
        "CS labels must partition eight odors",
    )
    require(0 < config["cs_encoding"]["kc_sparsity"] <= 1, "invalid KC sparsity")
    require(
        config["cs_encoding"]["active_count_rule"] == "nearest_integer"
        and config["cs_encoding"]["bit_generator"] == "PCG64",
        "unsupported frozen odor encoding policy",
    )
    require(
        config["plasticity"]["w_init"]["distribution"] == "uniform"
        and config["plasticity"]["w_init"]["assignment"] == "seed_and_root_id_pair_hash"
        and config["plasticity"]["post_activity"]
        == "weighted_sum_divided_by_structural_kc_indegree",
        "unsupported v1 initialization/activity policy",
    )
    require(
        config["readout"]["normalization"] == "fixed_affine_half_shift"
        and config["readout"]["discrimination"] == "balanced_cs_target_score_after_each_trial",
        "unsupported frozen readout/scoring policy",
    )
    return config


def load_source(root, manifest_path):
    root = Path(root)
    manifest = read_json(root / manifest_path)
    require(
        manifest and manifest.get("status") == "PASS", "task requires a completed, passing MB graph"
    )
    for item in manifest["artifacts"].values():
        require(sha256(root / item["path"]) == item["sha256"], "task graph checksum mismatch")
    nodes = pl.read_parquet(root / manifest["artifacts"]["nodes"]["path"])
    matrix = sparse.load_npz(root / manifest["artifacts"]["csr"]["path"])
    require(
        sparse.isspmatrix_csr(matrix) and matrix.shape == (len(nodes), len(nodes)),
        "task adjacency must be sparse CSR",
    )
    return manifest, nodes, matrix


def node_identity(nodes):
    return fingerprint(nodes.to_dict(as_series=False))


def make_patterns(n_kc, n_odors, fraction, seed, stream=0):
    require(n_kc > 0 and n_odors > 0 and 0 < fraction <= 1, "invalid odor dimensions/sparsity")
    active = min(n_kc, max(1, int(np.floor(n_kc * fraction + 0.5))))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    return [np.sort(rng.choice(n_kc, active, replace=False)).tolist() for _ in range(n_odors)]


def schedule(config, seed):
    task = config["task"]
    require(task["trial_order"] == "seed_shuffled_balanced_blocks", "unknown trial-order policy")
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([seed, config["cs_encoding"]["schedule_seed_stream"]])
        )
    )
    blocks = [
        rng.permutation(task["n_odors"])
        for _ in range((task["trials_max"] + task["n_odors"] - 1) // task["n_odors"])
    ]
    return np.concatenate(blocks)[: task["trials_max"]].astype(np.int64)


def initial_weights(pre_ids, post_ids, seed, low=0.8, high=1.0):
    require(
        len(pre_ids) == len(post_ids) and 0 <= low < high <= 1, "invalid initial weight request"
    )

    def uniform(pre, post):
        key = f"task-v1:weights:{seed}:{int(pre)}:{int(post)}".encode()
        number = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") >> 11
        return low + (high - low) * (number * 2.0**-53)

    return np.fromiter(
        (uniform(a, b) for a, b in zip(pre_ids, post_ids)), dtype=np.float64, count=len(pre_ids)
    )


def freeze_file(path, document):
    existing = read_json(path)
    if existing is not None:
        require(
            fingerprint(existing) == fingerprint(document),
            f"frozen input changed: {Path(path).name}; create a new version after documenting the change",
        )
    else:
        atomic_json(path, document)


def prepare(root, config, run):
    root = Path(root)
    _, nodes, matrix = load_source(root, config["source_manifest"])
    seeds = [
        int(line)
        for line in (root / config["seeds_file"]).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    require(seeds == list(range(30)), "scientific seed list must be exactly 0..29")
    kc = nodes.filter(pl.col("role") == "KC")
    mbon = nodes.filter(pl.col("role") == "MBON")
    encoding = config["cs_encoding"]
    odors = {
        "version": "odor_sets_v1",
        "kc_root_ids": [str(x) for x in kc["root_id"]],
        "n_odors": config["task"]["n_odors"],
        "kc_sparsity": encoding["kc_sparsity"],
        "rounding": encoding["active_count_rule"],
        "bit_generator": "PCG64",
        "seed_stream": encoding["odor_seed_stream"],
        "seeds": {},
    }
    for seed in seeds:
        patterns = make_patterns(
            len(kc), odors["n_odors"], odors["kc_sparsity"], seed, encoding["odor_seed_stream"]
        )
        odors["seeds"][str(seed)] = patterns
        run.save(prepared_seed=seed, odor_count=len(patterns))
    freeze_file(root / encoding["odor_sets"], odors)
    readout_config = config["readout"]
    avoidance = mbon["nt_type"].is_in(readout_config["avoidance_nt"])
    approach = mbon["nt_type"].is_in(readout_config["approach_nt"])
    require(
        not (avoidance & approach).any() and avoidance.any() and approach.any(),
        "invalid MBON readout groups",
    )
    readout = {
        "version": "mbon_readout_v1",
        "status": "PROPOSED",
        "policy": readout_config,
        "mbon_root_ids": [str(x) for x in mbon["root_id"]],
        "avoidance_indices": np.flatnonzero(avoidance.to_numpy()).tolist(),
        "approach_indices": np.flatnonzero(approach.to_numpy()).tolist(),
        "excluded_indices": np.flatnonzero((~(avoidance | approach)).to_numpy()).tolist(),
    }
    freeze_file(root / readout_config["frozen_membership"], readout)
    roles = nodes["role"].to_numpy()
    kc_indices, mbon_indices = np.flatnonzero(roles == "KC"), np.flatnonzero(roles == "MBON")
    plastic = matrix[kc_indices][:, mbon_indices].tocsr()
    report = {
        "status": "PREPARED_NO_BIOLOGICAL_LEARNING_RUN",
        "n_seeds": len(seeds),
        "n_odors": odors["n_odors"],
        "kc_count": len(kc),
        "active_kcs_per_odor": len(odors["seeds"]["0"][0]),
        "actual_active_fraction": len(odors["seeds"]["0"][0]) / len(kc),
        "readout_group_counts": {
            "avoidance": int(avoidance.sum()),
            "approach": int(approach.sum()),
            "excluded_unknown": len(readout["excluded_indices"]),
        },
        "plastic_kc_mbon_edges": plastic.nnz,
        "mbons_without_kc_inputs": int(np.count_nonzero(plastic.getnnz(axis=0) == 0)),
        "canonical_node_identity_sha256": node_identity(nodes),
        "task_config_sha256": fingerprint(config),
        "odor_sets_sha256": sha256(root / encoding["odor_sets"]),
        "readout_sha256": sha256(root / readout_config["frozen_membership"]),
        "seeds_sha256": sha256(root / config["seeds_file"]),
        "model_metrics_computed": False,
    }
    atomic_json(root / "reports" / "task_inputs.json", report)
    return report
