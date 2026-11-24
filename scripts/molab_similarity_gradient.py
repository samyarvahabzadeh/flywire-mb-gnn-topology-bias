"""CUDA batch runner for the exploratory graded-similarity conditioning assay."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import sparse


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2))
    os.replace(tmp, path)


def projected_coefficients(projection, patterns, presynaptic, postsynaptic):
    active_pns = np.zeros((len(patterns), projection.shape[0]), dtype=np.float32)
    for row, pattern in enumerate(patterns):
        active_pns[row, pattern] = 1.0
    drives = (sparse.csr_matrix(active_pns) @ projection).tocsr()
    x = drives[:, presynaptic].toarray().astype(np.float32, copy=False)
    coeff = np.zeros_like(x)
    for side in (0, 1):
        group = postsynaptic == side
        denom = x[:, group].sum(axis=1, keepdims=True)
        if np.any(denom <= 0):
            raise ValueError("PN cue produced an MBON11 group with no KC drive")
        coeff[:, group] = 0.5 * x[:, group] / denom
    return x, coeff


def make_patterns(projection, seed, shared, cfg):
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(projection.getnnz(axis=1))
    size = cfg["cue_size"]
    order = rng.permutation(active)
    common = order[:shared]
    first = np.r_[common, order[shared:size]]
    second = np.r_[common, order[size : 2 * size - shared]]
    training = [np.sort(first), np.sort(second)]
    pool = order[2 * size - shared :]
    if len(np.intersect1d(first, second)) != shared:
        raise ValueError("training PN overlap is not the configured dose")
    families = [[[] for _ in cfg["probe_shared_pns"]] for _ in (0, 1)]
    seen = {tuple(training[0]), tuple(training[1])}
    for class_id, cue in enumerate(training):
        for q_index, probe_shared in enumerate(cfg["probe_shared_pns"]):
            for _ in range(cfg["probes_per_class_per_level"]):
                for attempt in range(100):
                    probe = np.sort(
                        np.r_[
                            rng.choice(cue, probe_shared, replace=False),
                            rng.choice(pool, size - probe_shared, replace=False),
                        ]
                    )
                    key = tuple(probe)
                    if key not in seen:
                        seen.add(key)
                        families[class_id][q_index].append(probe)
                        break
                else:
                    raise ValueError("could not draw unique held-out PN probes")
    flat = [*training]
    for q_index in range(len(cfg["probe_shared_pns"])):
        for class_id in (0, 1):
            flat.extend(families[class_id][q_index])
    return flat, families


@torch.no_grad()
def fit_batch(x_plus, x_minus, c_plus, c_minus, probe, rule, schedule, cfg, device):
    x_plus = torch.as_tensor(x_plus, dtype=torch.float32, device=device)
    x_minus = torch.as_tensor(x_minus, dtype=torch.float32, device=device)
    c_plus = torch.as_tensor(c_plus, dtype=torch.float32, device=device)
    c_minus = torch.as_tensor(c_minus, dtype=torch.float32, device=device)
    probe = torch.as_tensor(probe, dtype=torch.float32, device=device)
    batch, edges = x_plus.shape
    w = torch.full((batch, edges), cfg["initial_weight"], dtype=torch.float32, device=device)
    lo, hi = cfg["weight_bounds"]
    if rule == "adam_bce":
        m = torch.zeros_like(w)
        v2 = torch.zeros_like(w)
    for step in range(1, cfg["training_cycles"] + 1):
        vp = (c_plus * (1 - w)).sum(dim=1)
        if rule == "pulse":
            w.sub_(cfg["local_eta"] * x_plus)
        elif rule == "prediction_error":
            w.sub_(cfg["local_eta"] * x_plus * (1 - vp[:, None]))
            if schedule == "differential":
                vm = (c_minus * (1 - w)).sum(dim=1)
                w.add_(cfg["local_eta"] * x_minus * vm[:, None])
        elif rule == "adam_bce":
            pp = 0.25 + 0.5 * vp
            gp = -0.25 * c_plus * ((pp - 1) / (pp * (1 - pp)))[:, None]
            if schedule == "differential":
                vm = (c_minus * (1 - w)).sum(dim=1)
                pm = 0.25 + 0.5 * vm
                gm = -0.25 * c_minus * (pm / (pm * (1 - pm)))[:, None]
                grad = gp + gm
            else:
                grad = gp
            m.mul_(0.9).add_(grad, alpha=0.1)
            v2.mul_(0.999).addcmul_(grad, grad, value=0.001)
            corrected_m = m / (1 - 0.9**step)
            corrected_v = v2 / (1 - 0.999**step)
            w.addcdiv_(corrected_m, corrected_v.sqrt() + 1e-8, value=-cfg["adam_learning_rate"])
        else:
            raise ValueError(f"unknown rule {rule}")
        w.clamp_(lo, hi)
    response = (1 - w) @ probe.T
    response = response.reshape(batch, len(cfg["probe_shared_pns"]), 2)
    cs_plus = (c_plus * (1 - w)).sum(dim=1)
    cs_minus = (c_minus * (1 - w)).sum(dim=1)
    return response.cpu().numpy(), cs_plus.cpu().numpy(), cs_minus.cpu().numpy()


def load_mask_arrays(path):
    data = np.load(path, allow_pickle=False)
    names = sorted({key.removesuffix("_pre") for key in data.files if key.endswith("_pre")})
    if "real" not in names or len(names) != 13:
        raise ValueError("expected the frozen real mask and 12 input controls")
    return data, names


def build_mask_inputs(projection, cfg, pres, post, seeds, shared):
    e = len(pres)
    x_train = np.empty((len(seeds), 2, e), dtype=np.float32)
    c_train = np.empty_like(x_train)
    c_probe = np.zeros((len(seeds), 2, len(cfg["probe_shared_pns"]), 2, e), dtype=np.float32)
    for i, seed in enumerate(seeds):
        patterns, _families = make_patterns(projection, seed, shared, cfg)
        x, coeff = projected_coefficients(projection, patterns, pres, post)
        x_train[i] = x[:2]
        c_train[i] = coeff[:2]
        cursor = 2
        for q_index in range(len(cfg["probe_shared_pns"])):
            for class_id in (0, 1):
                n = cfg["probes_per_class_per_level"]
                c_probe[i, class_id, q_index, class_id] = coeff[cursor : cursor + n].mean(axis=0)
                cursor += n
    return x_train, c_train, c_probe


def run_shard(projection, arrays, mask, shared, seeds, cfg, device):
    pres = arrays[mask + "_pre"].astype(np.int64)
    post = arrays[mask + "_post"].astype(np.int64)
    # The cues and their projection are fixed across learning rules, schedules and reciprocal labels.
    x_train, c_train, c_probe = build_mask_inputs(projection, cfg, pres, post, seeds, shared)
    rows = []
    for direction in (0, 1):
        plus_id, minus_id = direction, 1 - direction
        xp = x_train[:, plus_id]
        xm = x_train[:, minus_id]
        cp = c_train[:, plus_id]
        cm = c_train[:, minus_id]
        q_indices = np.arange(len(cfg["probe_shared_pns"]))
        probe = np.stack(
            (
                c_probe[:, plus_id, q_indices, plus_id],
                c_probe[:, minus_id, q_indices, minus_id],
            ),
            axis=2,
        )
        for rule in cfg["rules"]:
            for schedule in cfg["schedules"]:
                response, cs_plus, cs_minus = fit_batch(xp, xm, cp, cm, probe.reshape(len(seeds), -1, len(pres)), rule, schedule, cfg, device)
                response = response.reshape(len(seeds), len(cfg["probe_shared_pns"]), 2)
                for i, seed in enumerate(seeds):
                    for q_index, q in enumerate(cfg["probe_shared_pns"]):
                        rows.append(
                            {
                                "seed": seed,
                                "mask": mask,
                                "train_shared_pns": shared,
                                "probe_shared_pns": q,
                                "rule": rule,
                                "schedule": schedule,
                                "direction": direction,
                                "probe_csplus_response": float(response[i, q_index, 0]),
                                "probe_csminus_response": float(response[i, q_index, 1]),
                                "heldout_discrimination": float(response[i, q_index, 0] - response[i, q_index, 1]),
                                "trained_csplus_response": float(cs_plus[i]),
                                "trained_csminus_response": float(cs_minus[i]),
                            }
                        )
    return pl.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    work = args.work
    provenance = json.loads((work / "bundle.json").read_text())
    for name, expected in provenance["files"].items():
        if sha256(work / name) != expected:
            raise ValueError(f"input checksum mismatch: {name}")
    cfg = json.loads((work / "plan.json").read_text())
    projection = sparse.load_npz(work / "projection.npz").tocsr()
    arrays, masks = load_mask_arrays(work / "masks.npz")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Molab GPU run")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print("CUDA_READY", properties.name, properties.total_memory, flush=True)
    out = work / ("smoke" if args.smoke else "results")
    out.mkdir(exist_ok=True)
    seeds = [cfg["seed_start"]] if args.smoke else list(range(cfg["seed_start"], cfg["seed_start"] + cfg["n_seeds"]))
    selected_masks = ["real"] if args.smoke else masks
    selected_shared = [cfg["train_shared_pns"][2]] if args.smoke else cfg["train_shared_pns"]
    run_id = sha256(work / "bundle.json")
    for mask in selected_masks:
        for shared in selected_shared:
            name = f"mask-{mask}__train-{shared}.parquet"
            dest = out / name
            receipt = dest.with_suffix(".json")
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                if prior.get("run_id") != run_id or prior.get("sha256") != sha256(dest):
                    raise ValueError(f"checkpoint receipt differs: {name}")
                print("SHARD_VERIFIED", name, flush=True)
                continue
            started = time.monotonic()
            frame = run_shard(projection, arrays, mask, shared, seeds, cfg, device)
            if not frame.select(pl.col("heldout_discrimination").is_finite().all()).item():
                raise ValueError(f"non-finite output in {name}")
            tmp = dest.with_suffix(".partial")
            frame.write_parquet(tmp)
            os.replace(tmp, dest)
            atomic_json(receipt, {"run_id": run_id, "sha256": sha256(dest), "rows": frame.height})
            print("SHARD_COMPLETE", name, frame.height, f"{time.monotonic()-started:.2f}s", int(torch.cuda.max_memory_allocated(device)), flush=True)
    inventory = sorted(p.name for p in out.glob("mask-*.parquet"))
    if len(inventory) != len(selected_masks) * len(selected_shared):
        raise ValueError("shard inventory is incomplete")
    expected_rows = len(seeds) * 2 * len(cfg["rules"]) * len(cfg["schedules"]) * len(cfg["probe_shared_pns"])
    shards = {}
    for name in inventory:
        path = out / name
        receipt = json.loads(path.with_suffix(".json").read_text())
        if receipt["run_id"] != run_id or receipt["sha256"] != sha256(path):
            raise ValueError(f"final shard integrity failure: {name}")
        if pl.read_parquet(path).height != expected_rows:
            raise ValueError(f"wrong row count: {name}")
        shards[name] = receipt
    atomic_json(
        out / "MANIFEST.json",
        {
            "scope": cfg["scope"],
            "git_commit": provenance["git_commit"],
            "run_id": run_id,
            "config_sha256": provenance["files"]["plan.json"],
            "runner_sha256": provenance["files"]["run.py"],
            "device": properties.name,
            "device_total_vram_bytes": properties.total_memory,
            "jobs": sum(item["rows"] for item in shards.values()),
            "shards": shards,
            "completed_unix": time.time(),
            "smoke": args.smoke,
        },
    )
    print("RUN_COMPLETE", sum(item["rows"] for item in shards.values()), len(shards), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
