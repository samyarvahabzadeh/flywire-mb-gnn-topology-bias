"""Portable exploratory assay of discrimination after absolute or differential training."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import sparse

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cue_panel(projection, seed, cfg):
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.flatnonzero(projection.getnnz(axis=1)))
    n, common, r = cfg["size"], cfg["shared"], cfg["replacements"]
    shared, a, b = order[:common], order[common:n], order[n : 2 * n - common]
    other = order[2 * n - common :]
    cues = [np.r_[shared, a], np.r_[shared, b]]
    # Each novel pair has identical background and matched absolute edit distance.
    for _ in range(cfg["novel_pairs"]):
        background = rng.choice(other, r, replace=False)
        core = rng.choice(shared, common - r, replace=False)
        cues.extend([np.r_[core, a, background], np.r_[core, b, background]])
    cues = np.sort(np.stack(cues), axis=1)
    assert len(np.unique(cues, axis=0)) == len(cues)
    stimuli = sparse.csr_matrix(
        (np.ones(cues.size), cues.ravel(), np.arange(0, cues.size + 1, n)),
        shape=(len(cues), projection.shape[0]),
    )
    total = np.asarray(projection.sum(axis=0)).ravel()
    return (stimuli @ projection).multiply(1 / np.maximum(total, 1)).tocsr()


def assay(x, post, cfg, rule, differential, direction):
    # Linear readout equals twice the baseline-subtracted avoidance score of the original circuit.
    coefficients = torch.zeros_like(x)
    for side in (0, 1):
        sub = x[:, post == side]
        assert torch.all(sub.sum(1) > 0)
        coefficients[:, post == side] = 0.5 * sub / sub.sum(1, keepdim=True)
    w = torch.nn.Parameter(torch.ones(x.shape[1], dtype=torch.float64))
    opt = torch.optim.Adam([w], lr=cfg["adam_lr"]) if rule == "adam" else None
    ids = [direction, 1 - direction] if differential else [direction]
    for _ in range(cfg["pairs"]):
        if opt is not None:
            v = coefficients[ids] @ (1 - w)
            labels = torch.tensor([1.0, 0.0][: len(ids)], dtype=w.dtype)
            # Fixed divisor keeps the CS+ loss coefficient identical across protocols.
            loss = (
                torch.nn.functional.binary_cross_entropy(0.25 + 0.5 * v, labels, reduction="sum")
                / 2
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w.clamp_(0, 1)
        else:
            with torch.no_grad():
                for k in ids:
                    us = float(k == direction)
                    prediction = float(coefficients[k] @ (1 - w))
                    signal = us - prediction if rule == "prediction_error" else us
                    w.add_(x[k], alpha=-cfg["eta"] * signal).clamp_(0, 1)
    with torch.no_grad():
        response = (coefficients @ (1 - w)).numpy()
    novel = response[2:].reshape(-1, 2).mean(axis=0)
    assert np.isfinite(response).all() and response[direction] > 0
    return {
        "cs_plus": response[direction],
        "cs_minus": response[1 - direction],
        "near_plus": novel[direction],
        "near_minus": novel[1 - direction],
        "gap": novel[direction] - novel[1 - direction],
        "normalized_gap": (novel[direction] - novel[1 - direction]) / response[direction],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((HERE / "bundle.json").read_text())
    for name, h in provenance["files"].items():
        if digest(HERE / name) != h:
            raise ValueError(f"Bundle checksum differs: {name}")
    cfg = json.loads((HERE / "plan.json").read_text())
    projection = sparse.load_npz(HERE / "projection.npz")
    data = np.load(HERE / "masks.npz", allow_pickle=False)
    masks = sorted(k.removesuffix("_pre") for k in data.files if k.endswith("_pre"))
    assert "real" in masks and len(masks) == 13
    seeds = cfg["seeds"]
    if args.smoke:
        seeds, masks = [cfg["smoke_seed"]], ["real", next(x for x in masks if x != "real")]
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = digest(HERE / "bundle.json")
    pieces = []
    for seed in seeds:
        dest = args.output / f"seed-{seed}.parquet"
        meta = dest.with_suffix(".json")
        if meta.exists():
            receipt = json.loads(meta.read_text())
            assert receipt["run_id"] == run_id and receipt["sha256"] == digest(dest)
        else:
            activity = cue_panel(projection, seed, cfg)
            rows = []
            for mask in masks:
                x = torch.from_numpy(activity[:, data[mask + "_pre"]].toarray()).double()
                post = data[mask + "_post"]
                for direction in (0, 1):
                    for rule in ("pulse", "prediction_error", "adam"):
                        prev = None
                        for differential in (False, True):
                            result = assay(x, post, cfg, rule, differential, direction)
                            if rule == "pulse" and prev is not None:
                                assert result == prev, "unreinforced pulse changed the memory"
                            rows.append(
                                dict(
                                    seed=seed,
                                    mask=mask,
                                    direction=direction,
                                    rule=rule,
                                    differential=differential,
                                    **result,
                                )
                            )
                            prev = result
            tmp = dest.with_suffix(".partial")
            pl.DataFrame(rows).write_parquet(tmp)
            os.replace(tmp, dest)
            receipt = {
                "run_id": run_id,
                "sha256": digest(dest),
                "git_commit": provenance["git_commit"],
                "config_sha256": provenance["files"]["plan.json"],
            }
            tmp.write_text(json.dumps(receipt))
            os.replace(tmp, meta)
        pieces.append(pl.read_parquet(dest))
        print("VERIFIED_SEED", seed, flush=True)
    table = pl.concat(pieces)
    assert table.height == len(seeds) * len(masks) * 12
    gaps = table.group_by("seed", "mask", "rule", "differential").agg(pl.col("gap").mean())
    delta = (
        gaps.filter(pl.col("differential"))
        .join(gaps.filter(~pl.col("differential")), on=["seed", "mask", "rule"], suffix="_abs")
        .with_columns((pl.col("gap") - pl.col("gap_abs")).alias("gain"))
    )
    keys = ["seed", "rule"]
    summary = (
        delta.filter(pl.col("mask") == "real")
        .select(*keys, pl.col("gain").alias("real_gain"))
        .join(
            delta.filter(pl.col("mask") != "real")
            .group_by(*keys)
            .agg(pl.col("gain").median().alias("null_gain")),
            on=keys,
        )
        .with_columns((pl.col("real_gain") - pl.col("null_gain")).alias("topology_effect"))
    )
    dest = args.output / "seed-contrasts.parquet"
    summary.sort(keys).write_parquet(dest)
    (args.output / "MANIFEST.json").write_text(
        json.dumps(
            dict(
                **provenance,
                smoke=args.smoke,
                jobs=table.height,
                result_sha256=digest(dest),
                shards={
                    f"seed-{s}.parquet": digest(args.output / f"seed-{s}.parquet") for s in seeds
                },
            ),
            indent=2,
        )
    )
    print("COMPLETE", table.height, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
