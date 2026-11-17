from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
import torch
import yaml

from .runtime import atomic_file, atomic_json, now, read_json, require, sha256


class SessionYield(RuntimeError):
    pass


def save_yaml(path, document):
    def write(target):
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)

    atomic_file(path, write)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        require(torch.cuda.is_available(), "CUDA checkpoint cannot silently resume on CPU")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(directory, signature, state):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"training-{state['cursor']:06d}-{uuid4().hex}.pt"
    atomic_file(target, lambda p: torch.save({**state, "rng": rng_state()}, p))
    atomic_json(
        directory / "checkpoint.json",
        {
            "signature": signature,
            "file": target.name,
            "sha256": sha256(target),
            "cursor": state["cursor"],
            "created_at": now(),
        },
    )


def load_training_checkpoint(directory, signature):
    directory = Path(directory)
    pointer = read_json(directory / "checkpoint.json")
    if pointer is None:
        return None
    require(
        pointer["signature"] == signature,
        "model resume scientific/code/graph/device signature differs",
    )
    path = directory / pointer["file"]
    require(sha256(path) == pointer["sha256"], "model checkpoint checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(
        state["cursor"] == pointer["cursor"] == len(state["history"]),
        "model checkpoint cursor/history mismatch",
    )
    return state


def cached_result(path, run_config_sha256, bundle_sha256):
    path = Path(path)
    manifest = read_json(path.with_suffix(".manifest.json"))
    if manifest is None:
        return False
    require(
        manifest.get("complete")
        and manifest["run_config_sha256"] == run_config_sha256
        and manifest["bundle_sha256"] == bundle_sha256,
        "existing result belongs to a different run configuration",
    )
    require(
        path.exists() and sha256(path) == manifest["parquet_sha256"],
        "result parquet missing/corrupt",
    )
    return True


def write_result(root, path, frame, metadata):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(
        path.is_relative_to(root) and path.suffix == ".parquet",
        "result path must stay under the run root",
    )
    require(
        metadata.get("git_commit")
        and metadata.get("run_config_sha256")
        and metadata.get("bundle_sha256"),
        "result manifest requires git, run-config and scientific-bundle hashes",
    )
    if cached_result(path, metadata["run_config_sha256"], metadata["bundle_sha256"]):
        require(
            pl.read_parquet(path).equals(frame),
            "completed result differs from newly supplied history",
        )
        return read_json(path.with_suffix(".manifest.json"))
    if path.exists():
        require(
            pl.read_parquet(path).equals(frame),
            "orphan result differs from recovered checkpoint history",
        )
    else:
        atomic_file(path, lambda p: frame.write_parquet(p, compression="zstd"))
    manifest = {
        **metadata,
        "complete": True,
        "created_at": now(),
        "rows": len(frame),
        "parquet_path": path.relative_to(root).as_posix(),
        "parquet_bytes": path.stat().st_size,
        "parquet_sha256": sha256(path),
    }
    atomic_json(path.with_suffix(".manifest.json"), manifest)
    return manifest
