from __future__ import annotations

import hashlib
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import sparse

from .runtime import atomic_file, atomic_json, read_json, require, sha256


def signs_from_nt(nt_types, mapping):
    values = ["" if value is None else value for value in nt_types]
    missing = sorted(set(values) - set(mapping))
    require(not missing, f"NT labels missing from explicit sign policy: {missing}")
    return np.array([mapping[value] for value in values], dtype=np.int8)


class DirectedSwapChain:
    def __init__(self, matrix, edge_signs, variant, seed):
        require(
            sparse.isspmatrix_csr(matrix)
            and matrix.has_canonical_format
            and matrix.shape[0] == matrix.shape[1],
            "null input must be canonical square CSR",
        )
        require(variant in ("A", "B"), "unknown null variant")
        coo = matrix.tocoo(copy=False)
        require(
            matrix.nnz >= 2 and not np.any(coo.row == coo.col),
            "null input must have >=2 edges and no loops",
        )
        require(
            len(edge_signs) == matrix.nnz and np.isin(edge_signs, [-1, 0, 1]).all(),
            "invalid null sign vector",
        )
        require(np.isfinite(coo.data).all() and np.all(coo.data > 0), "invalid null edge weights")
        self.n = matrix.shape[0]
        self.pre = coo.row.astype(np.int64, copy=True)
        self.post = coo.col.astype(np.int64, copy=True)
        self.original_post = self.post.copy()
        self.weights = coo.data.copy()
        self.original_weights = self.weights.copy()
        self.signs = np.asarray(edge_signs, dtype=np.int8).copy()
        self.variant, self.seed = variant, seed
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.accepted, self.attempts = 0, 0
        self.edges = set((self.pre * self.n + self.post).tolist())
        self.original_edges = self.edges.copy()
        require(len(self.edges) == matrix.nnz, "parallel edges in null source")
        self.buckets = {
            int(value): np.flatnonzero(self.signs == value) for value in np.unique(self.signs)
        }
        self.original_in = np.bincount(self.post, minlength=self.n)
        self.original_out = np.bincount(self.pre, minlength=self.n)
        self.original_sign_in = {
            value: np.bincount(self.post[self.signs == value], minlength=self.n)
            for value in self.buckets
        }
        self.original_sign_out = {
            value: np.bincount(self.pre[self.signs == value], minlength=self.n)
            for value in self.buckets
        }
        digest = hashlib.sha256(f"directed-swaps-v1:{variant}:{seed}:{self.n}".encode())
        for array in (self.pre, self.post, self.weights, self.signs):
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        self.signature = digest.hexdigest()

    def step(self):
        self.attempts += 1
        i = int(self.rng.integers(len(self.pre)))
        if self.variant == "B":
            bucket = self.buckets[int(self.signs[i])]
            j = int(bucket[self.rng.integers(len(bucket))])
        else:
            j = int(self.rng.integers(len(self.pre)))
        a, b, c, d = int(self.pre[i]), int(self.post[i]), int(self.pre[j]), int(self.post[j])
        if i == j or a == c or b == d or a == d or c == b:
            return False
        first, second = a * self.n + d, c * self.n + b
        if first in self.edges or second in self.edges:
            return False
        self.edges.remove(a * self.n + b)
        self.edges.remove(c * self.n + d)
        self.edges.update((first, second))
        self.post[i], self.post[j] = d, b
        self.accepted += 1
        return True

    def validate(self):
        require(
            np.array_equal(self.weights, self.original_weights),
            "null changed transported edge weights",
        )
        require(
            len(self.edges) == len(self.pre)
            and len(set((self.pre * self.n + self.post).tolist())) == len(self.pre),
            "null created parallel edges",
        )
        require(not np.any(self.pre == self.post), "null created self loops")
        require(
            np.array_equal(np.bincount(self.pre, minlength=self.n), self.original_out),
            "null out-degree sequence changed",
        )
        require(
            np.array_equal(np.bincount(self.post, minlength=self.n), self.original_in),
            "null in-degree sequence changed",
        )
        if self.variant == "B":
            for value, expected in self.original_sign_in.items():
                require(
                    np.array_equal(
                        np.bincount(self.post[self.signs == value], minlength=self.n), expected
                    ),
                    f"null-B incoming degree changed in sign class {value}",
                )
                require(
                    np.array_equal(
                        np.bincount(self.pre[self.signs == value], minlength=self.n),
                        self.original_sign_out[value],
                    ),
                    "null-B outgoing sign degree changed",
                )
        return {
            "status": "PASS",
            "variant": self.variant,
            "edges": len(self.pre),
            "accepted_swaps": self.accepted,
            "attempts": self.attempts,
            "edge_overlap_fraction": len(self.edges & self.original_edges) / len(self.pre),
            "exact_nodewise_in_out_degree": True,
            "exact_signed_degree": self.variant == "B",
        }

    def checkpoint(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"swap-{self.accepted:09d}-{uuid4().hex}.npz"
        atomic_file(target, lambda p: np.savez_compressed(p, post=self.post))
        pointer = {
            "signature": self.signature,
            "file": target.name,
            "sha256": sha256(target),
            "accepted": self.accepted,
            "attempts": self.attempts,
            "rng_state": self.rng.bit_generator.state,
        }
        atomic_json(directory / "checkpoint.json", pointer)
        return pointer

    def restore(self, directory):
        directory = Path(directory)
        pointer = read_json(directory / "checkpoint.json")
        if pointer is None:
            return False
        require(
            pointer["signature"] == self.signature, "null checkpoint source/variant/seed changed"
        )
        path = directory / pointer["file"]
        require(sha256(path) == pointer["sha256"], "null checkpoint checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            post = data["post"].copy()
        require(
            post.shape == self.post.shape
            and np.issubdtype(post.dtype, np.integer)
            and post.min() >= 0
            and post.max() < self.n,
            "invalid null checkpoint endpoints",
        )
        self.post = post
        self.edges = set((self.pre * self.n + self.post).tolist())
        self.accepted, self.attempts = pointer["accepted"], pointer["attempts"]
        require(0 <= self.accepted <= self.attempts, "invalid null counters")
        self.rng.bit_generator.state = pointer["rng_state"]
        self.validate()
        return True

    def run(
        self, target_swaps, max_attempts, checkpoint_dir=None, checkpoint_seconds=900, progress=None
    ):
        require(
            target_swaps >= self.accepted and max_attempts >= target_swaps,
            "invalid null mixing budget",
        )
        last_save = time.monotonic()
        try:
            while self.accepted < target_swaps:
                require(
                    self.attempts < max_attempts,
                    f"null swap budget exhausted: accepted={self.accepted}, target={target_swaps}, attempts={self.attempts}",
                )
                self.step()
                if (
                    checkpoint_dir is not None
                    and time.monotonic() - last_save >= checkpoint_seconds
                ):
                    self.checkpoint(checkpoint_dir)
                    if progress:
                        progress(accepted_swaps=self.accepted, swap_attempts=self.attempts)
                    last_save = time.monotonic()
        finally:
            if checkpoint_dir is not None:
                self.checkpoint(checkpoint_dir)
        return self.validate()

    def as_csr(self):
        self.validate()
        result = sparse.coo_matrix(
            (self.weights, (self.pre, self.post)), shape=(self.n, self.n)
        ).tocsr()
        require(result.nnz == len(self.pre), "null CSR coalesced unexpected edges")
        return result
