"""Pilot cue generation and sparse projection through real PN-to-KC connections."""

import hashlib

import numpy as np
import polars as pl
from scipy import sparse

from .cue_families import check_disjoint
from .runtime import require


def binary_cues(adjacency, seed, trials, pairs, config):
    """Draw four balanced pairs per block from one shared A+/A- layout."""
    reachable = np.flatnonzero(adjacency.getnnz(axis=1))
    rng = np.random.default_rng(config["layout_seed"])
    order = rng.permutation(reachable)
    shared = config["pair_shared_pns"]
    private = config["family_specific_pns"]
    variable = config["instance_variable_pns"]
    stop = shared + 2 * private
    require(len(order) >= stop + variable, "not enough connected PNs for the real cue families")
    cores = np.stack(
        (order[: shared + private], np.r_[order[:shared], order[shared + private : stop]])
    )
    pool = order[stop:]
    splits = []
    for stream in (config["train_stream"], config["heldout_stream"]):
        draw = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        table = np.empty((trials, pairs * 2, shared + private + variable), dtype=np.int32)
        for i, row in enumerate(table.reshape(-1, table.shape[-1])):
            row[:] = np.sort(np.r_[cores[i % 2], draw.choice(pool, variable, replace=False)])
        splits.append(table)
    check_disjoint(*splits, "PN")
    return splits, cores, pool


def project_cues(adjacency, patterns):
    """Return normalized KC drive, cue-induced edge degree, and raw synaptic strength."""
    require(sparse.isspmatrix_csr(adjacency), "PN-to-KC adjacency must be CSR")
    require((adjacency.data > 0).all() and adjacency.has_canonical_format, "invalid PN-to-KC edges")
    flat = patterns.reshape(-1, patterns.shape[-1])
    stimulus = sparse.csr_matrix(
        (
            np.ones(flat.size, dtype=np.int32),
            flat.ravel(),
            np.arange(0, flat.size + 1, flat.shape[1]),
        ),
        shape=(len(flat), adjacency.shape[0]),
    )
    structure = adjacency.copy().astype(np.int32)
    structure.data.fill(1)
    degree = (stimulus @ structure).tocsr()
    weight = (stimulus @ adjacency.astype(np.int64)).tocsr()
    degree.sort_indices()
    weight.sort_indices()
    require(
        np.array_equal(degree.indptr, weight.indptr)
        and np.array_equal(degree.indices, weight.indices)
        and np.all(np.diff(weight.indptr) > 0),
        "silent cue or inconsistent real edge and weight supports",
    )
    total_input = np.asarray(adjacency.sum(axis=0)).ravel()
    activity = weight.astype(np.float32)
    activity.data /= total_input[activity.indices]
    require(np.all((activity.data > 0) & (activity.data <= 1)), "invalid real KC activity")
    return activity, degree, weight


def check_projected_disjoint(training, heldout):
    def keys(matrix):
        for a, b in zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True):
            yield hashlib.sha256(
                matrix.indices[a:b].tobytes() + matrix.data[a:b].tobytes()
            ).digest()

    seen = set(keys(training))
    require(
        not any(key in seen for key in keys(heldout)), "held-out KC drive duplicates a training cue"
    )


def cue_audit(patterns, root_ids, degree, weight, prefix):
    """Store exact per-presentation KC counts, true degrees, synapse counts and PN IDs."""
    trials, presentations, n_active_pns = patterns.shape
    counts = degree.getnnz(axis=1).reshape(trials, presentations)
    edge_counts = np.asarray(degree.sum(axis=1)).reshape(trials, presentations)
    weights = np.asarray(weight.sum(axis=1)).reshape(trials, presentations)
    histogram = np.zeros((trials * presentations, n_active_pns + 1), dtype=np.int32)
    unique = []
    for i in range(trials):
        start = i * presentations
        end = start + presentations
        unique.append(len(np.unique(degree.indices[degree.indptr[start] : degree.indptr[end]])))
        for row in range(start, end):
            low, high = degree.indptr[row : row + 2]
            histogram[row] = np.bincount(degree.data[low:high], minlength=n_active_pns + 1)
            histogram[row, 0] = degree.shape[1] - counts.ravel()[row]
    require(
        np.array_equal(histogram[:, 1:].sum(axis=1), counts.ravel())
        and np.array_equal(histogram @ np.arange(n_active_pns + 1), edge_counts.ravel()),
        "KC in-degree audit did not reconcile with the real edges",
    )
    return {
        f"{prefix}_pn_root_ids": pl.Series(
            root_ids[patterns].tolist(), dtype=pl.List(pl.List(pl.UInt64))
        ),
        f"{prefix}_kc_active": counts.tolist(),
        f"{prefix}_kc_active_union": unique,
        f"{prefix}_pn_kc_edges": edge_counts.tolist(),
        f"{prefix}_pn_kc_synapses": weights.tolist(),
        f"{prefix}_kc_indegree_histogram": histogram.reshape(trials, presentations, -1).tolist(),
    }
