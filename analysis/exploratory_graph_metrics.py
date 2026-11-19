"""Sparse graph descriptors for the exploratory topology analysis."""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

COMMUNITY_SEED = 20261119
DISTANCE_BATCH = 32


def path_summary(arcs):
    reachable, total = 0, 0.0
    n = arcs.shape[0]
    for first in range(0, n, DISTANCE_BATCH):
        sources = np.arange(first, min(n, first + DISTANCE_BATCH), dtype=np.int32)
        distances = csgraph.dijkstra(arcs, directed=True, indices=sources, unweighted=True)
        finite = np.isfinite(distances)
        reachable += int(finite.sum()) - len(sources)
        total += float(distances.sum(where=finite))
    if not reachable:
        raise ValueError("no reachable distinct vertices for distance calculation")
    return total / reachable, reachable / (n * (n - 1))


def connected_gap(adjacency):
    if adjacency.shape[0] < 3:
        return 2.0
    laplacian = csgraph.laplacian(adjacency, normed=True).tocsr()
    rng = np.random.default_rng(COMMUNITY_SEED)
    values, vectors = eigsh(laplacian, k=2, which="SA", v0=rng.normal(size=laplacian.shape[0]))
    residual = np.linalg.norm(laplacian @ vectors - vectors * values, axis=0)
    if abs(values[0]) > 1e-7 or values[1] <= 0 or residual.max() > 1e-7:
        raise ValueError(f"invalid connected Laplacian eigenpairs: {values}, residual {residual}")
    return float(values[1])


def graph_metrics(matrix):
    arcs = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if arcs.shape[0] != arcs.shape[1] or arcs.shape[0] < 2:
        raise ValueError(f"invalid graph shape: {arcs.shape}")
    arcs.sum_duplicates()
    arcs.eliminate_zeros()
    if not np.isfinite(arcs.data).all() or np.any(arcs.data <= 0):
        raise ValueError("edge weights must be positive and finite")
    self_loops = int(np.count_nonzero(arcs.diagonal()))
    if self_loops:
        arcs.setdiag(0)
        arcs.eliminate_zeros()
    if not arcs.nnz:
        raise ValueError("empty directed edge set")
    arcs.data.fill(1.0)
    arcs.sort_indices()
    undirected = arcs.maximum(arcs.T).tocsr()
    count, labels = csgraph.connected_components(undirected, directed=False)
    largest = np.flatnonzero(labels == np.bincount(labels).argmax())
    core_gap = connected_gap(undirected[largest][:, largest])
    mean_path, reachable = path_summary(arcs)
    network = nx.from_scipy_sparse_array(undirected)
    partition = nx.community.louvain_communities(
        network, weight=None, resolution=1, seed=COMMUNITY_SEED
    )
    return {
        "vertices": arcs.shape[0],
        "directed_arcs": arcs.nnz,
        "ignored_self_loops": self_loops,
        "weak_components": count,
        "largest_component_vertices": int(largest.size),
        "spectral_gap": core_gap if count == 1 else 0.0,
        "largest_component_gap": core_gap,
        "mean_directed_path": mean_path,
        "directed_reachability": reachable,
        "clustering": nx.average_clustering(network),
        "modularity": nx.community.modularity(network, partition, weight=None, resolution=1),
        "reciprocity": arcs.multiply(arcs.T).nnz / arcs.nnz,
    }
