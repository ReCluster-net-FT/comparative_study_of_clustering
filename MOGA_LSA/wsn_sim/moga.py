"""Multi-Objective GA for cluster-head (CH) selection — paper Eq. (1) / Fig. 2.

Chromosome = a fixed-length vector of node indices chosen as CHs (Fig. 2).
Fitness combines (after min-max normalization, as the paper specifies):
    + sum of intra-cluster distances (CM -> its CH)
    + sum of CH -> sink distances
    - sum of CH residual energies
lower is better.
"""
from __future__ import annotations

import numpy as np


def _cluster_cost(chs: np.ndarray, network, alive_idx: np.ndarray, embed_map: dict | None = None):
    """Assign every alive non-CH node to its nearest CH; return costs.

    embed_map, if given, maps node id -> GraphSAGE embedding vector. Intra-
    cluster cohesion (which CH each CM joins, and how "close" it is) is then
    computed in embedding space instead of the network's native distance —
    the whole point of the MOGA+GraphSAGE hybrid: the GA's search itself is
    unchanged, only what "close" means changes. CH-to-sink distance always
    uses network.dist_to_sink (the sink isn't a graph node with an
    embedding) and residual energy stays a plain scalar either way.

    Without embed_map, "close" falls back to network.pair_cost — Euclidean
    (x, y) distance by default, or a graph-native cost (e.g. hop count) when
    the network was built with a cost_matrix (see graph_data.py).
    """
    cm_idx = np.setdiff1d(alive_idx, chs)
    if embed_map is not None:
        ch_vec = np.array([embed_map[c] for c in chs])
        if len(cm_idx):
            cm_vec = np.array([embed_map[c] for c in cm_idx])
            d = np.linalg.norm(cm_vec[:, None, :] - ch_vec[None, :, :], axis=-1)
            intra = d.min(axis=1).sum()
        else:
            intra = 0.0
    elif len(cm_idx) == 0:
        intra = 0.0
    elif network.cost_matrix is not None:
        d = network.cost_matrix[np.ix_(cm_idx, chs)]
        intra = d.min(axis=1).sum()
    else:
        pos = network.pos
        d = np.linalg.norm(pos[cm_idx][:, None, :] - pos[chs][None, :, :], axis=-1)
        intra = d.min(axis=1).sum()
    ds = network.dist_to_sink(chs).sum()
    eres = network.energy[chs].sum()
    return intra, ds, eres


def select_cluster_heads(
    network,
    pop_size: int = 20,
    generations: int = 10,
    mutation_rate: float = 0.2,
    rng: np.random.Generator | None = None,
    embed_map: dict | None = None,
) -> np.ndarray:
    """Run the GA for one round; return chosen CH node indices.

    embed_map: optional node id -> embedding vector (see _cluster_cost).
    The GA process itself (population/crossover/mutation/selection) is
    identical either way — only the fitness function's notion of
    intra-cluster distance changes.
    """
    rng = rng or np.random.default_rng()
    alive_idx = network.alive_idx()
    k = max(1, round(network.ch_prob * len(alive_idx)))
    k = min(k, len(alive_idx))

    population = [rng.choice(alive_idx, size=k, replace=False) for _ in range(pop_size)]

    def evaluate(pop):
        raw = np.array([_cluster_cost(chs, network, alive_idx, embed_map) for chs in pop])
        norm = np.zeros_like(raw)
        for col in range(3):
            lo, hi = raw[:, col].min(), raw[:, col].max()
            norm[:, col] = 0.0 if hi == lo else (raw[:, col] - lo) / (hi - lo)
        return norm[:, 0] + norm[:, 1] - norm[:, 2]

    for _ in range(generations):
        fitness = evaluate(population)
        order = np.argsort(fitness)
        survivors = [population[i] for i in order[: pop_size // 2]]

        children = []
        while len(children) < pop_size - len(survivors):
            p1, p2 = survivors[rng.integers(len(survivors))], survivors[rng.integers(len(survivors))]
            cut = rng.integers(1, k) if k > 1 else 1
            child = np.concatenate([p1[:cut], p2[cut:]])
            child = np.unique(child)
            while len(child) < k:
                candidate = rng.choice(alive_idx)
                if candidate not in child:
                    child = np.append(child, candidate)
            if rng.random() < mutation_rate:
                pos = rng.integers(k)
                candidate = rng.choice(alive_idx)
                if candidate not in child:
                    child[pos] = candidate
            children.append(child)
        population = survivors + children

    fitness = evaluate(population)
    return population[int(np.argmin(fitness))]
