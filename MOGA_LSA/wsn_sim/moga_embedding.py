"""Hybrid CH selector: MOGA's genetic search, unsupervised GraphSAGE
embedding space (graphsage_embed.py) instead of raw (x, y) distance.

Isolates one question: does replacing MOGA's raw intra-cluster distance
with distance in a learned GraphSAGE embedding improve the GA's search,
holding the GA process itself (population, crossover, mutation, selection)
fixed? See wsn_sim/moga.py's embed_map and graphsage_embed.py.
"""
from __future__ import annotations

import numpy as np

from . import graphsage_embed, moga


def make_selector(pop_size: int = 20, generations: int = 10, mutation_rate: float = 0.2,
                   embed_kwargs: dict | None = None):
    embedder = graphsage_embed.make_embedder(**(embed_kwargs or {}))

    def selector(network, rng: np.random.Generator | None = None) -> np.ndarray:
        embed_map = embedder(network)
        return moga.select_cluster_heads(
            network, pop_size=pop_size, generations=generations,
            mutation_rate=mutation_rate, rng=rng, embed_map=embed_map,
        )

    return selector
