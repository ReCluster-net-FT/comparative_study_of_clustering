"""Unsupervised GraphSAGE node embeddings for the MOGA+GraphSAGE hybrid.

Trains a 2-layer GraphSAGE encoder on a k-NN graph of the network's nodes
with an unsupervised link-prediction objective (Hamilton et al. 2017, eq.
1: maximize similarity across real edges, minimize it across random
negative pairs) -- no heuristic hand-designed target, no labels. Node
features are the dataset's own sensor readings (network.node_features),
plus position and residual energy.

Embeddings are trained once (the graph's structure/features barely move
round to round -- only energy decays) and reused for the whole run via
moga.py's embed_map: MOGA's genetic search is completely unchanged, only
its notion of "how close are these two nodes" now comes from the learned
embedding space instead of raw (x, y) distance. See wsn_sim/moga.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, embed_dim, aggr="mean")

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def _build_features_and_graph(network, alive_idx: np.ndarray, k: int):
    d = network.dist_matrix()[np.ix_(alive_idx, alive_idx)]
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    knn = np.argsort(d, axis=1)[:, :k]
    src = np.repeat(np.arange(len(alive_idx)), k)
    dst = knn.flatten()
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long
    )

    pos = network.pos[alive_idx]
    energy = network.energy[alive_idx] / network.init_energy
    feats = [pos[:, 0] / network.area, pos[:, 1] / network.area, energy]
    if network.node_features is not None:
        real = network.node_features[alive_idx]
        lo, hi = real.min(axis=0), real.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        real = (real - lo) / span
        feats = [np.stack(feats, axis=1), real]
        x = np.concatenate(feats, axis=1)
    else:
        x = np.stack(feats, axis=1)
    return torch.tensor(x, dtype=torch.float32), edge_index


def _unsupervised_loss(z, edge_index, n_neg, rng):
    src, dst = edge_index
    pos_score = (z[src] * z[dst]).sum(-1)
    neg_dst = torch.randint(0, z.shape[0], (n_neg,), generator=rng)
    neg_src = src[torch.randint(0, src.shape[0], (n_neg,), generator=rng)]
    neg_score = (z[neg_src] * z[neg_dst]).sum(-1)
    return -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()


def make_embedder(hidden_dim: int = 32, embed_dim: int = 16, k_neighbors: int = 8,
                   epochs: int = 100, lr: float = 1e-2, seed: int = 0):
    """Returns fn(network) -> {node_id: embedding vector}, trained once on first call."""
    state: dict = {}

    def embedder(network) -> dict:
        alive_idx = network.alive_idx()
        if "embeddings" in state:
            return {int(i): state["embeddings"][pos] for pos, i in enumerate(state["alive_idx"])
                     if i in set(alive_idx.tolist())}

        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        k = min(k_neighbors, len(alive_idx) - 1)
        x, edge_index = _build_features_and_graph(network, alive_idx, k=k)

        model = GraphSAGEEncoder(x.shape[1], hidden_dim, embed_dim)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n_neg = edge_index.shape[1]
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            z = model(x, edge_index)
            loss = _unsupervised_loss(z, edge_index, n_neg, rng)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            z = model(x, edge_index).numpy()
        state["embeddings"] = z
        state["alive_idx"] = alive_idx
        return {int(i): z[pos] for pos, i in enumerate(alive_idx)}

    return embedder


if __name__ == "__main__":
    from .wsn_dataset import build_network
    net = build_network(n_nodes=30, seed=1)
    embed = make_embedder(embed_dim=8, epochs=20)
    m = embed(net)
    assert len(m) == 30
    assert next(iter(m.values())).shape == (8,)
    m2 = embed(net)  # second call should reuse the cached embedding, not retrain
    assert (m[0] == m2[0]).all()
    print(f"graphsage_embed ok: {len(m)} node embeddings, dim={next(iter(m.values())).shape[0]}")
