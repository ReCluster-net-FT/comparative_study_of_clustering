"""
graphsage_model.py
===================

A real GraphSAGE (Hamilton et al., 2017) model used to score Cluster
Head candidates from graph-structured features, implemented in
PyTorch.

Design
------
Per cluster and per round we build a small graph:

    nodes = alive member nodes of that cluster
    edges = every pair of members (the OCHSAT paper's clustering
            design guarantees single-hop membership, i.e. every
            member is within radio range of every other member of
            its own cluster - see Section 4.1.2 of the paper - so a
            fully-connected intra-cluster graph is a faithful model
            of "who can actually hear whom")

Node features (4-dim, each min-max normalized within the cluster):
    [ residual_energy, centrality, distance_to_BS, node_degree ]

The model outputs one scalar "CH suitability" score per node via two
SAGE-style layers (mean-aggregate neighbor features, concatenate with
the node's own features, linear transform, non-linearity) followed by
a linear scoring head.

Two backends
------------
- If `torch_geometric` is importable, we use its `SAGEConv` layers
  directly (the standard library implementation).
- Otherwise we fall back to a small hand-written SAGE layer using
  plain `torch` tensor ops (mean neighbor aggregation via the
  adjacency matrix). Both paths expose the exact same
  `GraphSAGENet.forward(x, edge_index)` signature, so the rest of the
  codebase (`ch_selector.py`) doesn't need to know which backend is
  active.

GPU
---
`get_device()` returns `cuda` automatically if `torch.cuda.is_available()`
and `config.GRAPHSAGE_DEVICE == "cuda"`, otherwise `cpu`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def get_device():
    want_cuda = getattr(cfg, "GRAPHSAGE_DEVICE", "cuda") == "cuda"
    if want_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Backend A: torch_geometric SAGEConv (used automatically if installed)
# ---------------------------------------------------------------------------

class _PyGGraphSAGENet(nn.Module):

    def __init__(self, in_dim, hidden_dims=(32, 16)):
        super().__init__()
        h1, h2 = hidden_dims
        self.conv1 = SAGEConv(in_dim, h1)
        self.conv2 = SAGEConv(h1, h2)
        self.head = nn.Linear(h2, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Backend B: pure-PyTorch fallback (no torch_geometric dependency)
# ---------------------------------------------------------------------------

class _ManualSAGELayer(nn.Module):
    """
    One GraphSAGE 'mean aggregator' layer, written from scratch:

        h_v = ReLU( W . concat(h_v, mean_{u in N(v)} h_u) )

    edge_index is a [2, E] tensor of (src, dst) pairs. We build a
    dense adjacency + degree normalisation once per forward pass -
    perfectly fine at this scale (a handful to a few dozen nodes per
    cluster, called once per round).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)
        adj = torch.zeros((n, n), device=x.device, dtype=x.dtype)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = 1.0

        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neighbor_mean = (adj @ x) / degree

        combined = torch.cat([x, neighbor_mean], dim=1)
        return F.relu(self.linear(combined))


class _ManualGraphSAGENet(nn.Module):

    def __init__(self, in_dim, hidden_dims=(32, 16)):
        super().__init__()
        h1, h2 = hidden_dims
        self.layer1 = _ManualSAGELayer(in_dim, h1)
        self.layer2 = _ManualSAGELayer(h1, h2)
        self.head = nn.Linear(h2, 1)

    def forward(self, x, edge_index):
        h = self.layer1(x, edge_index)
        h = self.layer2(h, edge_index)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Ablation baseline: node-only model (Experiment A, PRD section 21)
# ---------------------------------------------------------------------------
# Deliberately NOT a GraphSAGE model -- no edge_index is ever consulted,
# no message passing happens. Same input feature vector, same depth/
# width and same output head as the real GraphSAGE model above, so that
# any performance difference between the two is attributable to the
# neighbourhood aggregation itself and not to a confound like model
# capacity. Used to answer: "does actual neighbourhood aggregation help,
# compared to just using the node's own features?"
# ---------------------------------------------------------------------------

class NodeOnlyNet(nn.Module):

    def __init__(self, in_dim, hidden_dims=(32, 16)):
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.head = nn.Linear(h2, 1)

    def forward(self, x, edge_index=None):
        # edge_index is accepted (and ignored) only so this class is a
        # drop-in replacement for GraphSAGENet at every call site.
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model(in_dim=4, hidden_dims=None, kind="graphsage"):
    """
    Construct a scoring model on the correct device.

    kind = "graphsage" (default) -> real GraphSAGE with message passing;
           uses torch_geometric.SAGEConv if importable, else a
           hand-written mean-aggregator fallback (graphsage_model.py
           module docstring). Both expose identical forward(x,edge_index).
    kind = "node_only"           -> NodeOnlyNet ablation baseline, no
           message passing at all (edge_index accepted but unused).
    """

    hidden_dims = hidden_dims or getattr(cfg, "GRAPHSAGE_HIDDEN_DIMS", (32, 16))
    device = get_device()

    if kind == "node_only":
        model = NodeOnlyNet(in_dim, hidden_dims)
        backend = "node-only MLP (no message passing, ablation baseline)"
    elif _HAS_PYG:
        model = _PyGGraphSAGENet(in_dim, hidden_dims)
        backend = "torch_geometric.SAGEConv"
    else:
        model = _ManualGraphSAGENet(in_dim, hidden_dims)
        backend = "manual (pure PyTorch, no torch_geometric found)"

    model.to(device)

    return model, device, backend


def save_checkpoint(model, path, in_dim, hidden_dims, kind):
    """Save a frozen checkpoint with everything needed to rebuild the
    model architecture later (train_graphsage_offline.py writes this;
    ch_selector.py reads it back for inference-only use)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dims": tuple(hidden_dims),
            "kind": kind,
            "backend": "torch_geometric.SAGEConv" if _HAS_PYG else
                       ("node_only" if kind == "node_only" else "manual"),
        },
        path,
    )


def load_checkpoint(path, map_location=None):
    """
    Rebuild a model exactly as it was saved and load frozen weights.
    Returns (model, device, backend) -- model is already in eval() mode.
    Raises FileNotFoundError if no checkpoint exists at `path` (callers
    should catch this and fall back to pure AHP-TOPSIS rather than
    silently using a randomly-initialized, untrained model).
    """
    device = get_device()
    ckpt = torch.load(path, map_location=map_location or device)

    model, _, backend = build_model(
        in_dim=ckpt["in_dim"],
        hidden_dims=ckpt["hidden_dims"],
        kind=ckpt.get("kind", "graphsage"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, device, backend


def build_fully_connected_edge_index(num_nodes, device):
    """
    All-pairs directed edges (both directions), excluding self-loops.
    Fine at cluster scale (paper's clusters are single-hop / small).
    """

    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)

    return torch.tensor([src, dst], dtype=torch.long, device=device)
