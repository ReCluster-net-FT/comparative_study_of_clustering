"""
radio_graph.py
===============

Turns "which nodes can actually hear each other" from an assumption
(every node in a cluster hears every other node - a fully-connected
graph) into something computed from real geometry:

    rho = N / A                  node density
    R   = sqrt(k / (pi * rho))   radius giving ~k neighbours/node

Two nodes are neighbours only if their Euclidean distance is <= R.
This is used in two places:

1. protocols.build_features() - "Centrality" and node "degree" are now
   computed only over each node's real in-range neighbours, not the
   whole cluster.
2. ch_selector.GraphSAGECHSelector - the graph fed to the GraphSAGE
   model's message passing is this real (sparse, not fully connected)
   radio graph, restricted to whichever cluster is being scored.

Everything here is pure geometry / numpy - no torch dependency - so it
works identically whether or not GraphSAGE/torch is installed.
"""

import math

import numpy as np

from energy_model import distance


def compute_communication_radius(node_count, field_x, field_y, target_degree):
    """
    R = sqrt(k / (pi * rho)),  rho = N / A,  A = field_x * field_y

    Parameters
    ----------
    node_count : int
        N - number of nodes actually deployed in this network.
    field_x, field_y : float
        Deployment field dimensions (metres).
    target_degree : float
        k - desired average number of neighbours per node.

    Returns
    -------
    float
        Communication radius R (metres). Falls back to the larger field
        dimension (i.e. "everyone in range") if node_count <= 1, since
        density is undefined for a single node.
    """

    area = float(field_x) * float(field_y)

    if node_count <= 1 or area <= 0:
        return max(field_x, field_y)

    rho = node_count / area

    radius = math.sqrt(target_degree / (math.pi * rho))

    # Never exceed the field diagonal - a radius bigger than that is
    # meaningless (would already connect everyone).
    diagonal = math.sqrt(field_x ** 2 + field_y ** 2)

    return min(radius, diagonal)


def neighbors_within_radius(node, candidates, radius):
    """
    Every node in `candidates` (excluding `node` itself) whose Euclidean
    distance to `node` is <= radius.

    Returns
    -------
    list[Node]
    """

    return [
        other
        for other in candidates
        if other.node_id != node.node_id
        and distance(node.position, other.position) <= radius
    ]


def build_adjacency(cluster_nodes, radius):
    """
    Build the radio-range adjacency for one cluster.

    Returns
    -------
    dict[node_id -> list[Node]]
        Real in-range neighbours for every node in the cluster (can be
        an empty list for an isolated node - that is the whole point:
        NOT every node is connected to every other node any more).
    """

    adjacency = {}

    for node in cluster_nodes:
        adjacency[node.node_id] = neighbors_within_radius(
            node, cluster_nodes, radius
        )

    return adjacency


def build_edge_index_from_positions(cluster_nodes, radius, device=None):
    """
    Build a torch [2, E] edge_index (both directions, no self-loops) for
    `cluster_nodes`, connecting node i and node j iff their Euclidean
    distance is <= radius. Node indices in the returned edge_index refer
    to POSITIONS IN THE `cluster_nodes` LIST (0..len-1), matching the
    row order used to build the feature tensor `x` in ch_selector.py -
    NOT node.node_id.

    Requires torch (only imported lazily so this module has no hard
    torch dependency for the non-GraphSAGE code paths that only need
    build_adjacency/neighbors_within_radius/compute_communication_radius).
    """

    import torch

    n = len(cluster_nodes)

    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    positions = np.array([node.position for node in cluster_nodes])

    src, dst = [], []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.dist(positions[i], positions[j])
            if d <= radius:
                src.append(i)
                dst.append(j)

    return torch.tensor([src, dst], dtype=torch.long, device=device)


def degree_within_radius(cluster_nodes, radius):
    """
    Real node degree (count of in-range neighbours) for every node in
    `cluster_nodes`, in the SAME order as `cluster_nodes` - used as the
    GraphSAGE "degree" feature instead of the old constant
    `len(features) - 1` (which silently assumed full connectivity).

    Returns
    -------
    numpy.ndarray, shape (len(cluster_nodes),)
    """

    positions = np.array([node.position for node in cluster_nodes])
    n = len(cluster_nodes)

    degrees = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if math.dist(positions[i], positions[j]) <= radius:
                degrees[i] += 1

    return degrees
