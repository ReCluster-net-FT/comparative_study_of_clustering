"""
protocols.py
=============

Implements Cluster Head selection for

1. OCHSAT
2. ESCA
3. IGHND
4. MOFCA
"""

import math

import numpy as np

import config as cfg

from energy_model import distance
from radio_graph import neighbors_within_radius

from ahp_topsis import (
    select_ch_ahp_topsis,
    topsis_rank
)


# -------------------------------------------------------
# Helper
# -------------------------------------------------------

# Sentinel "unreachable" distance used when a node has zero neighbours
# within the communication radius (radius-aware mode only). Keeps
# centrality well-defined (very low, not undefined/zero-division) for an
# isolated node instead of pretending it can still hear the whole cluster.
_UNREACHABLE_DISTANCE = math.sqrt(2) * max(
    getattr(cfg, "FIELD_X", 100), getattr(cfg, "FIELD_Y", 100)
) * 2


def average_distance(node, members, radius=None):
    """
    Average distance from one node to its cluster neighbours.

    If `radius` is None (legacy behaviour): average distance to EVERY
    other node in the cluster (assumes full connectivity).

    If `radius` is given: average distance only over the node's REAL
    in-range neighbours (Euclidean distance <= radius) - i.e. not every
    node in the cluster is treated as reachable any more. A node with no
    in-range neighbours gets a large sentinel distance (very low
    centrality) rather than an undefined/zero value.
    """

    if radius is None:
        others = [m for m in members if m.node_id != node.node_id]
    else:
        others = neighbors_within_radius(node, members, radius)

    if len(others) == 0:
        return 0 if radius is None else _UNREACHABLE_DISTANCE

    return np.mean([
        distance(node.position, m.position)
        for m in others
    ])


def build_features(cluster_nodes, bs_position, radius=None):
    """
    Build RE, Centrality, DBS and (real) Degree for every node.

    `radius`: communication radius (metres). When provided, Centrality
    and Degree are computed only over each node's real in-range
    neighbours (Euclidean distance <= radius) rather than assuming every
    node in the cluster can hear every other node.
    """

    features = []

    for node in cluster_nodes:

        neighbors = (
            neighbors_within_radius(node, cluster_nodes, radius)
            if radius is not None
            else [m for m in cluster_nodes if m.node_id != node.node_id]
        )

        avg_distance = average_distance(
            node,
            cluster_nodes,
            radius=radius
        )

        centrality = 1 / (avg_distance + 1e-6)

        dbs = distance(
            node.position,
            bs_position
        )

        features.append({

            "node": node,

            "RE": node.energy,

            "Centrality": centrality,

            "DBS": dbs,

            "Degree": len(neighbors),

        })

    return features


# -------------------------------------------------------
# OCHSAT
# -------------------------------------------------------

def select_ch_ochsat(
    cluster_nodes,
    bs_position,
    weights,
    graphsage_selector=None,
    radius=None,
    current_round=0,
):
    """
    Select the OCHSAT Cluster Head.

    If `graphsage_selector` is provided (and enabled), the decision
    is a blend of AHP-TOPSIS and a trained GraphSAGE model's score
    (see ch_selector.GraphSAGECHSelector). Otherwise this behaves
    exactly as the original paper-exact AHP-TOPSIS selection.

    `radius`: communication radius (metres) used to restrict Centrality/
    Degree to real in-range neighbours (see build_features). None keeps
    the legacy fully-connected assumption.

    Returns
    -------
    Node
        The selected Cluster Head.
    Node or None
        The GraphSAGE model's own top pick (before blending), for
        visualization/logging. None if GraphSAGE is not active.
    """

    if graphsage_selector is not None and graphsage_selector.enabled:
        chosen_node, predicted_node = graphsage_selector.select(
            cluster_nodes, bs_position, weights, radius=radius,
            current_round=current_round,
        )
        return chosen_node, predicted_node

    features = build_features(
        cluster_nodes,
        bs_position,
        radius=radius
    )

    index = select_ch_ahp_topsis(
        features,
        weights
    )

    return features[index]["node"], None


# -------------------------------------------------------
# ESCA
# -------------------------------------------------------

def select_ch_esca(cluster_nodes):

    return max(

        cluster_nodes,

        key=lambda node: node.energy

    )


# -------------------------------------------------------
# IGHND
# -------------------------------------------------------

IGHND_WEIGHTS = np.array([0.34,0.33,0.33])


def select_ch_ighnd(
    cluster_nodes,
    bs_position,
    radius=None
):

    features = build_features(
        cluster_nodes,
        bs_position,
        radius=radius
    )

    matrix = np.array([

        [

            f["RE"],

            f["Centrality"],

            f["DBS"]

        ]

        for f in features

    ])

    scores = topsis_rank(

        matrix,

        IGHND_WEIGHTS,

        [True,True,False]

    )

    return features[np.argmax(scores)]["node"]


# -------------------------------------------------------
# MOFCA
# -------------------------------------------------------

def select_ch_mofca(
    cluster_nodes,
    rng
):

    energies = np.array([

        max(node.energy,1e-12)

        for node in cluster_nodes

    ])

    probabilities = energies / energies.sum()

    probabilities = (

        0.7 * probabilities +

        0.3 * (1 / len(cluster_nodes))

    )

    probabilities /= probabilities.sum()

    index = rng.choice(

        len(cluster_nodes),

        p=probabilities

    )

    return cluster_nodes[index]


# -------------------------------------------------------
# Main Interface
# -------------------------------------------------------

def select_cluster_head(
    protocol,
    cluster_nodes,
    bs_position,
    weights,
    rng,
    graphsage_selector=None,
    radius=None,
    current_round=0,
):
    """
    `radius`: communication radius (metres), see build_features(). Only
    OCHSAT and IGHND use it (they're the only protocols that consult
    Centrality/Degree); ESCA and MOFCA are untouched, energy-only
    baselines and ignore it, exactly as before.

    Returns
    -------
    Node
        The selected Cluster Head.
    Node or None
        For OCHSAT with an active GraphSAGE selector: the model's own
        top pick (before blending), used only for visualization/logging.
        None for every other protocol, and for OCHSAT when GraphSAGE is
        disabled.
    """

    if protocol == "OCHSAT":

        return select_ch_ochsat(

            cluster_nodes,

            bs_position,

            weights,

            graphsage_selector,

            radius,

            current_round,

        )

    elif protocol == "ESCA":

        return select_ch_esca(

            cluster_nodes

        ), None

    elif protocol == "IGHND":

        return select_ch_ighnd(

            cluster_nodes,

            bs_position,

            radius,

        ), None

    elif protocol == "MOFCA":

        return select_ch_mofca(

            cluster_nodes,

            rng

        ), None

    else:

        raise ValueError(
            f"Unknown protocol {protocol}"
        )


# -------------------------------------------------------
# Rotation
# -------------------------------------------------------

def needs_rotation(
    protocol,
    cluster_head,
    current_round,
    last_rotation
):

    if cluster_head is None:

        return True

    if not cluster_head.alive:

        return True

    if protocol == "MOFCA":

        return (

            current_round -

            last_rotation

        ) >= 20

    # Use the cluster head's OWN starting energy (identical to
    # cfg.INITIAL_ENERGY in synthetic mode; the real per-node CSV value
    # in dataset mode) rather than the global constant, so rotation
    # thresholds stay correct when nodes don't all start equal.
    return (

        cluster_head.energy

        <

        cfg.ROTATION_THRESHOLD *

        cluster_head.initial_energy

    )

def build_feature_table(
    cluster_nodes,
    bs_position,
    weights=None,
    radius=None,
):
    """
    Build a printable table containing all
    Cluster Head selection parameters.

    Used only for debugging and visualization.
    """

    features = build_features(
        cluster_nodes,
        bs_position,
        radius=radius
    )

    table = []

    for f in features:

        table.append({

            "Node ID": f["node"].node_id,

            "Residual Energy": round(
                f["RE"], 4
            ),

            "Centrality": round(
                f["Centrality"], 6
            ),

            "Distance to BS": round(
                f["DBS"], 2
            )

        })

    return table
