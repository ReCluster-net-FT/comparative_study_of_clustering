"""
clustering.py

Implements the clustering phase of the OCHSAT paper.

Algorithm 1:
    Find the optimal number of clusters using Silhouette Score.

Algorithm 2:
    Perform K-Means clustering.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import config as cfg

def find_optimal_k(
    positions,
    k_min=cfg.MIN_CLUSTERS,
    k_max=cfg.MAX_CLUSTERS,
    random_state=42
):
    """
    Finds the best number of clusters using
    the Silhouette Score.

    Returns
    -------
    (best_k, best_score)
    """

    n = len(positions)

    # Cannot have more clusters than nodes - 1
    k_max = min(k_max, n - 1)

    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(positions)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(
            positions,
            labels
        )

        if score > best_score:

            best_score = score
            best_k = k

    return best_k, best_score

def run_kmeans(
    positions,
    k,
    random_state=42
):
    """
    Runs K-Means clustering.

    Returns
    -------
    labels
    centroids
    """

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(positions)

    return labels, model.cluster_centers_


def assign_clusters(
    network,
    labels
):
    """
    Assign cluster IDs to every node.
    """

    network.clusters.clear()

    for node, label in zip(network.nodes, labels):

        node.cluster_id = int(label)

        if label not in network.clusters:

            network.clusters[label] = []

        network.clusters[label].append(node)


def cluster_network(
    network,
    random_state=42
):
    """
    Complete clustering pipeline.

    Returns
    -------
    k
    silhouette score
    centroids
    """

    positions = np.array(
        [node.position for node in network.nodes]
    )

    best_k, score = find_optimal_k(
        positions,
        random_state=random_state
    )

    labels, centroids = run_kmeans(
        positions,
        best_k,
        random_state=random_state
    )

    assign_clusters(
        network,
        labels
    )

    return best_k, score, centroids



