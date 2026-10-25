"""
ahp_topsis.py
=============

Implements the AHP-TOPSIS based Cluster Head Selection
described in Section 4.2 of the OCHSAT paper.

Criteria Order:
1. Residual Energy (Benefit)
2. Centrality (Benefit)
3. Distance to Base Station (Cost)
"""

import numpy as np
import config as cfg


# Saaty's Random Index values
_RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
}


def compute_ahp_weights(pairwise_matrix=None):
    """
    Compute AHP criteria weights.

    Parameters
    ----------
    pairwise_matrix : list[list] or None

    Returns
    -------
    weights : numpy.ndarray
    consistency_ratio : float
    """

    if pairwise_matrix is None:
        pairwise_matrix = cfg.AHP_PAIRWISE_MATRIX

    P = np.array(pairwise_matrix, dtype=float)

    n = P.shape[0]

    # ----------------------------
    # Step 1 : Normalize columns
    # ----------------------------
    column_sum = P.sum(axis=0)

    normalized = P / column_sum

    # ----------------------------
    # Step 2 : Row averages
    # ----------------------------
    weights = normalized.mean(axis=1)

    weights = weights / weights.sum()

    # ----------------------------
    # Step 3 : Consistency Ratio
    # ----------------------------
    weighted_sum = P @ weights

    lambda_max = np.mean(weighted_sum / weights)

    CI = (lambda_max - n) / (n - 1)

    RI = _RANDOM_INDEX.get(n, 1.49)

    if RI == 0:
        CR = 0
    else:
        CR = CI / RI

    return weights, CR


def topsis_rank(decision_matrix, weights, benefit_flags):
    """
    Perform TOPSIS ranking.

    Parameters
    ----------
    decision_matrix : ndarray
        Shape = (candidates, criteria)

    weights : ndarray

    benefit_flags : list
        True  -> Benefit criterion
        False -> Cost criterion

    Returns
    -------
    closeness_scores
    """

    X = np.array(decision_matrix, dtype=float)

    if X.shape[0] == 1:
        return np.array([1.0])

    # ----------------------------
    # Step 1 : Vector normalization
    # ----------------------------
    denominator = np.sqrt((X ** 2).sum(axis=0))

    denominator[denominator == 0] = 1e-12

    R = X / denominator

    # ----------------------------
    # Step 2 : Weighted matrix
    # ----------------------------
    V = R * weights

    # ----------------------------
    # Step 3 : Ideal solutions
    # ----------------------------
    ideal_best = np.zeros(X.shape[1])

    ideal_worst = np.zeros(X.shape[1])

    for j, benefit in enumerate(benefit_flags):

        if benefit:

            ideal_best[j] = np.max(V[:, j])

            ideal_worst[j] = np.min(V[:, j])

        else:

            ideal_best[j] = np.min(V[:, j])

            ideal_worst[j] = np.max(V[:, j])

    # ----------------------------
    # Step 4 : Separation measures
    # ----------------------------
    S_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))

    S_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # ----------------------------
    # Step 5 : Closeness coefficient
    # ----------------------------
    scores = S_minus / (S_plus + S_minus + 1e-12)

    return scores


def select_ch_ahp_topsis(candidate_features, weights):
    """
    Select Cluster Head using AHP-TOPSIS.

    Parameters
    ----------
    candidate_features : list

        Example

        [
            {
                "RE":0.45,
                "Centrality":0.13,
                "DBS":52.3
            },
            ...
        ]

    weights : ndarray

    Returns
    -------
    int
        Index of selected Cluster Head.
    """

    if len(candidate_features) == 1:
        return 0

    decision_matrix = np.array([

        [
            c["RE"],
            c["Centrality"],
            c["DBS"]
        ]

        for c in candidate_features

    ])

    scores = topsis_rank(

        decision_matrix,

        weights,

        benefit_flags=[True, True, False]

    )

    return int(np.argmax(scores))
