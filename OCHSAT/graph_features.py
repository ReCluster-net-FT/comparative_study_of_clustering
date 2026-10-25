"""
graph_features.py
==================

Single source of truth for turning `protocols.build_features()` output
into the numeric feature matrix GraphSAGE (and the node-only ablation
model) actually consume.

Why this exists as its own module
----------------------------------
The old ch_selector.py min-max normalized every feature PER CLUSTER, PER
CALL. That is fine for a model that trains online on that same
distribution, but it is wrong for an OFFLINE-trained, FROZEN model: a
frozen model must see features on the exact same scale at inference time
that it saw during training, otherwise "0.7" can mean something
different in every cluster it is applied to. So normalization
statistics (mean/std per feature) are:

  1. computed ONCE from the OFFLINE TRAINING dataset only
     (fit_normalization, called from train_graphsage_offline.py),
  2. saved to disk (config.FEATURE_NORM_CONFIG),
  3. loaded and applied verbatim, unchanged, at inference time
     (ch_selector.py) and during offline data generation for the
     validation/test splits.

Feature order is fixed by config.GRAPHSAGE_FEATURES =
["RE", "Centrality", "DBS", "Degree"]. RE is expressed as a ratio
(node.energy / node.initial_energy) rather than raw Joules so it stays
comparable across synthetic vs dataset-driven runs where nodes don't
all start with the same initial energy.
"""

import json
import os

import numpy as np


FEATURE_ORDER = ["RE", "Centrality", "DBS", "Degree"]


def raw_feature_matrix(features):
    """
    features: list of dicts as returned by protocols.build_features(),
    each with keys "node", "RE", "Centrality", "DBS", "Degree".

    Returns
    -------
    numpy.ndarray, shape (len(features), 4), columns in FEATURE_ORDER.
    RE is expressed as energy/initial_energy (a 0..1 ratio), NOT raw
    Joules, so it is meaningful across different initial-energy regimes.
    """
    rows = []
    for f in features:
        node = f["node"]
        init_e = getattr(node, "initial_energy", None) or 1e-9
        re_ratio = max(0.0, min(1.0, f["RE"] / init_e))
        rows.append([re_ratio, f["Centrality"], f["DBS"], f["Degree"]])
    return np.asarray(rows, dtype=float)


class FeatureNormalizer:
    """Fixed (frozen) z-score normalization: (x - mean) / std, with
    mean/std computed ONCE offline and never updated again."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.std[self.std < 1e-8] = 1e-8

    def apply(self, raw_matrix):
        return (raw_matrix - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "feature_order": FEATURE_ORDER}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        if d.get("feature_order") != FEATURE_ORDER:
            raise ValueError(
                f"feature_norm.json was fit with a different feature "
                f"order ({d.get('feature_order')}) than the current "
                f"FEATURE_ORDER ({FEATURE_ORDER}); regenerate the "
                f"offline dataset and retrain."
            )
        return cls(d["mean"], d["std"])

    @classmethod
    def fit(cls, raw_matrices):
        """raw_matrices: list of (n_i, 4) arrays from many training
        snapshots, concatenated to compute one global mean/std per
        feature column."""
        stacked = np.concatenate(raw_matrices, axis=0)
        return cls(stacked.mean(axis=0), stacked.std(axis=0))
