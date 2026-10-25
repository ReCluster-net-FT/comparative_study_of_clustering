"""
dataset_loader.py
===================

Loads real sensor-node records from the Kaggle "Wireless Sensor Network
Dataset" (WSN_Dataset.csv, ziya07) and turns a sample of them into
`energy_model.Node` objects that the rest of the simulator (clustering,
protocols, GraphSAGE, ...) can use exactly like the synthetic
randomly-generated nodes.

Expected columns (from the dataset's own documentation)
---------------------------------------------------------
Node_ID, Timestamp, X_Coordinate, Y_Coordinate, Z_Coordinate,
Initial_Energy, Residual_Energy, Transmission_Power, Signal_Strength,
Noise_Level, Energy_Consumption, Packet_Loss_Rate, Network_Lifetime,
Optimization_Algorithm, Adaptive_Learning_Rate, Temperature, Humidity,
Detection_Accuracy

Each ROW is treated as one independent, static sensor-node record (the
dataset is a flat table of 10,000 simulated node snapshots, not a
single fixed set of nodes evolving over time - Node_ID/Timestamp just
increment row by row). To reproduce the paper's scenario structure
(100 or 200 nodes per scenario) we draw a random sample of `node_count`
rows - a different, but seed-reproducible, subset per simulation seed.

Only X_Coordinate, Y_Coordinate and Initial_Energy actually feed the
radio/energy model (2D positions + starting energy, matching the rest
of the codebase). Every other column is kept on the node under
`node.extra[...]` purely for reference/analysis - it is not used by
the simulation logic.
"""

import os

import numpy as np
import pandas as pd

import config as cfg
from energy_model import Node

REQUIRED_COLUMNS = [
    "Node_ID",
    "X_Coordinate",
    "Y_Coordinate",
    "Initial_Energy",
]

OPTIONAL_COLUMNS = [
    "Timestamp",
    "Z_Coordinate",
    "Residual_Energy",
    "Transmission_Power",
    "Signal_Strength",
    "Noise_Level",
    "Energy_Consumption",
    "Packet_Loss_Rate",
    "Network_Lifetime",
    "Optimization_Algorithm",
    "Adaptive_Learning_Rate",
    "Temperature",
    "Humidity",
    "Detection_Accuracy",
]


def load_dataset(path=None):
    """
    Read WSN_Dataset.csv into a pandas DataFrame and validate that the
    columns the simulator actually needs are present.

    Raises
    ------
    FileNotFoundError
        With a clear message if `path` doesn't exist (the most common
        mistake: config.DATASET_PATH still points at the placeholder).
    ValueError
        If a required column is missing.
    """

    path = path or cfg.DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Set config.DATASET_PATH to "
            f"the actual location of your downloaded WSN_Dataset.csv "
            f"(Kaggle: ziya07/wireless-sensor-network-dataset)."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at '{path}' is missing required column(s): {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}."
        )

    return df


def sample_rows(df, node_count, rng):
    """
    Deterministically sample `node_count` rows (without replacement)
    using the network's own seeded numpy Generator `rng`, so the same
    simulation seed always draws the same subset of real nodes.

    Raises
    ------
    ValueError
        If the dataset has fewer rows than requested nodes.
    """

    if len(df) < node_count:
        raise ValueError(
            f"Requested {node_count} nodes but the dataset only has "
            f"{len(df)} rows."
        )

    indices = rng.choice(len(df), size=node_count, replace=False)

    return df.iloc[indices].reset_index(drop=True)


def build_nodes_from_dataframe(df_subset, energy_scale):
    """
    Convert a DataFrame of sampled rows into a list of `Node` objects,
    renumbered 0..len-1 (so the rest of the codebase, which assumes
    contiguous node_ids, keeps working unchanged). The dataset's own
    Node_ID/Timestamp/etc. are preserved under node.extra for reference.

    Parameters
    ----------
    df_subset : pandas.DataFrame
        Output of sample_rows().
    energy_scale : float
        Multiplier applied to Initial_Energy. Must already be resolved
        to a concrete number by the caller (build_nodes()) - this
        function does NOT auto-calibrate, since it only sees the sampled
        subset, not the whole dataset the calibration should be based on.

    Returns
    -------
    list[Node]
    """

    nodes = []

    for new_id, row in enumerate(df_subset.itertuples(index=False)):
        row_dict = row._asdict()

        x = float(row_dict["X_Coordinate"])
        y = float(row_dict["Y_Coordinate"])

        initial_energy = float(row_dict["Initial_Energy"]) * energy_scale
        if initial_energy <= 0:
            # Guard against a degenerate/zero-energy row killing the
            # node before round 0.
            initial_energy = 1e-6

        extra = {
            k: v
            for k, v in row_dict.items()
            if k not in ("X_Coordinate", "Y_Coordinate", "Initial_Energy")
        }
        extra["source_node_id"] = row_dict.get("Node_ID")

        nodes.append(
            Node(
                node_id=new_id,
                x=x,
                y=y,
                energy=initial_energy,
                initial_energy=initial_energy,
                extra=extra,
            )
        )

    return nodes


def dataset_field_extent(df):
    """
    Min/max X and Y actually present in the (whole, unsampled) dataset -
    useful for sanity-checking against config.FIELD_X/FIELD_Y, which is
    what the communication-radius density formula assumes as the
    deployment area.

    Returns
    -------
    (x_min, x_max, y_min, y_max)
    """

    return (
        float(df["X_Coordinate"].min()),
        float(df["X_Coordinate"].max()),
        float(df["Y_Coordinate"].min()),
        float(df["Y_Coordinate"].max()),
    )


# Cache so the auto-calibration message (and the calibration factor
# itself) is computed/printed once per dataset path, not on every single
# Network() construction (which would otherwise print the same message
# dozens of times across scenarios x seeds x with/without-GraphSAGE).
_calibration_cache = {}


def resolve_energy_scale(df, path, energy_scale=None):
    """
    Returns a concrete numeric energy_scale.

    If `energy_scale` (or cfg.DATASET_ENERGY_SCALE) is None: auto-
    calibrate from the WHOLE dataset (not just whatever subset happens
    to be sampled for one scenario/seed) so the mean Initial_Energy,
    after scaling, equals cfg.INITIAL_ENERGY. Cached per path so it's
    only computed/printed once.
    """

    if energy_scale is None:
        energy_scale = getattr(cfg, "DATASET_ENERGY_SCALE", None)

    if energy_scale is not None:
        return energy_scale

    cache_key = path or cfg.DATASET_PATH
    if cache_key in _calibration_cache:
        return _calibration_cache[cache_key]

    raw_mean = float(df["Initial_Energy"].mean())
    if raw_mean <= 0:
        raise ValueError(
            f"Dataset's Initial_Energy column has a non-positive mean "
            f"({raw_mean}); cannot auto-calibrate. Set "
            f"config.DATASET_ENERGY_SCALE explicitly instead."
        )

    factor = cfg.INITIAL_ENERGY / raw_mean

    print(
        f"[dataset_loader] Auto-calibrated DATASET_ENERGY_SCALE={factor:.6g} "
        f"(dataset mean Initial_Energy={raw_mean:.4g} -> scaled mean="
        f"{cfg.INITIAL_ENERGY:.4g} J, matching cfg.INITIAL_ENERGY). "
        f"Set config.DATASET_ENERGY_SCALE to a specific number instead of "
        f"None if you want to override this."
    )

    _calibration_cache[cache_key] = factor
    return factor


def build_nodes(node_count, rng, path=None, energy_scale=None):
    """
    Convenience one-shot entry point used by Network.generate_nodes():
    load the CSV, resolve/auto-calibrate the energy scale from the FULL
    dataset, sample `node_count` rows with `rng`, and return Node
    objects. Also prints a one-time sanity warning if the CSV's real
    coordinate extent doesn't match config.FIELD_X/FIELD_Y (since the
    communication-radius formula assumes the field size in config.py IS
    the dataset's deployment area).
    """

    df = load_dataset(path)

    x_min, x_max, y_min, y_max = dataset_field_extent(df)
    if x_max > cfg.FIELD_X * 1.05 or y_max > cfg.FIELD_Y * 1.05:
        print(
            f"[dataset_loader] WARNING: dataset X/Y extent is "
            f"[{x_min:.1f},{x_max:.1f}] x [{y_min:.1f},{y_max:.1f}] but "
            f"config.FIELD_X/FIELD_Y = {cfg.FIELD_X}/{cfg.FIELD_Y}. "
            f"The communication-radius density formula uses FIELD_X*FIELD_Y "
            f"as the deployment area - update config.py if these should "
            f"match your actual dataset."
        )

    resolved_scale = resolve_energy_scale(df, path, energy_scale=energy_scale)

    subset = sample_rows(df, node_count, rng)

    return build_nodes_from_dataframe(subset, energy_scale=resolved_scale)
