"""Load the Kaggle WSN dataset (ziya07/wireless-sensor-network-dataset) into
a Network: a fixed-size sample of nodes gets its real (X, Y) position,
Initial_Energy, and the rest of its sensor-reading columns are carried as
node_features for graphsage_embed.py -- mirrors real_data.py's pattern for
the Intel Lab dataset, but for this project's actual dataset.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .network import Network

# override with the WSN_DATASET_CSV env var so the repo runs on any machine
DEFAULT_CSV = os.environ.get(
    "WSN_DATASET_CSV",
    "/home/siddhartha/Final Year Project/archive/WSN_Dataset.csv",
)

FEATURE_COLS = [
    "Residual_Energy", "Transmission_Power", "Signal_Strength", "Noise_Level",
    "Energy_Consumption", "Packet_Loss_Rate", "Network_Lifetime",
    "Adaptive_Learning_Rate", "Temperature", "Humidity", "Detection_Accuracy",
]


def build_network(csv_path: str = DEFAULT_CSV, n_nodes: int = 100, init_energy: float = 0.5,
                   seed: int = 0) -> Network:
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(df), size=n_nodes, replace=False)
    rows = df.iloc[sample].reset_index(drop=True)

    net = Network(n_nodes=n_nodes, seed=seed)
    net.pos = rows[["X_Coordinate", "Y_Coordinate"]].to_numpy()
    xs, ys = net.pos[:, 0], net.pos[:, 1]
    net.area = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
    net.sink = (float(xs.mean()), float(ys.mean()))

    # real per-node initial energy from the dataset, rescaled to keep the
    # mean at init_energy so radio energy constants (tuned for ~0.5J nodes)
    # stay meaningful regardless of the dataset's own Initial_Energy units
    raw_energy = rows["Initial_Energy"].to_numpy()
    net.init_energy = init_energy
    net.energy = init_energy * (raw_energy / raw_energy.mean())

    net.node_features = rows[FEATURE_COLS].to_numpy()
    return net


if __name__ == "__main__":
    net = build_network(n_nodes=30, seed=1)
    assert net.pos.shape == (30, 2)
    assert net.energy.shape == (30,)
    assert net.node_features.shape == (30, len(FEATURE_COLS))
    assert np.isclose(net.energy.mean(), net.init_energy)
    print(f"wsn_dataset ok: {net.n_nodes} nodes, area={net.area:.1f}, sink={net.sink}")
