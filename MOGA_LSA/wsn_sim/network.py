"""WSN network model: nodes, first-order radio energy model, distances.

Table 1 in the paper leaves most values as "[Specify ...]" placeholders, so
concrete numbers below are the standard first-order radio model constants
used across the LEACH/MOGA-GSA/PSO literature this paper compares against
(Heinzelman et al.). Swap them out if a specific dataset later fixes them.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class RadioParams:
    e_elec: float = 50e-9        # J/bit, TX/RX electronics
    e_fs: float = 10e-12         # J/bit/m^2, free-space amplifier
    e_mp: float = 0.0013e-12     # J/bit/m^4, multipath amplifier
    e_da: float = 5e-9           # J/bit/signal, data aggregation at CH
    packet_size: int = 512       # bits, sensed data packet
    msg_size: int = 1024         # bits, CH-to-CH / CH-to-sink message
    hop_delay: float = 10e-6     # s, fixed per-hop propagation+MAC delay

    @property
    def d0(self) -> float:
        return np.sqrt(self.e_fs / self.e_mp)

    def tx_energy(self, bits: int, dist: float) -> float:
        amp = self.e_fs * dist ** 2 if dist < self.d0 else self.e_mp * dist ** 4
        return bits * self.e_elec + bits * amp

    def rx_energy(self, bits: int) -> float:
        return bits * self.e_elec


@dataclass
class Network:
    """Node positions are the Euclidean-space default. Set `cost_matrix`
    (+ `sink_idx`) to override every distance computation with a
    graph-native cost instead (e.g. hop count on a real, non-spatial graph
    like PPI) — see graph_data.py. `pos` is then optional and, if present,
    only used for visualization (NetAnim), never for cost.
    """
    n_nodes: int = 100
    area: float = 100.0
    sink: tuple[float, float] = (50.0, 50.0)
    init_energy: float = 0.5      # J
    ch_prob: float = 0.1          # target fraction of alive nodes as CH
    seed: int = 0
    radio: RadioParams = field(default_factory=RadioParams)
    cost_matrix: np.ndarray | None = None   # (N,N) graph-native pairwise cost, overrides Euclidean
    sink_idx: int | None = None             # index of the sink among the N nodes, when cost_matrix is set
    node_features: np.ndarray | None = None  # (N,F) real external features (e.g. PPI), for graphsage_ch

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.pos = rng.uniform(0, self.area, size=(self.n_nodes, 2))
        self.energy = np.full(self.n_nodes, self.init_energy)
        self.alive = np.ones(self.n_nodes, dtype=bool)

    def pair_cost(self, a, b) -> float:
        if self.cost_matrix is not None:
            return self.cost_matrix[a, b]
        return np.linalg.norm(self.pos[a] - self.pos[b])

    def dist_to_sink(self, idx) -> np.ndarray:
        if self.cost_matrix is not None:
            return self.cost_matrix[idx, self.sink_idx]
        return np.linalg.norm(self.pos[idx] - np.array(self.sink), axis=-1)

    def dist_matrix(self) -> np.ndarray:
        if self.cost_matrix is not None:
            return self.cost_matrix
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def alive_idx(self) -> np.ndarray:
        return np.nonzero(self.alive)[0]

    def kill_dead(self):
        self.alive &= self.energy > 0
