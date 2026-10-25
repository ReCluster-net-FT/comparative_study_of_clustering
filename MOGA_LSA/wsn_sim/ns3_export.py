"""Write a Network + chosen cluster heads + LSA routing as the roster CSV
ns3_wsn_sim.py (top-level, cppyy-driven ns-3 script) consumes to produce a
NetAnim trace of the topology.
"""
from __future__ import annotations

import csv

# ns-3's WifiRadioEnergyModel also draws energy continuously while idle/CCA-busy
# listening, on top of TX/RX -- much higher overhead than the round-based
# engine's discrete per-transmission accounting. The dataset's real per-node
# energy (~0.3-0.8J after rescaling) would deplete mid-animation under that
# model. This run is topology visualization only (simulate.run() is the
# source of truth for actual energy/lifetime numbers), so scale up headroom
# here; relative ordering between nodes is preserved.
NS3_ENERGY_HEADROOM = 20.0


def write_roster(network, chs, next_hop: dict, out_path: str):
    chs = set(int(c) for c in chs)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "z", "initial_energy", "is_cluster_head", "relay_to"])
        for i in range(network.n_nodes):
            is_ch = i in chs
            relay_to = int(next_hop.get(i, -1)) if is_ch else -1
            w.writerow([i, float(network.pos[i, 0]), float(network.pos[i, 1]), 0.0,
                        float(network.energy[i]) * NS3_ENERGY_HEADROOM, int(is_ch), relay_to])
    print(f"wrote {network.n_nodes} nodes ({len(chs)} cluster heads) to {out_path}")
