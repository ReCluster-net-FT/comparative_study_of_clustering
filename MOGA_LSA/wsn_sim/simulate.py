"""Round-based stock MOGA+LSA simulation loop (paper Fig. 1)."""
from __future__ import annotations

import numpy as np

from .network import Network
from . import moga, lsa


def run(
    network: Network,
    rounds: int = 3500,
    seed: int = 0,
    ch_selector=moga.select_cluster_heads,
    router=lsa.find_routes,
    on_round=None,
    connectivity: dict | None = None,
) -> dict:
    """Run `rounds` rounds; return per-round metric arrays.

    If given, on_round(r, {"chs", "edges", "alive"}) is called each round
    with the chosen CHs, the (src, dst) transmission edges used that round
    (CM->CH and CH-chain->sink), and the alive node set — for visualization
    (see netanim_export.py), without changing the simulation itself.

    `connectivity`, if given, maps (src_row, dst_row) -> measured delivery
    probability and is used instead of assuming every in-range transmission
    succeeds — a hop is dropped (energy still spent — the radio still
    transmitted) with probability 1 - connectivity[(src, dst)]. Hops to the
    sink aren't in the table and stay deterministic.
    """
    rng = np.random.default_rng(seed)
    radio = network.radio

    def hop_ok(src: int, dst: int) -> bool:
        if connectivity is None or dst == lsa.SINK:
            return True
        return rng.random() < connectivity.get((src, dst), 0.0)

    hist = {k: [] for k in ("avg_energy", "avg_residual", "alive", "dropped_total",
                             "delivery_rate", "avg_delay", "throughput", "e2e_delivery_rate")}
    dropped_total = 0
    sent_total = 0

    for r in range(rounds):
        network.kill_dead()
        alive_idx = network.alive_idx()
        if len(alive_idx) == 0:
            break

        chs = ch_selector(network, rng=rng)
        cm_idx = np.setdiff1d(alive_idx, chs)
        edges = []

        # CM -> CH (data collection)
        cm_hops = []  # (cm, ch, reached_ch) for the end-to-end delivery metric below
        if len(cm_idx):
            if network.cost_matrix is not None:
                d = network.cost_matrix[np.ix_(cm_idx, chs)]
            else:
                d = np.linalg.norm(network.pos[cm_idx][:, None, :] - network.pos[chs][None, :, :], axis=-1)
            nearest = d.argmin(axis=1)
            for cm, ci in zip(cm_idx, nearest):
                dist = d[np.where(cm_idx == cm)[0][0], ci]
                network.energy[cm] -= radio.tx_energy(radio.packet_size, dist)
                reached = hop_ok(int(cm), int(chs[ci]))
                if reached:
                    network.energy[chs[ci]] -= radio.rx_energy(radio.packet_size) + radio.e_da * radio.packet_size
                    edges.append((int(cm), int(chs[ci])))
                cm_hops.append((int(cm), int(chs[ci]), reached))

        # CH -> ... -> sink (global routing)
        next_hop, _ = router(network, chs, rng=rng)
        delay = energy_used = 0.0
        delivered = 0
        ch_reached_sink = {}
        for ch in chs:
            node, hops = ch, 0
            sent_total += 1
            while node != lsa.SINK and hops < len(chs) + 1:
                nxt = next_hop.get(node, lsa.SINK)
                dist = (network.dist_to_sink([node])[0] if nxt == lsa.SINK
                        else network.pair_cost(node, nxt))
                tx = radio.tx_energy(radio.msg_size, dist)
                if network.energy[node] < tx:
                    break  # can't forward: packet dropped
                network.energy[node] -= tx
                delay += radio.hop_delay
                energy_used += tx
                if not hop_ok(int(node), int(nxt) if nxt != lsa.SINK else -1):
                    break  # real link failed: packet dropped, tx energy already spent
                if nxt != lsa.SINK:
                    network.energy[nxt] -= radio.rx_energy(radio.msg_size)
                    energy_used += radio.rx_energy(radio.msg_size)
                edges.append((int(node), int(nxt) if nxt != lsa.SINK else -1))
                hops += 1
                node = nxt
            ch_reached_sink[int(ch)] = node == lsa.SINK
            if node == lsa.SINK:
                delivered += 1
            else:
                dropped_total += 1

        e2e_delivered = sum(1 for _, ch, reached in cm_hops if reached and ch_reached_sink.get(ch, False))

        network.kill_dead()
        alive_now = network.alive_idx()

        hist["avg_energy"].append(network.init_energy - network.energy[alive_now].mean() if len(alive_now) else network.init_energy)
        hist["avg_residual"].append(network.energy[alive_now].mean() if len(alive_now) else 0.0)
        hist["alive"].append(len(alive_now))
        hist["dropped_total"].append(dropped_total)
        hist["delivery_rate"].append(delivered / len(chs) if len(chs) else 1.0)
        hist["avg_delay"].append(delay / len(chs) if len(chs) else 0.0)
        hist["throughput"].append((sent_total - dropped_total) / sent_total if sent_total else 1.0)
        hist["e2e_delivery_rate"].append(e2e_delivered / len(cm_hops) if cm_hops else 1.0)

        if on_round is not None:
            metrics = {k: v[-1] for k, v in hist.items()}
            on_round(r, {"chs": chs, "edges": edges, "alive": alive_now, "metrics": metrics})

    return {k: np.array(v) for k, v in hist.items()}
