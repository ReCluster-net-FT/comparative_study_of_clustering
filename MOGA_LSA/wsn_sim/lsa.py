"""Lightning Search Algorithm for CH->sink global routing — paper Eq. (2).

A "channel" here is a next-hop assignment for every CH: next_hop[i] is either
another CH or the sink. LSA population = candidate channel assignments
(step-leader "projectiles"). Each round the best-so-far leader is kept
(elitism) and new candidates are produced by exploring around it — the
random/exponential "shot" from the paper's description of LSA exploration.
"""
from __future__ import annotations

import numpy as np

SINK = -1


def _repair_acyclic(next_hop: dict, chs: np.ndarray) -> dict:
    """Force every CH's forwarding chain to terminate at the sink."""
    fixed = dict(next_hop)
    for start in chs:
        seen = {start}
        node = fixed[start]
        while node != SINK:
            if node in seen:
                fixed[start] = SINK  # break cycle: fall back to direct-to-sink
                break
            seen.add(node)
            node = fixed[node]
    return fixed


def _path_cost(next_hop: dict, chs: np.ndarray, network) -> tuple[float, float, float, int]:
    """Return (total_delay, total_energy, delivered_count, total_count)."""
    radio = network.radio
    delay = energy = 0.0
    delivered = 0
    for start in chs:
        node, hops = start, 0
        while node != SINK and hops < len(chs) + 1:
            nxt = next_hop[node]
            d = (network.dist_to_sink([node])[0] if nxt == SINK
                 else network.pair_cost(node, nxt))
            energy += radio.tx_energy(radio.msg_size, d)
            if nxt != SINK:
                energy += radio.rx_energy(radio.msg_size)
            delay += radio.hop_delay
            hops += 1
            node = nxt
        if node == SINK:
            delivered += 1
    return delay, energy, delivered, len(chs)


def _fitness(next_hop: dict, chs: np.ndarray, network) -> float:
    delay, energy, delivered, total = _path_cost(next_hop, chs, network)
    ddr = delivered / total if total else 0.0
    # normalize energy/delay into comparable range so DDR term has real weight
    return (delay * 1e3) + (energy * 1e2) - ddr


def _random_channel(chs: np.ndarray, network, rng) -> dict:
    next_hop = {}
    for i, ch in enumerate(chs):
        options = [c for c in chs if c != ch]
        if rng.random() < 0.3 or not options:
            next_hop[ch] = SINK
        else:
            next_hop[ch] = rng.choice(options)
    return _repair_acyclic(next_hop, chs)


def find_routes(
    network,
    chs: np.ndarray,
    pop_size: int = 15,
    iterations: int = 30,
    rng: np.random.Generator | None = None,
) -> tuple[dict, list[float]]:
    """Return (next_hop mapping, fitness-per-iteration history)."""
    rng = rng or np.random.default_rng()
    if len(chs) == 0:
        return {}, []

    population = [_random_channel(chs, network, rng) for _ in range(pop_size)]
    history = []
    leader, leader_fit = None, np.inf

    for _ in range(iterations):
        fits = [_fitness(ind, chs, network) for ind in population]
        best_i = int(np.argmin(fits))
        if fits[best_i] < leader_fit:
            leader, leader_fit = population[best_i], fits[best_i]
        history.append(leader_fit)

        new_pop = [leader]  # elitism
        while len(new_pop) < pop_size:
            child = dict(leader)
            for ch in chs:
                if rng.random() < 0.25:  # exploratory "shot"
                    options = [c for c in chs if c != ch]
                    child[ch] = SINK if (rng.random() < 0.3 or not options) else rng.choice(options)
            new_pop.append(_repair_acyclic(child, chs))
        population = new_pop

    return leader, history
