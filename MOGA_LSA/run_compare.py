"""Stock MOGA+LSA vs MOGA+GraphSAGE-embeddings+LSA, on the Kaggle WSN
dataset. Round-based energy-depletion simulation (wsn_sim.simulate.run,
first-order radio model) across several seeds for the lifetime/energy
comparison, plus one ns-3 (cppyy) run per method for a NetAnim topology
snapshot -- ns-3's continuous-listening energy model isn't suited to
running thousands of rounds, so it's used for visualization only; the
round-based engine is the source of truth for the numbers.
"""
import json
import subprocess
import sys
import time

import numpy as np

from wsn_sim import moga, lsa, simulate, wsn_dataset, moga_embedding, ns3_export

N_NODES = 100
ROUNDS = 3000
SEEDS = [1, 2, 3]
SAMPLE_EVERY = 20
RESULTS_PATH = "multi_seed_wsn_dataset_results.json"
STOCK_PREFIX = "MOGA_LSA_WSN_Dataset"
HYBRID_PREFIX = "MOGA_GraphSAGE_LSA_WSN_Dataset"


def _sample(hist: dict) -> dict:
    return {k: v[::SAMPLE_EVERY].tolist() for k, v in hist.items()}


def _first_death_round(alive: np.ndarray) -> int:
    below = np.nonzero(alive < alive[0])[0]
    return int(below[0]) if len(below) else -1


def run_one_seed(seed: int) -> dict:
    result = {"seed": seed}
    t0 = time.time()

    net = wsn_dataset.build_network(n_nodes=N_NODES, seed=seed)
    hist = simulate.run(net, rounds=ROUNDS, seed=seed,
                         ch_selector=moga.select_cluster_heads, router=lsa.find_routes)
    result["moga"] = {
        "final_alive": int(hist["alive"][-1]),
        "final_avg_residual": float(hist["avg_residual"][-1]),
        "first_death_round": _first_death_round(hist["alive"]),
        "trajectory": _sample(hist),
    }

    net2 = wsn_dataset.build_network(n_nodes=N_NODES, seed=seed)
    sel = moga_embedding.make_selector()
    hist2 = simulate.run(net2, rounds=ROUNDS, seed=seed, ch_selector=sel, router=lsa.find_routes)
    result["moga_graphsage"] = {
        "final_alive": int(hist2["alive"][-1]),
        "final_avg_residual": float(hist2["avg_residual"][-1]),
        "first_death_round": _first_death_round(hist2["alive"]),
        "trajectory": _sample(hist2),
    }

    result["wall_time_s"] = time.time() - t0
    return result


def run_ns3_snapshot(seed: int, sim_time: float = 15.0):
    net = wsn_dataset.build_network(n_nodes=N_NODES, seed=seed)
    rng = np.random.default_rng(seed)
    chs = moga.select_cluster_heads(net, rng=rng)
    next_hop, _ = lsa.find_routes(net, chs, rng=rng)
    ns3_export.write_roster(net, chs, next_hop, f"{STOCK_PREFIX}_nodes.csv")

    net2 = wsn_dataset.build_network(n_nodes=N_NODES, seed=seed)
    rng2 = np.random.default_rng(seed)
    sel = moga_embedding.make_selector()
    chs2 = sel(net2, rng=rng2)
    next_hop2, _ = lsa.find_routes(net2, chs2, rng=rng2)
    ns3_export.write_roster(net2, chs2, next_hop2, f"{HYBRID_PREFIX}_nodes.csv")

    sink_x, sink_y = net.sink
    for prefix in (STOCK_PREFIX, HYBRID_PREFIX):
        subprocess.run([sys.executable, "ns3_wsn_sim.py",
                         "--nodes-csv", f"{prefix}_nodes.csv", "--anim-out", f"{prefix}_netanim.xml",
                         "--sim-time", str(sim_time), "--sink-x", str(sink_x), "--sink-y", str(sink_y)],
                        check=True)


if __name__ == "__main__":
    results = []
    t0 = time.time()
    for s in SEEDS:
        r = run_one_seed(s)
        results.append(r)
        print(f"seed {s} done in {r['wall_time_s']:.1f}s "
              f"(moga alive={r['moga']['final_alive']}, "
              f"moga_graphsage alive={r['moga_graphsage']['final_alive']})", flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"rounds": ROUNDS, "seeds": SEEDS, "n_nodes": N_NODES, "results": results}, f)
    print(f"all {len(SEEDS)} seeds done in {time.time()-t0:.1f}s, saved {RESULTS_PATH}")

    print("running ns-3 NetAnim snapshots...")
    run_ns3_snapshot(SEEDS[0])
