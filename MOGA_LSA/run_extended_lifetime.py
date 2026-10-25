"""Single-seed, long-horizon run of stock MOGA+LSA vs MOGA+GraphSAGE+LSA,
purely to capture the full network-lifetime curve (FND/HND/LND) that the
3000-round, 3-seed comparison (run_compare.py) didn't reach -- avg residual
energy depleted linearly there (~5.6e-5 J/round) with no deaths yet.
"""
import json
import time

from wsn_sim import moga, lsa, simulate, wsn_dataset, moga_embedding

N_NODES = 100
ROUNDS = 13000
SEED = 1
SAMPLE_EVERY = 25
OUT_PATH = "wsn_dataset_extended_lifetime.json"


def fnd_hnd_lnd(alive: list, n_nodes: int) -> dict:
    """First/Half/Last Node Death round, from a (possibly subsampled) alive trajectory."""
    fnd = next((i for i, a in enumerate(alive) if a < n_nodes), -1)
    hnd = next((i for i, a in enumerate(alive) if a <= n_nodes // 2), -1)
    lnd = next((i for i, a in enumerate(alive) if a == 0), -1)
    return {"fnd": fnd, "hnd": hnd, "lnd": lnd}


if __name__ == "__main__":
    t0 = time.time()
    out = {"rounds": ROUNDS, "seed": SEED, "n_nodes": N_NODES, "sample_every": SAMPLE_EVERY}

    net = wsn_dataset.build_network(n_nodes=N_NODES, seed=SEED)
    hist = simulate.run(net, rounds=ROUNDS, seed=SEED,
                         ch_selector=moga.select_cluster_heads, router=lsa.find_routes)
    traj = {k: v[::SAMPLE_EVERY].tolist() for k, v in hist.items()}
    fhl = fnd_hnd_lnd(traj["alive"], N_NODES)
    fhl = {k: (v * SAMPLE_EVERY if v >= 0 else v) for k, v in fhl.items()}
    out["moga"] = {"trajectory": traj, **fhl}
    print(f"stock MOGA+LSA: FND={fhl['fnd']} HND={fhl['hnd']} LND={fhl['lnd']} "
          f"({time.time()-t0:.1f}s)", flush=True)

    net2 = wsn_dataset.build_network(n_nodes=N_NODES, seed=SEED)
    sel = moga_embedding.make_selector()
    hist2 = simulate.run(net2, rounds=ROUNDS, seed=SEED, ch_selector=sel, router=lsa.find_routes)
    traj2 = {k: v[::SAMPLE_EVERY].tolist() for k, v in hist2.items()}
    fhl2 = fnd_hnd_lnd(traj2["alive"], N_NODES)
    fhl2 = {k: (v * SAMPLE_EVERY if v >= 0 else v) for k, v in fhl2.items()}
    out["moga_graphsage"] = {"trajectory": traj2, **fhl2}
    print(f"MOGA+GraphSAGE+LSA: FND={fhl2['fnd']} HND={fhl2['hnd']} LND={fhl2['lnd']} "
          f"({time.time()-t0:.1f}s)", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"saved {OUT_PATH}, total {time.time()-t0:.1f}s")
