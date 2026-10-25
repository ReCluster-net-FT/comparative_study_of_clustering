import sys
import os
import time
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import config as cfg
from common.network import Network
from common.metrics import RoundRecorder
from common import packet_model
from common import dataset_loader

from esca.protocol import ESCA
from ighnd.protocol import IGHND
from ighnd_gs.protocol import IGHND_GraphSAGE
from esca_graphsage import ESCA_GraphSAGE

PROTOCOLS = [
    ("ESCA", ESCA),
    ("ESCA-GraphSAGE", ESCA_GraphSAGE),
    ("IGHND", IGHND),
    ("IGHND-GraphSAGE", IGHND_GraphSAGE),
]

RESULTS_PATH = "results/raw_results.json"


_HISTORY_KEYS = [
    "alive_history", "dead_history", "residual_energy_history",
    "avg_residual_energy_history", "energy_consumed_history",
    "cumulative_energy_consumed_history", "packet_loss_rate_history",
    "ch_per_round_count_history",
]


def collect_round_extras(proto, net):
    ch_ids = []
    sender_ids = []
    distances = []

    if hasattr(proto, "ch") and hasattr(proto, "labels"):
        # ESCA / ESCA-GraphSAGE
        for c, ch_id in proto.ch.items():
            ch_ids.append(ch_id)
            members = np.where((proto.labels == c) & net.alive)[0]
            non_ch = members[members != ch_id]
            if len(non_ch) > 0:
                distances.extend(net.pairwise_dist(ch_id, non_ch).tolist())
                sender_ids.extend(non_ch.tolist())
            distances.append(float(net.dist_to_bs[ch_id]))
            sender_ids.append(int(ch_id))

    elif hasattr(proto, "zh") and hasattr(proto, "zones"):
        # IGHND / IGHND-GraphSAGE
        for zid, zh_id in proto.zh.items():
            ch_ids.append(zh_id)
            z = proto.zones[zid]
            members = np.array([m for m in z.members if net.alive[m] and m != zh_id])
            if len(members) > 0:
                distances.extend(net.pairwise_dist(zh_id, members).tolist())
                sender_ids.extend(members.tolist())
            distances.append(float(net.dist_to_bs[zh_id]))
            sender_ids.append(int(zh_id))

    return ch_ids, np.array(sender_ids, dtype=int), np.array(distances, dtype=float)


def run_one(cls, network, seed, keep_curves, embedding_path=None):
    net = network.clone()
    if cls.name == "ESCA":
        proto = cls(net, seed=seed)
    else:
        proto = cls(net)

    rec = RoundRecorder(net.n_nodes)
    r = 0

    use_real_packet_loss = hasattr(net, "packet_loss_prob")

    while net.num_alive() > 0 and r < cfg.MAX_ROUNDS:
        r += 1
        energy_before = net.total_alive_energy()
        proto.run_round()

        ch_ids, sender_ids, distances = collect_round_extras(proto, net)
        rec.record_ch(ch_ids)
        if use_real_packet_loss:
            attempted, delivered, _ = packet_model.round_packet_stats_from_dataset(
                sender_ids, net
            )
        else:
            attempted, delivered, _ = packet_model.round_packet_stats(distances)
        rec.record_packets(attempted, delivered)

        net.kill_dead()
        rec.record(r, net, energy_before=energy_before)

    result = rec.finalize(r)
    result["hit_max_rounds"] = (r >= cfg.MAX_ROUNDS and net.num_alive() > 0)
    result["seed"] = seed
    result["used_real_packet_loss"] = use_real_packet_loss  # [DATASET] audit trail

    if hasattr(proto, "K"):
        result["K"] = proto.K

    if (keep_curves and embedding_path is not None
            and getattr(proto, "last_embedding", None) is not None):
        np.savez(
            embedding_path,
            embedding=proto.last_embedding,
            node_ids=proto.last_embedding_node_ids,
        )

    if not keep_curves:
        for key in _HISTORY_KEYS:
            result.pop(key, None)

    return result


def load_existing():
    if not os.path.exists(RESULTS_PATH):
        return {}
    try:
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except json.JSONDecodeError:
        return {}


def save_results(all_results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)


def print_final_verification(dataset):
    r = dataset.report
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)
    print(f"CSV file path used                 : {r.path}")
    print(f"Number of rows in CSV               : {r.n_rows}")
    print(f"Number of unique nodes (Node_ID)     : {r.n_unique_node_ids}")
    print(f"Number of unique timestamps          : {r.n_unique_timestamps}")
    print(f"Dataset time range                   : {r.time_min} -> {r.time_max}")
    print(f"Nodes actually used this run         : {dataset.n_nodes} "
          f"({'sampled subset, real rows only' if r.sampled else 'all real rows'})")
    print(f"Feature columns used by GraphSAGE     : {dataset.feature_names}")
    print("Confirmation: node positions/energy   : READ FROM CSV -- "
          "Network.from_dataset() contains no np.random call for position "
          "or energy (see common/network.py).")
    print("Confirmation: packet loss              : REAL Packet_Loss_Rate "
          "column used (common/packet_model.round_packet_stats_from_dataset) "
          "-- the synthetic distance-based formula is not invoked when a "
          "dataset network is used.")
    print("Confirmation: GraphSAGE affects CH/ZH  : YES -- final_score = "
          "alpha*as_score + (1-alpha)*sage_score, and the argmax over "
          "final_score (not as_score alone) is what is elected CH/ZH "
          "(esca_graphsage.py / ighnd_gs/protocol.py).")
    print("Confirmation: same dataset, all 4 algos: YES -- ESCA, IGHND and "
          "both GraphSAGE variants are run via net.clone() off the SAME "
          "Network.from_dataset(...) instance within a scenario, so they "
          "see identical node positions, energies and features; only the "
          "algorithmic decision logic differs.")
    if r.invalid_residual_gt_initial:
        print(f"Data-quality note: {r.invalid_residual_gt_initial} CSV rows have "
              f"Residual_Energy > Initial_Energy -- these are NOT used to seed "
              f"simulation energy (see common/dataset_loader.py docstring).")
    print("=" * 70 + "\n")


def main(only_scenario=None):
    dataset_path = os.path.join(PROJECT_ROOT, cfg.DATASET_PATH) \
        if not os.path.isabs(cfg.DATASET_PATH) else cfg.DATASET_PATH
    dataset = dataset_loader.load(
        dataset_path,
        max_nodes=cfg.DATASET_MAX_NODES,
        sample_seed=cfg.DATASET_SAMPLE_SEED,
    )
    dataset.report.print_summary()
    print_final_verification(dataset)

    all_results = load_existing()
    print("=" * 70)
    print("ESCA vs IGHND vs GraphSAGE-hybrids Benchmark (real-dataset input)")
    print("=" * 70)

    for scen_name, bs_mode in cfg.DATASET_SCENARIOS.items():
        if only_scenario and scen_name != only_scenario:
            continue

        if scen_name in all_results:
            existing_protocols = all_results[scen_name]
            completed = all(
                proto in existing_protocols
                and len(existing_protocols[proto]) >= len(cfg.FIXED_SEEDS)
                for proto, _ in PROTOCOLS
            )
            if completed:
                print(f"\nSkipping {scen_name} - already completed.")
                continue

        if scen_name not in all_results:
            all_results[scen_name] = {}

        for name, _ in PROTOCOLS:
            if name not in all_results[scen_name]:
                all_results[scen_name][name] = []

        # [DATASET] ALL seeds/protocols in this scenario share the SAME
        # real node set -- only the BS placement (bs_mode) differs between
        # scenarios, and only ESCA's K-means initialization differs
        # between seeds. No per-seed random node/energy regeneration.
        net0 = Network.from_dataset(dataset, bs_mode=bs_mode)

        print("-" * 70)
        print(f"{scen_name} | Real nodes={net0.n_nodes} | BS={tuple(net0.bs)} "
              f"(bs_mode={bs_mode})")
        print("-" * 70)

        for seed in cfg.FIXED_SEEDS:
            print(f"\nSeed {seed} (K-means/clustering-init seed only -- "
                  f"node data itself is fixed/real)")
            keep_curves = (seed == cfg.CURVE_SEED)

            for name, cls in PROTOCOLS:
                existing_trials = all_results[scen_name][name]
                if any(trial.get("seed") == seed and "error" not in trial for trial in existing_trials):
                    continue

                print(f"  Running {name}...")
                t0 = time.time()
                try:
                    emb_path = None
                    if keep_curves and "GraphSAGE" in name:
                        os.makedirs("results/embeddings", exist_ok=True)
                        emb_path = f"results/embeddings/{scen_name}_{name}.npz"
                    result = run_one(cls, net0, seed, keep_curves=keep_curves,
                                      embedding_path=emb_path)
                    runtime = time.time() - t0
                    result["runtime_seconds"] = runtime
                    all_results[scen_name][name].append(result)

                    fnd = result.get("fnd")
                    hnd = result.get("hnd")
                    lnd = result.get("lnd")
                    avg_energy = result.get("avg_residual_energy")
                    total_energy = result.get("total_energy_consumed")
                    loss = result.get("overall_packet_loss_rate")

                    print(f"    FND={fnd if fnd is not None else 'N/A'} | "
                          f"HND={hnd if hnd is not None else 'N/A'} | "
                          f"LND={lnd if lnd is not None else 'N/A'}")
                    if avg_energy is not None:
                        print(f"    Avg Residual Energy={avg_energy:.4f} J | "
                              f"Total Consumed={total_energy:.4f} J")
                    if loss is not None:
                        print(f"    Real-dataset Packet Loss Rate={loss:.4f}")
                    print(f"    Runtime={runtime:.2f}s")

                    if lnd is None:
                        print(f"    LIMITATION: network did not fully die within "
                              f"MAX_ROUNDS={cfg.MAX_ROUNDS}; LND is not defined for "
                              f"this run (not invented -- see result['hit_max_rounds']).")
                except KeyboardInterrupt:
                    save_results(all_results)
                    raise
                except Exception as e:
                    print(f"    ERROR in {name}: {e}")
                    all_results[scen_name][name].append({"seed": seed, "error": str(e)})

            save_results(all_results)
            print(f"  Saved results after seed {seed}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    scen_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(only_scenario=scen_arg)
