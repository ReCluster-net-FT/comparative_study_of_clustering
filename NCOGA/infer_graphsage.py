"""Frozen GraphSAGE utility inference for the C++ socket harness."""

import argparse
import os
import socket
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from train_graphsage import GraphSAGE, UtilityMLP
from graph_features import build_node_features, feature_columns_for

def load_model(path, mode):
    checkpoint = torch.load(path, map_location="cpu")
    if "feature_set" not in checkpoint:
        raise ValueError(
            "Checkpoint has no feature_set metadata; retrain required."
        )
    if "feature_columns" not in checkpoint:
        raise ValueError(
            "Checkpoint has no feature_columns metadata; retrain required."
        )
    assert checkpoint["feature_columns"] == feature_columns_for(
        checkpoint["feature_set"]
    ), "Checkpoint feature_columns do not match feature_set; use the matching model."
    if checkpoint["input_dim"] != len(checkpoint["feature_columns"]):
        raise ValueError(
            "Checkpoint input_dim does not match its feature_columns."
        )
    if "feature_spec" not in checkpoint:
        raise ValueError(
            "Checkpoint has no feature_spec metadata; retrain required."
        )
    if "capacity_k" not in checkpoint["feature_spec"]:
        raise ValueError(
            "Checkpoint feature_spec has no capacity_k; retrain required."
        )
    if mode == "mlp":
        model = UtilityMLP(checkpoint["input_dim"])
    else:
        model = GraphSAGE(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint["model"] = model
    checkpoint["_mean"] = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
    checkpoint["_std"] = torch.tensor(checkpoint["feature_std"], dtype=torch.float32)
    return checkpoint

_PROFILE_PATH = os.environ.get("INFER_PROFILE_LOG")
_STAGES = ("read", "filter", "pairs", "features", "forward", "total")
_stats = {k: 0.0 for k in _STAGES}
_calls = 0

def _record(times):
    """Append per-stage average timings to INFER_PROFILE_LOG every 200 calls."""
    global _calls
    _calls += 1
    for key, value in times.items():
        _stats[key] += value
    if _PROFILE_PATH and _calls % 200 == 0:
        parts = " ".join(
            f"{k}={1000.0 * _stats[k] / _calls:.1f}ms" for k in _STAGES
        )
        with open(_PROFILE_PATH, "a", encoding="utf-8") as log:
            log.write(f"pid={os.getpid()} calls={_calls} avg {parts}\n")

def _edge_pairs(frame, edges):
    """Same result as the original dict/zip loop, but vectorised."""
    ids = frame.node_id.to_numpy()
    src = edges.source.to_numpy()
    dst = edges.destination.to_numpy()
    if len(ids) == 0 or len(src) == 0:
        return []
    if len(np.unique(ids)) != len(ids):
        positions = {int(node): i for i, node in enumerate(ids)}
        return [(positions[a], positions[b]) for a, b in zip(src, dst)
                if a in positions and b in positions]

    si = np.searchsorted(ids, src)
    di = np.searchsorted(ids, dst)
    si_c = np.minimum(si, len(ids) - 1)
    di_c = np.minimum(di, len(ids) - 1)
    keep = (ids[si_c] == src) & (ids[di_c] == dst)
    return list(zip(si[keep].tolist(), di[keep].tolist()))

def _read_edges(path):

    try:
        return pd.read_csv(path, usecols=["source", "destination"],
                           engine="pyarrow")
    except Exception:
        return pd.read_csv(path, usecols=["source", "destination"])

def predict(nodes_path, edges_path, checkpoint, seed, round_id, mode):
    t0 = time.perf_counter()
    nodes = pd.read_csv(nodes_path)
    edges = _read_edges(edges_path)
    t1 = time.perf_counter()
    frame = nodes[(nodes.seed == seed) & (nodes["round"] == round_id)].sort_values("node_id").reset_index(drop=True)
    t2 = time.perf_counter()
    pairs = _edge_pairs(frame, edges)
    edge_index = (torch.from_numpy(np.asarray(pairs, dtype=np.int64).T.copy())
                  if pairs else torch.empty((2, 0), dtype=torch.long))
    t3 = time.perf_counter()
    spec = checkpoint["feature_spec"]
    x = torch.from_numpy(
        build_node_features(
            frame,
            pairs,
            checkpoint["feature_set"],
            spec["capacity_k"],
        )
    )
    t4 = time.perf_counter()
    mean = checkpoint["_mean"]
    std = checkpoint["_std"]
    with torch.inference_mode():
        features = (x - mean) / std
        if mode == "mlp":
            prediction = checkpoint["model"](features)
        else:
            _, prediction = checkpoint["model"](features, edge_index)
    t5 = time.perf_counter()
    _record({
        "read": t1 - t0, "filter": t2 - t1, "pairs": t3 - t2,
        "features": t4 - t3, "forward": t5 - t4, "total": t5 - t0,
    })
    return list(zip(frame.node_id.tolist(), prediction.tolist()))

def run_server(model_path, port, mode):
    torch.set_num_threads(int(os.environ.get("INFER_THREADS", "1")))
    checkpoint = load_model(model_path, mode)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port)); server.listen(1)
        connection, _ = server.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with connection, connection.makefile("r", encoding="utf-8") as stream:
            for request in stream:
                if request.rstrip("\n") == "QUIT": break
                nodes, edges, seed, round_id = request.rstrip("\n").split("\t")
                payload = "".join(f"{node},{score}\n" for node, score in predict(nodes, edges, checkpoint, int(seed), int(round_id), mode))
                connection.sendall((payload + "END\n").encode())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--port", type=int, default=39000)
    parser.add_argument("--model", default="graphsage_model.pt")
    parser.add_argument("--mode", choices=["graphsage", "mlp"], default="graphsage")
    args = parser.parse_args()
    if args.server:
        run_server(args.model, args.port, args.mode)
        return
    raise SystemExit("Use --server for persistent inference.")

if __name__ == "__main__":
    main()
