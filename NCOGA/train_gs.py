import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from graph_features import (
    GRAPH_FEATURE_COLUMNS,
    LEGACY_FEATURE_COLUMNS,
    build_node_features,
    compute_graph_features,
    feature_columns_for,
    selftest as graph_features_selftest,
)

BASE_DIR = Path(__file__).resolve().parent

NS3_DIR = BASE_DIR.parent

NS3_EXECUTABLE = NS3_DIR / "ns3"
NS3_PROGRAM = "scratch/ncoga"

INFERENCE_SCRIPT = BASE_DIR / "infer_graphsage.py"

GRAPHSAGE_MODEL = BASE_DIR / "graphsage_model.pt"
MLP_MODEL = BASE_DIR / "mlp_model.pt"

TRAINING_NODES = BASE_DIR / "graphsage_training_nodes.csv"
TRAINING_EDGES = BASE_DIR / "graphsage_training_edges.csv"
TRAINING_STATE = BASE_DIR / "graphsage_training_graph_nodes.csv"

FEATURE_COLUMNS = GRAPH_FEATURE_COLUMNS.copy()

UTILITY_COLUMNS = [
    "U_E",
    "U_C",
    "U_CE",
    "C_relay",
]
UTILITY_COEFFS = (0.25, 0.25, 0.25, -0.25)
PROGRESS_EVERY = 500

DTYPE_MAP = {
    "seed": "int32",
    "round": "int32",
    "node_id": "int32",
    "source": "int32",
    "destination": "int32",
    "neighbor_count": "int32",
    "sample_weight": "float32",
    "F": "float32",
    "residual_ratio": "float32",
    "utility": "float32",
    **{
        c: "float32"
        for c in LEGACY_FEATURE_COLUMNS + FEATURE_COLUMNS
        if c != "neighbor_count"
    },
    **{c: "float32" for c in UTILITY_COLUMNS},
}

def _dtype_for_columns(path):
    header = pd.read_csv(path, nrows=0)
    return {col: dt for col, dt in DTYPE_MAP.items() if col in header.columns}

def read_csv_slim(path, **kwargs):
    dtype = _dtype_for_columns(path)
    return pd.read_csv(path, dtype=dtype, **kwargs)

def _resident_memory_gb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None

class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, embedding_dim=16):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
        self.regressor = nn.Linear(embedding_dim, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()

        embedding = self.conv2(x, edge_index).relu()

        return embedding, self.regressor(embedding).squeeze(-1)

class UtilityMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

def add_utility(nodes, norm_stats=None):
    values = nodes[UTILITY_COLUMNS].to_numpy(np.float64)

    if norm_stats is None:
        mean, std = values.mean(axis=0), values.std(axis=0)
    else:
        mean, std = norm_stats

    std = np.maximum(std, 1e-8)

    normalized = (values - mean) / std

    nodes = nodes.copy()

    nodes["utility"] = sum(
        coefficient * normalized[:, index]
        for index, coefficient in enumerate(UTILITY_COEFFS)
    ).astype(np.float32)

    return nodes, (mean, std)

def make_feature_spec(capacity_k):
    return {
        "capacity_k": capacity_k,
        "ap_scope": "full_snapshot_graph",
        "betweenness": "exact",
        "edge_radius_m": 30.0,
        "graph_symmetrized": True,
    }

def load_graphs(
    nodes_path,
    edges_path,
    seeds,
    graph_state_path=None,
    feature_set="slide",
    capacity_k=None,
    feature_cache_dir=None,
):
    load_started = time.perf_counter()
    samples = read_csv_slim(nodes_path)

    state_path = Path(graph_state_path or nodes_path)
    nodes = read_csv_slim(state_path)
    if not isinstance(nodes.index, pd.RangeIndex):
        raise ValueError(
            "The graph-state CSV header/row field-count mismatch caused pandas "
            "to use the first column as the index. Append ',F' to the "
            "g_graphStateCsv header string in the C++ file, then regenerate "
            "the data."
        )

    edges = read_csv_slim(edges_path)

    seeds = list(seeds)

    nodes = nodes[nodes.seed.isin(seeds)]
    samples = samples[samples.seed.isin(seeds)]

    edge_lookup = {
        (int(seed), int(round_id)):
            frame[["source", "destination"]].to_numpy(
                dtype=np.int64
            )
        for (seed, round_id), frame
        in edges.groupby(["seed", "round"], sort=False)
    }

    del edges

    sample_cols = ["utility", "sample_weight", "F", "residual_ratio"]
    samples_indexed = (
        samples
        .set_index(["seed", "round", "node_id"])[sample_cols]
        .sort_index()
    )

    del samples

    graphs = []

    cache_enabled = feature_cache_dir is not None and feature_set != "legacy"
    state_stat = state_path.stat() if cache_enabled else None
    seed_round_counts = nodes.groupby("seed")["round"].nunique().to_dict()
    seed_ids = sorted(seed_round_counts)
    total_rounds = sum(seed_round_counts.values())
    print(
        f"[load] {len(seed_ids)} seeds, {total_rounds} total rounds",
        flush=True,
    )
    for seed_index, (seed, seed_frame) in enumerate(
        nodes.groupby("seed", sort=True),
        start=1,
    ):
        seed_started = time.perf_counter()
        round_ids = np.sort(seed_frame["round"].unique().astype(np.int64))
        cache_path = None
        cached_features = {}
        cache_loaded = False
        if cache_enabled:
            cache_path = Path(feature_cache_dir) / (
                f"{state_path.name}.seed_{int(seed)}.K_{capacity_k}."
                f"features_{feature_set}.size_{state_stat.st_size}.npz"
            )
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as cache:
                    if "round_ids" not in cache.files:
                        raise ValueError(
                            f"Feature cache {cache_path} is missing round_ids"
                        )
                    if np.array_equal(cache["round_ids"], round_ids):
                        for index, round_id in enumerate(round_ids):
                            key = f"feature_{index}"
                            if key not in cache.files:
                                raise ValueError(
                                    f"Feature cache {cache_path} is missing {key}"
                                )
                            cached_features[int(round_id)] = cache[key].copy()
                        cache_loaded = True

        if not cache_enabled:
            cache_status = "caching disabled"
        elif cache_loaded:
            cache_status = "cache hit"
        else:
            cache_status = "cache miss"
        print(
            f"[load] seed {int(seed)} ({seed_index}/{len(seed_ids)}), "
            f"{len(round_ids)} rounds, {cache_status}",
            flush=True,
        )

        computed_features = []
        computed_round_ids = []
        for round_index, round_id in enumerate(round_ids, start=1):
            frame = (
                seed_frame[seed_frame["round"] == round_id]
                .sort_values("node_id")
                .reset_index(drop=True)
            )
            positions = {
                int(node): i
                for i, node in enumerate(frame.node_id)
            }
            pairs = [
                (positions[src], positions[dst])
                for src, dst in edge_lookup.get(
                    (int(seed), int(round_id)),
                    []
                )
                if src in positions and dst in positions
            ]
            edge_index = (
                torch.from_numpy(
                    np.asarray(pairs, dtype=np.int64).T.copy()
                )
                if pairs
                else torch.empty((2, 0), dtype=torch.long)
            )

            precomputed = None
            if feature_set != "legacy":
                if int(round_id) in cached_features:
                    precomputed = cached_features[int(round_id)]
                elif cache_enabled:
                    precomputed = compute_graph_features(
                        len(frame),
                        pairs,
                        capacity_k,
                    )
                    computed_round_ids.append(int(round_id))
                    computed_features.append(precomputed)
            node_features = build_node_features(
                frame,
                pairs,
                feature_set,
                capacity_k,
                precomputed_graph_features=precomputed,
            )

            idx = pd.MultiIndex.from_arrays([
                np.full(len(frame), int(seed), dtype=np.int64),
                np.full(len(frame), int(round_id), dtype=np.int64),
                frame.node_id.to_numpy(dtype=np.int64),
            ])
            matched = samples_indexed.reindex(idx)
            labels = matched["utility"].to_numpy(dtype=np.float32, na_value=0.0)
            weights = matched["sample_weight"].to_numpy(dtype=np.float32, na_value=0.0)
            fitness = matched["F"].to_numpy(dtype=np.float32, na_value=0.0)
            residual = matched["residual_ratio"].to_numpy(dtype=np.float32, na_value=0.0)

            graphs.append(
                Data(
                    x=torch.from_numpy(node_features),
                    y=torch.from_numpy(labels),
                    weight=torch.from_numpy(weights),
                    fitness=torch.from_numpy(fitness),
                    residual=torch.from_numpy(residual),
                    edge_index=edge_index,
                    seed=int(seed),
                    round_id=int(round_id),
                )
            )

            if round_index % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - seed_started
                average_per_round = elapsed / round_index
                remaining = average_per_round * (len(round_ids) - round_index)
                percent = 100.0 * round_index / max(len(round_ids), 1)
                memory_gb = _resident_memory_gb()
                memory_text = (
                    f" rss={memory_gb:.2f}GB"
                    if memory_gb is not None
                    else ""
                )
                print(
                    f"[load] seed {int(seed)}: "
                    f"{round_index}/{len(round_ids)} rounds "
                    f"({percent:.1f}%), elapsed={elapsed:.1f}s "
                    f"eta={remaining:.1f}s{memory_text}",
                    flush=True,
                )

        seed_elapsed = time.perf_counter() - seed_started
        memory_gb = _resident_memory_gb()
        memory_text = (
            f" rss={memory_gb:.2f}GB"
            if memory_gb is not None
            else ""
        )
        print(
            f"[load] seed {int(seed)} done in {seed_elapsed:.1f}s; "
            f"{len(graphs)} graphs built{memory_text}",
            flush=True,
        )

        if cache_enabled and not cache_loaded:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_cache_path = cache_path.with_name(cache_path.name + ".tmp.npz")
            cache_arrays = {
                f"feature_{index}": feature_array
                for index, feature_array in enumerate(computed_features)
            }
            np.savez(
                temp_cache_path,
                round_ids=np.asarray(computed_round_ids, dtype=np.int64),
                **cache_arrays,
            )
            os.replace(temp_cache_path, cache_path)
            print(
                f"[load] cache saved: {cache_path}",
                flush=True,
            )

    del nodes, samples_indexed, edge_lookup

    if not graphs:
        raise ValueError(
            "No utility samples found"
        )

    print(
        f"[load] {len(graphs)} graphs built in "
        f"{time.perf_counter() - load_started:.1f}s",
        flush=True,
    )
    return graphs

def normalize_features(
    graphs,
    skip_columns=None,
    feature_columns=None,
):
    skip_columns = set(skip_columns or [])
    count = 0
    dim = graphs[0].x.shape[1]
    mean = torch.zeros(dim, dtype=torch.float64)
    m2 = torch.zeros(dim, dtype=torch.float64)

    for graph in graphs:
        x = graph.x.double()
        batch_count = x.shape[0]

        if batch_count == 0:
            continue

        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)

        delta = batch_mean - mean
        total = count + batch_count

        mean = mean + delta * (batch_count / total)
        m2 = (
            m2
            + batch_var * batch_count
            + delta.pow(2) * count * batch_count / total
        )

        count = total

    std = (m2 / max(count, 1)).sqrt().clamp_min(1e-8).float()
    mean = mean.float()

    raw_variance = m2 / max(count, 1)
    for index in range(dim):
        if raw_variance[index] == 0:
            name = feature_columns[index] if feature_columns else f"feature_{index}"
            print(
                f"WARNING: raw feature {name} has zero variance",
                flush=True,
            )
        if index in skip_columns:
            mean[index] = 0.0
            std[index] = 1.0

    for graph in graphs:
        graph.x = (graph.x - mean) / std

    normalized_count = 0
    normalized_sum = torch.zeros(dim, dtype=torch.float64)
    normalized_sum_sq = torch.zeros(dim, dtype=torch.float64)
    for graph in graphs:
        x = graph.x.double()
        normalized_count += x.shape[0]
        normalized_sum += x.sum(0)
        normalized_sum_sq += x.pow(2).sum(0)
    normalized_mean = normalized_sum / max(normalized_count, 1)
    normalized_std = (
        normalized_sum_sq / max(normalized_count, 1)
        - normalized_mean.pow(2)
    ).clamp_min(0).sqrt()
    for index in range(dim):
        name = feature_columns[index] if feature_columns else f"feature_{index}"
        print(
            f"normalized feature {name}: "
            f"mean={normalized_mean[index].item():.6f} "
            f"std={normalized_std[index].item():.6f}",
            flush=True,
        )

    return mean, std

def run_epoch(
    model,
    loader,
    optimizer=None
):
    training = optimizer is not None

    model.train(training)

    loss_sum = 0.0
    node_count = 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    n = 0

    for batch in loader:

        with torch.set_grad_enabled(training):

            _, prediction = model(
                batch.x,
                batch.edge_index
            )

            values = (
                prediction - batch.y
            ).pow(2)

            loss = (
                values * batch.weight
            ).sum() / batch.weight.sum().clamp_min(1e-8)

            if training:

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        loss_sum += float(
            values.sum()
        )

        node_count += batch.num_nodes

        p = prediction.detach()
        t = batch.y

        sum_x += float(p.sum())
        sum_y += float(t.sum())
        sum_xy += float((p * t).sum())
        sum_x2 += float((p * p).sum())
        sum_y2 += float((t * t).sum())
        n += p.numel()

    if n > 1:
        mean_x = sum_x / n
        mean_y = sum_y / n
        covariance = sum_xy / n - mean_x * mean_y
        var_x = max(sum_x2 / n - mean_x * mean_x, 0.0)
        var_y = max(sum_y2 / n - mean_y * mean_y, 0.0)
        denom = (var_x * var_y) ** 0.5
        correlation = covariance / denom if denom > 1e-12 else 0.0
    else:
        correlation = 0.0

    return (
        loss_sum / max(node_count, 1),
        correlation
    )

def graph_predictions(
    model,
    graphs
):
    total_nodes = sum(int(g.num_nodes) for g in graphs)

    loader = DataLoader(
        graphs,
        batch_size=16,
        shuffle=False
    )

    model.eval()

    values = np.empty(total_nodes, dtype=np.float32)
    targets = np.empty(total_nodes, dtype=np.float32)
    weights = np.empty(total_nodes, dtype=np.float32)

    offset = 0

    with torch.no_grad():

        for batch in loader:

            _, prediction = model(
                batch.x,
                batch.edge_index
            )

            n = prediction.numel()

            values[offset:offset + n] = prediction.numpy()
            targets[offset:offset + n] = batch.y.numpy()
            weights[offset:offset + n] = batch.weight.numpy()

            offset += n

    return values[:offset], targets[:offset], weights[:offset]

def regression_metrics(
    prediction,
    target
):

    error = prediction - target

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (target - target.mean()) ** 2
        )
    )

    r2 = (
        1.0
        - float(
            np.sum(error ** 2)
        ) / denominator

        if denominator > 1e-12

        else 0.0
    )

    rank = float(
        pd.Series(prediction).corr(
            pd.Series(target),
            method="spearman"
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman":
            0.0 if np.isnan(rank)
            else rank,
    }

def train_mlp(
    graphs,
    epochs=1
):

    model = UtilityMLP(
        graphs[0].num_node_features
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loader = DataLoader(
        graphs,
        batch_size=16,
        shuffle=True
    )

    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):

        model.train()

        for batch in loader:

            prediction = model(
                batch.x
            )

            loss = (
                (prediction - batch.y) ** 2
                * batch.weight
            ).sum() / batch.weight.sum().clamp_min(1e-8)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print(
            f"[load] MLP epoch {epoch}/{epochs}: "
            f"batch_loss={float(loss.detach()):.6f} "
            f"elapsed={time.perf_counter() - training_started:.1f}s",
            flush=True,
        )

    del loader

    return model

def evaluate_gate(
    model_path,
    validation_nodes,
    validation_edges,
    validation_state,
    training_nodes,
    training_edges,
    training_state,
    feature_set,
    capacity_k,
    feature_cache_dir=None,
):

    print("[load] validation gate started", flush=True)
    train_frame = read_csv_slim(training_nodes)

    utility_stats = (
        train_frame[
            UTILITY_COLUMNS
        ].to_numpy().mean(0),

        np.maximum(
            train_frame[
                UTILITY_COLUMNS
            ].to_numpy().std(0),
            1e-8
        )
    )

    del train_frame

    validation_frame, _ = add_utility(
        read_csv_slim(validation_nodes),
        utility_stats
    )

    validation_frame.to_csv(
        validation_nodes,
        index=False
    )

    del validation_frame

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )
    if "feature_set" not in checkpoint:
        raise ValueError(
            "GraphSAGE checkpoint has no feature_set metadata; retrain required."
        )
    if "feature_columns" not in checkpoint:
        raise ValueError(
            "GraphSAGE checkpoint has no feature_columns metadata; retrain required."
        )
    assert checkpoint["feature_columns"] == feature_columns_for(
        checkpoint["feature_set"]
    ), "GraphSAGE checkpoint feature_columns do not match its feature_set"
    if checkpoint["input_dim"] != len(checkpoint["feature_columns"]):
        raise ValueError(
            "GraphSAGE checkpoint input_dim does not match feature_columns."
        )
    if checkpoint["feature_set"] != feature_set:
        raise ValueError(
            "GraphSAGE checkpoint feature_set does not match the requested "
            "feature_set; retrain with the selected feature set."
        )
    if "feature_spec" not in checkpoint:
        raise ValueError(
            "GraphSAGE checkpoint has no feature_spec metadata; retrain required."
        )
    if checkpoint["feature_spec"].get("capacity_k") != capacity_k:
        raise ValueError(
            "GraphSAGE checkpoint capacity_k does not match the requested "
            "capacity_k; retrain with the selected capacity."
        )

    train_graphs = load_graphs(
        training_nodes,
        training_edges,
        range(1, 13),
        graph_state_path=training_state,
        feature_set=feature_set,
        capacity_k=capacity_k,
        feature_cache_dir=feature_cache_dir,
    )

    validation_graphs = load_graphs(
        validation_nodes,
        validation_edges,
        range(13, 17),
        validation_state,
        feature_set=feature_set,
        capacity_k=capacity_k,
        feature_cache_dir=feature_cache_dir,
    )

    mean = torch.tensor(
        checkpoint["feature_mean"]
    )

    std = torch.tensor(
        checkpoint["feature_std"]
    )

    for graph in (
        train_graphs + validation_graphs
    ):

        graph.x = (
            graph.x - mean
        ) / std

    model = GraphSAGE(
        checkpoint["input_dim"],
        checkpoint["hidden_dim"],
        checkpoint["embedding_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    graph_pred, target, weights = graph_predictions(
        model,
        validation_graphs
    )

    mask = weights > 0

    graph_pred = graph_pred[mask]
    target = target[mask]

    mlp = train_mlp(
        train_graphs
    )

    torch.save(
        {
            "model_state_dict":
                mlp.state_dict(),

            "input_dim":
                train_graphs[0].num_node_features,

            "feature_columns":
                FEATURE_COLUMNS.copy(),

            "feature_set":
                feature_set,

            "feature_spec":
                checkpoint["feature_spec"],

            "feature_mean":
                checkpoint["feature_mean"],

            "feature_std":
                checkpoint["feature_std"],
        },

        MLP_MODEL
    )

    del train_graphs
    gc.collect()

    mlp_pred = np.empty(sum(int(g.num_nodes) for g in validation_graphs), dtype=np.float32)

    mlp.eval()

    offset = 0

    with torch.no_grad():

        for graph in validation_graphs:

            out = mlp(graph.x).numpy()
            n = out.shape[0]
            mlp_pred[offset:offset + n] = out
            offset += n

    mlp_pred = mlp_pred[:offset][mask]

    residual = np.concatenate(
        [
            graph.residual.numpy()
            for graph in validation_graphs
        ]
    )[mask]

    fp_proxy = np.concatenate(
        [
            graph.fitness.numpy()
            for graph in validation_graphs
        ]
    )[mask]

    del validation_graphs
    gc.collect()

    metrics = {
        "graphsage":
            regression_metrics(
                graph_pred,
                target
            ),

        "mlp":
            regression_metrics(
                mlp_pred,
                target
            ),

        "residual_ratio":
            regression_metrics(
                residual,
                target
            ),

        "fp_term_proxy":
            regression_metrics(
                fp_proxy,
                target
            ),
    }

    pd.DataFrame(
        metrics
    ).to_csv(
        BASE_DIR / "validation_gate_metrics.csv"
    )

    gate_passed = (
        all(
            metrics["graphsage"][key]
            <= metrics[name][key]

            for name in (
                "mlp",
                "residual_ratio",
                "fp_term_proxy"
            )

            for key in (
                "mae",
                "rmse"
            )
        )

        and

        metrics["graphsage"]["r2"]
        >= max(
            metrics[name]["r2"]

            for name in (
                "mlp",
                "residual_ratio",
                "fp_term_proxy"
            )
        )
    )

    if not gate_passed:

        print("[load] validation gate passed=False", flush=True)
        raise RuntimeError(
            "GraphSAGE validation gate failed; "
            "test seeds will not run."
        )

    print("[load] validation gate passed=True", flush=True)
    return metrics

def train_model(
    args,
    nodes_path,
    edges_path,
    model_path,
    state_path=None
):

    nodes = read_csv_slim(nodes_path)

    nodes, utility_stats = add_utility(
        nodes[
            nodes.seed.isin(
                range(1, 13)
            )
        ]
    )

    target_mean = float(
        nodes.utility.mean()
    )

    target_std = float(
        max(
            nodes.utility.std(),
            1e-8
        )
    )

    fitness_mean = float(
        nodes.F.mean()
    )

    fitness_std = float(
        max(
            nodes.F.std(),
            1e-8
        )
    )

    nodes.to_csv(
        nodes_path,
        index=False
    )

    del nodes
    gc.collect()

    graphs = load_graphs(
        nodes_path,
        edges_path,
        range(1, 13),
        graph_state_path=state_path,
        feature_set=args.feature_set,
        capacity_k=args.capacity_k,
        feature_cache_dir=args.feature_cache_dir,
    )

    skip_columns = (
        [FEATURE_COLUMNS.index("ap")]
        if "ap" in FEATURE_COLUMNS
        else []
    )
    feature_mean, feature_std = (
        normalize_features(
            graphs,
            skip_columns=skip_columns,
            feature_columns=FEATURE_COLUMNS,
        )
    )

    loader = DataLoader(
        graphs,
        batch_size=16,
        shuffle=True
    )

    mlp = train_mlp(
        graphs
    )

    torch.save(
        {
            "model_state_dict":
                mlp.state_dict(),

            "input_dim": graphs[0].num_node_features,

            "feature_columns":
                FEATURE_COLUMNS.copy(),

            "feature_set":
                args.feature_set,

            "feature_spec":
                make_feature_spec(args.capacity_k),

            "feature_mean":
                feature_mean.tolist(),

            "feature_std":
                feature_std.tolist(),
        },

        MLP_MODEL
    )

    del mlp

    model = GraphSAGE(
        graphs[0].num_node_features,
        args.hidden_dim,
        args.embedding_dim
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    print(
        f"Training GraphSAGE regression: "
        f"graphs={len(graphs)} "
        f"epochs={args.epochs}",
        flush=True
    )

    training_started = time.perf_counter()
    for epoch in range(
        1,
        args.epochs + 1
    ):

        loss, correlation = run_epoch(
            model,
            loader,
            optimizer
        )

        print(
            f"[load] GraphSAGE epoch {epoch}/{args.epochs}: "
            f"mse={loss:.6f} "
            f"correlation={correlation:.4f} "
            f"elapsed={time.perf_counter() - training_started:.1f}s",
            flush=True
        )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim": graphs[0].num_node_features,

            "hidden_dim":
                args.hidden_dim,

            "embedding_dim":
                args.embedding_dim,

            "feature_columns":
                FEATURE_COLUMNS.copy(),

            "feature_set":
                args.feature_set,

            "feature_spec":
                make_feature_spec(args.capacity_k),

            "feature_mean":
                feature_mean.tolist(),

            "feature_std":
                feature_std.tolist(),

            "utility_mean":
                utility_stats[0].tolist(),

            "utility_std":
                utility_stats[1].tolist(),

            "training_seeds":
                list(range(1, 13)),

            "target_mean":
                target_mean,

            "target_std":
                target_std,

            "fitness_mean":
                fitness_mean,

            "fitness_std":
                fitness_std,
        },

        model_path
    )
    with open(
        Path(model_path).with_name("feature_spec.json"),
        "w",
        encoding="utf-8",
    ) as feature_spec_file:
        json.dump(
            make_feature_spec(args.capacity_k),
            feature_spec_file,
            indent=2,
        )

    print(
        f"saved frozen model: {Path(model_path)}",
        flush=True
    )

    del graphs, loader, model, optimizer
    gc.collect()

def run_command(template, **values):
    command = template.format(**values)

    print(f"running: {command}", flush=True)

    result = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )

    output = result.stdout + "\n" + result.stderr

    patterns = {
        "First Node Dead":
            r"First Node Dead \(Stability Period\):\s*([0-9]+)",

        "Half Nodes Dead":
            r"Half Nodes Dead:\s*([0-9]+)",

        "Last Node Dead":
            r"Last Node Dead \(Network Lifetime\):\s*([0-9]+)",

        "Total Throughput":
            r"Total packets delivered to sink \(Throughput\):\s*([0-9]+)"
    }

    metrics = {}

    for name, pattern in patterns.items():
        match = re.search(pattern, output)

        if match:
            metrics[name] = float(match.group(1))
        else:
            metrics[name] = None

    print(f"  First Node Dead: {metrics['First Node Dead']}")
    print(f"  Half Nodes Dead: {metrics['Half Nodes Dead']}")
    print(f"  Last Node Dead: {metrics['Last Node Dead']}")
    print(f"  Total Throughput: {metrics['Total Throughput']}")

    return metrics

def paired_test(
    summary,
    left,
    right
):

    left_values = (
        summary[
            summary.condition == left
        ]
        .sort_values("seed")
        .lnd
        .to_numpy()
    )

    right_values = (
        summary[
            summary.condition == right
        ]
        .sort_values("seed")
        .lnd
        .to_numpy()
    )

    differences = (
        left_values - right_values
    )

    normal = (
        len(differences) >= 3
        and stats.shapiro(
            differences
        ).pvalue >= 0.05
    )

    if normal:

        result = stats.ttest_rel(
            left_values,
            right_values,
            alternative="greater"
        )

        method = "paired_t_test"

    else:

        result = stats.wilcoxon(
            left_values,
            right_values,
            alternative="greater"
        )

        method = "wilcoxon"

    mean_difference = float(
        np.mean(differences)
    )

    standard_error = (
        float(
            stats.sem(
                differences
            )
        )

        if len(differences) > 1

        else 0.0
    )

    critical = (
        float(
            stats.t.ppf(
                0.975,
                len(differences) - 1
            )
        )

        if len(differences) > 1

        else 0.0
    )

    return {
        "comparison":
            f"{left}_vs_{right}",

        "method":
            method,

        "p_value":
            float(result.pvalue),

        "mean_difference":
            mean_difference,

        "ci95_low":
            mean_difference
            - critical * standard_error,

        "ci95_high":
            mean_difference
            + critical * standard_error,
    }

def write_statistical_results(
    summary
):

    results = [
        paired_test(
            summary,
            "graphsage",
            "baseline"
        ),

        paired_test(
            summary,
            "mlp",
            "baseline"
        ),

        paired_test(
            summary,
            "graphsage",
            "mlp"
        ),
    ]

    ordered = sorted(
        enumerate(results),
        key=lambda item:
            item[1]["p_value"]
    )

    count = len(results)

    for rank, (
        index,
        result
    ) in enumerate(
        ordered,
        1
    ):

        result[
            "holm_adjusted_p_value"
        ] = min(
            1.0,
            result["p_value"]
            * (count - rank + 1)
        )

        result["primary"] = (
            index == 0
        )

    (
        BASE_DIR /
        "statistical_results.json"
    ).write_text(
        json.dumps(
            results,
            indent=2
        )
    )

def combine_csvs(prefixes, suffix, out_path):
    first = True

    for prefix in prefixes:

        chunk = read_csv_slim(
            BASE_DIR / f"{prefix}_{suffix}.csv"
        )

        chunk.to_csv(
            out_path,
            mode="w" if first else "a",
            header=first,
            index=False
        )

        del chunk

        first = False

    gc.collect()

def run_experiment(args):

    print(
        "Training seeds: 1-12; "
        "validation seeds: 13-16; "
        "test seeds: 17-36",
        flush=True
    )

    prefixes = [f"training_seed_{seed}" for seed in range(1, 13)]

    if args.skip_seed_generation:

        print(
            "Skipping seed 1-12 generation; "
            "reusing existing training_seed_* CSVs.",
            flush=True
        )

        missing = [
            prefix
            for prefix in prefixes
            for suffix in (
                "graphsage_training_nodes",
                "graphsage_training_edges",
                "graphsage_training_graph_nodes",
            )
            if not (BASE_DIR / f"{prefix}_{suffix}.csv").exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing expected per-seed CSVs for --skip-seed-generation: "
                + ", ".join(sorted(set(missing)))
            )

    else:

        for seed in range(1, 13):

            prefix = (
                f"training_seed_{seed}"
            )

            run_command(
                args.sim_command,

                seed=seed,

                data_mode="true",

                enabled="false",

                model=args.model,

                prefix=prefix,

                k=args.graphsage_k,

                exploration=
                    args.exploration_fraction,

                weight=0.0,

                mlp_enabled="false",

                mlp_model=args.mlp_model,

                fitness_mean=
                    args.fitness_mean,

                fitness_std=
                    args.fitness_std,

                utility_mean=
                    args.utility_mean,

                utility_std=
                    args.utility_std,
            )

    if args.skip_training:
        print(
            "Skipping training data combine + training; "
            "reusing saved models and combined training CSVs.",
            flush=True
        )
        for required in (GRAPHSAGE_MODEL, MLP_MODEL, TRAINING_NODES,
                         TRAINING_EDGES, TRAINING_STATE):
            if not Path(required).exists():
                raise FileNotFoundError(
                    f"--skip-training needs {required}"
                )
    else:
        combine_csvs(
            prefixes,
            "graphsage_training_nodes",
            TRAINING_NODES
        )

        combine_csvs(
            prefixes,
            "graphsage_training_edges",
            TRAINING_EDGES
        )

        combine_csvs(
            prefixes,
            "graphsage_training_graph_nodes",
            TRAINING_STATE
        )

        train_model(
            args,
            TRAINING_NODES,
            TRAINING_EDGES,
            GRAPHSAGE_MODEL,
            TRAINING_STATE
        )

    checkpoint = torch.load(
        GRAPHSAGE_MODEL,
        map_location="cpu"
    )

    common = dict(
        model=str(GRAPHSAGE_MODEL),

        k=args.graphsage_k,

        exploration=0.0,

        data_mode="false",

        weight=args.graphsage_weight,

        fitness_mean=
            checkpoint.get(
                "fitness_mean",
                args.fitness_mean
            ),

        fitness_std=
            checkpoint.get(
                "fitness_std",
                args.fitness_std
            ),

        utility_mean=
            checkpoint.get(
                "target_mean",
                args.utility_mean
            ),

        utility_std=
            checkpoint.get(
                "target_std",
                args.utility_std
            ),

        mlp_model=str(MLP_MODEL),
    )

    validation = []

    validation_prefixes = []

    for seed in range(13, 17):

        data_prefix = (
            f"validation_data_{seed}"
        )

        validation_prefixes.append(
            data_prefix
        )

        data_done = args.resume and all(
            (BASE_DIR / f"{data_prefix}_{sfx}.csv").exists()
            for sfx in (
                "graphsage_training_nodes",
                "graphsage_training_edges",
                "graphsage_training_graph_nodes",
            )
        )
        if data_done:
            print(f"resume: reusing {data_prefix}", flush=True)
        else:
            run_command(
                args.sim_command,

                seed=seed,

                enabled="false",

                prefix=data_prefix,

                model=str(GRAPHSAGE_MODEL),

                k=args.graphsage_k,

                exploration=0.0,

                data_mode="true",

                weight=0.0,

                fitness_mean=
                    args.fitness_mean,

                fitness_std=
                    args.fitness_std,

                utility_mean=
                    args.utility_mean,

                utility_std=
                    args.utility_std,

                mlp_enabled="false",

                mlp_model=str(MLP_MODEL),
            )

        for enabled, mlp_enabled, name in [

            ("false", "false", "baseline"),

            ("false", "true", "mlp"),

            ("true", "false", "graphsage"),

        ]:

            prefix = (
                f"validation_{name}_{seed}"
            )

            if args.resume and (
                BASE_DIR / f"{prefix}_summary.csv"
            ).exists():
                print(f"resume: reusing {prefix}", flush=True)
            else:
                run_command(
                    args.sim_command,

                    seed=seed,

                    enabled=enabled,

                    mlp_enabled=mlp_enabled,

                    prefix=prefix,

                    **common
                )

            summary = read_csv_slim(
                BASE_DIR /
                f"{prefix}_summary.csv"
            ).iloc[0].to_dict()

            summary.update(
                seed=seed,
                condition=name
            )

            validation.append(
                summary
            )

    pd.DataFrame(
        validation
    ).to_csv(
        BASE_DIR /
        "validation_summary.csv",
        index=False
    )

    combine_csvs(
        validation_prefixes,
        "graphsage_training_nodes",
        BASE_DIR / "validation_utility_nodes.csv"
    )

    combine_csvs(
        validation_prefixes,
        "graphsage_training_edges",
        BASE_DIR / "validation_utility_edges.csv"
    )

    combine_csvs(
        validation_prefixes,
        "graphsage_training_graph_nodes",
        BASE_DIR / "validation_graph_nodes.csv"
    )

    gate_metrics = evaluate_gate(
        GRAPHSAGE_MODEL,

        BASE_DIR /
        "validation_utility_nodes.csv",

        BASE_DIR /
        "validation_utility_edges.csv",

        BASE_DIR /
        "validation_graph_nodes.csv",

        TRAINING_NODES,

        TRAINING_EDGES,

        TRAINING_STATE,

        args.feature_set,

        args.capacity_k,

        args.feature_cache_dir,
    )

    print(
        json.dumps(
            {
                "validation_gate":
                    gate_metrics
            }
        ),
        flush=True
    )

    print(
        "Model frozen after training; "
        "validation seeds completed "
        "before test seeds.",
        flush=True
    )

    results = []

    test_conditions = set(args.test_conditions)

    for seed in range(17, 37):

        for enabled, mlp_enabled, name in [

            ("false", "false", "baseline"),

            ("false", "true", "mlp"),

            ("true", "false", "graphsage"),

        ]:

            if name not in test_conditions:
                continue

            prefix = (
                f"test_{name}_{seed}"
            )

            if args.resume and (
                BASE_DIR / f"{prefix}_summary.csv"
            ).exists():
                print(f"resume: reusing {prefix}", flush=True)
            else:
                run_command(
                    args.sim_command,

                    seed=seed,

                    enabled=enabled,

                    mlp_enabled=mlp_enabled,

                    prefix=prefix,

                    **common
                )

            summary = read_csv_slim(
                BASE_DIR /
                f"{prefix}_summary.csv"
            ).iloc[0].to_dict()

            summary.update(
                seed=seed,
                condition=name
            )

            results.append(
                summary
            )

            print(
                f"[test] {name} seed={seed} "
                f"FND={summary['fnd']} "
                f"HND={summary['hnd']} "
                f"LND={summary['lnd']}",
                flush=True
            )

    pd.DataFrame(
        results
    ).to_csv(
        BASE_DIR /
        "experiment_summary.csv",
        index=False
    )

    frame = pd.DataFrame(
        results
    )

    averages = (
        frame
        .groupby("condition")[
            [
                "fnd",
                "hnd",
                "lnd",
                "total_throughput",
                "final_energy",
                "avg_ch",
                "num_rounds",
            ]
        ]
        .mean()
    )

    averages.to_csv(
        BASE_DIR /
        "experiment_averages.csv"
    )

    print(
        "\n[test] mean FND/HND/LND per condition:\n"
        + averages[["fnd", "hnd", "lnd"]].to_string(),
        flush=True
    )

    if {"baseline", "mlp", "graphsage"} <= set(frame.condition):
        write_statistical_results(
            frame
        )
    else:
        print(
            "Skipping statistical tests: need baseline, mlp and "
            "graphsage conditions (ran: "
            f"{sorted(set(frame.condition))})",
            flush=True
        )

def main():
    global FEATURE_COLUMNS

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        action="store_true"
    )

    parser.add_argument(
        "--skip-seed-generation",
        action="store_true",
        help=(
            "Reuse existing training_seed_1..12 CSVs instead of "
            "regenerating them via the ns-3 simulation. Use this to "
            "resume the pipeline after seed generation already ran."
        )
    )

    parser.add_argument(
        "--sim-command",
        default=(
            f'"{NS3_EXECUTABLE}" '
            f'run {NS3_PROGRAM} -- '
            '--seed={seed} '

            '--graphSageDataMode={data_mode} '

            '--enableGraphSAGEData={data_mode} '

            '--enableGraphSAGE={enabled} '

            '--enableMLP={mlp_enabled} '

            '--graphSageModel="{model}" '

            '--mlpModel="{mlp_model}" '

            f'--graphSageInferenceScript="{INFERENCE_SCRIPT}" '

            '--graphSageK={k} '

            '--explorationFraction={exploration} '

            '--graphSageWeight={weight} '

            '--graphFitnessMean={fitness_mean} '

            '--graphFitnessStd={fitness_std} '

            '--graphUtilityMean={utility_mean} '

            '--graphUtilityStd={utility_std} '

            f'--graphSagePython="{sys.executable}" '

            f'--outputPrefix="{BASE_DIR}/{{prefix}}"'
        )
    )

    parser.add_argument(
        "--model",
        default=str(
            GRAPHSAGE_MODEL
        )
    )

    parser.add_argument(
        "--mlp-model",
        default=str(
            MLP_MODEL
        )
    )

    parser.add_argument(
        "--training-nodes",
        default=str(
            TRAINING_NODES
        )
    )

    parser.add_argument(
        "--training-edges",
        default=str(
            TRAINING_EDGES
        )
    )

    parser.add_argument(
        "--training-state",
        default=str(
            TRAINING_STATE
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345
    )

    parser.add_argument(
        "--graphsage-k",
        type=int,
        default=250
    )

    parser.add_argument(
        "--feature-set",
        choices=["slide", "legacy", "both"],
        default="slide",
    )

    parser.add_argument(
        "--capacity-k",
        type=float,
    )

    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--selftest",
        action="store_true",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse saved graphsage_model.pt / mlp_model.pt and the "
             "combined training CSVs; skip combine + training.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ns-3 runs whose output files already exist "
             "(validation and test phases).",
    )

    parser.add_argument(
        "--test-conditions",
        nargs="+",
        choices=["baseline", "mlp", "graphsage"],
        default=["baseline", "mlp", "graphsage"],
        help="Which conditions to run in the test phase (seeds 17-36).",
    )

    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--graphsage-weight",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--fitness-mean",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--fitness-std",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--utility-mean",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--utility-std",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    if args.selftest:
        graph_features_selftest()
        return

    FEATURE_COLUMNS = feature_columns_for(args.feature_set)
    if args.feature_set in ("slide", "both") and (
        args.capacity_k is None
        or not np.isfinite(args.capacity_k)
        or args.capacity_k <= 0
    ):
        raise ValueError(
            "--capacity-k is required for --feature-set slide or both "
            "and must be finite and greater than 0."
        )
    forbidden_features = set(
        UTILITY_COLUMNS + ["utility", "F", "sample_weight"]
    )
    if not set(FEATURE_COLUMNS).isdisjoint(forbidden_features):
        raise AssertionError(
            "Active feature columns overlap labels, fitness, or sample weights"
        )

    if args.experiment:

        run_experiment(args)

    else:

        train_model(
            args,
            args.training_nodes,
            args.training_edges,
            args.model,
            args.training_state
        )

if __name__ == "__main__":
    main()
