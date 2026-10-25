# ============================================================
# GraphSAGE for Dynamic WSN
#
# Uses:
#   1. Current network topology
#   2. Dataset/node features
#
# Output:
#   64-dimensional node embeddings
#
# Protocol:
#   Round 1:
#       GraphSAGE is trained for INITIAL_EPOCHS.
#
#   Every later update round:
#       GraphSAGE is used in INFERENCE-ONLY mode.
#
#   Therefore:
#       GraphSAGE weights are learned initially and then kept fixed.
#       Embeddings change because the network topology and node
#       features change as nodes die and energy decreases.
# ============================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling


# ============================================================
# CONFIGURATION
# ============================================================

HIDDEN_DIM = 32
EMBEDDING_DIM = 64

INITIAL_EPOCHS = 200

LEARNING_RATE = 0.01

RANDOM_SEED = 42

# Base station
BASE_STATION = (50.0, 50.0)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GRAPHSAGE MODEL
# ============================================================

class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=EMBEDDING_DIM
    ):

        super().__init__()

        # First GraphSAGE layer
        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        # Second GraphSAGE layer
        self.conv2 = SAGEConv(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        x,
        edge_index
    ):

        # ----------------------------------------------------
        # GraphSAGE Layer 1
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        # ----------------------------------------------------
        # GraphSAGE Layer 2
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# BUILD CURRENT GRAPH
# ============================================================

def build_current_graph(
    positions,
    alive,
    communication_radius
):

    num_nodes = len(positions)

    graph = nx.Graph()

    # --------------------------------------------------------
    # Only currently alive nodes are included
    # --------------------------------------------------------

    alive_nodes = np.where(alive)[0]

    graph.add_nodes_from(
        alive_nodes.tolist()
    )

    # --------------------------------------------------------
    # Euclidean communication graph
    # --------------------------------------------------------

    for i_index in range(
        len(alive_nodes)
    ):

        node_i = int(
            alive_nodes[i_index]
        )

        for j_index in range(
            i_index + 1,
            len(alive_nodes)
        ):

            node_j = int(
                alive_nodes[j_index]
            )

            distance = np.linalg.norm(
                positions[node_i]
                -
                positions[node_j]
            )

            if distance <= communication_radius:

                graph.add_edge(
                    node_i,
                    node_j
                )

    return graph


# ============================================================
# TOPOLOGY FEATURES
# ============================================================

def create_topology_features(
    graph,
    num_nodes
):

    # Six topology features:
    #
    # 1. Degree
    # 2. Clustering coefficient
    # 3. Average neighbor degree
    # 4. Betweenness centrality
    # 5. Closeness centrality
    # 6. K-core number

    features = np.zeros(
        (num_nodes, 6),
        dtype=np.float32
    )

    alive_nodes = list(
        graph.nodes()
    )

    if len(alive_nodes) == 0:

        return features

    # --------------------------------------------------------
    # 1. Degree
    # --------------------------------------------------------

    degree = dict(
        graph.degree()
    )

    # --------------------------------------------------------
    # 2. Clustering coefficient
    # --------------------------------------------------------

    clustering = nx.clustering(
        graph
    )

    # --------------------------------------------------------
    # 3. Average neighbor degree
    # --------------------------------------------------------

    average_neighbor_degree = (
        nx.average_neighbor_degree(
            graph
        )
    )

    # --------------------------------------------------------
    # 4. Betweenness centrality
    # --------------------------------------------------------

    betweenness = (
        nx.betweenness_centrality(
            graph,
            normalized=True
        )
    )

    # --------------------------------------------------------
    # 5. Closeness centrality
    # --------------------------------------------------------

    closeness = (
        nx.closeness_centrality(
            graph
        )
    )

    # --------------------------------------------------------
    # 6. K-core number
    # --------------------------------------------------------

    if graph.number_of_edges() > 0:

        core_number = (
            nx.core_number(
                graph
            )
        )

    else:

        core_number = {
            node: 0
            for node in alive_nodes
        }

    # --------------------------------------------------------
    # Construct feature matrix
    # --------------------------------------------------------

    for node in alive_nodes:

        features[node, 0] = degree.get(
            node,
            0
        )

        features[node, 1] = clustering.get(
            node,
            0
        )

        features[node, 2] = (
            average_neighbor_degree.get(
                node,
                0
            )
        )

        features[node, 3] = betweenness.get(
            node,
            0
        )

        features[node, 4] = closeness.get(
            node,
            0
        )

        features[node, 5] = core_number.get(
            node,
            0
        )

    return features


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

def standardize_alive_features(
    features,
    alive
):

    alive_indices = np.where(alive)[0]

    if len(alive_indices) == 0:

        return features

    alive_values = (
        features[alive_indices]
    )

    mean = alive_values.mean(
        axis=0
    )

    std = alive_values.std(
        axis=0
    )

    # Avoid division by zero
    std[std == 0] = 1.0

    features[alive_indices] = (
        alive_values - mean
    ) / std

    return features


# ============================================================
# DATASET FEATURES
# ============================================================

def create_dataset_features(
    df,
    alive,
    positions,
    base_station=BASE_STATION
):

    num_nodes = len(df)

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "Residual_Energy",
        "X_Coordinate",
        "Y_Coordinate"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required dataset columns: "
            +
            str(missing_columns)
        )

    # --------------------------------------------------------
    # Create feature matrix
    #
    # 4 features:
    #
    # 1. Residual Energy
    # 2. X Coordinate
    # 3. Y Coordinate
    # 4. Distance to Base Station
    # --------------------------------------------------------

    features = np.zeros(
        (num_nodes, 4),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Residual energy
    # --------------------------------------------------------

    features[:, 0] = (
        df["Residual_Energy"]
        .astype(np.float32)
        .to_numpy()
    )

    # --------------------------------------------------------
    # X coordinate
    # --------------------------------------------------------

    features[:, 1] = (
        df["X_Coordinate"]
        .astype(np.float32)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Y coordinate
    # --------------------------------------------------------

    features[:, 2] = (
        df["Y_Coordinate"]
        .astype(np.float32)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Distance to Base Station
    # --------------------------------------------------------

    bs_x = float(
        base_station[0]
    )

    bs_y = float(
        base_station[1]
    )

    x = features[:, 1]

    y = features[:, 2]

    distance_to_bs = np.sqrt(
        (x - bs_x) ** 2
        +
        (y - bs_y) ** 2
    )

    features[:, 3] = (
        distance_to_bs
    )

    # --------------------------------------------------------
    # Standardize using alive nodes only
    # --------------------------------------------------------

    features = standardize_alive_features(
        features,
        alive
    )

    # --------------------------------------------------------
    # Dead nodes receive zero features
    # --------------------------------------------------------

    features[~alive] = 0.0

    return features


# ============================================================
# COMBINE TOPOLOGY + DATASET FEATURES
# ============================================================

def create_node_features(
    graph,
    df,
    alive,
    positions,
    base_station=BASE_STATION
):

    num_nodes = len(df)

    # --------------------------------------------------------
    # Topology features
    # --------------------------------------------------------

    topology_features = (
        create_topology_features(
            graph=graph,
            num_nodes=num_nodes
        )
    )

    # Standardize topology features
    topology_features = (
        standardize_alive_features(
            topology_features,
            alive
        )
    )

    # --------------------------------------------------------
    # Dataset/state features
    # --------------------------------------------------------

    dataset_features = (
        create_dataset_features(
            df=df,
            alive=alive,
            positions=positions,
            base_station=base_station
        )
    )

    # --------------------------------------------------------
    # Concatenate
    #
    # 6 topology + 4 dataset = 10 features
    # --------------------------------------------------------

    combined_features = np.concatenate(
        [
            topology_features,
            dataset_features
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Dead nodes receive zero features
    # --------------------------------------------------------

    combined_features[~alive] = 0.0

    return combined_features


# ============================================================
# CREATE EDGE INDEX
# ============================================================

def create_edge_index(graph):

    edges = list(
        graph.edges()
    )

    if not edges:

        return torch.empty(
            (2, 0),
            dtype=torch.long
        )

    edge_list = []

    for source, target in edges:

        # Graph is undirected,
        # so explicitly add both directions.

        edge_list.append(
            [source, target]
        )

        edge_list.append(
            [target, source]
        )

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long
    ).t().contiguous()

    return edge_index


# ============================================================
# UNSUPERVISED LINK-PREDICTION LOSS
# ============================================================

def compute_loss(
    embedding,
    edge_index
):

    if edge_index.shape[1] == 0:

        return None

    # ========================================================
    # POSITIVE EDGES
    # ========================================================

    source_nodes = (
        edge_index[0]
    )

    target_nodes = (
        edge_index[1]
    )

    positive_score = torch.sum(
        embedding[source_nodes]
        *
        embedding[target_nodes],
        dim=-1
    )

    positive_loss = (
        -F.logsigmoid(
            positive_score
        ).mean()
    )

    # ========================================================
    # NEGATIVE EDGES
    # ========================================================

    negative_edge_index = (
        negative_sampling(
            edge_index=edge_index,
            num_nodes=embedding.size(0),
            num_neg_samples=edge_index.size(1),
            method="sparse"
        )
    )

    if negative_edge_index.shape[1] == 0:

        return positive_loss

    negative_source = (
        negative_edge_index[0]
    )

    negative_target = (
        negative_edge_index[1]
    )

    negative_score = torch.sum(
        embedding[negative_source]
        *
        embedding[negative_target],
        dim=-1
    )

    negative_loss = (
        -F.logsigmoid(
            -negative_score
        ).mean()
    )

    # ========================================================
    # TOTAL LOSS
    # ========================================================

    return (
        positive_loss
        +
        negative_loss
    )


# ============================================================
# GENERATE GRAPH SAGE EMBEDDINGS
# ============================================================

def generate_graphsage_embeddings(
    positions,
    alive,
    communication_radius,
    df,
    model=None,
    optimizer=None,
    output_file=None,
    inference_only=False,
    base_station=BASE_STATION
):

    num_nodes = len(
        positions
    )

    # ========================================================
    # 1. BUILD CURRENT GRAPH
    # ========================================================

    graph = build_current_graph(
        positions=positions,
        alive=alive,
        communication_radius=communication_radius
    )

    # ========================================================
    # 2. CREATE CURRENT FEATURES
    # ========================================================

    feature_matrix = (
        create_node_features(
            graph=graph,
            df=df,
            alive=alive,
            positions=positions,
            base_station=base_station
        )
    )

    # ========================================================
    # 3. CREATE EDGE INDEX
    # ========================================================

    edge_index = (
        create_edge_index(
            graph
        )
    )

    if edge_index.shape[1] == 0:

        raise RuntimeError(
            "Current WSN graph contains no edges."
        )

    # ========================================================
    # 4. CREATE PYTORCH DATA
    # ========================================================

    x = torch.tensor(
        feature_matrix,
        dtype=torch.float32
    )

    data = Data(
        x=x,
        edge_index=edge_index
    )

    # ========================================================
    # 5. INITIALIZE MODEL
    # ========================================================

    if model is None:

        # ----------------------------------------------------
        # Initial training
        # ----------------------------------------------------

        set_seed()

        model = GraphSAGE(
            input_dim=x.shape[1],
            hidden_dim=HIDDEN_DIM,
            output_dim=EMBEDDING_DIM
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        training_mode = True

        epochs = INITIAL_EPOCHS

        print(
            "\n============================================"
        )

        print(
            "Initial GraphSAGE training"
        )

        print(
            "============================================"
        )

    else:

        # ----------------------------------------------------
        # Existing model
        # ----------------------------------------------------

        if inference_only:

            training_mode = False

            epochs = 0

            print(
                "\n============================================"
            )

            print(
                "GraphSAGE inference-only"
            )

            print(
                "============================================"
            )

        else:

            raise ValueError(
                "An existing GraphSAGE model was provided "
                "with inference_only=False. "
                "For this experiment, the existing model "
                "must remain fixed."
            )

    # ========================================================
    # INFORMATION
    # ========================================================

    print(
        "Alive nodes:",
        int(np.sum(alive))
    )

    print(
        "Graph nodes:",
        graph.number_of_nodes()
    )

    print(
        "Graph edges:",
        graph.number_of_edges()
    )

    print(
        "Input feature dimension:",
        x.shape[1]
    )

    print(
        "Expected input dimension:",
        10
    )

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    # ========================================================
    # 6. INITIAL TRAINING ONLY
    # ========================================================

    if training_mode:

        for epoch in range(
            1,
            epochs + 1
        ):

            model.train()

            optimizer.zero_grad()

            embedding = model(
                data.x,
                data.edge_index
            )

            loss = compute_loss(
                embedding,
                data.edge_index
            )

            if loss is None:

                raise RuntimeError(
                    "Unable to compute GraphSAGE loss."
                )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Progress output
            # ------------------------------------------------

            if (
                epoch == 1
                or
                epoch % 25 == 0
                or
                epoch == epochs
            ):

                print(
                    f"Epoch {epoch:3d} / "
                    f"{epochs} | "
                    f"Loss = {loss.item():.6f}"
                )

    # ========================================================
    # 7. INFERENCE
    # ========================================================

    model.eval()

    with torch.no_grad():

        embedding = (
            model(
                data.x,
                data.edge_index
            )
            .cpu()
            .numpy()
        )

    # ========================================================
    # 8. DEAD NODES = ZERO EMBEDDING
    # ========================================================

    embedding[~alive] = 0.0

    # ========================================================
    # 9. SAVE EMBEDDINGS
    # ========================================================

    if output_file is not None:

        embedding_df = pd.DataFrame(
            embedding,
            columns=[
                f"GSAGE_{i}"
                for i in range(
                    EMBEDDING_DIM
                )
            ]
        )

        embedding_df.insert(
            0,
            "Node_ID",
            np.arange(num_nodes)
        )

        embedding_df.to_csv(
            output_file,
            index=False
        )

    # ========================================================
    # 10. RETURN
    # ========================================================

    return (
        embedding,
        model,
        optimizer
    )
