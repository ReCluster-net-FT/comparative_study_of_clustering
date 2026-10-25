import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ==========================================================
# RESEARCH PARAMETERS
# ==========================================================

DATASET = "WSN_Dataset.csv"

NUM_NODES = 200
FIELD_LENGTH = 100          # metres
FIELD_WIDTH = 100           # metres

TARGET_AVG_DEGREE = 12
SEED = 42

BASE_STATION = (50, 50)

# ==========================================================
# STEP 1 : LOAD DATASET
# ==========================================================

df = pd.read_csv(DATASET)
df.columns = df.columns.str.strip()

print("Original dataset records :", len(df))

# Reproducible sampling
df = (
    df.sample(n=NUM_NODES, random_state=SEED)
      .reset_index(drop=True)
)

# New node IDs
df["Node_ID"] = range(NUM_NODES)

print("Experimental nodes :", len(df))

# ==========================================================
# STEP 2 : GENERATE WSN DEPLOYMENT
# ==========================================================

np.random.seed(SEED)

df["X_Coordinate"] = np.random.uniform(
    0,
    FIELD_LENGTH,
    NUM_NODES
)

df["Y_Coordinate"] = np.random.uniform(
    0,
    FIELD_WIDTH,
    NUM_NODES
)

# ==========================================================
# STEP 3 : CALCULATE COMMUNICATION RADIUS
# ==========================================================

AREA = FIELD_LENGTH * FIELD_WIDTH

RHO = NUM_NODES / AREA

COMM_RADIUS = np.sqrt(
    TARGET_AVG_DEGREE /
    (np.pi * RHO)
)

print(f"Communication Radius : {COMM_RADIUS:.2f} m")

# ==========================================================
# STEP 4 : DISTANCE MATRIX
# ==========================================================

coords = df[
    ["X_Coordinate", "Y_Coordinate"]
].values

distance_matrix = cdist(coords, coords)

adjacency = (
    distance_matrix <= COMM_RADIUS
).astype(np.uint8)

np.fill_diagonal(adjacency, 0)

# ==========================================================
# STEP 5 : BUILD GRAPH
# ==========================================================

G = nx.Graph()

for _, row in df.iterrows():

    G.add_node(
        int(row["Node_ID"]),
        x=row["X_Coordinate"],
        y=row["Y_Coordinate"],
        energy=row["Residual_Energy"]
    )

edge_data = []

for i in range(NUM_NODES):

    for j in range(i + 1, NUM_NODES):

        if adjacency[i, j]:

            dist = float(distance_matrix[i, j])

            G.add_edge(i, j, weight=dist)

            edge_data.append([
                i,
                j,
                dist
            ])

# ==========================================================
# STEP 6 : NODE FEATURES
# ==========================================================

degree = dict(G.degree())

df["Degree"] = df["Node_ID"].map(degree)

bx, by = BASE_STATION

df["BS_Distance"] = np.sqrt(
    (df["X_Coordinate"] - bx) ** 2 +
    (df["Y_Coordinate"] - by) ** 2
)

# ==========================================================
# STEP 7 : SAVE FILES
# ==========================================================

pd.DataFrame(
    edge_data,
    columns=[
        "Source",
        "Target",
        "Distance"
    ]
).to_csv(
    "edge_list.csv",
    index=False
)

pd.DataFrame(adjacency).to_csv(
    "adjacency_matrix.csv",
    index=False
)

df.to_csv(
    "node_features.csv",
    index=False
)

# ==========================================================
# STEP 8 : TOPOLOGY IMAGE
# ==========================================================

plt.figure(figsize=(8, 8))

pos = {
    int(r["Node_ID"]):
    (
        r["X_Coordinate"],
        r["Y_Coordinate"]
    )
    for _, r in df.iterrows()
}

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.25,
    width=0.6,
    edge_color="gray"
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=35,
    node_color=df["Residual_Energy"],
    cmap="viridis"
)

plt.scatter(
    bx,
    by,
    marker="*",
    c="red",
    s=260,
    label="Base Station"
)

plt.title("200-Node Random Geometric WSN")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.grid(alpha=0.2)
plt.legend()

plt.savefig(
    "topology.png",
    dpi=300
)

plt.close()

# ==========================================================
# STEP 9 : SUMMARY
# ==========================================================

print("\n========== TOPOLOGY SUMMARY ==========")

print("Nodes               :", G.number_of_nodes())
print("Edges               :", G.number_of_edges())

avg_degree = np.mean(
    list(degree.values())
)

print("Average Degree      :", round(avg_degree, 2))

print("Connected Graph     :", nx.is_connected(G))

print("\nGenerated Files")
print(" edge_list.csv")
print(" adjacency_matrix.csv")
print(" node_features.csv")
print(" topology.png")
