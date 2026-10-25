# ============================================================
# main_pipeline.py
# Centralized WSN Master Simulation
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch

from simulation1 import execute_simulation_round


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

NODE_FILE = "node_features.csv"
RESULT_FILE = "wsn_simulation_results.csv"

MAX_ROUNDS = 20000

NUM_CLUSTER_HEADS = 6

INITIAL_ENERGY = 2.0

BASE_STATION = (
    50.0,
    50.0
)

# ============================================================
# COMMUNICATION RADIUS
# ============================================================
# Must be consistent with the topology-generation code.
#
# Topology:
#   Number of nodes       = 200
#   Field                 = 100 x 100 m
#   Target average degree = 12
#
# R = sqrt(Target_Avg_Degree / (pi * node_density))
# ============================================================

NUM_NODES_FOR_RADIUS = 200

FIELD_LENGTH = 100.0
FIELD_WIDTH = 100.0

TARGET_AVG_DEGREE = 12

AREA = (
    FIELD_LENGTH *
    FIELD_WIDTH
)

NODE_DENSITY = (
    NUM_NODES_FOR_RADIUS /
    AREA
)

COMMUNICATION_RADIUS = np.sqrt(
    TARGET_AVG_DEGREE /
    (
        np.pi *
        NODE_DENSITY
    )
)

# ============================================================
# GRAPH SAGE + GA UPDATE
# ============================================================

CH_UPDATE_INTERVAL = 500

RANDOM_SEED = 42


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# LOAD NODE TOPOLOGY
# ============================================================

def load_node_data(
    filepath=NODE_FILE
):

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"Could not find {filepath}"
        )

    df = pd.read_csv(
        filepath
    )

    df.columns = (
        df.columns
        .str.strip()
    )

    required_columns = {
        "Node_ID",
        "X_Coordinate",
        "Y_Coordinate"
    }

    missing = (
        required_columns
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df["Node_ID"] = (
        df["Node_ID"]
        .astype(int)
    )

    df = (
        df
        .sort_values("Node_ID")
        .reset_index(drop=True)
    )

    expected_ids = set(
        range(len(df))
    )

    actual_ids = set(
        df["Node_ID"]
    )

    if actual_ids != expected_ids:

        raise ValueError(
            "Node_ID values must be exactly "
            "0 ... N-1."
        )

    return df


# ============================================================
# DISTANCE MATRIX
# ============================================================

def calculate_distance_matrix(
    positions
):

    positions = np.asarray(
        positions,
        dtype=float
    )

    difference = (
        positions[:, None, :]
        -
        positions[None, :, :]
    )

    return np.linalg.norm(
        difference,
        axis=2
    )


# ============================================================
# MASTER PIPELINE
# ============================================================

def run_master_pipeline():

    print()
    print("=" * 65)
    print("CENTRALIZED WSN SIMULATION")
    print("=" * 65)

    print(
        f"Node file              : {NODE_FILE}"
    )

    print(
        f"Maximum rounds         : {MAX_ROUNDS}"
    )

    print(
        f"Cluster heads          : {NUM_CLUSTER_HEADS}"
    )

    print(
        f"Initial energy         : "
        f"{INITIAL_ENERGY} J"
    )

    print(
        f"Field                  : "
        f"{FIELD_LENGTH} x {FIELD_WIDTH} m"
    )

    print(
        f"Target average degree  : "
        f"{TARGET_AVG_DEGREE}"
    )

    print(
        f"Communication radius   : "
        f"{COMMUNICATION_RADIUS:.4f} m"
    )

    print(
        f"Base station           : "
        f"{BASE_STATION}"
    )

    print(
        f"CH/GraphSAGE interval  : "
        f"{CH_UPDATE_INTERVAL} rounds"
    )

    print(
        f"Random seed            : "
        f"{RANDOM_SEED}"
    )

    print("=" * 65)


    # ========================================================
    # LOAD FIXED TOPOLOGY
    # ========================================================

    df = load_node_data()

    positions = (
        df[
            [
                "X_Coordinate",
                "Y_Coordinate"
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    num_nodes = len(df)

    print(
        f"\nNumber of nodes       : "
        f"{num_nodes}"
    )

    # --------------------------------------------------------
    # Verify that the topology size matches the experiment
    # configuration used to calculate the radius.
    # --------------------------------------------------------

    if num_nodes != NUM_NODES_FOR_RADIUS:

        raise ValueError(
            f"Topology contains {num_nodes} nodes, "
            f"but communication radius was calculated "
            f"for {NUM_NODES_FOR_RADIUS} nodes."
        )


    # ========================================================
    # STATIC DISTANCE MATRIX
    # ========================================================

    distance_matrix = (
        calculate_distance_matrix(
            positions
        )
    )


    # ========================================================
    # INITIAL NETWORK STATE
    # ========================================================

    energy = np.full(
        num_nodes,
        INITIAL_ENERGY,
        dtype=float
    )

    alive = np.ones(
        num_nodes,
        dtype=bool
    )

    cluster_heads = None

    # GraphSAGE model
    graphsage_model = None

    graphsage_optimizer = None

    graphsage_embeddings = None


    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    fnd = None
    hnd = None
    lnd = None

    network_lifetime = None
    stopping_reason = None


    # ========================================================
    # MAIN SIMULATION LOOP
    # ========================================================

    for round_num in range(
        1,
        MAX_ROUNDS + 1
    ):

        # ----------------------------------------------------
        # Current number of alive nodes
        # ----------------------------------------------------

        alive_count_before = int(
            np.sum(alive)
        )

        # ----------------------------------------------------
        # Complete network death
        # ----------------------------------------------------

        if alive_count_before == 0:

            if lnd is None:

                lnd = (
                    round_num - 1
                )

            network_lifetime = (
                round_num - 1
            )

            stopping_reason = (
                "All nodes are dead"
            )

            break


        # ====================================================
        # EXECUTE ONE COMPLETE ROUND
        # ====================================================

        try:

            (
                energy,
                alive,
                cluster_heads,
                graphsage_model,
                graphsage_optimizer,
                graphsage_embeddings,
                success

            ) = execute_simulation_round(

                round_num=round_num,

                positions=positions,

                distance_matrix=distance_matrix,

                energy=energy,

                alive=alive,

                cluster_heads=cluster_heads,

                graphsage_model=graphsage_model,

                graphsage_optimizer=graphsage_optimizer,

                graphsage_embeddings=graphsage_embeddings,

                df=df,

                comm_radius=COMMUNICATION_RADIUS,

                num_cluster_heads=NUM_CLUSTER_HEADS,

                base_station=BASE_STATION,

                update_interval=CH_UPDATE_INTERVAL,

                use_graphsage=False

            )

        except RuntimeError as e:

                # ------------------------------------------------
                # A runtime failure occurred during the round.
                # Do NOT assume that every RuntimeError is caused
                # by GA infeasibility.
                # ------------------------------------------------

                network_lifetime = (
                    round_num - 1
                )

                stopping_reason = (
                    "Runtime failure during simulation"
                )

                print()
                print("=" * 65)
                print("NETWORK SIMULATION FAILURE")
                print("=" * 65)

                print(
                    f"Failure detected at round : "
                    f"{round_num}"
                )

                print(
                    f"Last successful round     : "
                    f"{network_lifetime}"
                )

                print(
                    f"Exception type            : "
                    f"{type(e).__name__}"
                )

                print(
                    f"Exact error               : "
                    f"{e}"
                )

                print("=" * 65)

                break


        # ----------------------------------------------------
        # Explicit unsuccessful round
        # ----------------------------------------------------

        if not success:

            network_lifetime = (
                round_num - 1
            )

            stopping_reason = (
                "Simulation could not complete "
                "the current round"
            )

            print()
            print(
                f"Simulation stopped at "
                f"round {round_num}."
            )

            break


        # ====================================================
        # CURRENT NETWORK STATE
        # ====================================================

        alive_count = int(
            np.sum(alive)
        )

        dead_count = (
            num_nodes
            -
            alive_count
        )

        total_residual_energy = float(
            np.sum(energy)
        )

        average_alive_energy = (

            total_residual_energy
            /
            alive_count

            if alive_count > 0

            else 0.0

        )


        # ====================================================
        # FND
        # ====================================================

        if (
            fnd is None
            and
            dead_count >= 1
        ):

            fnd = round_num


        # ====================================================
        # HND
        # ====================================================

        if (
            hnd is None
            and
            alive_count <= num_nodes / 2
        ):

            hnd = round_num


        # ====================================================
        # LND
        # ====================================================

        if (
            alive_count == 0
            and
            lnd is None
        ):

            lnd = round_num


        # ====================================================
        # STORE RESULTS
        # ========================================================

        results.append({

            "Round":
                round_num,

            "Alive_Nodes":
                alive_count,

            "Dead_Nodes":
                dead_count,

            "Total_Residual_Energy":
                total_residual_energy,

            "Average_Alive_Node_Energy":
                average_alive_energy,

            "Cluster_Heads":

                ",".join(
                    map(
                        str,
                        sorted(
                            cluster_heads
                        )
                    )
                )

                if cluster_heads is not None

                else ""

        })


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            round_num == 1
            or
            round_num % 100 == 0
        ):

            print(

                f"Round {round_num:5d} | "
                f"Alive = {alive_count:3d} | "
                f"Dead = {dead_count:3d} | "
                f"Residual Energy = "
                f"{total_residual_energy:.6f} J"

            )


        # ----------------------------------------------------
        # All nodes dead
        # ----------------------------------------------------

        if alive_count == 0:

            lnd = round_num

            network_lifetime = round_num

            stopping_reason = (
                "All nodes are dead"
            )

            break


    # ========================================================
    # IF MAXIMUM ROUNDS WERE REACHED
    # ========================================================

    if network_lifetime is None:

        network_lifetime = MAX_ROUNDS

        stopping_reason = (
            "Maximum simulation rounds reached"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_alive = int(
        np.sum(alive)
    )

    final_dead = (
        num_nodes
        -
        final_alive
    )

    final_residual_energy = float(
        np.sum(energy)
    )

    final_average_alive_energy = (

        final_residual_energy /
        final_alive

        if final_alive > 0

        else 0.0

    )


    print()
    print("=" * 65)
    print("SIMULATION COMPLETE")
    print("=" * 65)

    print(
        f"FND                         : "
        f"{fnd if fnd is not None else 'Not reached'}"
    )

    print(
        f"HND                         : "
        f"{hnd if hnd is not None else 'Not reached'}"
    )

    print(
        f"LND                         : "
        f"{lnd if lnd is not None else 'Not reached'}"
    )

    print(
        f"Network Lifetime            : "
        f"{network_lifetime} rounds"
    )

    print(
        f"Final Alive Nodes           : "
        f"{final_alive}"
    )

    print(
        f"Final Dead Nodes            : "
        f"{final_dead}"
    )

    print(
        f"Final Residual Energy       : "
        f"{final_residual_energy:.6f} J"
    )

    print(
        f"Final Average Alive Energy  : "
        f"{final_average_alive_energy:.6f} J"
    )

    print(
        f"Stopping Reason             : "
        f"{stopping_reason}"
    )

    print(
        f"Results saved to            : "
        f"{RESULT_FILE}"
    )

    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_master_pipeline()
