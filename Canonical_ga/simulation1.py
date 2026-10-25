# ============================================================
# simulation1.py
#
# Centralized WSN Simulation
#
# Communication model:
#
#   Ordinary Node -> Assigned CH only
#   Ordinary Node -> BS              NOT ALLOWED
#   Ordinary Node -> Ordinary Node   NOT ALLOWED
#   CH -> CH                         ALLOWED
#   CH -> BS                         ALLOWED
#
# Routing:
#
#   CH-only multi-hop routing
#   BFS-based
#   No "must be closer to BS" restriction
#
# CH selection:
#
#   GraphSAGE + GA
#   Periodic reselection
#   Emergency reselection at beginning of a round
#
# Runtime routing failure:
#
#   DOES NOT terminate simulation.
#   Unroutable CH packets are lost for that round.
#   Next round performs emergency CH reselection.
# ============================================================

import math
import numpy as np

import ga_cluster_head
import gat_embedding

from ga_router import WSN_GA_Router


# ============================================================
# RADIO PARAMETERS
# ============================================================

PACKET_SIZE = 4000

E_ELEC = 50e-9
E_FS = 10e-12
E_MP = 0.0013e-12
E_DA = 5e-9


# ============================================================
# RADIO MODEL
# ============================================================

def calculate_d0():

    return math.sqrt(
        E_FS / E_MP
    )


def tx_energy(
    distance,
    bits=PACKET_SIZE
):
    """
    First-order radio model.
    """

    distance = float(distance)

    if distance < 0:
        raise ValueError(
            "Distance cannot be negative."
        )

    d0 = calculate_d0()

    if distance <= d0:

        return bits * (
            E_ELEC +
            E_FS * distance ** 2
        )

    return bits * (
        E_ELEC +
        E_MP * distance ** 4
    )


def rx_energy(
    bits=PACKET_SIZE
):

    return bits * E_ELEC


def aggregation_energy(
    number_of_packets,
    bits=PACKET_SIZE
):

    return (
        number_of_packets *
        bits *
        E_DA
    )


# ============================================================
# DISTANCE FUNCTIONS
# ============================================================

def node_distance(
    df,
    node_a,
    node_b
):

    node_a = int(node_a)
    node_b = int(node_b)

    xa = float(
        df.loc[node_a, "X_Coordinate"]
    )

    ya = float(
        df.loc[node_a, "Y_Coordinate"]
    )

    xb = float(
        df.loc[node_b, "X_Coordinate"]
    )

    yb = float(
        df.loc[node_b, "Y_Coordinate"]
    )

    return math.sqrt(
        (xa - xb) ** 2 +
        (ya - yb) ** 2
    )


def distance_to_bs(
    df,
    node,
    base_station
):

    node = int(node)

    x = float(
        df.loc[node, "X_Coordinate"]
    )

    y = float(
        df.loc[node, "Y_Coordinate"]
    )

    bx, by = base_station

    return math.sqrt(
        (x - bx) ** 2 +
        (y - by) ** 2
    )


# ============================================================
# UPDATE ALIVE STATUS
# ============================================================

def update_alive_status(
    energy,
    alive
):

    energy = np.asarray(
        energy,
        dtype=float
    ).copy()

    alive = np.asarray(
        alive,
        dtype=bool
    ).copy()

    dead = energy <= 0.0

    energy[energy < 0] = 0.0

    alive[dead] = False

    return energy, alive


# ============================================================
# CHECK CH CONFIGURATION
# ============================================================

def check_ch_configuration_feasibility(
    df,
    alive,
    cluster_heads,
    base_station,
    communication_radius
):
    """
    Strict feasibility test.

    Requirements:

    1. Selected CH must be alive.
    2. At least one CH must directly reach BS.
    3. Every selected CH must have a CH-to-CH path to a
       BS-reachable CH.

    No monotonic BS-distance restriction.
    """

    if cluster_heads is None:
        return False

    alive_chs = [
        int(ch)
        for ch in cluster_heads
        if alive[int(ch)]
    ]

    if len(alive_chs) == 0:
        return False

    # --------------------------------------------------------
    # CHs directly reachable from BS
    # --------------------------------------------------------

    bs_reachable = []

    for ch in alive_chs:

        d = distance_to_bs(
            df,
            ch,
            base_station
        )

        if d <= communication_radius:

            bs_reachable.append(ch)

    if len(bs_reachable) == 0:

        return False

    # --------------------------------------------------------
    # Build CH graph
    # --------------------------------------------------------

    graph = {
        ch: []
        for ch in alive_chs
    }

    for i in range(
        len(alive_chs)
    ):

        ch_a = alive_chs[i]

        for j in range(
            i + 1,
            len(alive_chs)
        ):

            ch_b = alive_chs[j]

            d = node_distance(
                df,
                ch_a,
                ch_b
            )

            if d <= communication_radius:

                graph[ch_a].append(ch_b)
                graph[ch_b].append(ch_a)

    # --------------------------------------------------------
    # Multi-source BFS from BS-connected CHs
    # --------------------------------------------------------

    visited = set(
        bs_reachable
    )

    queue = list(
        bs_reachable
    )

    while queue:

        current = queue.pop(0)

        for neighbour in graph[current]:

            if neighbour in visited:
                continue

            visited.add(neighbour)

            queue.append(neighbour)

    return all(
        ch in visited
        for ch in alive_chs
    )


# ============================================================
# MEMBER ASSIGNMENT
# ============================================================

def assign_members_to_cluster_heads(
    df,
    alive,
    cluster_heads,
    communication_radius
):
    """
    Ordinary nodes communicate only with CHs.

    Each ordinary alive node is assigned to its closest
    alive CH within communication range.

    No ordinary node communicates directly with BS.
    """

    assignments = {}

    alive_chs = [
        int(ch)
        for ch in cluster_heads
        if alive[int(ch)]
    ]

    if len(alive_chs) == 0:

        return assignments

    for node in range(
        len(alive)
    ):

        if not alive[node]:
            continue

        if node in alive_chs:
            continue

        best_ch = None
        best_distance = float("inf")

        for ch in alive_chs:

            d = node_distance(
                df,
                node,
                ch
            )

            if d <= communication_radius:

                if d < best_distance:

                    best_distance = d
                    best_ch = ch

        assignments[node] = best_ch

    return assignments


# ============================================================
# MEMBER -> CH TRANSMISSION
# ============================================================

def transmit_members_to_cluster_heads(
    df,
    energy,
    alive,
    assignments,
    communication_radius
):

    energy = np.asarray(
        energy,
        dtype=float
    ).copy()

    received_packets = {}

    for node, ch in assignments.items():

        node = int(node)

        if ch is None:
            continue

        ch = int(ch)

        if not alive[node]:
            continue

        if not alive[ch]:
            continue

        distance = node_distance(
            df,
            node,
            ch
        )

        if distance > communication_radius:
            continue

        tx_required = tx_energy(
            distance,
            PACKET_SIZE
        )

        rx_required = rx_energy(
            PACKET_SIZE
        )

        # ----------------------------------------------------
        # Sender cannot transmit
        # ----------------------------------------------------

        if energy[node] < tx_required:

            energy[node] = 0.0
            alive[node] = False

            continue

        # ----------------------------------------------------
        # CH cannot receive
        # ----------------------------------------------------

        if energy[ch] < rx_required:

            energy[ch] = 0.0
            alive[ch] = False

            continue

        # ----------------------------------------------------
        # Perform communication
        # ----------------------------------------------------

        energy[node] -= tx_required

        energy[ch] -= rx_required

        received_packets[ch] = (
            received_packets.get(ch, 0)
            + 1
        )

    return (
        energy,
        received_packets
    )


# ============================================================
# ROUTE CHs
# ============================================================

def route_cluster_heads(
    df,
    energy,
    alive,
    cluster_heads,
    base_station,
    communication_radius,
    distance_matrix=None
):

    energy = np.asarray(
        energy,
        dtype=float
    ).copy()

    alive_chs = [
        int(ch)
        for ch in cluster_heads
        if alive[int(ch)]
    ]

    if len(alive_chs) == 0:

        return (
            energy,
            {},
            None
        )

    router = WSN_GA_Router(
        df=df,
        cluster_heads=alive_chs,
        base_station=base_station,
        max_tx_range=communication_radius,
        packet_size=PACKET_SIZE,
        E_elec=E_ELEC,
        E_fs=E_FS,
        E_mp=E_MP,
        E_DA=E_DA,
        dist_matrix=distance_matrix
    )

    # IMPORTANT:
    # ga_router.py does not terminate when some CHs
    # are disconnected.
    next_hop_table, routes = (
        router.build_all_routes()
    )

    # --------------------------------------------------------
    # Report routing status
    # --------------------------------------------------------

    unroutable = [
        ch
        for ch in alive_chs
        if routes.get(ch) is None
    ]

    if unroutable:

        print(
            "\nWARNING: Unroutable CHs this round:",
            unroutable
        )

    return (
        energy,
        routes,
        router
    )


# ============================================================
# CH -> CH -> BS TRANSMISSION
# ============================================================

def transmit_cluster_head_data(
    df,
    energy,
    alive,
    cluster_heads,
    routes,
    router,
    base_station,
    communication_radius
):

    energy = np.asarray(
        energy,
        dtype=float
    ).copy()

    if router is None:

        return energy

    alive_chs = [
        int(ch)
        for ch in cluster_heads
        if alive[int(ch)]
    ]

    for source_ch in alive_chs:

        source_ch = int(source_ch)

        # ----------------------------------------------------
        # No route
        # ----------------------------------------------------

        route = routes.get(
            source_ch
        )

        if route is None:

            print(
                f"WARNING: CH {source_ch} "
                f"has no BS route. "
                f"Aggregate packet lost."
            )

            continue

        if len(route) < 2:

            print(
                f"WARNING: Invalid route for "
                f"CH {source_ch}."
            )

            continue

        if route[-1] != "BS":

            print(
                f"WARNING: Route for CH "
                f"{source_ch} does not end at BS."
            )

            continue

        packet_delivered = True

        # ----------------------------------------------------
        # Traverse route
        # ----------------------------------------------------

        for i in range(
            len(route) - 1
        ):

            current = int(
                route[i]
            )

            next_hop = route[
                i + 1
            ]

            # ------------------------------------------------
            # Current CH must be alive
            # ------------------------------------------------

            if not alive[current]:

                packet_delivered = False

                break

            # ------------------------------------------------
            # CH -> BS
            # ------------------------------------------------

            if next_hop == "BS":

                distance = (
                    router.distance_to_bs(
                        current
                    )
                )

                if distance > communication_radius:

                    packet_delivered = False

                    break

                required = tx_energy(
                    distance,
                    PACKET_SIZE
                )

                if energy[current] < required:

                    energy[current] = 0.0
                    alive[current] = False

                    packet_delivered = False

                    break

                energy[current] -= required

                continue

            # ------------------------------------------------
            # CH -> CH
            # ------------------------------------------------

            next_ch = int(
                next_hop
            )

            if next_ch not in cluster_heads:

                packet_delivered = False

                break

            if not alive[next_ch]:

                packet_delivered = False

                break

            distance = router.distance(
                current,
                next_ch
            )

            if distance > communication_radius:

                packet_delivered = False

                break

            tx_required = tx_energy(
                distance,
                PACKET_SIZE
            )

            rx_required = rx_energy(
                PACKET_SIZE
            )

            # ------------------------------------------------
            # Sender energy
            # ------------------------------------------------

            if energy[current] < tx_required:

                energy[current] = 0.0
                alive[current] = False

                packet_delivered = False

                break

            # ------------------------------------------------
            # Receiver energy
            # ------------------------------------------------

            if energy[next_ch] < rx_required:

                energy[next_ch] = 0.0
                alive[next_ch] = False

                packet_delivered = False

                break

            energy[current] -= tx_required

            energy[next_ch] -= rx_required

    return energy


# ============================================================
# GRAPH SAGE + GA SELECTION
# ============================================================

def select_new_cluster_heads(
    df,
    energy,
    alive,
    positions,
    num_cluster_heads,
    base_station,
    communication_radius,
    graphsage_model=None,
    graphsage_optimizer=None,
    graphsage_embeddings=None,
    use_graphsage=True
):
    """
    Generate/update GraphSAGE embeddings and then select
    cluster heads using the centralized GA.

    GraphSAGE:
        - First call: train a new model.
        - Later calls: reuse the existing model and perform
          inference on the current topology/state.

    GA:
        - Receives the current positions.
        - Receives current residual energy.
        - Receives current alive mask.
        - Receives the current GraphSAGE embeddings.
        - Returns (best_chromosome, fitness).
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if positions is None:

        raise ValueError(
            "select_new_cluster_heads(): "
            "positions is None."
        )

    positions = np.asarray(
        positions,
        dtype=float
    )

    energy = np.asarray(
        energy,
        dtype=float
    )

    alive = np.asarray(
        alive,
        dtype=bool
    )

    if len(positions) != len(alive):

        raise ValueError(
            "select_new_cluster_heads(): "
            "positions and alive have different lengths."
        )

    if len(energy) != len(alive):

        raise ValueError(
            "select_new_cluster_heads(): "
            "energy and alive have different lengths."
        )

    df = df.copy()

    # --------------------------------------------------------
    # Synchronize residual energy
    # --------------------------------------------------------

    df["Residual_Energy"] = energy

    # ========================================================
    # GRAPH SAGE
    # ========================================================

    if use_graphsage:

        print(
            "\nGenerating GraphSAGE embeddings..."
        )

        # ----------------------------------------------------
        # If model is None:
        #
        #   GraphSAGE module creates and trains the model.
        #
        # If model already exists:
        #
        #   Reuse trained weights and perform inference on
        #   the current graph/state.
        # ----------------------------------------------------

        inference_only = (
            graphsage_model is not None
        )

        (
            graphsage_embeddings,
            graphsage_model,
            graphsage_optimizer
        ) = gat_embedding.generate_graphsage_embeddings(
            positions=positions,
            alive=alive,
            communication_radius=communication_radius,
            df=df,
            model=graphsage_model,
            optimizer=graphsage_optimizer,
            inference_only=inference_only,
            base_station=base_station
        )

        print(
            "GraphSAGE embedding shape:",
            np.asarray(graphsage_embeddings).shape
        )

        # Tell the GA to include GraphSAGE embeddings
        # in its fitness calculation.
        ga_cluster_head.USE_GRAPHSAGE = True

    else:

        graphsage_embeddings = None

        ga_cluster_head.USE_GRAPHSAGE = False

    # ========================================================
    # GENETIC ALGORITHM
    # ========================================================

    print(
        "\nRunning Genetic Algorithm for CH selection..."
    )

    (
        best_chromosome,
        ga_fitness
    ) = ga_cluster_head.select_cluster_heads(
        positions=positions,
        residual_energy=energy,
        embeddings=graphsage_embeddings,
        alive_mask=alive,
        num_heads=num_cluster_heads,
        communication_radius=communication_radius,
        base_station=base_station
    )

    cluster_heads = [
        int(ch)
        for ch in best_chromosome
    ]

    print(
        "GA selected CHs:",
        cluster_heads
    )

    print(
        "GA fitness:",
        ga_fitness
    )

    # ========================================================
    # STRICT VALIDATION
    #
    # GA is NOT allowed to return an invalid configuration.
    # ========================================================

    feasible = (
        check_ch_configuration_feasibility(
            df=df,
            alive=alive,
            cluster_heads=cluster_heads,
            base_station=base_station,
            communication_radius=communication_radius
        )
    )

    if not feasible:

        raise RuntimeError(
            "\nGA returned an invalid CH configuration.\n"
            f"Selected CHs: {cluster_heads}\n"
            "The configuration is not fully connected "
            "to the BS."
        )

    return (
        cluster_heads,
        graphsage_model,
        graphsage_optimizer,
        graphsage_embeddings
    )


# ============================================================
# PERIODIC CH UPDATE
# ============================================================

def should_update_cluster_heads(
    round_num,
    update_interval
):

    if round_num == 1:
        return True

    return (
        round_num % update_interval == 0
    )


# ============================================================
# MAIN SIMULATION ROUND
# ============================================================

def execute_simulation_round(
    round_num,
    energy,
    alive,
    cluster_heads,
    graphsage_model=None,
    graphsage_optimizer=None,
    graphsage_embeddings=None,
    df=None,
    positions=None,
    comm_radius=None,
    num_cluster_heads=None,
    base_station=None,
    update_interval=None,
    use_graphsage=True,
    **kwargs
):
    """
    Execute one complete WSN round.

    `positions` is required by the GraphSAGE + GA pipeline.

    Return order is kept fixed:

        energy
        alive
        cluster_heads
        graphsage_model
        graphsage_optimizer
        graphsage_embeddings
        success
    """

    # ========================================================
    # COPY STATE
    # ========================================================

    energy = np.asarray(
        energy,
        dtype=float
    ).copy()

    alive = np.asarray(
        alive,
        dtype=bool
    ).copy()

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if df is None:

        raise ValueError(
            "execute_simulation_round(): df is None."
        )

    if positions is None:

        raise ValueError(
            "execute_simulation_round(): "
            "positions is None."
        )

    if comm_radius is None:

        raise ValueError(
            "execute_simulation_round(): "
            "comm_radius is None."
        )

    if num_cluster_heads is None:

        raise ValueError(
            "execute_simulation_round(): "
            "num_cluster_heads is None."
        )

    if base_station is None:

        raise ValueError(
            "execute_simulation_round(): "
            "base_station is None."
        )

    if update_interval is None:

        raise ValueError(
            "execute_simulation_round(): "
            "update_interval is None."
        )

    # ========================================================
    # STEP 1
    # Update death status from previous round
    # ========================================================

    energy, alive = (
        update_alive_status(
            energy,
            alive
        )
    )

    # ========================================================
    # LND
    # ========================================================

    if np.sum(alive) == 0:

        return (
            energy,
            alive,
            cluster_heads,
            graphsage_model,
            graphsage_optimizer,
            graphsage_embeddings,
            False
        )

    # ========================================================
    # STEP 2
    # Synchronize dataframe
    # ========================================================

    df = df.copy()

    df["Residual_Energy"] = energy

    # ========================================================
    # STEP 3
    # Determine CH update
    # ========================================================

    periodic_update = (
        cluster_heads is None
        or
        should_update_cluster_heads(
            round_num,
            update_interval
        )
    )

    emergency_update = False

    # --------------------------------------------------------
    # Check current CH set
    # --------------------------------------------------------

    if cluster_heads is not None:

        # Check whether a selected CH has died.
        for ch in cluster_heads:

            ch = int(ch)

            if not alive[ch]:

                emergency_update = True

                break

        # ----------------------------------------------------
        # If all CHs are alive, check BS connectivity.
        # ----------------------------------------------------

        if not emergency_update:

            configuration_ok = (
                check_ch_configuration_feasibility(
                    df=df,
                    alive=alive,
                    cluster_heads=cluster_heads,
                    base_station=base_station,
                    communication_radius=comm_radius
                )
            )

            if not configuration_ok:

                emergency_update = True

    # ========================================================
    # STEP 4
    # Reselect CHs
    # ========================================================

    if (
        periodic_update
        or
        emergency_update
    ):

        reasons = []

        if periodic_update:
            reasons.append("periodic")

        if emergency_update:
            reasons.append("emergency")

        reason_text = " + ".join(
            reasons
        )

        print(
            f"\nCH RESELECTION at round "
            f"{round_num} ({reason_text})"
        )

        (
            cluster_heads,
            graphsage_model,
            graphsage_optimizer,
            graphsage_embeddings
        ) = select_new_cluster_heads(
            df=df,
            energy=energy,
            alive=alive,
            positions=positions,
            num_cluster_heads=num_cluster_heads,
            base_station=base_station,
            communication_radius=comm_radius,
            graphsage_model=graphsage_model,
            graphsage_optimizer=graphsage_optimizer,
            graphsage_embeddings=graphsage_embeddings,
            use_graphsage=use_graphsage
        )

        print(
            "New selected CHs:",
            cluster_heads
        )

    # ========================================================
    # STEP 5
    # Verify CH set before communication
    # ========================================================

    if not check_ch_configuration_feasibility(
        df=df,
        alive=alive,
        cluster_heads=cluster_heads,
        base_station=base_station,
        communication_radius=comm_radius
    ):

        # One final emergency GA attempt.
        print(
            "\nWARNING: Current CH configuration "
            "is not feasible."
        )

        print(
            "Performing emergency GA reselection..."
        )

        (
            cluster_heads,
            graphsage_model,
            graphsage_optimizer,
            graphsage_embeddings
        ) = select_new_cluster_heads(
            df=df,
            energy=energy,
            alive=alive,
            positions=positions,
            num_cluster_heads=num_cluster_heads,
            base_station=base_station,
            communication_radius=comm_radius,
            graphsage_model=graphsage_model,
            graphsage_optimizer=graphsage_optimizer,
            graphsage_embeddings=graphsage_embeddings,
            use_graphsage=use_graphsage
        )

    # ========================================================
    # STEP 6
    # Debug
    # ========================================================

    alive_cluster_heads = [
        int(ch)
        for ch in cluster_heads
        if alive[int(ch)]
    ]

    print(
        "\n============================================================"
    )

    print(
        "DEBUG BEFORE MEMBER COMMUNICATION"
    )

    print(
        "============================================================"
    )

    print(
        "Selected CHs:",
        cluster_heads
    )

    print(
        "Alive selected CHs:",
        alive_cluster_heads
    )

    print(
        "Number of selected CHs:",
        len(cluster_heads)
    )

    print(
        "Number of alive selected CHs:",
        len(alive_cluster_heads)
    )

    for ch in cluster_heads:

        ch = int(ch)

        x = float(
            df.loc[
                ch,
                "X_Coordinate"
            ]
        )

        y = float(
            df.loc[
                ch,
                "Y_Coordinate"
            ]
        )

        bs_distance = distance_to_bs(
            df,
            ch,
            base_station
        )

        print(
            f"CH {ch:3d}: "
            f"position=({x:.4f}, {y:.4f}), "
            f"BS distance={bs_distance:.4f} m, "
            f"within range="
            f"{bs_distance <= comm_radius}"
        )

    print(
        "Communication radius:",
        comm_radius
    )

    print(
        "============================================================"
    )

    # ========================================================
    # STEP 7
    # Ordinary Node -> CH assignment
    # ========================================================

    assignments = (
        assign_members_to_cluster_heads(
            df=df,
            alive=alive,
            cluster_heads=cluster_heads,
            communication_radius=comm_radius
        )
    )

    # ========================================================
    # STEP 8
    # Ordinary Node -> CH transmission
    # ========================================================

    (
        energy,
        received_packets
    ) = transmit_members_to_cluster_heads(
        df=df,
        energy=energy,
        alive=alive,
        assignments=assignments,
        communication_radius=comm_radius
    )

    # Update deaths.
    energy, alive = (
        update_alive_status(
            energy,
            alive
        )
    )

    # ========================================================
    # STEP 9
    # CH DATA AGGREGATION
    # ========================================================

    for ch, packet_count in (
        received_packets.items()
    ):

        ch = int(ch)

        if not alive[ch]:
            continue

        aggregation_cost = (
            aggregation_energy(
                packet_count,
                PACKET_SIZE
            )
        )

        if energy[ch] < aggregation_cost:

            energy[ch] = 0.0
            alive[ch] = False

        else:

            energy[ch] -= aggregation_cost

    # Update deaths.
    energy, alive = (
        update_alive_status(
            energy,
            alive
        )
    )

    # ========================================================
    # STEP 10
    # SURVIVING CHs
    # ========================================================

    alive_cluster_heads = [
        int(ch)
        for ch in cluster_heads
        if alive[int(ch)]
    ]

    # ========================================================
    # STEP 11
    # CH -> CH -> BS ROUTING
    # ========================================================

    if len(alive_cluster_heads) > 0:

        (
            energy,
            routes,
            router
        ) = route_cluster_heads(
            df=df,
            energy=energy,
            alive=alive,
            cluster_heads=alive_cluster_heads,
            base_station=base_station,
            communication_radius=comm_radius,
            distance_matrix=kwargs.get(
                "distance_matrix",
                None
            )
        )

        # ----------------------------------------------------
        # Transmit aggregate packets.
        #
        # If a CH became disconnected this round, its packet
        # is lost. Simulation continues.
        # ----------------------------------------------------

        energy = (
            transmit_cluster_head_data(
                df=df,
                energy=energy,
                alive=alive,
                cluster_heads=alive_cluster_heads,
                routes=routes,
                router=router,
                base_station=base_station,
                communication_radius=comm_radius
            )
        )

    # ========================================================
    # STEP 12
    # FINAL DEATH UPDATE
    # ========================================================

    energy, alive = (
        update_alive_status(
            energy,
            alive
        )
    )

    # ========================================================
    # STEP 13
    # LND CHECK
    # ========================================================

    alive_count = int(
        np.sum(alive)
    )

    if alive_count == 0:

        print(
            f"\nAll nodes are dead at "
            f"round {round_num}."
        )

        return (
            energy,
            alive,
            cluster_heads,
            graphsage_model,
            graphsage_optimizer,
            graphsage_embeddings,
            False
        )

    # ========================================================
    # STEP 14
    # ROUND SUCCESSFULLY COMPLETED
    # ========================================================

    return (
        energy,
        alive,
        cluster_heads,
        graphsage_model,
        graphsage_optimizer,
        graphsage_embeddings,
        True
    )


# ============================================================
# END OF simulation1.py
# ============================================================
