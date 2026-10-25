"""
simulator.py
============

Main simulation engine for the OCHSAT paper.

Responsibilities
----------------
1. Create network clusters
2. Select Cluster Heads
3. Simulate every communication round
4. Rotate Cluster Heads
5. Collect simulation metrics
6. Report network lifetime statistics
"""

import config as cfg
from trace import TraceLogger
from clustering import cluster_network
from ahp_topsis import compute_ahp_weights

from protocols import (
    select_cluster_head,
    needs_rotation,
)

from energy_model import (
    distance,
    transmission_energy,
    reception_energy,
    aggregation_energy,
)

from ch_selector import GraphSAGECHSelector

from export_simulation import export_network


class Simulator:

    def __init__(self, network, protocol, scenario_name=None):

        self.network = network
        self.protocol = protocol

        # Scenario name kept for reference/printing only. Per your
        # choice, trace/network files always save as "trace.json" /
        # "network.json" (simplest - each new run overwrites the
        # previous one; see FIXES_GUIDE for the tradeoff).
        self.scenario_name = scenario_name

        # How often (in rounds) a 'graphsage_prediction' trace event
        # is emitted for NetAnim visualization. The model itself is
        # trained/blended into the real CH decision every round
        # regardless (see ch_selector.py) - this interval only
        # controls how often we additionally *log* its standalone
        # top pick for the animation.
        self.graphsage_log_interval = getattr(
            cfg, "GRAPHSAGE_PREDICTION_LOG_INTERVAL", 25
        )

        # Real, trained GraphSAGE model - only meaningfully used for
        # the OCHSAT protocol (see protocols.select_ch_ochsat). For
        # other protocols this is created but simply never consulted.
        #
        # IMPORTANT (reproducibility fix): the model's initial weights
        # are drawn from PyTorch's global RNG. Left unseeded, that RNG
        # state is not tied to this network's own seed at all, so two
        # runs of the SAME seed would silently produce DIFFERENT
        # GraphSAGE weights - and therefore different CH choices -
        # every time. We derive a deterministic torch seed from this
        # network's own (already-seeded) RNG so that "with GraphSAGE"
        # runs become exactly as reproducible as every other protocol.
        if protocol == "OCHSAT":
            try:
                import torch
                torch_seed = int(network.rng.integers(0, 2**31 - 1))
                torch.manual_seed(torch_seed)
            except ImportError:
                pass
            self.graphsage_selector = GraphSAGECHSelector()
        else:
            self.graphsage_selector = None

        self.current_round = 0
        # ----------------------------------------------------
        # Trace Logger
        #
        # Stores every event occurring during simulation.
        # The generated trace will later be used by
        # ns-3 and NetAnim.
        # ----------------------------------------------------

        self.trace = TraceLogger()
        self.weights, self.consistency_ratio = compute_ahp_weights()

        # Current Cluster Heads
        self.cluster_heads = {}

        # Most recent GraphSAGE-model-only top pick per cluster
        # (before blending), refreshed every CH selection/rotation.
        self.last_graphsage_prediction = {}

        # Last rotation round of each cluster
        self.last_rotation = {}

        # Used for energy-consumption calculation. Sums each node's OWN
        # starting energy (identical to cfg.INITIAL_ENERGY in synthetic
        # mode; the real per-node CSV value in dataset mode, where nodes
        # do NOT all start with the same energy).
        self.previous_total_energy = sum(
            node.initial_energy for node in network.nodes
        )

        # Metrics collected every round
        self.metrics = {

            # Network state
            "alive_nodes": [],
            "dead_nodes": [],

            # Energy
            "total_energy": [],
            "energy_consumed": [],

            # Cluster information
            "cluster_heads": [],
            "cluster_sizes": [],
            "rotations": [],

            # Communication
            "packets_sent": [],

            # Lifetime metrics
            "fnd": None,
            "hnd": None,
            "lnd": None,
        }

    ####################################################################
    # INITIALIZATION
    ####################################################################

    def initialize(self):
        """
        Build clusters and choose the first
        Cluster Head for every cluster.
        """

        cluster_network(self.network)

        # Store every node position for NetAnim.

        for node in self.network.nodes:
            self.trace.node_position(
                node.node_id,
                node.position[0],
                node.position[1]
            )

        # Record every generated cluster.
        for cluster_id, members in self.network.clusters.items():
            self.trace.cluster_created(
                self.current_round,
                cluster_id,
                [node.node_id for node in members]
            )

        # ----------------------------------------------------
        # Record the simulation configuration.
        #
        # This information allows ns-3 to reconstruct the
        # original network before replaying events.
        # ----------------------------------------------------

        self.trace.simulation_start(

            self.protocol,

            len(self.network.nodes),

            self.network.base_station.position

        )

        self.cluster_heads.clear()
        self.last_rotation.clear()

        for cluster_id, members in self.network.clusters.items():

            ch, predicted = select_cluster_head(
                protocol=self.protocol,
                cluster_nodes=members,
                bs_position=self.network.base_station.position,
                weights=self.weights,
                rng=self.network.rng,
                graphsage_selector=self.graphsage_selector,
                radius=self.network.communication_radius,
            )

            ch.is_cluster_head = True

            self.cluster_heads[cluster_id] = ch

            # Keep the model's own (pre-blend) top pick per cluster so
            # emit_graphsage_predictions() can log it without redoing
            # a forward pass.
            self.last_graphsage_prediction[cluster_id] = predicted

            # Record the selected Cluster Head.
            self.trace.cluster_head_selected(
                self.current_round,
                cluster_id,
                ch.node_id
            )

            self.last_rotation[cluster_id] = 0

    ####################################################################
    # SIMULATE ONE ROUND
    ####################################################################

    def simulate_round(self):
        """
        Simulate one complete communication round.

        Member Nodes
                ↓
        Cluster Head
                ↓
        Data Aggregation
                ↓
        Base Station
        """

        packets = 0

        for cluster_id, members in self.network.clusters.items():

            ch = self.cluster_heads[cluster_id]

            if not ch.alive:
                continue

            alive_members = [
                node
                for node in members
                if node.alive
            ]

            # ---------------------------------------------------------
            # Member Nodes → Cluster Head
            # ---------------------------------------------------------

            for node in alive_members:

                if node.node_id == ch.node_id:
                    continue

                d = distance(
                    node.position,
                    ch.position
                )

                tx = transmission_energy(
                    cfg.PACKET_SIZE,
                    d
                )

                rx = reception_energy(
                    cfg.PACKET_SIZE
                )

                node.consume_energy(
                    tx,
                    self.current_round
                )
        
                # Member transmits a packet to its Cluster Head.
                self.trace.member_to_ch(
                    self.current_round,
                    node.node_id,
                    ch.node_id
                )
                
                ch.consume_energy(
                    rx,
                    self.current_round
                )

                packets += 1

            # ---------------------------------------------------------
            # Data Aggregation
            # ---------------------------------------------------------

            aggregation = aggregation_energy(
                cfg.PACKET_SIZE,
                max(0, len(alive_members) - 1)
            )

            ch.consume_energy(
                aggregation,
                self.current_round
            )

            # Cluster Head aggregates all received packets.
            self.trace.aggregation(
                self.current_round,
                ch.node_id
            )

            # ---------------------------------------------------------
            # Cluster Head → Base Station
            # ---------------------------------------------------------

            d_bs = distance(
                ch.position,
                self.network.base_station.position
            )

            tx_bs = transmission_energy(
                cfg.PACKET_SIZE,
                d_bs
            )

            ch.consume_energy(
                tx_bs,
                self.current_round
            )


            # Cluster Head forwards the aggregated packet to the Base Station.
            self.trace.ch_to_bs(
                self.current_round,
                ch.node_id
            )

            packets += 1

        self.metrics["packets_sent"].append(
            packets
        )

        self.current_round += 1

    ####################################################################
    # CLUSTER HEAD ROTATION
    ####################################################################

    def rotate_cluster_heads(self):
        """
        Rotate Cluster Heads whenever
        the protocol decides it is necessary.
        """

        rotations = 0

        for cluster_id, members in self.network.clusters.items():

            current_ch = self.cluster_heads[cluster_id]

            if not needs_rotation(
                self.protocol,
                current_ch,
                self.current_round,
                self.last_rotation[cluster_id]
            ):
                continue

            current_ch.is_cluster_head = False

            alive_nodes = [
                node
                for node in members
                if node.alive
            ]

            if len(alive_nodes) == 0:
                continue

            new_ch, predicted = select_cluster_head(
                protocol=self.protocol,
                cluster_nodes=alive_nodes,
                bs_position=self.network.base_station.position,
                weights=self.weights,
                rng=self.network.rng,
                graphsage_selector=self.graphsage_selector,
                radius=self.network.communication_radius,
                current_round=self.current_round,
            )

            self.last_graphsage_prediction[cluster_id] = predicted

            # Record Cluster Head rotation.
            self.trace.rotate_ch(
                self.current_round,
                cluster_id,
                current_ch.node_id,
                new_ch.node_id
            )

            new_ch.is_cluster_head = True

            self.cluster_heads[cluster_id] = new_ch

            self.last_rotation[cluster_id] = (
                self.current_round
            )

            rotations += 1

        self.metrics["rotations"].append(
            rotations
        )

    ####################################################################
    # GRAPHSAGE PREDICTION LOGGING (uses the real, trained model)
    ####################################################################

    def emit_graphsage_predictions(self):
        """
        Logs a 'graphsage_prediction' event per cluster using the
        real GraphSAGE model's own top pick (captured the last time
        select_cluster_head ran for that cluster - see ch_selector.py).

        This is purely a logging/visualization step; the model
        already influenced the *actual* CH decision at selection
        time (blended with AHP-TOPSIS per config.GRAPHSAGE_BLEND_ALPHA).
        """

        if self.graphsage_selector is None or not self.graphsage_selector.enabled:
            return

        for cluster_id, predicted_node in self.last_graphsage_prediction.items():

            if predicted_node is None or not predicted_node.alive:
                continue

            self.trace.graphsage_prediction(
                self.current_round,
                predicted_node.node_id,
                prediction="graphsage_top_pick",
            )

    ####################################################################
    # METRICS
    ####################################################################

    def update_metrics(self):
        """
        Update all simulation statistics after every round.
        """

        alive = sum(node.alive for node in self.network.nodes)
        dead = len(self.network.nodes) - alive

        total_energy = sum(
            node.energy
            for node in self.network.nodes
        )

        # Record newly dead nodes.

        for node in self.network.nodes:
            if (
                not node.alive
                and
                getattr(node, "death_logged", False) is False
            ):
                self.trace.node_dead(
                    self.current_round,
                    node.node_id
                )
                node.death_logged = True

        energy_used = (
            self.previous_total_energy
            - total_energy
        )

        self.previous_total_energy = total_energy

        # ------------------------------
        # Store metrics
        # ------------------------------

        self.metrics["alive_nodes"].append(alive)

        self.metrics["dead_nodes"].append(dead)

        self.metrics["total_energy"].append(total_energy)

        self.metrics["energy_consumed"].append(
            energy_used
        )

        self.metrics["cluster_heads"].append(
            {
                cid: ch.node_id
                for cid, ch in self.cluster_heads.items()
            }
        )

        self.metrics["cluster_sizes"].append(
            {
                cid: len(nodes)
                for cid, nodes in self.network.clusters.items()
            }
        )

        # ------------------------------
        # Lifetime metrics
        # ------------------------------

        if (
            self.metrics["fnd"] is None
            and dead >= 1
        ):
            self.metrics["fnd"] = self.current_round

        if (
            self.metrics["hnd"] is None
            and dead >= len(self.network.nodes) / 2
        ):
            self.metrics["hnd"] = self.current_round

        if (
            self.metrics["lnd"] is None
            and alive == 0
        ):
            self.metrics["lnd"] = self.current_round

    ####################################################################
    # HELPERS
    ####################################################################

    def all_nodes_dead(self):
        """
        Returns True when every node has died.
        """

        return all(
            not node.alive
            for node in self.network.nodes
        )

    ####################################################################
    # MAIN SIMULATION
    ####################################################################

    def run(self):
        """
        Execute the complete simulation.
        """

        self.initialize()

        while (
            self.current_round < cfg.MAX_ROUNDS
            and not self.all_nodes_dead()
        ):

            self.simulate_round()

            # NOTE: GraphSAGE is frozen (offline-trained) during this
            # entire run -- there is no online training step here or
            # anywhere else. See ch_selector.py / train_graphsage_offline.py.

            self.rotate_cluster_heads()

            self.update_metrics()

            if self.current_round % self.graphsage_log_interval == 0:
                self.emit_graphsage_predictions()

        # Export the complete execution trace.
        self.trace.simulation_end(
            self.current_round
        )

        # ----------------------------------------------------
        # Trace/network filenames: always the same name, per your
        # choice (simplest). NOTE: this means the next run you do
        # (any scenario/protocol/seed) overwrites these two files -
        # copy them out before starting another run if you want to
        # keep this result.
        # ----------------------------------------------------
        trace_filename = "trace.json"
        network_filename = "network.json"

        self.trace.save(trace_filename)
        export_network(self, network_filename)

        self.metrics["trace_file"] = trace_filename
        self.metrics["network_file"] = network_filename

        # Per-CH-selection-event diagnostic log (PRD section 19): AHP
        # pick vs GraphSAGE pick vs final pick, for every decision this
        # run made. Empty list if GraphSAGE was never enabled/available.
        if self.graphsage_selector is not None and self.graphsage_selector.enabled:
            self.metrics["ch_selection_log"] = self.graphsage_selector.decision_log
            self.metrics["graphsage_agreement_rate"] = self.graphsage_selector.agreement_rate()
            self.metrics["graphsage_change_rate"] = self.graphsage_selector.decision_change_rate()
            self.metrics["graphsage_score_correlation"] = self.graphsage_selector.score_correlation()
        else:
            self.metrics["ch_selection_log"] = []
            self.metrics["graphsage_agreement_rate"] = None
            self.metrics["graphsage_change_rate"] = None
            self.metrics["graphsage_score_correlation"] = None

        return self.metrics

    ####################################################################
    # SUMMARY
    ####################################################################

    def summary(self):
        """
        Print a simulation summary.
        """

        print("=" * 60)
        print("Simulation Summary")
        print("=" * 60)

        print(f"Protocol : {self.protocol}")

        print(f"Rounds   : {self.current_round}")

        print(f"FND      : {self.metrics['fnd']}")

        print(f"HND      : {self.metrics['hnd']}")

        print(f"LND      : {self.metrics['lnd']}")

        if len(self.metrics["alive_nodes"]) > 0:

            print(
                f"Alive    : {self.metrics['alive_nodes'][-1]}"
            )

            print(
                f"Dead     : {self.metrics['dead_nodes'][-1]}"
            )

            print(
                "Remaining Energy : "
                f"{self.metrics['total_energy'][-1]:.6f} J"
            )

            print(
                "Energy Consumed  : "
                f"{sum(self.metrics['energy_consumed']):.6f} J"
            )

            print(
                "Packets Sent     : "
                f"{sum(self.metrics['packets_sent'])}"
            )

            print(
                "CH Rotations     : "
                f"{sum(self.metrics['rotations'])}"
            )
