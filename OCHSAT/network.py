"""
network.py

Creates and manages the wireless sensor network.
"""

from dataclasses import dataclass
import numpy as np

import config as cfg
from energy_model import Node
from radio_graph import compute_communication_radius


@dataclass
class BaseStation:
    """
    Represents the sink/base station.
    """
    x: float
    y: float

    @property
    def position(self):
        return (self.x, self.y)


class Network:
    """
    Represents the complete wireless sensor network.
    """

    def __init__(self, node_count, bs_position, seed=42, data_source=None,
                 dataset_path=None):

        self.node_count = node_count

        self.base_station = BaseStation(*bs_position)

        self.round_number = 0

        self.rng = np.random.default_rng(seed)

        self.nodes = []

        self.clusters = {}

        self.cluster_heads = {}

        # "synthetic" (default, uniform-random) or "dataset" (real rows
        # sampled from a CSV such as WSN_Dataset.csv - see
        # dataset_loader.py). Falls back to config.DATA_SOURCE so every
        # existing call site (scenario_runner.py, compare_graphsage.py,
        # graphsage_all_scenarios.py, ...) picks this up automatically
        # just by flipping the config flag - no call-site changes needed.
        self.data_source = data_source or getattr(cfg, "DATA_SOURCE", "synthetic")
        self.dataset_path = dataset_path or getattr(cfg, "DATASET_PATH", None)

        self.generate_nodes()

        # Real (not "assume everyone hears everyone") communication
        # radius for THIS network, based on its actual node count and
        # config.FIELD_X/FIELD_Y. Recomputed here (not hardcoded) so it
        # adapts to 100 vs 200 nodes, or to however many nodes a dataset
        # run ends up using.
        self.communication_radius = compute_communication_radius(
            node_count=self.node_count,
            field_x=cfg.FIELD_X,
            field_y=cfg.FIELD_Y,
            target_degree=getattr(cfg, "TARGET_NODE_DEGREE", 12),
        )

    def generate_nodes(self):
        """
        Deploy sensor nodes, either synthetically (uniform-random, as
        before) or from a real dataset (see dataset_loader.py),
        depending on self.data_source.
        """

        self.nodes.clear()

        if self.data_source == "dataset":

            from dataset_loader import build_nodes

            self.nodes = build_nodes(
                node_count=self.node_count,
                rng=self.rng,
                path=self.dataset_path,
            )
            return

        for node_id in range(self.node_count):

            x = self.rng.uniform(0, cfg.FIELD_X)

            y = self.rng.uniform(0, cfg.FIELD_Y)

            node = Node(
                node_id=node_id,
                x=x,
                y=y
            )

            self.nodes.append(node)

    def alive_nodes(self):

        return [node for node in self.nodes if node.alive]

    def dead_nodes(self):

        return [node for node in self.nodes if not node.alive]

    def alive_count(self):

        return len(self.alive_nodes())

    def dead_count(self):

        return len(self.dead_nodes())

    def reset(self):
        """
        Restore every node to its initial state.
        Useful when running multiple protocols.
        """

        self.round_number = 0

        self.clusters.clear()

        self.cluster_heads.clear()

        for node in self.nodes:

            # Use the node's OWN starting energy (identical to
            # cfg.INITIAL_ENERGY in synthetic mode; the real per-node
            # value from the CSV in dataset mode) rather than the
            # global constant.
            node.energy = node.initial_energy

            node.cluster_id = -1

            node.is_cluster_head = False

            node.alive = True

            node.rounds_as_ch = 0

            node.death_round = -1
