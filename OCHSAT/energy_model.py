"""
energy_model.py

Implements the first-order radio energy model used in the OCHSAT paper.
"""

import math
from dataclasses import dataclass, field

import config as cfg


def distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    point1 : tuple (x, y)
    point2 : tuple (x, y)

    Returns
    -------
    float
    """
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def transmission_energy(bits, dist):
    """
    Energy required to transmit a packet.

    Uses:
    - Free-space model if d < d0
    - Multipath model otherwise
    """

    if dist < cfg.D0:
        return (
            bits * cfg.E_ELEC +
            bits * cfg.E_FS * (dist ** 2)
        )

    return (
        bits * cfg.E_ELEC +
        bits * cfg.E_MP * (dist ** 4)
    )


def reception_energy(bits):
    """
    Energy required to receive a packet.
    """

    return bits * cfg.E_ELEC


def aggregation_energy(bits, member_count):
    """
    Energy spent by Cluster Head
    while aggregating data.
    """

    return bits * member_count * cfg.E_DA


@dataclass
class Node:
    """
    Represents one sensor node.
    """

    node_id: int

    x: float
    y: float

    energy: float = field(default_factory=lambda: cfg.INITIAL_ENERGY)

    # Per-node starting energy. Defaults to the global cfg.INITIAL_ENERGY
    # (synthetic mode, where every node starts identical), but dataset-
    # driven nodes each carry their own real Initial_Energy value from the
    # CSV, so rotation thresholds / energy bookkeeping must reference THIS
    # per-node value rather than the global constant.
    initial_energy: float = field(default_factory=lambda: cfg.INITIAL_ENERGY)

    cluster_id: int = -1

    is_cluster_head: bool = False

    alive: bool = True

    rounds_as_ch: int = 0

    death_round: int = -1

    # Free-form bag for extra per-node attributes pulled in from external
    # datasets (e.g. Detection_Accuracy, Temperature, Humidity, ... from
    # WSN_Dataset.csv) that the radio/energy model itself doesn't need but
    # that are handy to keep around for later analysis/export.
    extra: dict = field(default_factory=dict)

    @property
    def position(self):
        return (self.x, self.y)

    def consume_energy(self, amount, current_round):
        """
        Deduct energy from node.
        """

        if not self.alive:
            return

        self.energy -= amount

        if self.energy <= 0:

            self.energy = 0.0

            self.alive = False

            self.death_round = current_round
