import numpy as np
from . import config as cfg


def dataset_delivery_probability(node_indices, network):
    node_indices = np.asarray(node_indices, dtype=int)
    if node_indices.size == 0:
        return np.array([], dtype=float)
    return np.clip(1.0 - network.packet_loss_prob[node_indices], 0.0, 1.0)


def round_packet_stats_from_dataset(sender_indices, network):
    p = dataset_delivery_probability(sender_indices, network)
    attempted = int(p.size)
    if attempted == 0:
        return 0, 0.0, 0.0
    expected_delivered = float(p.sum())
    loss_rate = 1.0 - (expected_delivered / attempted)
    return attempted, expected_delivered, loss_rate



def delivery_probability(d):
    d = np.asarray(d, dtype=float)
    d0 = cfg.D0
    alpha = getattr(cfg, "PACKET_LOSS_ALPHA", 0.02)
    p = np.where(d <= d0, 1.0, np.exp(-alpha * (d - d0)))
    return np.clip(p, 0.0, 1.0)


def round_packet_stats(distances):
    distances = np.asarray(distances, dtype=float)
    attempted = int(distances.size)
    if attempted == 0:
        return 0, 0.0, 0.0
    p = delivery_probability(distances)
    expected_delivered = float(p.sum())
    loss_rate = 1.0 - (expected_delivered / attempted)
    return attempted, expected_delivered, loss_rate
