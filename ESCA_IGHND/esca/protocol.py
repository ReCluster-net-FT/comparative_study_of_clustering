import numpy as np
import random
from sklearn.cluster import KMeans

from common import config as cfg
from common import energy_model as em


def _wss(positions, labels, centroids):
    total = 0.0
    for k in range(len(centroids)):
        pts = positions[labels == k]
        if len(pts) == 0:
            continue
        total += np.sum((pts - centroids[k]) ** 2)
    return total


def choose_k_elbow(positions, k_max, random_state):
    """Elbow method operationalized as max perpendicular distance from the
    chord joining the first and last (K, WSS) points."""
    k_values = list(range(2, k_max + 1))
    wss_values = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        labels = km.fit_predict(positions)
        wss_values.append(_wss(positions, labels, km.cluster_centers_))

    wss = np.array(wss_values, dtype=float)
    ks = np.array(k_values, dtype=float)
    x = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    y = (wss - wss.min()) / (wss.max() - wss.min() + 1e-12)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / (line_len + 1e-12)
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]]) - p1
        proj_len = np.dot(p, line_unit)
        proj = proj_len * line_unit
        perp = p - proj
        distances.append(np.linalg.norm(perp))
    best_idx = int(np.argmax(distances))
    return k_values[best_idx]


class ESCA:
    name = "ESCA"

    def __init__(self, network, seed=None):
        self.net = network
        n = network.n_nodes
        k_max = max(3, int(n * cfg.ESCA_KMAX_FRACTION))
        k_max = min(k_max, n - 1)
        self.K = choose_k_elbow(network.pos, k_max, random_state=seed)

        km = KMeans(n_clusters=self.K, n_init=10, random_state=seed)
        self.labels = km.fit_predict(network.pos)  # static for entire run

        self.ch = {}
        for c in range(self.K):
            members = np.where(self.labels == c)[0]
            if len(members) == 0:
                continue
            self.ch[c] = int(members.min())

    def _rotate_ch(self):
        net = self.net
        for c in range(self.K):
            members = np.where((self.labels == c) & net.alive)[0]
            if len(members) == 0:
                self.ch.pop(c, None)
                continue
            current_ch = self.ch.get(c)
            if current_ch is None or not net.alive[current_ch]:
                self.ch[c] = int(members[np.argmax(net.energy[members])])
                continue
            best_member = members[np.argmax(net.energy[members])]
            if net.energy[best_member] > net.energy[current_ch]:
                self.ch[c] = int(best_member)

    def run_round(self):
        net = self.net
        self._rotate_ch()

        for c, ch_id in self.ch.items():
            members = np.where((self.labels == c) & net.alive)[0]
            non_ch_members = members[members != ch_id]

            if len(non_ch_members) > 0:
                d_to_ch = net.pairwise_dist(ch_id, non_ch_members)
                tx_cost = em.e_tx(cfg.PACKET_BITS, d_to_ch)
                net.energy[non_ch_members] -= tx_cost

                rx_cost = em.e_rx(cfg.PACKET_BITS) * len(non_ch_members)
                da_cost = em.e_da(cfg.PACKET_BITS, num_signals=len(non_ch_members))
                net.energy[ch_id] -= (rx_cost + da_cost)

            d_ch_to_bs = net.dist_to_bs[ch_id]
            net.energy[ch_id] -= em.e_tx(cfg.PACKET_BITS, np.array([d_ch_to_bs]))[0]

        net.energy[net.alive] = np.maximum(net.energy[net.alive], 0.0)
