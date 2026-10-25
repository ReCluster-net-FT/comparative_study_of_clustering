import numpy as np
from collections import defaultdict

from common import config as cfg
from common import energy_model as em


class Zone:
    __slots__ = ("id", "members", "center", "cells")

    def __init__(self, zid, members, center, cells):
        self.id = zid
        self.members = set(members)
        self.center = np.array(center, dtype=float)
        self.cells = set(cells)


def _initial_grid_zones(positions, M):
    n = len(positions)
    cell_w = cfg.AREA_X / M
    cell_h = cfg.AREA_Y / M
    zx = np.clip((positions[:, 0] // cell_w).astype(int), 0, M - 1)
    zy = np.clip((positions[:, 1] // cell_h).astype(int), 0, M - 1)
    zone_id = zx + zy * M
    zones = {}
    for zid in range(M * M):
        members = np.where(zone_id == zid)[0].tolist()
        cell_x = (zid % M + 0.5) * cell_w
        cell_y = (zid // M + 0.5) * cell_h
        zones[zid] = Zone(zid, members, (cell_x, cell_y), cells={zid})
    return zones, zx, zy, M


def _grid_neighbors(zid, M):
    x, y = zid % M, zid // M
    cand = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    out = []
    for cx, cy in cand:
        if 0 <= cx < M and 0 <= cy < M:
            out.append(cx + cy * M)
    return out


def _zone_neighbors(zid, zones, cell_adjacency):
    z = zones[zid]
    candidate_cells = set()
    for cell in z.cells:
        candidate_cells.update(cell_adjacency[cell])
    candidate_cells -= z.cells
    neighbor_zids = set()
    for other_zid, other in zones.items():
        if other_zid == zid:
            continue
        if other.cells & candidate_cells:
            neighbor_zids.add(other_zid)
    return neighbor_zids


def _build_topology(positions, M):
    n = len(positions)
    zones, zx, zy, M = _initial_grid_zones(positions, M)
    node_zone = np.zeros(n, dtype=int)
    for zid, z in zones.items():
        for m in z.members:
            node_zone[m] = zid
    native = np.ones(n, dtype=bool)

    cell_adjacency = {zid: set(_grid_neighbors(zid, M)) for zid in range(M * M)}

    expected_density = n / (M * M)
    lower_bound = cfg.LOWER_BOUND_FACTOR * expected_density
    upper_bound = cfg.UPPER_BOUND_FACTOR * expected_density
    diag = np.hypot(cfg.AREA_X, cfg.AREA_Y)

    changed = True
    guard = 0
    while changed and guard < (M * M) * 4:
        changed = False
        guard += 1
        for zid in list(zones.keys()):
            z = zones.get(zid)
            if z is None or len(z.members) == 0:
                continue
            if len(z.members) >= lower_bound:
                continue
            neighbors = _zone_neighbors(zid, zones, cell_adjacency)
            if not neighbors:
                continue
            best_nz, best_wms = None, None
            for nz_id in neighbors:
                nz = zones[nz_id]
                dist = np.linalg.norm(z.center - nz.center) / diag
                den = len(nz.members) / max(expected_density, 1e-9)
                wms = cfg.WMS_D * dist + cfg.WMS_SIGMA * den
                if best_wms is None or wms < best_wms:
                    best_wms, best_nz = wms, nz_id

            target = zones[best_nz]
            for m in z.members:
                node_zone[m] = target.id
                native[m] = False
                target.members.add(m)
            target.cells |= z.cells
            del zones[zid]
            changed = True

    next_id = max(zones.keys(), default=-1) + 1
    to_process = [zid for zid, z in zones.items() if len(z.members) > upper_bound]
    while to_process:
        zid = to_process.pop()
        z = zones.get(zid)
        if z is None or len(z.members) <= upper_bound:
            continue
        member_list = list(z.members)
        pts = positions[member_list]
        mid_x = pts[:, 0].mean()
        mid_y = pts[:, 1].mean()
        quads = defaultdict(list)
        for idx, p in zip(member_list, pts):
            qx = 0 if p[0] < mid_x else 1
            qy = 0 if p[1] < mid_y else 1
            quads[(qx, qy)].append(idx)

        if len(quads) <= 1:
            continue

        parent_cells = z.cells
        del zones[zid]
        for (qx, qy), members in quads.items():
            new_zone = Zone(next_id, members, positions[members].mean(axis=0), cells=parent_cells)
            zones[next_id] = new_zone
            for m in members:
                node_zone[m] = next_id
            if len(members) > upper_bound:
                to_process.append(next_id)
            next_id += 1

    return zones, node_zone, native


class IGHND:
    name = "IGHND"

    def __init__(self, network):
        self.net = network
        self.zones, self.node_zone, self.native = _build_topology(network.pos, cfg.GRID_M)
        self.tzh = np.zeros(network.n_nodes, dtype=int)
        self.zh = {}

    def _select_zh(self):
        net = self.net
        w = cfg.ANP_WEIGHTS
        for zid, z in self.zones.items():
            members = np.array([m for m in z.members if net.alive[m]])
            if len(members) == 0:
                self.zh.pop(zid, None)
                continue
            if len(members) == 1:
                self.zh[zid] = int(members[0])
                self.tzh[members[0]] += 1
                continue

            rel = net.energy[members] / net.init_energy_per_node[members]
            n = len(members)
            adv = np.zeros(n)
            for k, i in enumerate(members):
                others = members[members != i]
                d_nodes = np.linalg.norm(net.pos[others] - net.pos[i], axis=1).sum()
                d_cent = np.linalg.norm(net.pos[i] - z.center)
                adv[k] = (cfg.ADV_ALPHA / (n - 1)) * d_nodes + cfg.ADV_BETA * d_cent
            adv = np.maximum(adv, 1e-6)

            prtzh = 1.0 / (1.0 + self.tzh[members])
            mnode = np.where(self.native[members], cfg.MNODE_NATIVE, cfg.MNODE_MERGED)

            as_score = (w["W1"] * rel + w["W2"] * (1.0 / adv)
                        + w["W3"] * prtzh + w["W4"] * mnode)

            best = members[int(np.argmax(as_score))]
            self.zh[zid] = int(best)
            self.tzh[best] += 1

    def run_round(self):
        net = self.net
        self._select_zh()

        for zid, zh_id in self.zh.items():
            z = self.zones[zid]
            members = np.array([m for m in z.members if net.alive[m] and m != zh_id])

            if len(members) > 0:
                d_to_zh = net.pairwise_dist(zh_id, members)
                tx_cost = em.e_tx(cfg.PACKET_BITS, d_to_zh)
                net.energy[members] -= tx_cost

                rx_cost = em.e_rx(cfg.PACKET_BITS) * len(members)
                da_cost = em.e_da(cfg.PACKET_BITS, num_signals=len(members))
                net.energy[zh_id] -= (rx_cost + da_cost)

            d_zh_to_bs = net.dist_to_bs[zh_id]
            net.energy[zh_id] -= em.e_tx(cfg.PACKET_BITS, np.array([d_zh_to_bs]))[0]

        net.energy[net.alive] = np.maximum(net.energy[net.alive], 0.0)
