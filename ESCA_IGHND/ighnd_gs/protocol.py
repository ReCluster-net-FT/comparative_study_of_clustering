import numpy as np

from common import config as cfg
from ighnd.protocol import IGHND


def _build_adjacency(positions):
    n = len(positions)
    if n <= 1:
        return np.zeros((n, n))

    dmat = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)

    comm_range = getattr(cfg, "COMM_RANGE", None)
    if comm_range is not None:
        adj = (dmat <= comm_range).astype(float)
        isolated = adj.sum(axis=1) == 0
        if isolated.any():
            nn = np.argmin(dmat[isolated], axis=1)
            adj[np.where(isolated)[0], nn] = 1.0
    else:
        k = min(getattr(cfg, "GRAPHSAGE_KNN_K", 6), n - 1)
        adj = np.zeros((n, n))
        nn_idx = np.argpartition(dmat, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        adj[rows, nn_idx.ravel()] = 1.0

    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


class _NumpySAGE:
    def __init__(self, in_dim, hid1, hid2, seed=0):
        rng = np.random.default_rng(seed)

        def xavier(fan_in, fan_out):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-bound, bound, size=(fan_in, fan_out))

        self.W1 = xavier(2 * in_dim, hid1)
        self.b1 = np.zeros(hid1)
        self.W2 = xavier(2 * hid1, hid2)
        self.b2 = np.zeros(hid2)
        self.W3 = xavier(hid2, 1)
        self.b3 = np.zeros(1)

    def forward(self, X, A, cache=False):
        agg0 = A @ X
        concat1 = np.hstack([X, agg0])
        z1 = concat1 @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)

        agg1 = A @ h1
        concat2 = np.hstack([h1, agg1])
        z2 = concat2 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0.0)

        out = (h2 @ self.W3 + self.b3).ravel()

        if cache:
            return out, dict(X=X, A=A, concat1=concat1, z1=z1, h1=h1,
                              concat2=concat2, z2=z2, h2=h2)
        return out

    def train_step(self, X, A, y, lr):
        n = X.shape[0]
        out, c = self.forward(X, A, cache=True)

        d_out = (2.0 / n) * (out - y).reshape(-1, 1)

        dW3 = c["h2"].T @ d_out
        db3 = d_out.sum(axis=0)
        dh2 = d_out @ self.W3.T

        dz2 = dh2 * (c["z2"] > 0)
        dW2 = c["concat2"].T @ dz2
        db2 = dz2.sum(axis=0)
        dconcat2 = dz2 @ self.W2.T
        hid1 = c["h1"].shape[1]
        dh1_direct = dconcat2[:, :hid1]
        dagg1 = dconcat2[:, hid1:]
        dh1 = dh1_direct + c["A"].T @ dagg1

        dz1 = dh1 * (c["z1"] > 0)
        dW1 = c["concat1"].T @ dz1
        db1 = dz1.sum(axis=0)

        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

        return float(np.mean((out - y) ** 2))


class IGHND_GraphSAGE(IGHND):
    name = "IGHND-GraphSAGE"

    def __init__(self, network):
        super().__init__(network)

        # [DATASET] real feature width, if present, else 0 (legacy path).
        extra_dim = getattr(network, "extra_features", None)
        extra_dim = extra_dim.shape[1] if extra_dim is not None else 0
        in_dim = 5 + extra_dim  # [rel, 1/adv (scaled), prtzh, mnode, degree] + real CSV features

        self.sage = _NumpySAGE(
            in_dim,
            getattr(cfg, "GRAPHSAGE_HIDDEN1", 8),
            getattr(cfg, "GRAPHSAGE_HIDDEN2", 8),
            seed=getattr(cfg, "GRAPHSAGE_SEED", 0),
        )
        self.last_embedding = None
        self.last_embedding_node_ids = None

    def _compute_global_features(self):
        net = self.net
        alive_idx = np.where(net.alive)[0]
        if len(alive_idx) == 0:
            return alive_idx, None, None

        rel = np.zeros(len(alive_idx))
        adv = np.zeros(len(alive_idx))
        prtzh = np.zeros(len(alive_idx))
        mnode = np.zeros(len(alive_idx))
        pos_lookup = {g: k for k, g in enumerate(alive_idx)}

        for zid, z in self.zones.items():
            members = np.array([m for m in z.members if net.alive[m]])
            if len(members) == 0:
                continue
            local = np.array([pos_lookup[m] for m in members])

            # [DATASET fix -- see common/config.py] normalize by each
            # node's OWN starting energy, not one shared constant.
            rel[local] = net.energy[members] / net.init_energy_per_node[members]

            n_m = len(members)
            if n_m == 1:
                adv[local] = 1e-6
            else:
                for k, i in enumerate(members):
                    others = members[members != i]
                    d_nodes = np.linalg.norm(net.pos[others] - net.pos[i], axis=1).sum()
                    d_cent = np.linalg.norm(net.pos[i] - z.center)
                    a = (cfg.ADV_ALPHA / (n_m - 1)) * d_nodes + cfg.ADV_BETA * d_cent
                    adv[local[k]] = max(a, 1e-6)

            prtzh[local] = 1.0 / (1.0 + self.tzh[members])
            mnode[local] = np.where(self.native[members], cfg.MNODE_NATIVE, cfg.MNODE_MERGED)

        return alive_idx, dict(rel=rel, inv_adv=1.0 / adv, prtzh=prtzh, mnode=mnode), pos_lookup

    def _select_zh(self):
        net = self.net
        w = cfg.ANP_WEIGHTS
        alpha = getattr(cfg, "GRAPHSAGE_BLEND_ALPHA", 0.5)

        alive_idx, feats, pos_lookup = self._compute_global_features()
        if len(alive_idx) == 0:
            self.zh = {}
            return

        as_score = (w["W1"] * feats["rel"] + w["W2"] * feats["inv_adv"]
                    + w["W3"] * feats["prtzh"] + w["W4"] * feats["mnode"])

        A = _build_adjacency(net.pos[alive_idx])
        degree = A.astype(bool).sum(axis=1).astype(float)
        degree = degree / max(degree.max(), 1.0)

        X_dynamic = np.column_stack([
            feats["rel"], feats["inv_adv"] / max(feats["inv_adv"].max(), 1e-6),
            feats["prtzh"], feats["mnode"], degree,
        ])

        # [DATASET] concatenate real, static per-node CSV features when
        # available (see module docstring) -- purely additive, falls back
        # to the old dynamic-only X on a legacy synthetic network.
        extra = getattr(net, "extra_features", None)
        X = np.hstack([X_dynamic, extra[alive_idx]]) if extra is not None else X_dynamic

        epochs = getattr(cfg, "GRAPHSAGE_EPOCHS", 20)
        lr = getattr(cfg, "GRAPHSAGE_LR", 0.05)
        for _ in range(epochs):
            self.sage.train_step(X, A, as_score, lr)
        sage_score, cache = self.sage.forward(X, A, cache=True)

        # [NEW] expose embedding for visualization only.
        self.last_embedding = cache["h2"]
        self.last_embedding_node_ids = alive_idx.copy()

        final_score = alpha * as_score + (1.0 - alpha) * sage_score

        for zid, z in self.zones.items():
            members = np.array([m for m in z.members if net.alive[m]])
            if len(members) == 0:
                self.zh.pop(zid, None)
                continue
            if len(members) == 1:
                self.zh[zid] = int(members[0])
                self.tzh[members[0]] += 1
                continue

            local = np.array([pos_lookup[m] for m in members])
            best = members[int(np.argmax(final_score[local]))]
            self.zh[zid] = int(best)
            self.tzh[best] += 1


