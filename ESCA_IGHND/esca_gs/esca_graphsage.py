
import numpy as np

from common import config as cfg
from esca import ESCA


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


class ESCA_GraphSAGE(ESCA):
    name = "ESCA-GraphSAGE"

    def __init__(self, network, seed=None):
        super().__init__(network, seed=seed)
        self._centroids = np.array([
            network.pos[self.labels == c].mean(axis=0) if np.any(self.labels == c)
            else np.zeros(2)
            for c in range(self.K)
        ])

      
        extra_dim = getattr(network, "extra_features", None)
        extra_dim = extra_dim.shape[1] if extra_dim is not None else 0
        in_dim = 3 + extra_dim  

        self.sage = _NumpySAGE(
            in_dim,
            getattr(cfg, "GRAPHSAGE_HIDDEN1", 8),
            getattr(cfg, "GRAPHSAGE_HIDDEN2", 8),
            seed=getattr(cfg, "GRAPHSAGE_SEED", 0),
        )
       
        self.last_embedding = None
        self.last_embedding_node_ids = None

    def _rotate_ch(self):
        net = self.net
        alpha = getattr(cfg, "GRAPHSAGE_BLEND_ALPHA", 0.5)

        alive_idx = np.where(net.alive)[0]
        if len(alive_idx) == 0:
            self.ch = {}
            return
        pos_lookup = {g: k for k, g in enumerate(alive_idx)}

    
        rel = net.energy[alive_idx] / net.init_energy_per_node[alive_idx]

        A = _build_adjacency(net.pos[alive_idx])
        degree = A.astype(bool).sum(axis=1).astype(float)
        degree = degree / max(degree.max(), 1.0)

        centroids = self._centroids[self.labels[alive_idx]]
        dist_to_centroid = np.linalg.norm(net.pos[alive_idx] - centroids, axis=1)
        diag = np.hypot(cfg.AREA_X, cfg.AREA_Y)
        dist_to_centroid = dist_to_centroid / diag

        X_dynamic = np.column_stack([rel, degree, dist_to_centroid])

      
        extra = getattr(net, "extra_features", None)
        X = np.hstack([X_dynamic, extra[alive_idx]]) if extra is not None else X_dynamic

        epochs = getattr(cfg, "GRAPHSAGE_EPOCHS", 20)
        lr = getattr(cfg, "GRAPHSAGE_LR", 0.05)
        for _ in range(epochs):
            self.sage.train_step(X, A, rel, lr)
        sage_score, cache = self.sage.forward(X, A, cache=True)


        self.last_embedding = cache["h2"]
        self.last_embedding_node_ids = alive_idx.copy()

        final_score = alpha * rel + (1.0 - alpha) * sage_score

        for c in range(self.K):
            members = np.where((self.labels == c) & net.alive)[0]
            if len(members) == 0:
                self.ch.pop(c, None)
                continue

            local = np.array([pos_lookup[m] for m in members])
            current_ch = self.ch.get(c)

            if current_ch is None or not net.alive[current_ch]:
                self.ch[c] = int(members[int(np.argmax(final_score[local]))])
                continue

            best_local_idx = int(np.argmax(final_score[local]))
            best_member = members[best_local_idx]
            cur_score = final_score[pos_lookup[current_ch]]
            if final_score[local[best_local_idx]] > cur_score:
                self.ch[c] = int(best_member)

