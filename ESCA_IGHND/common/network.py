import numpy as np
from . import config as cfg


class Network:

    def __init__(self, n_nodes: int, bs_xy, rng: np.random.Generator):
        self.n_nodes = n_nodes
        self.bs = np.array(bs_xy, dtype=float)
        self.pos = rng.uniform(
            low=[0.0, 0.0], high=[cfg.AREA_X, cfg.AREA_Y], size=(n_nodes, 2)
        )
        self.energy = np.full(n_nodes, cfg.INIT_ENERGY, dtype=float)
        self.alive = np.ones(n_nodes, dtype=bool)
        self.dist_to_bs = np.linalg.norm(self.pos - self.bs, axis=1)

        self.init_energy_per_node = np.full(n_nodes, cfg.INIT_ENERGY, dtype=float)

    @classmethod
    def from_dataset(cls, dataset, bs_mode="centroid"):
        net = cls.__new__(cls)
        net.n_nodes = dataset.n_nodes
        net.bs = np.array(dataset.bs_position(bs_mode), dtype=float)
        net.pos = dataset.pos_xy.copy()                       
        net.energy = dataset.initial_energy.copy()
        net.alive = np.ones(net.n_nodes, dtype=bool)
        net.dist_to_bs = np.linalg.norm(net.pos - net.bs, axis=1)

        net.init_energy_per_node = dataset.initial_energy.copy()

        net.node_ids = dataset.node_ids.copy()
        net.z_coordinate = dataset.z.copy()
        net.extra_features = dataset.features.copy()          
        net.extra_feature_names = list(dataset.feature_names)
        net.packet_loss_prob = dataset.packet_loss_prob.copy()  
        net.residual_energy_raw = dataset.residual_energy_raw.copy()  

        return net

    def clone(self):
        net = Network.__new__(Network)
        net.n_nodes = self.n_nodes
        net.bs = self.bs.copy()
        net.pos = self.pos.copy()
        net.energy = self.energy.copy()
        net.alive = self.alive.copy()
        net.dist_to_bs = self.dist_to_bs.copy()
        net.init_energy_per_node = self.init_energy_per_node.copy()

        for attr in ("node_ids", "z_coordinate", "extra_features",
                     "extra_feature_names", "packet_loss_prob",
                     "residual_energy_raw"):
            if hasattr(self, attr):
                val = getattr(self, attr)
                setattr(net, attr, val.copy() if hasattr(val, "copy") else val)

        return net

    def alive_ids(self):
        return np.where(self.alive)[0]

    def num_alive(self):
        return int(self.alive.sum())

    def total_alive_energy(self):
        return float(self.energy[self.alive].sum())

    def pairwise_dist(self, i, j_indices):
        return np.linalg.norm(self.pos[j_indices] - self.pos[i], axis=1)

    def kill_dead(self):
        newly_dead = self.alive & (self.energy <= 0)
        self.alive[newly_dead] = False
        return newly_dead
