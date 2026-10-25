from collections import defaultdict
import numpy as np


class RoundRecorder:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

        self.alive_history = []
        self.dead_history = []
        self.residual_energy_history = []          # total J alive, per round
        self.avg_residual_energy_history = []       # mean J per alive node
        self.energy_consumed_history = []            # J consumed THIS round
        self.cumulative_energy_consumed_history = []

        self.packets_attempted_history = []
        self.packets_delivered_history = []          # expected (fractional)
        self.packet_loss_rate_history = []            # 1 - delivered/attempted

        self.ch_counts = defaultdict(int)             # node_id -> times CH/ZH
        self.ch_per_round_count_history = []           # num distinct CH/ZH this round

        self.fnd = None   # First Node Death (round index)
        self.hna = None   # Half Node(s) Dead (round index)
        self.and_round = None  # All Nodes Dead (round index)

        self._cumulative_consumed = 0.0

    def record(self, round_idx, network, energy_before=None):
        alive = network.num_alive()
        dead = self.n_nodes - alive
        residual_total = float(network.energy[network.alive].sum())
        residual_avg = residual_total / alive if alive > 0 else 0.0

        consumed = None
        if energy_before is not None:
            consumed = max(energy_before - residual_total, 0.0)
            self._cumulative_consumed += consumed

        self.alive_history.append(alive)
        self.dead_history.append(dead)
        self.residual_energy_history.append(residual_total)
        self.avg_residual_energy_history.append(residual_avg)
        self.energy_consumed_history.append(consumed)
        self.cumulative_energy_consumed_history.append(self._cumulative_consumed)

        if self.fnd is None and alive < self.n_nodes:
            self.fnd = round_idx
        if self.hna is None and alive <= self.n_nodes / 2:
            self.hna = round_idx
        if self.and_round is None and alive == 0:
            self.and_round = round_idx

    def record_ch(self, ch_ids):
      
        ch_ids = list(ch_ids)
        for node_id in ch_ids:
            self.ch_counts[int(node_id)] += 1
        self.ch_per_round_count_history.append(len(set(ch_ids)))

    def record_packets(self, attempted, delivered):
    
        self.packets_attempted_history.append(attempted)
        self.packets_delivered_history.append(delivered)
        loss_rate = 1.0 - (delivered / attempted) if attempted > 0 else 0.0
        self.packet_loss_rate_history.append(loss_rate)


    def finalize(self, last_round):
        avg_residual = (float(np.mean(self.residual_energy_history))
                         if self.residual_energy_history else 0.0)
        total_energy_consumed = (self.cumulative_energy_consumed_history[-1]
                                  if self.cumulative_energy_consumed_history else 0.0)

        ch_fairness_cv = None  # coefficient of variation of CH/ZH load;
                                 # lower = more evenly rotated (fairer)
        if len(self.ch_counts) > 1:
            vals = np.array(list(self.ch_counts.values()), dtype=float)
            ch_fairness_cv = float(np.std(vals) / (np.mean(vals) + 1e-12))

        total_attempted = int(np.sum(self.packets_attempted_history)) if self.packets_attempted_history else 0
        total_delivered = float(np.sum(self.packets_delivered_history)) if self.packets_delivered_history else 0.0
        overall_packet_loss_rate = (
            1.0 - (total_delivered / total_attempted) if total_attempted > 0 else None
        )
        throughput = total_delivered / last_round if last_round > 0 else 0.0

        return dict(
            
            fnd=self.fnd,
            hnd=self.hna,
            hna=self.hna,
            lnd=self.and_round,
            and_round=self.and_round,
            rounds_run=last_round,

            # energy
            avg_residual_energy=avg_residual,
            total_energy_consumed=total_energy_consumed,

            # packets / throughput
            overall_packet_loss_rate=overall_packet_loss_rate,
            throughput_packets_per_round=throughput,

            # CH/ZH fairness
            ch_counts=dict(self.ch_counts),
            ch_fairness_cv=ch_fairness_cv,

    
            alive_history=self.alive_history,
            dead_history=self.dead_history,
            residual_energy_history=self.residual_energy_history,
            avg_residual_energy_history=self.avg_residual_energy_history,
            energy_consumed_history=self.energy_consumed_history,
            cumulative_energy_consumed_history=self.cumulative_energy_consumed_history,
            packet_loss_rate_history=self.packet_loss_rate_history,
            ch_per_round_count_history=self.ch_per_round_count_history,
        )
