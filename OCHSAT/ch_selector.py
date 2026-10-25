"""
ch_selector.py
==============

v4 REDESIGN -- frozen, offline-trained GraphSAGE inference only.

This file used to also train GraphSAGE ONLINE, during the very
simulation being evaluated (see git history for v1/v2/v3 -- all had the
same fundamental problem: train/test data were the same run, so there
was no way to "freeze" the model for a clean final evaluation, and the
outcome label kept re-deriving something close to the AHP-TOPSIS score
it was meant to be independent from).

The whole training pipeline has been moved OUT of this file and out of
the live simulation entirely:

    offline_data_generation.py  (train/val seeds only, counterfactual
                                  rollouts -- see PRD sections 10-14)
        -> train_graphsage_offline.py  (offline MSE training,
                                         best-checkpoint selection)
        -> models/graphsage_best.pt, models/node_only_best.pt,
           models/feature_norm.json

This file ONLY does inference with whatever frozen checkpoint it finds:

    model.eval()
    with torch.no_grad():
        scores = model(x, edge_index)

No optimizer, no backward(), no future rollout, no label generation, and
no per-round training step happen here or anywhere else during a real
OCHSAT run. If no trained checkpoint exists (or torch is missing),
`enabled` is False and OCHSAT falls back to plain AHP-TOPSIS -- exactly
as it did before GraphSAGE existed -- rather than silently scoring with
random, untrained weights.

Integration (config.GRAPHSAGE_MODE, GA/learned-fusion-weights excluded
per the PRD):

  "ahp_topsis_rerank" (PRIMARY, Approach B) -- AHP-TOPSIS ranks all
      eligible candidates with its existing three criteria; GraphSAGE
      re-ranks only the top-K of those candidates using its frozen
      neighbourhood-aware score computed over the REAL radio graph of
      the whole cluster; the final CH is the highest-GraphSAGE-scoring
      node among that top-K.
  "node_only_rerank" -- identical integration logic, but the frozen
      model loaded is the node-only MLP ablation baseline (no message
      passing) instead of real GraphSAGE. Used to isolate "does the
      graph actually matter?" (PRD section 21) at the full
      simulation/CH-selection level, not just at offline validation MSE.
  "graphsage_only" -- ablation: ignore AHP-TOPSIS, CH = argmax frozen
      GraphSAGE score over ALL alive cluster members.
  "fixed_blend" -- Approach A, kept only as an experimental baseline
      (0.5*AHP + 0.5*GraphSAGE on min-max normalized scores). NOT the
      primary method, NOT evidence 50/50 is optimal.

Diagnostic logging (PRD section 19-20): every call to select() that
actually reaches a trained model appends one record to
`self.decision_log` with the AHP-selected node, the GraphSAGE-selected
node (its own top pick, ignoring the blend/rerank restriction), the
final selected node, both raw scores, and whether GraphSAGE changed the
decision. `agreement_rate()`, `decision_change_rate()`, and
`score_correlation()` below summarize this log for
run_ablation_study.py -- this is how we tell whether GraphSAGE is doing
anything AHP-TOPSIS wasn't already doing, instead of just asserting it.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import config as cfg

from protocols import build_features
from ahp_topsis import topsis_rank
from radio_graph import build_edge_index_from_positions
from graph_features import raw_feature_matrix, FeatureNormalizer

if _HAS_TORCH:
    from graphsage_model import (
        load_checkpoint,
        build_fully_connected_edge_index,  # legacy fallback if radius=None
        get_device,
    )


def _minmax_normalize(values):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


class GraphSAGECHSelector:
    """
    One instance per (OCHSAT) simulation run. Holds a FROZEN, already-
    trained model loaded once at construction time -- nothing about this
    object's model state changes over the course of a run. Only
    `self.decision_log` grows, purely for diagnostics/reporting.
    """

    def __init__(self):
        self.mode = getattr(cfg, "GRAPHSAGE_MODE", "ahp_topsis_rerank")
        self.topk = max(1, getattr(cfg, "GRAPHSAGE_TOPK", 5))
        self.alpha = getattr(cfg, "GRAPHSAGE_BLEND_ALPHA", 0.5)
        self.decision_log = []
        # Pooled raw scores across every decision this run, for a single
        # end-of-run AHP/GraphSAGE score correlation (PRD section 20).
        self._pooled_ahp_scores = []
        self._pooled_graphsage_scores = []

        wants_model = _HAS_TORCH and getattr(cfg, "GRAPHSAGE_ENABLED", False)
        if not wants_model:
            self.enabled = False
            self.backend = "disabled (torch missing or GRAPHSAGE_ENABLED=False)"
            return

        checkpoint_path = (
            cfg.NODE_ONLY_CHECKPOINT if self.mode == "node_only_rerank"
            else cfg.GRAPHSAGE_CHECKPOINT
        )

        try:
            self.model, self.device, self.backend = load_checkpoint(checkpoint_path)
            self.normalizer = FeatureNormalizer.load(cfg.FEATURE_NORM_CONFIG)
            self.enabled = True
        except FileNotFoundError as exc:
            # No trained checkpoint yet -- fall back to pure AHP-TOPSIS
            # rather than scoring with random, untrained weights. Run
            # offline_data_generation.py + train_graphsage_offline.py
            # first.
            print(
                f"[GraphSAGE] WARNING: no frozen checkpoint found "
                f"({exc}). Falling back to pure AHP-TOPSIS. Run "
                f"offline_data_generation.py then "
                f"train_graphsage_offline.py first."
            )
            self.enabled = False
            self.backend = "disabled (no trained checkpoint found)"
            return

        # Model is frozen: eval mode + grads off (load_checkpoint already
        # does this, but re-assert here so it is impossible to miss).
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        print(
            f"[GraphSAGE] enabled (FROZEN inference only), "
            f"mode={self.mode}, backend={self.backend}, "
            f"device={self.device}, top_k={self.topk}, "
            f"checkpoint={checkpoint_path}"
        )

    # ------------------------------------------------------------
    # Feature / graph construction
    # ------------------------------------------------------------

    def _build_graph_tensors(self, features, radius):
        x_raw = raw_feature_matrix(features)
        x_norm = self.normalizer.apply(x_raw)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device)

        cluster_nodes = [f["node"] for f in features]

        if radius is not None:
            # Real radio graph: node i and node j are only connected if
            # their Euclidean distance is <= radius -- the same geometry
            # used to generate the offline training data (radio_graph.py).
            edge_index = build_edge_index_from_positions(
                cluster_nodes, radius, self.device
            )
        else:
            edge_index = build_fully_connected_edge_index(len(features), self.device)

        return x_tensor, edge_index

    # ------------------------------------------------------------
    # Main entry point: choose a Cluster Head
    # ------------------------------------------------------------

    def select(self, cluster_nodes, bs_position, weights, radius=None, current_round=0):
        """
        Returns
        -------
        chosen_node : Node
            The node actually selected as Cluster Head.
        predicted_node : Node
            GraphSAGE's OWN top pick (before any AHP filtering/blending)
            -- used for visualization/logging and for the diagnostic log,
            regardless of which node was actually chosen.
        """

        if len(cluster_nodes) == 1:
            return cluster_nodes[0], cluster_nodes[0]

        features = build_features(cluster_nodes, bs_position, radius=radius)
        decision_matrix = np.array(
            [[f["RE"], f["Centrality"], f["DBS"]] for f in features]
        )
        ahp_scores = topsis_rank(
            decision_matrix, weights, benefit_flags=[True, True, False]
        )
        ahp_rank_order = np.argsort(-ahp_scores)
        ahp_best_idx = int(ahp_rank_order[0])

        if not self.enabled:
            node = features[ahp_best_idx]["node"]
            return node, node

        x_tensor, edge_index = self._build_graph_tensors(features, radius)

        # ---- FROZEN forward pass: inference only, no gradient, no
        # training step of any kind happens here or anywhere else ----
        self.model.eval()
        with torch.no_grad():
            raw_scores = self.model(x_tensor, edge_index)
        graphsage_scores = raw_scores.detach().cpu().numpy()

        graphsage_rank_order = np.argsort(-graphsage_scores)
        graphsage_best_idx = int(graphsage_rank_order[0])

        # ---- decide the final CH per config.GRAPHSAGE_MODE ----
        n = len(features)
        if self.mode in ("ahp_topsis_rerank", "node_only_rerank"):
            top_k_idx = ahp_rank_order[: min(self.topk, n)]
            local_scores = graphsage_scores[top_k_idx]
            chosen_idx = int(top_k_idx[int(np.argmax(local_scores))])
        elif self.mode == "graphsage_only":
            chosen_idx = graphsage_best_idx
        elif self.mode == "fixed_blend":
            blended = (
                self.alpha * _minmax_normalize(graphsage_scores)
                + (1 - self.alpha) * _minmax_normalize(ahp_scores)
            )
            chosen_idx = int(np.argmax(blended))
        else:
            raise ValueError(f"Unknown config.GRAPHSAGE_MODE={self.mode!r}")

        chosen_node = features[chosen_idx]["node"]
        predicted_node = features[graphsage_best_idx]["node"]

        # ---- diagnostics (PRD section 19-20) ----
        ahp_node_id = features[ahp_best_idx]["node"].node_id
        graphsage_node_id = features[graphsage_best_idx]["node"].node_id
        final_node_id = chosen_node.node_id

        self.decision_log.append({
            "round": current_round,
            "mode": self.mode,
            "cluster_size": n,
            "ahp_selected_node": ahp_node_id,
            "graphsage_selected_node": graphsage_node_id,
            "final_selected_node": final_node_id,
            "ahp_score_of_final": float(ahp_scores[chosen_idx]),
            "graphsage_score_of_final": float(graphsage_scores[chosen_idx]),
            "ahp_rank_of_final": int(np.where(ahp_rank_order == chosen_idx)[0][0]) + 1,
            "graphsage_rank_of_final": int(np.where(graphsage_rank_order == chosen_idx)[0][0]) + 1,
            "graphsage_changed_decision": graphsage_node_id != ahp_node_id,
            "final_matches_ahp_top1": final_node_id == ahp_node_id,
        })
        self._pooled_ahp_scores.extend(ahp_scores.tolist())
        self._pooled_graphsage_scores.extend(graphsage_scores.tolist())

        return chosen_node, predicted_node

    # ------------------------------------------------------------
    # Diagnostics summary (PRD section 20)
    # ------------------------------------------------------------

    def agreement_rate(self):
        """% of decisions where GraphSAGE's own top pick == AHP-TOPSIS's
        top pick (PRD 'AHP-GraphSAGE agreement')."""
        if not self.decision_log:
            return None
        matches = sum(1 for d in self.decision_log if not d["graphsage_changed_decision"])
        return matches / len(self.decision_log)

    def decision_change_rate(self):
        """% of decisions where the FINAL selected CH differs from
        AHP-TOPSIS's own top pick (PRD 'GraphSAGE decision-change rate')."""
        if not self.decision_log:
            return None
        changed = sum(1 for d in self.decision_log if not d["final_matches_ahp_top1"])
        return changed / len(self.decision_log)

    def score_correlation(self):
        """Pearson correlation between AHP-TOPSIS scores and GraphSAGE
        scores, pooled across every candidate in every decision this
        run. A correlation near 1.0 is a warning sign that GraphSAGE
        learned essentially the same signal as AHP-TOPSIS (PRD 20)."""
        if len(self._pooled_ahp_scores) < 2:
            return None
        a = np.asarray(self._pooled_ahp_scores)
        b = np.asarray(self._pooled_graphsage_scores)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return None
        return float(np.corrcoef(a, b)[0, 1])
