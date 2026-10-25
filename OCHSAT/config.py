"""
config.py
=========
All simulation parameters, taken directly from Table 3 (Simulation Parameters)
and Section 5.1-5.3 of the OCHSAT paper:

  Al-Qurabat et al., "Optimizing cluster heads for energy efficiency in WSNs
  using MADM approaches", Telecommunication Systems (2026) 89:26.

Edit values here to re-run experiments with different settings -- nothing
else in the codebase needs to change.
"""

import math

# ---------------------------------------------------------------------------
# Radio / energy model (Section 3.2, Table 3)
# ---------------------------------------------------------------------------
E_ELEC = 50e-9          # J/bit      - energy to run transmitter/receiver electronics
E_FS = 10e-12           # J/bit/m^2  - free-space amplifier energy
E_MP = 0.0013e-12       # J/bit/m^4  - multipath amplifier energy
E_DA = 5e-9             # J/bit      - data aggregation energy
PACKET_SIZE_BITS = 4000 # bits       - data packet size
PACKET_SIZE = PACKET_SIZE_BITS
INITIAL_ENERGY = 0.5                # J          - initial energy per node

# distance threshold d0 = sqrt(E_FS / E_MP)  (~87 m, matches Table 3)
D0 = math.sqrt(E_FS / E_MP)

# ---------------------------------------------------------------------------
# Network / deployment (Section 5.1, 5.2)
# ---------------------------------------------------------------------------
FIELD_X = 100           # m
FIELD_Y = 100           # m
NODE_COUNTS = [100, 200]
BS_POSITIONS = {
    "center": (50, 50),
    "corner": (0, 0),
}

COMMUNICATION_RANGE = 30          # m  (TelosB-class motes, Table 3)
                                   # NOTE: this fixed value is now only a
                                   # fallback. The value actually used at
                                   # runtime is Network.communication_radius,
                                   # computed dynamically per network from
                                   # the node-density formula below.
DATA_RATE = 250000    # 250 kbps, IEEE 802.15.4

# ---------------------------------------------------------------------------
# Dynamic communication radius (node-density formula)
# ---------------------------------------------------------------------------
# Two nodes are only allowed to be treated as radio neighbours (i.e. an edge
# is drawn between them for centrality/degree features and for the
# GraphSAGE graph) if their Euclidean distance is <= this radius - NOT
# simply "every other node in the cluster" as before.
#
#   rho = N / A                      (node density, nodes/m^2)
#   R   = sqrt(k / (pi * rho))       (radius that yields ~k neighbours/node)
#
# where N = number of nodes actually deployed, A = FIELD_X * FIELD_Y, and
# k = TARGET_NODE_DEGREE. This is computed fresh for every Network (it
# depends on however many nodes that particular run actually has), so it
# adapts automatically to 100-node vs 200-node (vs any dataset-driven)
# scenarios instead of being one fixed number.
TARGET_NODE_DEGREE = 12    # k in the formula above (average neighbours/node)

# ---------------------------------------------------------------------------
# Dataset-driven network (real WSN_Dataset.csv from Kaggle) instead of the
# synthetic uniform-random node generator.
# ---------------------------------------------------------------------------
# DATA_SOURCE
#   "synthetic" -> Network.generate_nodes() behaves exactly as before
#                  (uniform-random placement in FIELD_X x FIELD_Y).
#   "dataset"   -> Network.generate_nodes() instead draws `node_count` rows
#                  from DATASET_PATH (see dataset_loader.py) - a different,
#                  deterministic subset per random seed - and uses each
#                  row's real X_Coordinate/Y_Coordinate/Initial_Energy as
#                  that node's starting position/energy. Every existing
#                  script (main.py, compare_graphsage.py,
#                  graphsage_all_scenarios.py, ...) picks this up
#                  automatically because they all go through Network(); no
#                  call-site changes are required, only this flag.
DATA_SOURCE = "synthetic"

# Path to the Kaggle "WSN_Dataset.csv" file. Edit this to point at your
# actual downloaded copy before switching DATA_SOURCE to "dataset".
DATASET_PATH = "WSN_Dataset.csv"

# The dataset's Initial_Energy column is in the dataset's own (unspecified)
# units, not necessarily Joules - and its typical magnitude (~5-15) is far
# larger than the radio model's per-round energy cost (calibrated around
# cfg.INITIAL_ENERGY = 0.5 J), so nodes would essentially never die within
# MAX_ROUNDS if used as-is.
#
# DATASET_ENERGY_SCALE = None (default): auto-calibrated. The loader
#   rescales the WHOLE dataset's Initial_Energy so its mean lands exactly
#   on cfg.INITIAL_ENERGY - nodes keep their relative differences (some
#   start with a bit more/less energy than others, preserving the
#   dataset's real heterogeneity), just centered on a regime where the
#   existing radio model actually produces deaths within MAX_ROUNDS. The
#   chosen factor is printed once so it's never a silent surprise.
# DATASET_ENERGY_SCALE = <number>: use that exact multiplier instead
#   (auto-calibration is skipped). Set this if you specifically want to
#   study a different energy regime (e.g. much shorter/longer lifetimes).
DATASET_ENERGY_SCALE = None

# ---------------------------------------------------------------------------
# CH rotation
# ---------------------------------------------------------------------------
ROTATION_THRESHOLD = 0.30   # rotate CH when residual energy < alpha * E0

# ---------------------------------------------------------------------------
# Clustering (silhouette search range)
# ---------------------------------------------------------------------------
MIN_CLUSTERS = 3
MAX_CLUSTERS = 12

# ---------------------------------------------------------------------------
# AHP pairwise comparison matrix for criteria [RE, Centrality, DBS]
# Reproduced exactly from Section 4.2.2 of the paper.
# ---------------------------------------------------------------------------
AHP_PAIRWISE_MATRIX = [
    [1,   5,   3],
    [1/5, 1,   1/3],
    [1/3, 3,   1],
]
CRITERIA = ["Residual Energy", "Centrality", "Distance to BS"]

# ---------------------------------------------------------------------------
# Simulation control
# ---------------------------------------------------------------------------
RANDOM_SEEDS = [42, 101, 202, 303, 404]   # Table 3
MAX_ROUNDS = 2500                          # hard safety cap per run
SIGNIFICANCE_ALPHA = 0.05                  # paired t-test threshold

PROTOCOLS = ["OCHSAT", "ESCA", "IGHND", "MOFCA"]

# Scenario definitions (Table 4)
SCENARIOS = [
    {"name": "SCEN#1", "nodes": 100, "bs": "center", "focus": "Balanced baseline performance"},
    {"name": "SCEN#2", "nodes": 100, "bs": "corner", "focus": "Asymmetric load handling"},
    {"name": "SCEN#3", "nodes": 200, "bs": "center", "focus": "Scalability with balanced load"},
    {"name": "SCEN#4", "nodes": 200, "bs": "corner", "focus": "Combined stress test"},
]

# Scalability sweep used for Table 7 (per-round energy vs node density)
SCALABILITY_NODE_COUNTS = [100, 200, 300, 400]

# ---------------------------------------------------------------------------
# GraphSAGE-based Cluster Head selection -- v4 (OFFLINE, FROZEN) REDESIGN
# ---------------------------------------------------------------------------
# Only applied to the OCHSAT protocol - ESCA/IGHND/MOFCA remain untouched
# baselines so the paper's comparison stays fair/unmodified.
#
# This replaces the earlier v1-v3 designs (all of which trained GraphSAGE
# ONLINE, during the very simulation being evaluated -- see git history /
# ch_selector.py docstring for why that was scientifically invalid: no
# train/test separation, and the model could never be "frozen" for a
# clean final evaluation). The pipeline is now strictly:
#
#   OFFLINE data generation (offline_data_generation.py, uses
#       GRAPHSAGE_TRAIN_SEEDS / GRAPHSAGE_VAL_SEEDS only)
#         -> OFFLINE training (train_graphsage_offline.py)
#         -> best checkpoint saved to GRAPHSAGE_MODEL_DIR
#         -> FROZEN at inference time in ch_selector.py (model.eval(),
#            no optimizer, no backward(), no future rollout) during the
#            actual OCHSAT simulation, which only ever uses
#            RANDOM_SEEDS (the final, untouched test seeds).
#
# GRAPHSAGE_ENABLED
#   Master on/off switch. If False (or torch isn't installed, or no
#   trained checkpoint exists), OCHSAT falls back to pure AHP-TOPSIS.
GRAPHSAGE_ENABLED = True

# GRAPHSAGE_MODE selects how the frozen model participates in the final
# CH decision (GA / learned-fusion-weight approaches are intentionally
# excluded -- see PRD). One of:
#   "ahp_topsis_rerank" (PRIMARY / default) - Approach B from the PRD.
#       AHP-TOPSIS ranks all eligible candidates using the existing
#       three criteria; GraphSAGE then re-ranks only the top-K of those
#       candidates using its frozen neighbourhood-aware score; the final
#       CH is the highest-GraphSAGE-scoring node among that top-K. This
#       is the primary method (chosen over an arbitrary fixed weighted
#       sum -- see GRAPHSAGE_TOPK below).
#   "fixed_blend" - Approach A, kept ONLY as an experimental baseline
#       (0.5*AHP + 0.5*GraphSAGE on min-max normalized scores), NOT
#       evidence that 50/50 is optimal, NOT the primary method.
#   "graphsage_only" - ablation: ignore AHP-TOPSIS entirely, CH =
#       argmax frozen GraphSAGE score over ALL alive cluster members.
#   "node_only_rerank" - ablation (Experiment A, "does the graph
#       actually matter?"): identical to "ahp_topsis_rerank" but the
#       frozen model loaded is the node-only MLP baseline (same input
#       features, no message passing) instead of real GraphSAGE.
GRAPHSAGE_MODE = "ahp_topsis_rerank"

# K for the re-ranking approaches: AHP-TOPSIS proposes its top-K
# candidates, GraphSAGE (or the node-only ablation model) re-ranks only
# those K. K=5 initially per the PRD; sweep this for a K-sensitivity
# study if compute allows (see run_ablation_study.py --k-sweep).
GRAPHSAGE_TOPK = 5

# Fallback fixed alpha, only used when GRAPHSAGE_MODE == "fixed_blend".
GRAPHSAGE_BLEND_ALPHA = 0.5

# ---------------------------------------------------------------------------
# Model architecture / where the frozen checkpoint lives
# ---------------------------------------------------------------------------
GRAPHSAGE_HIDDEN_DIMS = (32, 16)   # SAGEConv(in,32) -> SAGEConv(32,16) -> Linear(16,1)
GRAPHSAGE_DEVICE = "cuda"          # falls back to "cpu" automatically if no GPU
GRAPHSAGE_MODEL_DIR = "models"
GRAPHSAGE_CHECKPOINT = "models/graphsage_best.pt"
NODE_ONLY_CHECKPOINT = "models/node_only_best.pt"      # ablation baseline
FEATURE_NORM_CONFIG = "models/feature_norm.json"       # fixed, offline-fit
                                                        # normalization stats
                                                        # (mean/std per
                                                        # feature) -- NOT
                                                        # recomputed per
                                                        # cluster at
                                                        # inference time, so
                                                        # the frozen model
                                                        # sees the same
                                                        # feature scale it
                                                        # was trained on.

# Node feature order fed to the model (in this exact order):
#   [ residual_energy_ratio, centrality, distance_to_bs, degree ]
GRAPHSAGE_FEATURES = ["RE", "Centrality", "DBS", "Degree"]

# ---------------------------------------------------------------------------
# OFFLINE training-data generation (offline_data_generation.py)
# ---------------------------------------------------------------------------
# Disjoint seed pools -- the PRD requires train/validation/test separation
# so the final OCHSAT evaluation is on genuinely unseen data. RANDOM_SEEDS
# above (5 seeds) is kept EXACTLY as-is and reserved for the final test
# only; these are brand-new, additional seeds used purely to *generate*
# offline training/validation graph snapshots (never touched by, and
# never used to tune, the final evaluation).
GRAPHSAGE_TRAIN_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
GRAPHSAGE_VAL_SEEDS = [2001, 2002, 2003, 2004]

# Snapshot cadence: every this many rounds, sample a CH-selection event
# per active cluster and generate counterfactual rollouts for it (doing
# this every round for every cluster is unnecessary and expensive).
GRAPHSAGE_SNAPSHOT_INTERVAL = 10

# Counterfactual candidate sampling (PRD section 11/response to Q1):
# evaluate the top-N AHP candidates plus M deterministically-chosen
# "random" candidates (seeded from the network's own RNG, so the whole
# pipeline stays reproducible) -- not every member of every cluster,
# which would be far too expensive for large clusters.
GRAPHSAGE_TOPN_AHP_CANDIDATES = 3
GRAPHSAGE_RANDOM_CANDIDATES = 2

# Short rollout horizon (PRD section 12): 25 rounds, NOT ~150. At 150
# rounds "energy now" almost deterministically predicts "energy later"
# (nothing reverses it), so the label becomes redundant with AHP-TOPSIS's
# own Residual-Energy criterion. 25 rounds leaves more room for
# short-term topological/neighbourhood effects to actually matter.
GRAPHSAGE_UTILITY_HORIZON = 25

# Utility components (PRD section 13) -- only metrics the simulator
# actually exposes are used, nothing invented:
#   energy_ratio    : candidate's OWN residual energy / initial energy
#                     at the horizon (0 if dead by then).
#   survival        : 1.0 if the candidate itself is alive at the horizon.
#   cluster_health  : fraction of the candidate's CLUSTER-MATES (every
#                     other node in the same rollout, excluding the
#                     candidate) still alive at the horizon. This is the
#                     one component that is NOT a property of the
#                     candidate in isolation -- a node can be personally
#                     fine while its neighbourhood collapses, or vice
#                     versa -- so it is the signal GraphSAGE's neighbour-
#                     aggregating message passing can see that AHP-
#                     TOPSIS's per-node scoring structurally cannot.
# (Honest limitation, unchanged from earlier versions: this simulator has
# no packet-loss/failure model, so "communication performance" is
# operationalised as cluster-mate survival, not a literal delivery ratio.)
GRAPHSAGE_UTILITY_ENERGY_WEIGHT = 0.4
GRAPHSAGE_UTILITY_SURVIVAL_WEIGHT = 0.3
GRAPHSAGE_UTILITY_CLUSTER_WEIGHT = 0.3

# Residual-target formulation (PRD section 14, the "stronger approach"):
#   graphsage_target = future_ch_utility(candidate) - ahp_baseline(candidate)
# where ahp_baseline is that candidate's own (min-max normalized)
# AHP-TOPSIS closeness score AT SELECTION TIME. This explicitly frames the
# learning problem as "where should the AHP ranking be adjusted once
# neighbourhood/topology is taken into account", instead of asking
# GraphSAGE to re-derive future utility (and implicitly the AHP score)
# from scratch. Set to False to fall back to plain future-utility
# regression if you want to compare the two target formulations.
GRAPHSAGE_USE_RESIDUAL_TARGET = True

# Where offline_data_generation.py writes its output, and where
# train_graphsage_offline.py reads it from.
GRAPHSAGE_DATASET_DIR = "graphsage_offline_dataset"

# Diagnostic logging cadence for NetAnim visualization only (separate
# from the per-decision diagnostic log described below, which is always
# recorded for every CH-selection event regardless of this interval).
GRAPHSAGE_PREDICTION_LOG_INTERVAL = 25

