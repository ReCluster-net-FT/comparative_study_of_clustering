import os
import numpy as np
import pandas as pd
DATASET_PATH = "data/converted_dataset.csv"

REQUIRED_COLUMNS = [
    "Node_ID", "Timestamp", "X_Coordinate", "Y_Coordinate", "Z_Coordinate",
    "Initial_Energy", "Residual_Energy", "Transmission_Power",
    "Signal_Strength", "Noise_Level", "Energy_Consumption",
    "Packet_Loss_Rate", "Network_Lifetime", "Optimization_Algorithm",
    "Adaptive_Learning_Rate", "Temperature", "Humidity",
    "Detection_Accuracy",
]

                                
GRAPHSAGE_FEATURE_COLUMNS = [
    "Residual_Energy", "Transmission_Power", "Signal_Strength",
    "Noise_Level", "Energy_Consumption", "Packet_Loss_Rate",
    "Adaptive_Learning_Rate", "Temperature", "Humidity",
    "X_Coordinate", "Y_Coordinate", "Z_Coordinate",
]


class DatasetReport:


    def __init__(self):
        self.path = None
        self.n_rows = 0
        self.n_unique_node_ids = 0
        self.n_unique_timestamps = 0
        self.time_min = None
        self.time_max = None
        self.missing_counts = {}
        self.dropped_rows = 0
        self.invalid_residual_gt_initial = 0
        self.n_nodes_used = 0
        self.sampled = False
        self.sample_size = None

    def print_summary(self):
        print("=" * 70)
        print("DATASET VERIFICATION REPORT (common/dataset_loader.py)")
        print("=" * 70)
        print(f"CSV file path                 : {self.path}")
        print(f"Rows read                     : {self.n_rows}")
        print(f"Unique Node_ID values         : {self.n_unique_node_ids}")
        print(f"Unique Timestamp values       : {self.n_unique_timestamps}")
        print(f"Dataset time range            : {self.time_min}  ->  {self.time_max}")
        any_missing = any(v for v in self.missing_counts.values())
        if any_missing:
            print("Missing values found:")
            for c, n in self.missing_counts.items():
                if n:
                    print(f"    {c}: {n} missing")
        else:
            print("Missing values                : none found in required columns")
        print(f"Rows dropped (missing/unparseable): {self.dropped_rows}")
        print(f"Rows with Residual_Energy > Initial_Energy (data-quality flag,")
        print(f"  NOT used to seed simulation energy -- see module docstring): "
              f"{self.invalid_residual_gt_initial}")
        if self.sampled:
            print(f"Node sample used for simulation   : {self.sample_size} of "
                  f"{self.n_unique_node_ids} real nodes")
            print(f"  (DATASET_MAX_NODES cap in config.py -- a fixed-seed random")
            print(f"   SUBSET of real rows for O(n^2) feasibility; no values invented)")
        else:
            print(f"Node sample used for simulation   : all {self.n_nodes_used} nodes")
        print(f"NOTE: Node_ID<->Timestamp is 1:1 in this file (no node is measured")
        print(f"  more than once) -- see this module's docstring for what that means")
        print(f"  for how round-by-round simulation is done here.")
        print("=" * 70)


class WSNDataset:

    def __init__(self, df, report):
        self.df = df.reset_index(drop=True)
        self.report = report
        self.n_nodes = len(self.df)

        self.node_ids = self.df["Node_ID"].to_numpy()
        self.pos_xy = self.df[["X_Coordinate", "Y_Coordinate"]].to_numpy(dtype=float)
        self.z = self.df["Z_Coordinate"].to_numpy(dtype=float)

        # Simulation starting energy -- REAL, from the CSV, never randomized.
        self.initial_energy = self.df["Initial_Energy"].to_numpy(dtype=float)
        # Reporting-only / auxiliary-feature-only (see module docstring for
        # why this is never used to seed simulated energy).
        self.residual_energy_raw = self.df["Residual_Energy"].to_numpy(dtype=float)

        self.packet_loss_rate_raw = self.df["Packet_Loss_Rate"].to_numpy(dtype=float)

        raw_feats = self.df[GRAPHSAGE_FEATURE_COLUMNS].to_numpy(dtype=float)
        fmin = raw_feats.min(axis=0)
        fmax = raw_feats.max(axis=0)
        denom = np.where((fmax - fmin) > 1e-12, fmax - fmin, 1.0)
        self.features = (raw_feats - fmin) / denom          # (n, F) in [0,1]
        self.feature_names = list(GRAPHSAGE_FEATURE_COLUMNS)

        plr_min, plr_max = self.packet_loss_rate_raw.min(), self.packet_loss_rate_raw.max()
        span = max(plr_max - plr_min, 1e-12)
        self.packet_loss_prob = np.clip(
            (self.packet_loss_rate_raw - plr_min) / span, 0.0, 1.0
        )

    def bs_position(self, mode="centroid"):
    
        if mode == "centroid":
            return self.pos_xy.mean(axis=0)
        elif mode == "corner":
            return np.array([self.pos_xy[:, 0].max(), self.pos_xy[:, 1].max()])
        raise ValueError(f"Unknown bs mode: {mode!r}")


def load(csv_path, max_nodes=None, sample_seed=0):
  
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at '{csv_path}'. Set common/config.py "
            f"DATASET_PATH to the real preprocessed dataset -- this "
            f"pipeline does not fall back to synthetic data."
        )

    df_raw = pd.read_csv(csv_path)

    report = DatasetReport()
    report.path = os.path.abspath(csv_path)
    report.n_rows = len(df_raw)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_required:
        raise ValueError(
            f"Dataset at '{csv_path}' is missing required columns: "
            f"{missing_required}"
        )

    report.missing_counts = {
        c: int(df_raw[c].isna().sum()) for c in REQUIRED_COLUMNS if c != "Timestamp"
    }

    df_raw = df_raw.copy()
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], errors="coerce")
    report.missing_counts["Timestamp"] = int(df_raw["Timestamp"].isna().sum())

    before = len(df_raw)
    df = df_raw.dropna(subset=REQUIRED_COLUMNS).copy()
    report.dropped_rows = before - len(df)

    report.n_unique_node_ids = int(df["Node_ID"].nunique())
    report.n_unique_timestamps = int(df["Timestamp"].nunique())
    report.time_min = df["Timestamp"].min()
    report.time_max = df["Timestamp"].max()
    report.invalid_residual_gt_initial = int(
        (df["Residual_Energy"] > df["Initial_Energy"]).sum()
    )

    # Node_ID is supposed to be a unique node identifier; guard against
    # accidental duplicates rather than silently double-counting a node.
    df = df.drop_duplicates(subset=["Node_ID"], keep="first")
    df = df.sort_values("Node_ID").reset_index(drop=True)

    if max_nodes is not None and max_nodes < len(df):
        rng = np.random.default_rng(sample_seed)
        idx = np.sort(rng.choice(len(df), size=max_nodes, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        report.sampled = True
        report.sample_size = max_nodes

    report.n_nodes_used = len(df)

    return WSNDataset(df, report)
