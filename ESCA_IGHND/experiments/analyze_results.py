import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import config as cfg

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

RAW_PATH = "results/raw_results.json"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
EMBED_DIR = "results/embeddings"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(RAW_PATH, "r") as f:
    raw = json.load(f)

PROTOCOLS = ["ESCA", "ESCA-GraphSAGE", "IGHND", "IGHND-GraphSAGE"]
PROTOCOL_COLORS = {
    "ESCA": "#1f77b4",
    "ESCA-GraphSAGE": "#ff7f0e",
    "IGHND": "#2ca02c",
    "IGHND-GraphSAGE": "#d62728",
}
SCENARIOS = [s for s in raw.keys() if all(p in raw[s] for p in PROTOCOLS)]

SCALAR_METRICS = [
    ("fnd", "Rounds", "First Node Death (FND)"),
    ("hnd", "Rounds", "Half Node(s) Death (HND)"),
    ("lnd", "Rounds", "Last Node Death (LND / Network Lifetime)"),
    ("avg_residual_energy", "Energy (J)", "Average Residual Energy"),
    ("total_energy_consumed", "Energy (J)", "Total Energy Consumed"),
    ("overall_packet_loss_rate", "Loss rate", "Estimated Packet Loss Rate"),
    ("throughput_packets_per_round", "Packets/round", "Throughput"),
    ("ch_fairness_cv", "CV", "CH/ZH Load Fairness (lower = fairer)"),
]


def stat(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return np.nan, np.nan, 0
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std, len(values)


def get_metric_values(scen, proto, metric):
    return [t.get(metric) for t in raw[scen][proto] if "error" not in t and t.get(metric) is not None]


def get_curve_trial(scen, proto):
    """The single CURVE_SEED trial (has the full per-round histories)."""
    for t in raw[scen][proto]:
        if "error" not in t and t.get("seed") == cfg.CURVE_SEED and "alive_history" in t:
            return t
    return None


# ============================================================================
# 1. Summary table (all scalar metrics, mean/std/n, every scenario)
# ============================================================================

rows = []
for scen in SCENARIOS:
    row = {"Scenario": scen}
    for metric, _, _ in SCALAR_METRICS:
        for proto in PROTOCOLS:
            vals = get_metric_values(scen, proto, metric)
            m, s, n = stat(vals)
            row[f"{proto}_{metric}_mean"] = round(m, 4) if not np.isnan(m) else np.nan
            row[f"{proto}_{metric}_std"] = round(s, 4) if not np.isnan(s) else np.nan
            row[f"{proto}_{metric}_n"] = n
    rows.append(row)

df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
df.to_csv(csv_path, index=False)


# ============================================================================
# 2. Statistical comparison: ESCA vs IGHND, ESCA vs ESCA-GS, IGHND vs IGHND-GS
# ============================================================================

def paired_compare(scen, proto_a, proto_b, metric):
    """Match trials by seed, compute absolute + percentage improvement of
    proto_b over proto_a, and (if scipy is available) a paired t-test."""
    trials_a = {t["seed"]: t.get(metric) for t in raw[scen][proto_a]
                if "error" not in t and t.get(metric) is not None}
    trials_b = {t["seed"]: t.get(metric) for t in raw[scen][proto_b]
                if "error" not in t and t.get(metric) is not None}
    common_seeds = sorted(set(trials_a) & set(trials_b))
    if not common_seeds:
        return None

    a = np.array([trials_a[s] for s in common_seeds], dtype=float)
    b = np.array([trials_b[s] for s in common_seeds], dtype=float)

    abs_improve = float(np.mean(b - a))
    pct_improve = float(np.mean((b - a) / np.where(a == 0, np.nan, a)) * 100)
    consistent = bool(np.all(b > a) or np.all(b < a))

    p_value = None
    if _HAVE_SCIPY and len(common_seeds) > 1:
        try:
            p_value = float(_scipy_stats.ttest_rel(b, a).pvalue)
        except Exception:
            p_value = None

    return dict(
        scenario=scen, metric=metric, baseline=proto_a, compared=proto_b,
        n_seeds=len(common_seeds), abs_improvement=abs_improve,
        pct_improvement=pct_improve, consistent_direction=consistent,
        p_value=p_value,
    )


COMPARISON_PAIRS = [
    ("ESCA", "IGHND"),
    ("ESCA", "ESCA-GraphSAGE"),
    ("IGHND", "IGHND-GraphSAGE"),
]
COMPARISON_METRICS = ["fnd", "hnd", "lnd", "avg_residual_energy", "total_energy_consumed"]

comparison_rows = []
for scen in SCENARIOS:
    for proto_a, proto_b in COMPARISON_PAIRS:
        for metric in COMPARISON_METRICS:
            result = paired_compare(scen, proto_a, proto_b, metric)
            if result is not None:
                comparison_rows.append(result)

comparison_df = pd.DataFrame(comparison_rows)
comparison_csv_path = os.path.join(RESULTS_DIR, "statistical_comparison.csv")
comparison_df.to_csv(comparison_csv_path, index=False)


# ============================================================================
# 3. Excel workbook (Summary / Mean / Mean±STD / Comparison)
# ============================================================================

excel_path = os.path.join(RESULTS_DIR, "WSN_results_summary.xlsx")
mean_rows, formatted_rows = [], []
for scen in SCENARIOS:
    m_row, f_row = {"Scenario": scen}, {"Scenario": scen}
    for metric, _, _ in SCALAR_METRICS:
        for proto in PROTOCOLS:
            m_val = df.loc[df["Scenario"] == scen, f"{proto}_{metric}_mean"].iloc[0]
            s_val = df.loc[df["Scenario"] == scen, f"{proto}_{metric}_std"].iloc[0]
            m_row[f"{proto}_{metric}"] = m_val
            f_row[f"{proto}_{metric}"] = f"{m_val:.3f} ± {s_val:.3f}" if not np.isnan(m_val) else "N/A"
    mean_rows.append(m_row)
    formatted_rows.append(f_row)

mean_df = pd.DataFrame(mean_rows)
formatted_df = pd.DataFrame(formatted_rows)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Summary", index=False)
    mean_df.to_excel(writer, sheet_name="Mean", index=False)
    formatted_df.to_excel(writer, sheet_name="Mean ± STD", index=False)
    comparison_df.to_excel(writer, sheet_name="Statistical Comparison", index=False)


# ============================================================================
# 4. Bar-chart comparisons for every scalar metric
# ============================================================================

def make_bar_plot(metric, ylabel, title, filename):
    x = np.arange(len(SCENARIOS))
    width = 0.18
    plt.figure(figsize=(10, 6))
    for i, proto in enumerate(PROTOCOLS):
        means = []
        for scen in SCENARIOS:
            m, _, _ = stat(get_metric_values(scen, proto, metric))
            means.append(m)
        positions = x + (i - 1.5) * width
        plt.bar(positions, means, width=width, label=proto, color=PROTOCOL_COLORS[proto])
    plt.xticks(x, SCENARIOS)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


for metric, ylabel, title in SCALAR_METRICS:
    make_bar_plot(metric, ylabel, f"{title} Comparison", f"{metric}_comparison.png")


# ============================================================================
# 5. FND / HND / LND grouped comparison
# ============================================================================

def make_lifetime_milestone_plot(scen):
    metrics = ["fnd", "hnd", "lnd"]
    x = np.arange(len(metrics))
    width = 0.18
    plt.figure(figsize=(8, 6))
    for i, proto in enumerate(PROTOCOLS):
        means = [stat(get_metric_values(scen, proto, m))[0] for m in metrics]
        positions = x + (i - 1.5) * width
        plt.bar(positions, means, width=width, label=proto, color=PROTOCOL_COLORS[proto])
    plt.xticks(x, ["FND", "HND", "LND"])
    plt.ylabel("Rounds")
    plt.title(f"FND / HND / LND Comparison -- {scen}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"fnd_hnd_lnd_{scen}.png"), dpi=300, bbox_inches="tight")
    plt.close()


for scen in SCENARIOS:
    make_lifetime_milestone_plot(scen)


# ============================================================================
# 6. Per-round curves : alive, dead, avg energy, energy
#    consumption, packet loss -- all vs. round, from the CURVE_SEED run
# ============================================================================

def make_curve_plot(key, ylabel, title, filename, smooth_window=1):
    plt.figure(figsize=(10, 6))
    any_plotted = False
    for proto in PROTOCOLS:
        for scen in SCENARIOS:
            trial = get_curve_trial(scen, proto)
            if trial is None or key not in trial:
                continue
            series = [v if v is not None else 0.0 for v in trial[key]]
            if smooth_window > 1 and len(series) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                series = np.convolve(series, kernel, mode="valid")
            rounds = np.arange(1, len(series) + 1)
            plt.plot(rounds, series, label=f"{proto} ({scen})",
                     color=PROTOCOL_COLORS[proto],
                     linestyle="--" if scen != SCENARIOS[0] else "-",
                     alpha=0.85, linewidth=1.3)
            any_plotted = True
    if not any_plotted:
        plt.close()
        return
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


make_curve_plot("alive_history", "Alive nodes", "Alive Nodes vs Rounds",
                 "alive_vs_rounds.png")
make_curve_plot("dead_history", "Dead nodes", "Dead Nodes vs Rounds",
                 "dead_vs_rounds.png")
make_curve_plot("avg_residual_energy_history", "Avg residual energy (J)",
                 "Average Residual Energy vs Rounds", "avg_energy_vs_rounds.png")
make_curve_plot("energy_consumed_history", "Energy consumed this round (J)",
                 "Energy Consumption vs Rounds (20-round rolling mean)",
                 "energy_consumption_vs_rounds.png", smooth_window=20)
make_curve_plot("packet_loss_rate_history", "Estimated packet loss rate",
                 "Estimated Packet Loss Rate vs Rounds (20-round rolling mean, "
                 "OBSERVATIONAL model -- see common/packet_model.py)",
                 "packet_loss_vs_rounds.png", smooth_window=20)


# ============================================================================
# 7. CH / ZH selection distribution
# ============================================================================

def make_ch_distribution_plot(scen):
    plt.figure(figsize=(10, 6))
    plotted = False
    for proto in PROTOCOLS:
        trial = get_curve_trial(scen, proto)
        if trial is None or "ch_counts" not in trial or not trial["ch_counts"]:
            continue
        counts = sorted(trial["ch_counts"].values(), reverse=True)
        plt.plot(np.arange(1, len(counts) + 1), counts, marker="o", markersize=2,
                 label=proto, color=PROTOCOL_COLORS[proto])
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Node rank (most-selected CH/ZH first)")
    plt.ylabel("Times selected as CH / ZH")
    plt.title(f"CH/ZH Selection Distribution -- {scen} (fairer = flatter curve)")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"ch_distribution_{scen}.png"), dpi=300, bbox_inches="tight")
    plt.close()


for scen in SCENARIOS:
    make_ch_distribution_plot(scen)


# ============================================================================
# 8. GraphSAGE embedding visualization 
# ============================================================================

def _pca_2d(X):
    """Plain-numpy PCA (no sklearn dependency needed for a 2D projection)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2 or Xc.shape[1] < 2:
        return None
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def make_embedding_plot(scen, proto):
    path = os.path.join(EMBED_DIR, f"{scen}_{proto}.npz")
    if not os.path.exists(path):
        return
    data = np.load(path)
    emb = data["embedding"]
    if emb.shape[0] < 3 or emb.shape[1] < 2:
        return
    proj = _pca_2d(emb)
    if proj is None:
        return
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], s=18, alpha=0.75,
                      c=np.arange(emb.shape[0]), cmap="viridis")
    plt.colorbar(sc, label="node index (alive-node order)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"GraphSAGE Node Embeddings (PCA-2D) -- {proto}, {scen}\n"
              f"(final round captured, seed={cfg.CURVE_SEED})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"embedding_{scen}_{proto}.png"), dpi=300, bbox_inches="tight")
    plt.close()


for scen in SCENARIOS:
    for proto in ["ESCA-GraphSAGE", "IGHND-GraphSAGE"]:
        make_embedding_plot(scen, proto)


# ============================================================================
# 9. Overall performance comparison -- normalized radar chart 
# ============================================================================

RADAR_METRICS = [
    ("lnd", "Network\nLifetime", False),          # higher is better
    ("fnd", "FND", False),                         # higher is better
    ("avg_residual_energy", "Avg Residual\nEnergy", False),  # higher is better
    ("total_energy_consumed", "Energy\nEfficiency", True),    # lower consumed = better -> invert
    ("overall_packet_loss_rate", "Packet\nReliability", True),  # lower loss = better -> invert
]


def make_radar_plot():
    labels = [lbl for _, lbl, _ in RADAR_METRICS]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # aggregate each metric across ALL scenarios and ALL seeds per protocol
    raw_vals = {proto: [] for proto in PROTOCOLS}
    for metric, _, invert in RADAR_METRICS:
        all_vals = []
        per_proto = {}
        for proto in PROTOCOLS:
            vals = []
            for scen in SCENARIOS:
                vals.extend(get_metric_values(scen, proto, metric))
            m = float(np.mean(vals)) if vals else np.nan
            per_proto[proto] = m
            if not np.isnan(m):
                all_vals.append(m)
        if not all_vals:
            for proto in PROTOCOLS:
                raw_vals[proto].append(0.5)
            continue
        lo, hi = min(all_vals), max(all_vals)
        span = (hi - lo) if hi > lo else 1.0
        for proto in PROTOCOLS:
            v = per_proto[proto]
            if np.isnan(v):
                norm = 0.0
            else:
                norm = (v - lo) / span
                if invert:
                    norm = 1.0 - norm
            raw_vals[proto].append(norm)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for proto in PROTOCOLS:
        values = raw_vals[proto] + raw_vals[proto][:1]
        ax.plot(angles, values, label=proto, color=PROTOCOL_COLORS[proto], linewidth=2)
        ax.fill(angles, values, color=PROTOCOL_COLORS[proto], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Performance Comparison\n(min-max normalized across protocols, "
                 "all scenarios & seeds; outer = better)", pad=35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.25), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "overall_performance_radar.png"), dpi=300, bbox_inches="tight")
    plt.close()


make_radar_plot()


print("Analysis complete.")
print(f"  Tables : {csv_path}, {comparison_csv_path}, {excel_path}")
print(f"  Plots  : {PLOTS_DIR}/")
