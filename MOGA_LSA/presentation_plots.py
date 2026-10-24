"""Presentation-ready figures for the MOGA+LSA vs MOGA+GraphSAGE+LSA slides:
a lifetime (FND/HND/LND) bar chart, an alive-nodes-over-time curve, and a
residual-energy-over-time curve. Mirrors the single-metric-per-image style
the rest of the team's slides use.
"""
import json

import numpy as np
import matplotlib.pyplot as plt

IN_PATH = "wsn_dataset_extended_lifetime.json"
C_MOGA = "#1B6E77"
C_HYBRID = "#7A4FB5"

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
    "axes.edgecolor": "#57626C", "axes.labelcolor": "#12181F", "text.color": "#12181F",
    "xtick.color": "#57626C", "ytick.color": "#57626C",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#DDE2E7", "font.size": 12, "legend.frameon": False,
})

data = json.load(open(IN_PATH))
n_nodes = data["n_nodes"]
moga, hybrid = data["moga"], data["moga_graphsage"]
rounds_axis = np.arange(len(moga["trajectory"]["alive"])) * data["sample_every"]


def _fmt(v):
    return f"{v:,}" if v >= 0 else "n/a"


# 1) FND / HND / LND bar chart
fig, ax = plt.subplots(figsize=(5, 4.2))
labels = ["FND", "HND", "LND"]
x = np.arange(len(labels))
w = 0.35
moga_vals = [moga["fnd"], moga["hnd"], moga["lnd"]]
hybrid_vals = [hybrid["fnd"], hybrid["hnd"], hybrid["lnd"]]
ax.bar(x - w / 2, [max(v, 0) for v in moga_vals], w, label="MOGA+LSA", color=C_MOGA)
ax.bar(x + w / 2, [max(v, 0) for v in hybrid_vals], w, label="MOGA+GraphSAGE+LSA", color=C_HYBRID)
for xi, v in zip(x - w / 2, moga_vals):
    ax.text(xi, max(v, 0), _fmt(v), ha="center", va="bottom", fontsize=9)
for xi, v in zip(x + w / 2, hybrid_vals):
    ax.text(xi, max(v, 0), _fmt(v), ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("round")
ax.set_title("Network lifetime (rounds)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("mogasage_lifetime_bars.png", dpi=150, bbox_inches="tight")
print("saved mogasage_lifetime_bars.png")

# 2) alive nodes over time
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.plot(rounds_axis, moga["trajectory"]["alive"], color=C_MOGA, linewidth=2, label="MOGA+LSA")
ax.plot(rounds_axis, hybrid["trajectory"]["alive"], color=C_HYBRID, linewidth=2, label="MOGA+GraphSAGE+LSA")
ax.set_xlabel("round")
ax.set_ylabel(f"alive nodes (of {n_nodes})")
ax.set_title("Alive nodes over time")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("mogasage_alive_curve.png", dpi=150, bbox_inches="tight")
print("saved mogasage_alive_curve.png")

# 3) avg residual energy over time
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.plot(rounds_axis, moga["trajectory"]["avg_residual"], color=C_MOGA, linewidth=2, label="MOGA+LSA")
ax.plot(rounds_axis, hybrid["trajectory"]["avg_residual"], color=C_HYBRID, linewidth=2, label="MOGA+GraphSAGE+LSA")
ax.set_xlabel("round")
ax.set_ylabel("avg residual energy (J)")
ax.set_title("Residual energy over time")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("mogasage_residual_curve.png", dpi=150, bbox_inches="tight")
print("saved mogasage_residual_curve.png")

print(f"\nMOGA+LSA:          FND={_fmt(moga['fnd'])}  HND={_fmt(moga['hnd'])}  LND={_fmt(moga['lnd'])}")
print(f"MOGA+GraphSAGE+LSA: FND={_fmt(hybrid['fnd'])}  HND={_fmt(hybrid['hnd'])}  LND={_fmt(hybrid['lnd'])}")
