"""Single-method versions of the alive/residual curves, for the individual
'baseline observation' and 'baseline+GraphSAGE observation' slides (as
opposed to mogasage_*.png, which overlay both methods for the final
comparison slide).
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
rounds_axis = np.arange(len(data["moga"]["trajectory"]["alive"])) * data["sample_every"]


def make_pair(key, color, label, prefix):
    d = data[key]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(rounds_axis, d["trajectory"]["alive"], color=color, linewidth=2.2)
    ax.set_xlabel("round")
    ax.set_ylabel(f"alive nodes (of {n_nodes})")
    ax.set_title(f"{label}: alive nodes over time")
    fig.tight_layout()
    fig.savefig(f"{prefix}_alive_curve.png", dpi=150, bbox_inches="tight")
    print(f"saved {prefix}_alive_curve.png")

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(rounds_axis, d["trajectory"]["avg_residual"], color=color, linewidth=2.2)
    ax.set_xlabel("round")
    ax.set_ylabel("avg residual energy (J)")
    ax.set_title(f"{label}: residual energy over time")
    fig.tight_layout()
    fig.savefig(f"{prefix}_residual_curve.png", dpi=150, bbox_inches="tight")
    print(f"saved {prefix}_residual_curve.png")


make_pair("moga", C_MOGA, "MOGA+LSA (stock)", "baseline")
make_pair("moga_graphsage", C_HYBRID, "MOGA+GraphSAGE+LSA", "hybrid")
