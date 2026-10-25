"""Throughput over time, stock MOGA+LSA vs MOGA+GraphSAGE+LSA, WSN dataset."""
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
moga, hybrid = data["moga"], data["moga_graphsage"]
rounds_axis = np.arange(len(moga["trajectory"]["alive"])) * data["sample_every"]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(rounds_axis, moga["trajectory"]["throughput"], color=C_MOGA, linewidth=2.2, label="MOGA+LSA (stock)")
ax.plot(rounds_axis, hybrid["trajectory"]["throughput"], color=C_HYBRID, linewidth=2.2, label="MOGA+GraphSAGE+LSA")
ax.set_xlabel("round")
ax.set_ylabel("throughput (delivered / sent)")
ax.set_title("Throughput over time")
ax.set_ylim(0.9, 1.02)
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig("mogasage_throughput_curve.png", dpi=150, bbox_inches="tight")
print("saved mogasage_throughput_curve.png")

print(f"final throughput: MOGA+LSA={moga['trajectory']['throughput'][-1]:.4f}  "
      f"MOGA+GraphSAGE+LSA={hybrid['trajectory']['throughput'][-1]:.4f}")
