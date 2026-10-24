"""All tracked metrics, stock MOGA+LSA vs MOGA+GraphSAGE-embeddings+LSA, on the WSN dataset."""
import json

import numpy as np
import matplotlib.pyplot as plt

IN_PATH = "multi_seed_wsn_dataset_results.json"
C_MOGA = "#1B6E77"
C_HYBRID = "#7A4FB5"

plt.rcParams.update({
    "figure.facecolor": "#FBFAF7", "axes.facecolor": "#FBFAF7", "savefig.facecolor": "#FBFAF7",
    "axes.edgecolor": "#57626C", "axes.labelcolor": "#12181F", "text.color": "#12181F",
    "xtick.color": "#57626C", "ytick.color": "#57626C",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#DDE2E7", "font.size": 10.5, "legend.frameon": False,
})

data = json.load(open(IN_PATH))
results = data["results"]

METRICS = [
    ("alive", f"alive nodes (of {data['n_nodes']})"),
    ("avg_residual", "avg residual energy (J) — survivors only*"),
    ("avg_energy", "avg energy consumed (J) — survivors only*"),
    ("delivery_rate", "CH-chain delivery rate"),
    ("throughput", "throughput (delivered / sent)"),
    ("avg_delay", "avg end-to-end delay (s)"),
]

rounds = data["rounds"]
n = len(results[0]["moga"]["trajectory"]["alive"])
sample_every = rounds // n
rounds_axis = np.arange(n) * sample_every

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (key, ylabel) in zip(axes.flat, METRICS):
    for method, color, label in [("moga", C_MOGA, "MOGA + LSA (stock)"),
                                   ("moga_graphsage", C_HYBRID, "MOGA + GraphSAGE embeddings + LSA")]:
        mat = np.array([r[method]["trajectory"][key] for r in results])
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        ax.plot(rounds_axis, mean, color=color, linewidth=2, label=label)
        ax.fill_between(rounds_axis, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("round")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

fig.suptitle(f"All tracked metrics — stock MOGA+LSA vs MOGA+GraphSAGE embeddings+LSA, "
             f"WSN dataset ({len(results)} seeds, mean ± 1σ)\n"
             "*avg_residual/avg_energy are survivor-only and favor whichever method loses more nodes",
             fontsize=11)
fig.tight_layout()
fig.savefig("WSN_Dataset_all_metrics.png", dpi=130, bbox_inches="tight")
print("saved WSN_Dataset_all_metrics.png")
