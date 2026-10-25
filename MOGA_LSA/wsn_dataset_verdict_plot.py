"""One figure, three subplots: does MOGA+GraphSAGE-embeddings+LSA beat
stock MOGA+LSA on the WSN dataset?
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

IN_PATH = "multi_seed_wsn_dataset_results.json"
C_MOGA = "#1B6E77"
C_HYBRID = "#7A4FB5"
C_GOOD = "#2F8F5B"

plt.rcParams.update({
    "figure.facecolor": "#FBFAF7", "axes.facecolor": "#FBFAF7", "savefig.facecolor": "#FBFAF7",
    "axes.edgecolor": "#57626C", "axes.labelcolor": "#12181F", "text.color": "#12181F",
    "xtick.color": "#57626C", "ytick.color": "#57626C",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#DDE2E7", "font.size": 11, "legend.frameon": False,
})

data = json.load(open(IN_PATH))
results = data["results"]
rounds = data["rounds"]
n_nodes = data["n_nodes"]
n = len(results[0]["moga"]["trajectory"]["alive"])
sample_every = rounds // n
rounds_axis = np.arange(n) * sample_every
n_seeds = len(results)

moga_alive = np.array([r["moga"]["trajectory"]["alive"] for r in results])
hybrid_alive = np.array([r["moga_graphsage"]["trajectory"]["alive"] for r in results])

a = np.array([r["moga"]["final_alive"] for r in results])
b = np.array([r["moga_graphsage"]["final_alive"] for r in results])
if np.all(a == b):
    p = 1.0
else:
    _, p = wilcoxon(a, b)
wins = int(np.sum(b > a))
ties = int(np.sum(b == a))

deaths_moga = [r["moga"]["first_death_round"] for r in results]
deaths_hybrid = [r["moga_graphsage"]["first_death_round"] for r in results]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: alive-node count over time, mean ± std band
ax = axes[0]
for arr, color, label in [(moga_alive, C_MOGA, "MOGA + LSA (stock)"),
                            (hybrid_alive, C_HYBRID, "MOGA + GraphSAGE + LSA")]:
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    ax.plot(rounds_axis, mean, color=color, linewidth=2.2, label=label)
    ax.fill_between(rounds_axis, mean - std, mean + std, color=color, alpha=0.15)
ax.set_xlabel("round"); ax.set_ylabel(f"alive nodes (of {n_nodes})")
ax.set_title(f"Network lifetime over time\n(mean ± 1σ, {n_seeds} seeds, WSN dataset)")
ax.legend(fontsize=8, loc="lower left")

# Panel 2: first node death per seed, both methods
ax = axes[1]
rng = np.random.default_rng(0)
for i, (vals, color, label) in enumerate([(deaths_moga, C_MOGA, "MOGA + LSA"),
                                            (deaths_hybrid, C_HYBRID, "MOGA + GraphSAGE")]):
    no_death = sum(1 for v in vals if v < 0)
    y = [v for v in vals if v >= 0]
    jitter = rng.uniform(-0.08, 0.08, size=len(y))
    ax.scatter([i] * len(y) + jitter, y, color=color, s=70, zorder=3,
               edgecolor="#FBFAF7", linewidth=0.8,
               label=f"{label}  ({no_death}/{n_seeds} seeds: no death)")
ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["MOGA\n+LSA", "MOGA+\nGraphSAGE"])
ax.set_ylabel("round of first node death")
ax.set_title("First node death, by seed")
ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.12))

# Panel 3: the number that matters
ax = axes[2]
means = [a.mean(), b.mean()]
stds = [a.std(), b.std()]
ax.bar(["MOGA\n+LSA", "MOGA+\nGraphSAGE"], means, yerr=stds, capsize=8,
       color=[C_MOGA, C_HYBRID], error_kw={"ecolor": "#12181F", "linewidth": 1.4}, width=0.55)
ax.set_ylabel(f"final alive nodes, round {rounds}")
sig = "p < 0.01" if p < 0.01 else f"p = {p:.3f}"
winner = "GraphSAGE hybrid" if b.mean() > a.mean() else ("stock MOGA" if a.mean() > b.mean() else "tie")
ax.set_title(f"{winner} wins on final lifetime\n({sig}, Wilcoxon, {wins} wins / {ties} ties / {n_seeds} seeds)")
pct = b.mean() - a.mean()
if abs(pct) > 1e-9:
    target = 1 if pct > 0 else 0
    ax.annotate(f"{pct:+.1f} nodes kept alive", xy=(target, means[target]),
                xytext=(target, means[target] + max(stds) + 1), ha="center", fontsize=10,
                color=C_GOOD, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_GOOD, lw=1))

fig.suptitle(f"On the WSN dataset, {winner} beats the other on network lifetime",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("WSN_Dataset_verdict.png", dpi=140, bbox_inches="tight")
print("saved WSN_Dataset_verdict.png")
print(f"final_alive: moga={a.mean():.2f}+/-{a.std():.2f}  hybrid={b.mean():.2f}+/-{b.std():.2f}  "
      f"p={p:.4f}  hybrid wins {wins}/{n_seeds}, ties {ties}/{n_seeds}")
