"""Render the MOGA fitness equation (paper Eq. 1) with term-by-term definitions."""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axis("off")

ax.text(0.5, 0.93,
        r"$Fitness = \min\left(\sum_{i=1}^{M}\sum_{j=1}^{N} D_{ij} \;+\; \sum_{i=1}^{M} DS_i \;-\; \sum_{i=1}^{M} E_{res_i}\right)$",
        ha="center", va="top", fontsize=17, transform=ax.transAxes)

terms = [
    (r"$D_{ij}$", "Distance from source sensor node $i$ to next hop $j$ — summed over "
                   "all clusters/nodes, gives total intra-cluster distance (member-to-CH)."),
    (r"$DS_i$", "Estimated distance of cluster $i$ to the sink (CH-to-base-station distance)."),
    (r"$E_{res_i}$", "Residual energy of the cluster head in cluster $i$ — "
                      "subtracted, since more remaining energy is better (maximized)."),
    (r"$M,\ N$", "Number of clusters ($M$) and number of sensor nodes ($N$)."),
]

y = 0.62
for sym, desc in terms:
    ax.text(0.06, y, sym, ha="left", va="top", fontsize=14, transform=ax.transAxes)
    ax.text(0.22, y, desc, ha="left", va="top", fontsize=12.5, wrap=True, transform=ax.transAxes)
    y -= 0.13

fig.tight_layout()
fig.savefig("moga_fitness_terms.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
print("saved moga_fitness_terms.png")
