"""Diagrams for the MOGA+LSA pipeline explainer handed to the team."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#12181F"
TEAL = "#1B6E77"
PURPLE = "#7A4FB5"
GREY = "#57626C"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=9.5, tc=INK, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, linewidth=1.3, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------- Figure 1: the per-round pipeline ----------
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

y, h = 1.75, 0.95
boxes = [
    (0.15, 1.75, "Sensor network\n100 nodes, real\npositions + energy"),
    (2.15, 1.75, "MOGA\npicks ~10\ncluster heads"),
    (4.15, 1.75, "LSA\npicks routes\nCH → sink"),
    (6.15, 1.75, "Radio model\ndeducts energy\nfor every TX/RX"),
    (8.15, 1.75, "Log metrics\nalive, energy,\nthroughput"),
]
for x, w, label in boxes:
    fc = "#EAF2F3" if "MOGA" in label else "white"
    box(ax, x, y, w, h, label, fc=fc)

for x in (1.95, 3.95, 5.95, 7.95):
    arrow(ax, x, y + h / 2, x + 0.18, y + h / 2)

# loop-back
arrow(ax, 9.0, y, 1.05, y, color=TEAL, rad=-0.22)
ax.text(5.0, 0.38, "repeat every round until nodes run out of energy",
        ha="center", va="center", fontsize=9.5, color=TEAL, style="italic")

ax.text(0.15, 3.15, "One simulation round", fontsize=12, color=INK, fontweight="bold")
fig.savefig("fig_pipeline.png", dpi=220, bbox_inches="tight",
            pad_inches=0.18, facecolor="white")
plt.close(fig)
print("saved fig_pipeline.png")


# ---------- Figure 2: inside MOGA, one round ----------
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")

box(ax, 0.15, 2.5, 2.0, 1.0,
    "Start\n20 random teams\n(10 CHs each)", fc="#F2F0F8")
box(ax, 2.75, 2.5, 2.0, 1.0, "Score each team\nwith the fitness\nformula")
box(ax, 5.35, 2.5, 2.0, 1.0, "Keep the best 10\nthrow away\nthe worst 10")
box(ax, 7.95, 2.5, 1.9, 1.0, "Breed + mutate\nback up to 20")

arrow(ax, 2.15, 3.0, 2.73, 3.0)
arrow(ax, 4.75, 3.0, 5.33, 3.0)
arrow(ax, 7.35, 3.0, 7.93, 3.0)

# loop back round the bottom
arrow(ax, 8.9, 2.5, 3.75, 2.5, color=PURPLE, rad=-0.3)
ax.text(6.3, 1.32, "repeat 10 times (10 generations)", ha="center",
        fontsize=9.5, color=PURPLE, style="italic")

box(ax, 0.15, 0.45, 2.0, 0.85, "Best team wins\n= this round's\ncluster heads",
    fc="#EAF2F3", ec=TEAL, tc=TEAL)
arrow(ax, 1.15, 2.48, 1.15, 1.33, color=TEAL)
ax.text(1.32, 1.9, "after gen 10", fontsize=8.5, color=TEAL, ha="left")

ax.text(0.15, 3.95, "Inside MOGA: one round of cluster-head selection",
        fontsize=12, color=INK, fontweight="bold")
fig.savefig("fig_moga_loop.png", dpi=220, bbox_inches="tight",
            pad_inches=0.18, facecolor="white")
plt.close(fig)
print("saved fig_moga_loop.png")


# ---------- Figure 3: where GraphSAGE plugs in ----------
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")

box(ax, 0.15, 1.9, 2.3, 0.9, "Node features\n14 numbers\nper node", fs=9)
box(ax, 2.95, 1.9, 2.3, 0.9, "GraphSAGE\nencoder\n(trained once)", fs=9,
    fc="#F2F0F8", ec=PURPLE, tc=PURPLE)
box(ax, 5.75, 1.9, 2.3, 0.9, "16-number\n“fingerprint”\nper node", fs=9)

arrow(ax, 2.45, 2.35, 2.93, 2.35)
arrow(ax, 5.25, 2.35, 5.73, 2.35)

ax.text(0.15, 1.25,
        "Only one thing changes in MOGA: the member-to-CH distance in the fitness\n"
        "formula is measured between fingerprints instead of between map positions.\n"
        "The GA, the LSA routing and the energy model are untouched.",
        fontsize=9.5, color=INK, va="top", linespacing=1.6)

ax.text(0.15, 3.0, "The GraphSAGE variant", fontsize=12, color=INK,
        fontweight="bold")
fig.savefig("fig_graphsage.png", dpi=220, bbox_inches="tight",
            pad_inches=0.18, facecolor="white")
plt.close(fig)
print("saved fig_graphsage.png")
