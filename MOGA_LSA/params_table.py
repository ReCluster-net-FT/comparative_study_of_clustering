"""Parameters table (MOGA + radio model + LSA), matching the visual style
of the reference NCOGA parameters table: two columns, bold italic header,
italic body, simple black grid.
"""
import matplotlib.pyplot as plt

ROWS = [
    ("Size of Area", "100 m × 100 m (data-derived, WSN dataset coordinates)"),
    ("Nodes (N)", "100"),
    ("Sink", "1 (fixed base station)"),
    ("Initial energy (E₀) of nodes", "0.5 Joule (mean)"),
    ("Node heterogeneity", "Real per-node energy from dataset (continuous, not tiered)"),
    ("Essential transceiver energy (E$_{elec}$)", "50 nJ/bit"),
    ("Threshold distance (d₀)", "≈ 87.7 m"),
    ("Size of sensed-data packet", "512 bits"),
    ("Size of CH→CH / CH→sink message", "1024 bits"),
    ("Small-distance energy (E$_{fs}$), d ≤ d₀", "10 pJ/bit/m²"),
    ("Large-distance energy (E$_{mp}$), d > d₀", "0.0013 pJ/bit/m⁴"),
    ("Data aggregation energy (E$_{da}$)", "5 nJ/bit/signal"),
    ("CH probability (CH$_p$)", "0.10"),
    ("Size of the population (P)", "20"),
    ("Crossover", "Single-point (applied to all children)"),
    ("Mutation rate (P$_m$)", "0.20"),
    ("Selection method", "Elitist truncation (top 50%)"),
    ("Total chromosomes", "20"),
    ("No. of generations", "10"),
    ("LSA population / iterations", "15 / 30"),
    ("Simulation run", "3 seeds × 3,000 rounds + 1 seed × 13,000 rounds"),
]

fig, ax = plt.subplots(figsize=(9, 0.42 * (len(ROWS) + 1)))
ax.axis("off")

table = ax.table(
    cellText=[[k, v] for k, v in ROWS],
    colLabels=["Parameters", "Values"],
    cellLoc="center",
    loc="center",
    colWidths=[0.5, 0.5],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.55)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.8)
    text = cell.get_text()
    text.set_fontstyle("italic")
    if row == 0:
        text.set_fontweight("bold")
        text.set_fontstyle("normal")
        cell.set_facecolor("white")
    else:
        cell.set_facecolor("white")

fig.tight_layout()
fig.savefig("moga_params_table.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
print("saved moga_params_table.png")
