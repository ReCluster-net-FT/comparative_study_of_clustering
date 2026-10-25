"""GraphSAGE training parameters table, same visual style as moga_params_table.png."""
import matplotlib.pyplot as plt

ROWS = [
    ("Encoder architecture", "2-layer SAGEConv (mean-aggregation)"),
    ("Input feature dim", "14 (position x,y + residual energy + 11 real sensor features)"),
    ("Hidden layer dim", "32"),
    ("Output embedding dim", "16"),
    ("Neighbor graph", "k-NN, k = 8 (Euclidean, over alive nodes)"),
    ("Training objective", "Unsupervised link prediction (no labels)"),
    ("Positive samples", "Real k-NN graph edges"),
    ("Negative samples", "Random node pairs (count = |edges|)"),
    ("Loss function", "$-\\log\\sigma(\\mathrm{pos\\_score}) - \\log\\sigma(-\\mathrm{neg\\_score})$"),
    ("Score function", "Dot product of two nodes' embeddings"),
    ("Optimizer", "Adam"),
    ("Learning rate", "0.01"),
    ("Training epochs", "100"),
    ("Training frequency", "Once per run (frozen afterward, reused every round)"),
    ("Framework", "PyTorch Geometric (torch_geometric.nn.SAGEConv)"),
]

fig, ax = plt.subplots(figsize=(9, 0.42 * (len(ROWS) + 1)))
ax.axis("off")

table = ax.table(
    cellText=[[k, v] for k, v in ROWS],
    colLabels=["Parameters", "Values"],
    cellLoc="center",
    loc="center",
    colWidths=[0.42, 0.58],
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

fig.tight_layout()
fig.savefig("graphsage_params_table.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
print("saved graphsage_params_table.png")
