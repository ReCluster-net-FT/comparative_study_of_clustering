"""Equation-only formula images for the journal draft, Computer Modern to match
the LaTeX-rendered equations already in the document."""
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"

EQS = [
    ("eq_moga_fitness.png",
     r"$F = \min\left(\sum_{i=1}^{M}\sum_{j=1}^{N} D_{ij} \; + \; "
     r"\sum_{i=1}^{M} DS_i \; - \; \sum_{i=1}^{M} E_{res_i}\right)$", 9, 1.3),

    ("eq_radio_tx.png",
     r"$E_{TX}(k,d) = k \cdot E_{elec} + k \cdot E_{fs} \cdot d^2 \;\;\; (d \leq d_0)$"
     "\n"
     r"$E_{TX}(k,d) = k \cdot E_{elec} + k \cdot E_{mp} \cdot d^4 \;\;\; (d > d_0)$"
     "\n"
     r"$E_{RX}(k) = k \cdot E_{elec}$", 9, 2.0),

    ("eq_sage_loss.png",
     r"$\mathcal{L} = -\log\sigma(z_u^{\top} z_v) \; - \; \log\sigma(-z_u^{\top} z_n)$", 9, 1.1),

    ("eq_embed_substitution.png",
     r"$D_{ij}^{*} = \| h_i - h_j \|_2 , \quad h \in \mathbb{R}^{16}$", 9, 1.1),
]

for fname, eq, w, h in EQS:
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    ax.text(0.5, 0.5, eq, ha="center", va="center", fontsize=19,
            transform=ax.transAxes, linespacing=2.0)
    fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.25,
                facecolor="white")
    plt.close(fig)
    print("saved", fname)
