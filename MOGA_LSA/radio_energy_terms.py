"""Render the first-order radio energy model formulas with term-by-term definitions,
same visual style as moga_fitness_terms.png."""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.0))
ax.axis("off")

ax.text(0.5, 0.97,
        r"$E_{TX}(k,d) = k \cdot E_{elec} + k \cdot E_{fs} \cdot d^2$   (if $d \leq d_0$)",
        ha="center", va="top", fontsize=14, transform=ax.transAxes)

ax.text(0.5, 0.83,
        r"$E_{TX}(k,d) = k \cdot E_{elec} + k \cdot E_{mp} \cdot d^4$   (if $d > d_0$)",
        ha="center", va="top", fontsize=14, transform=ax.transAxes)

ax.text(0.5, 0.69,
        r"$E_{RX}(k) = k \cdot E_{elec}$",
        ha="center", va="top", fontsize=14, transform=ax.transAxes)

terms = [
    (r"$k$", "Number of bits in the message (512 bits sensor data, 1024 bits CH/relay message)."),
    (r"$d$", "Distance in metres between sender and receiver."),
    (r"$d_0$", "Threshold distance (≈ 87.7 m) that decides which amplifier cost applies."),
    (r"$E_{elec}$", "Base radio electronics cost, 50 nJ/bit — paid on both TX and RX."),
    (r"$E_{fs}$", "Free-space amplifier cost, 10 pJ/bit/m² — used when $d \\leq d_0$."),
    (r"$E_{mp}$", "Multi-path amplifier cost, 0.0013 pJ/bit/m⁴ — used when $d > d_0$."),
    (r"$E_{TX}, E_{RX}$", "Energy deducted for sending / receiving $k$ bits."),
]

y = 0.56
for sym, desc in terms:
    ax.text(0.06, y, sym, ha="left", va="top", fontsize=13, transform=ax.transAxes)
    ax.text(0.28, y, desc, ha="left", va="top", fontsize=11.5, wrap=True, transform=ax.transAxes)
    y -= 0.09

fig.tight_layout()
fig.savefig("radio_energy_terms.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
print("saved radio_energy_terms.png")
