# ESCA vs IGHND 

## Overview

This project evaluates GraphSAGE-enhanced versions of two Wireless
Sensor Network (WSN) protocols:

-   **ESCA - GraphSAGE**: integrates GraphSAGE into ESCA's cluster-head
    selection.
-   **IGHND - GraphSAGE**: integrates GraphSAGE into IGHND's zone-head
    selection.

The aim is to investigate whether graph-based node representations can
influence head selection and improve network-lifetime and performance
metrics under two base-station placements: **Center-BS** and
**Corner-BS**.


## Project Structure

```
ESCA_IGHND/
├── README.md                  ← this file
│
├── common/                    ← shared code used by all protocols
│   ├── config.py              ← all settings (energy values, grid size, weights, seeds, GraphSAGE settings)
│   ├── dataset_loader.py      ← reads and checks the CSV, picks 300 nodes, scales features
│   ├── network.py             ← the Network object: node positions, energy, alive/dead, BS position
│   ├── energy_model.py        ← First-Order Radio Model (send / receive / aggregate energy)
│   ├── packet_model.py        ← packet delivery and loss (uses the dataset's loss rate)
│   └── metrics.py             ← records FND/HND/LND, energy, throughput, fairness every round
│
├── esca/
│   └── protocol.py            ← ESCA: Elbow method + K-Means + highest-energy CH
│
├── esca_gs/
│   └── esca_graphsage.py      ← ESCA + GraphSAGE leader selection
│
├── ighnd/
│   └── protocol.py            ← IGHND: grid zones, merge/split, weighted ZH score
│
├── ighnd_gs/
│   └── protocol.py            ← IGHND + GraphSAGE leader selection
│
├── experiments/
│   ├── run_experiments.py     ← runs all 4 protocols × 2 scenarios × 5 seeds, saves results/raw_results.json
│   └── analyze_results.py     ← makes tables (CSV, Excel), statistical tests and all plots
│
└── plots/                     ← result graphs
    ├── fnd_hnd_lnd_DATASET-CenterBS.png
    ├── fnd_hnd_lnd_DATASET-CornerBS.png
    ├── alive_vs_rounds.png
    ├── avg_residual_energy_comparison.png
    ├── total_energy_consumed_comparison.png
    ├── throughput_packets_per_round_comparison.png
    └── ch_fairness_cv_comparison.png
```


## How to Run

**Requirements:** Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`, `scipy` 

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl scipy
```

1. Put the dataset at `data/converted_dataset.csv` (the path is set in `common/config.py`).
2. Run the experiments from the project folder:
   ```bash
   python experiments/run_experiments.py                 # both scenarios
   python experiments/run_experiments.py DATASET-CenterBS # only one scenario
   ```
   Results are saved after every seed. If you stop and restart, finished runs are skipped.
3. Make the tables and plots:
   ```bash
   python experiments/analyze_results.py
   ```
   Output goes to `results/`: `summary_table.csv`, `statistical_comparison.csv`, `WSN_results_summary.xlsx`, and `results/plots/`.

To change settings (number of nodes, grid size, IGHND weights, GraphSAGE blend, seeds), edit `common/config.py`.


## Results

ESCA + GraphSAGE achieves approximately 1.6% improvement in LND compared with standard ESCA across both center and corner base-station scenarios, indicating a modest improvement in overall network lifetime.
