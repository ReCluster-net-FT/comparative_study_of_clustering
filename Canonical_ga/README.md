# Canonical GA

## Overview
This project implements a canonical Genetic Algorithm for cluster head selection in Wireless Sensor Networks and evaluates the impact of integrating GraphSAGE for graph-based structural learning.

## Methodology
- **Network:** 200 sensor nodes deployed in a 100 × 100 m area.
- **Cluster Heads:** Six cluster heads selected per configuration.
- **Genetic Algorithm:** Uses selection, crossover, mutation, and a multi-objective fitness function.
- **GraphSAGE:** Learns node embeddings from the communication graph using unsupervised link prediction.
- **Comparison:** Baseline GA vs. GraphSAGE-enhanced GA under identical simulation settings.

## Results

| Metric | Canonical GA | GraphSAGE + GA |
|---|---:|---:|
| First Node Death (FND) | 1,931 rounds | 2,172 rounds |
| Final Residual Energy | 355.63 J | 359.05 J |

The GraphSAGE-enhanced GA achieved a **12.48% improvement in First Node Death (FND)** and higher final residual energy.

## Technologies
- Python
- PyTorch
- PyTorch Geometric
- Genetic Algorithm
- GraphSAGE
- Wireless Sensor Network Simulation

## Key Takeaway
Integrating GraphSAGE provides additional structural information about the communication network, improving early network stability and energy retention in the evaluated canonical GA configuration.
