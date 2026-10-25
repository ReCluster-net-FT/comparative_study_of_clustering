# NCOGA with GraphSAGE 

An ns-3 simulation of **NCOGA** (a genetic-algorithm-based cluster head selection protocol for heterogeneous wireless sensor networks, after Sahoo, Pandey and Amgoth), extended with a learned **utility score** that biases the GA toward better cluster head sets. The score comes from either a frozen **GraphSAGE** model or a frozen **MLP** baseline, both trained on data collected from the simulator itself.

The repository contains three parts:

1. **Simulator** (`scratch/ncoga.cc`): the NCOGA round-based energy simulation, data export, and inference client.
2. **Training / experiment pipeline** (`train_gs.py`): generates data, trains the models, validates, runs the test seeds, and computes statistics.
3. **Inference server** (`infer_graphsage.py`): a persistent Python process that the simulator queries over a local TCP socket.

A helper module, `graph_features.py`, builds the node features and is shared by training and inference.

---

## How it works

### 1. NCOGA simulation (C++)

- 100 nodes in a 100 m x 100 m area, sink at the centre.
- Three node types with different initial energy: normal (0.5 J, 70%), advanced (2.0 J, 20%), super (1.0 J, 10%).
- First-order radio energy model (`E_ELEC`, `E_FS`, `E_MP`, `E_DA`, 2000-bit packets).
- Each round, a GA (population 30, 30 generations, `PM = 0.031`) selects cluster heads using six fitness parameters FP1 to FP6, each weighted 1/6. FP2, FP3, FP4 and FP6 use modified definitions documented at the top of the `.cc` file.
- Reactive (TEEN-like) transmission: members report only when the sensed value crosses a hard threshold (0.5) or changes by more than a soft threshold (0.01).
- Up to `MAX_ROUNDS = 30000` rounds. Reported metrics: **FND** (first node dead), **HND** (half nodes dead), **LND** (last node dead), total throughput, final energy, average CH count.

### 2. Learned utility (GraphSAGE or MLP)

Each round, the simulator exports the alive-node graph (edges join nodes within 30 m) and per-node features. A frozen model predicts a **utility score** per node. During GA evaluation, a chromosome's fitness becomes:

```
fitness = (F - fitness_mean) / fitness_std  -  graphSageWeight * (mean_utility_of_CHs - utility_mean) / utility_std
```

FP1 to FP6 are unchanged. The learned score is only an additional preference term, scaled by `--graphSageWeight` (default in the experiment: 0.25).

The **utility label** used for training is a weighted combination of four normalised outcome columns measured over a forward horizon of `graphSageK` rounds (default 250), available for nodes that served as cluster heads:

```
utility = 0.25*U_E + 0.25*U_C + 0.25*U_CE - 0.25*C_relay
```

### 3. Models

| Model | Class | Architecture |
|---|---|---|
| GraphSAGE | `GraphSAGE` | 2 x `SAGEConv` (hidden 32, embedding 16) followed by a linear regressor |
| MLP | `UtilityMLP` | Linear(in, 32), ReLU, Linear(32, 16), ReLU, Linear(16, 1) |

Both are trained as regressors for the utility value. The MLP sees node features only; GraphSAGE additionally uses the neighbourhood graph.

### 4. Simulator and inference server communication

When `--enableGraphSAGE=true` or `--enableMLP=true`, the simulator launches the inference server itself:

```
<graphSagePython> infer_graphsage.py --server --port <39000 + seed % 1000> --mode <mlp|graphsage> --model <model.pt>
```

- The port is derived from the seed (seed 13 uses port 39013), so concurrent runs with different seeds do not collide.
- Protocol: the simulator writes `graphsage_live_nodes.csv` and `graphsage_live_edges.csv`, then sends one line `nodes<TAB>edges<TAB>seed<TAB>round`. The server replies with `node_id,score` lines ending in `END`. Sending `QUIT` shuts the server down.
- The server accepts **one** client connection per launch, so each ns-3 run starts its own server.
- The simulator discards the server's stdout and stderr (`>/dev/null 2>&1`), so startup failures are silent. See [Troubleshooting](#troubleshooting).

---

## Requirements

- Ubuntu (tested in a VirtualBox VM) with a working ns-3-dev build
- Python 3.12 virtual environment (`myEnv`) with:
  - `torch`
  - `torch_geometric`
  - `numpy`, `pandas`, `scipy`
  - `pyarrow` (optional; speeds up CSV reading in the inference server)
- The memory-optimised training script reads CSVs with narrow dtypes, but combining many seeds is still heavy; give the VM several GB of RAM.

---

## Usage

### Build the simulator

Copy the simulator source to `ns-3-dev/scratch/ncoga.cc`, then:

```bash
cd ~/ns-3-dev
source myEnv/bin/activate
./ns3 build
```

### Run a single baseline simulation

```bash
./ns3 run "scratch/ncoga -- --seed=1 --outputPrefix=/tmp/baseline_1"
```

### Run the full experiment

```bash
cd ~/ns-3-dev/GSTrain
INFER_PROFILE_LOG=$HOME /ns-3-dev/GSTrain/infer_profile.log \
python train_gs.py --experiment --capacity-k 20 \
    --feature-cache-dir ./feature_cache
```

The experiment runs these phases:

| Phase | Seeds | What happens |
|---|---|---|
| Training data | 1 to 12 | ns-3 runs with data export enabled, producing `training_seed_*` CSVs, then combined |
| Training | 1 to 12 | Train GraphSAGE and MLP, saving `graphsage_model.pt`, `mlp_model.pt`, `feature_spec.json` |
| Validation | 13 to 16 | Runs baseline, MLP and GraphSAGE; a **validation gate** checks regression quality (`validation_gate_metrics.csv`). Test seeds only run if the gate passes |
| Test | 17 to 36 | Runs the conditions selected by `--test-conditions` |

Resume a partly finished run (reuses existing CSVs and models):

```bash
python train_gs.py --experiment --capacity-k 20 \
    --skip-seed-generation --skip-training --resume \
    --feature-cache-dir ./feature_cache \
    --test-conditions graphsage
```

### Training script options

| Option | Default | Description |
|---|---|---|
| `--experiment` | off | Run the full pipeline. Without it, only trains a model from the given CSVs |
| `--skip-seed-generation` | off | Reuse existing `training_seed_1..12` CSVs |
| `--skip-training` | off | Reuse saved models and combined training CSVs |
| `--resume` | off | Skip ns-3 runs whose output files already exist |
| `--test-conditions` | all three | Any of `baseline`, `mlp`, `graphsage` |
| `--feature-set` | `slide` | `slide`, `legacy` or `both` |
| `--capacity-k` | none | Required for `slide` and `both`; must be finite and greater than 0 |
| `--feature-cache-dir` | none | Cache computed graph features as `.npz` files |
| `--epochs` | 100 | Training epochs |
| `--hidden-dim` / `--embedding-dim` | 32 / 16 | GraphSAGE sizes |
| `--graphsage-k` | 250 | Forward utility horizon in rounds |
| `--graphsage-weight` | 0.25 | Weight of the learned term in GA fitness |
| `--exploration-fraction` | 0.15 | Exploration used when collecting training data |
| `--fitness-mean/-std`, `--utility-mean/-std` | 0 / 1 | Normalisation constants (the experiment passes frozen values computed from training data) |
| `--seed` | 12345 | Seed for model training |
| `--selftest` | off | Run the `graph_features` self-test and exit |

### Simulator options

| Option | Description |
|---|---|
| `--seed` | Deterministic simulation seed |
| `--enableGraphSAGE` / `--enableMLP` | Use the frozen GraphSAGE or MLP score in the GA |
| `--enableGraphSAGEData` / `--graphSageDataMode` | Export training samples |
| `--graphSageModel`, `--mlpModel` | Checkpoint paths |
| `--graphSageInferenceScript`, `--graphSagePython` | Inference script and Python interpreter |
| `--graphSageK` | Forward utility horizon (rounds) |
| `--explorationFraction` | Training-only exploration fraction |
| `--graphSageWeight` | Hybrid fitness weight (lambda) |
| `--graphFitnessMean/Std`, `--graphUtilityMean/Std` | Frozen normalisation constants |
| `--outputPrefix` | Prefix for all output files of the run |

### Running the inference server by hand

Useful for debugging a checkpoint:

```bash
cd ~/ns-3-dev/GSTrain
python infer_graphsage.py --server --port 39013 --mode mlp --model mlp_model.pt
```

If it prints nothing and keeps running, it loaded successfully and is waiting for a client. `--mode` must match the checkpoint (`mlp` with `mlp_model.pt`, `graphsage` with `graphsage_model.pt`). Set `INFER_THREADS` to change the PyTorch thread count (default 1) and `INFER_PROFILE_LOG` to log per-stage timing every 200 calls.

---

## Outputs

Per run, with `--outputPrefix=<prefix>`:

| File | Content |
|---|---|
| `<prefix>_ncoga_results.csv` | Per-round alive and dead nodes, energy, cumulative throughput, CH count |
| `<prefix>_alive_nodes_vs_rounds.csv` | `round,alive_nodes` series for plotting |
| `<prefix>_summary.csv` | `seed,fnd,hnd,lnd,total_throughput,final_energy,avg_ch,num_rounds` |
| `<prefix>_graphsage_training_*.csv` | Training samples, graph state and edges (data-collection runs only) |

Experiment-level files in `GSTrain3/`:

| File | Content |
|---|---|
| `validation_summary.csv` | Validation-seed results per condition |
| `validation_gate_metrics.csv` | Regression metrics used for the gate |
| `experiment_summary.csv` | Per-seed test results |
| `experiment_averages.csv` | Averages per condition |
| `statistical_results.json` | Paired test results between conditions |
| `feature_spec.json` | Feature configuration saved with the models |

The experiment prints FND, HND, LND and total throughput after every run. Statistical comparisons use paired tests over matching seeds.

---

## Checkpoint requirements

`infer_graphsage.py` refuses to load a checkpoint unless it contains:

- `feature_set`, `feature_columns` (consistent with `feature_set`), and `input_dim` equal to the number of feature columns
- `feature_spec` containing `capacity_k`
- `model_state_dict`, `feature_mean`, `feature_std`, and for GraphSAGE `hidden_dim` and `embedding_dim`

Checkpoints from older training scripts lack this metadata and must be retrained.

---

## Troubleshooting

**The experiment appears to hang at "running: ns3 ..."**
The simulator starts the inference server in the background with its output discarded, so a crashing server looks like a hang (the client retries the connection for about 5 seconds, then continues without scores). Run the server by hand (see above) to see the real error. Check `pgrep -af -- "--server --port"` while a run is active: a PID that keeps changing means the server is crashing at startup.

Importing the training script must not start training; its work runs only under `if __name__ == "__main__":`.

**`RuntimeError: Error(s) in loading state_dict`**
The `--mode` does not match the checkpoint. `network.*` keys belong to the MLP; `conv1.*`, `conv2.*` and `regressor.*` belong to GraphSAGE.

**"retrain required" errors**
The checkpoint was saved without the metadata listed above. Retrain with `train_gs.py`.

**No `infer_profile.log` appears**
Use an absolute path for `INFER_PROFILE_LOG`. The server's working directory may differ from your shell's, and the variable must survive being passed through the ns-3 wrapper. A line is only written after every 200 inference calls.

**The graph-state CSV header/row mismatch error**
Append `,F` to the `g_graphStateCsv` header string in the C++ file and regenerate the data.

---

## Notes and limitations


- Each inference call re-reads and filters the exported CSVs, so inference cost grows with network size and is the main runtime bottleneck.
- Because the learned term only reweights the GA fitness, its effect on lifetime and throughput depends on `--graphSageWeight` and on how well the utility label reflects delivery as well as lifetime. Evaluate throughput alongside FND, HND and LND, and compare across all test seeds with the paired statistics rather than a single seed.

