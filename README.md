# flywire-mb-gnn-topology-bias

Preregistered test of whether the **real Drosophila mushroom-body connectome topology** confers an inductive bias for **associative learning** in graph neural networks, compared with degree-preserving null networks.

## Design at a glance

| Component | Choice |
|---|---|
| Data | FlyWire FAFB whole-brain connectome, snapshot **v783** (Codex downloads) |
| Substrate | Mushroom body (MB) subnetwork: KCs, PNs, MBONs, DANs (~2.2K KCs) |
| Task | Simulated olfactory associative conditioning (odor-shock protocol) |
| Models | GCN, GAT (fixed hyperparameter budget, 30 seeds) |
| Nulls | Degree-preserving directed rewiring, with and without E/I sign preservation |
| Primary metric | Trials-to-criterion (preregistered) |
| Secondary | Learning-curve AUC, noise robustness |

## Why this is falsifiable and publishable either way

- Real topology beats nulls -> structural explanation of biological learning.
- No advantage -> topology alone is insufficient; learning lives in dynamics/plasticity.

Both outcomes are reportable claims with a preregistered analysis plan.

## Repo layout

```
PREREGISTRATION.md   # frozen hypotheses + analysis plan (register on OSF)
EXPERIMENT_SPEC.md   # technical pipeline v1
DECISIONS.md         # timestamped log of semantic decisions/deviations
src/                 # pipeline code (agent-built)
configs/             # one yaml per run
data/                # gitignored
results/             # parquet + run manifests
```

## Roadmap (10 weeks)

- W1: env + data download + graph build + sanity checks
- W2: task engine + one end-to-end GNN run (feasibility gate)
- W3-5: full runs (real vs nulls, both architectures, 30 seeds each)
- W5-6: signal gate (difference or clean null)
- W6-8: robustness (noise), figures
- W9-10: write preprint + OSF registration link + submission

## Targets

preprint: bioRxiv -> then Network Neuroscience / eNeuro / Discover Neuroscience (ladder).

## Data sources

- Codex downloads: https://codex.flywire.ai (account page -> API token), dataset=fafb, snapshot v783 (Oct 2023)
- Cell typing: Schlegel et al. 2024 (Nature)
- MB reference: Li et al. 2020 (eLife) - adult MB connectome
