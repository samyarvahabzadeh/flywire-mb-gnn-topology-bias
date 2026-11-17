# Experiment spec v1

## Environment
- molab (marimo) or local. Python >= 3.11. torch, torch_geometric, polars, pyarrow, scipy, networkx (analysis only), matplotlib.
- GPU optional (molab RTX Pro 6000 / 96GB). Scripts MUST be resumable from checkpoints (sessions cap at ~12h).

## Pipeline
1. DATA: get Codex API token (account page). Download dataset=fafb: connections product + consolidated_cell_types. Convert CSV.gz -> parquet (polars). Record sha256 in data/MANIFEST.json.
2. GRAPH: build directed weighted graph (weight = synapse count) as scipy.sparse CSR + torch_geometric Data. Node attrs: cell_type, predicted neurotransmitter -> edge sign (+/-), super class if available.
3. MB SUBSET: select cell types KC, PN (olfactory projection neurons), MBON, DAN (and their subtypes) via type labels; induce subgraph; keep threshold=5 version as primary; verify bilateral KC count in [4400, 5500] and report PN->KC convergence. The original ~2.2K anchor referred to a single hemisphere; bilateral scope and the revised range were explicitly approved by the researcher on 2026-11-17 (DECISIONS W1-007).
4. TASK ENGINE v1: eight frozen sparse KC odor patterns, 5% active KCs, four CS+/CS- pairs, seeded trial order and max 2000 trials. Per the researcher's explicit simplification, US is a global scalar dopamine signal d=1 (otherwise 0), without an explicitly simulated DAN intermediary. KC->MBON plasticity is instantaneous anti-Hebbian depression, delta_w=-0.05*pre*post*d, unit weights initialized uniform[0.8,1.0] and clipped [0,1]. MBON avoidance-minus-approach readout and criterion remain as specified in configs/task_engine.yaml. Seeds 0..29 and odor sets are frozen in configs/. Unit-test with a synthetic 3-odor case. Biological learning and comparative runs are gated on OSF registration. See DECISIONS W1-010 and PROPOSED W1-P06..P09; reconcile the DRAFT's pathway/exclusion wording before registered runs.
5. MODELS: GCN and GAT on the fixed MB adjacency; node features: cell-type one-hot + input drive; trained end-to-end (gradient) to produce valence readout. Parameter budget matched across graphs.
6. NULLS: rewiring with preserved in/out degree sequences; variant B additionally preserves E/I sign per edge. Validate degree sequences after swap (assert equality).
7. RUNS: 30 seeds x {real, null-A x30, null-B x30} x {GCN, GAT}. One yaml config per run -> results/<run_id>.parquet + manifest JSON (git commit, config hash, seed).
8. ANALYSIS: per PREREGISTRATION.md only. Exploratory notebooks must be named exploratory_*.

## Sanity anchors (hard checks; fail loudly)
- Whole-brain neuron count ~139k (v783);
- bilateral KC count in [4400, 5500];
- Graph is sparse; NEVER densify (140k x 140k dense ~ 78GB).

## Definition of done (week-1)
parquet data + sparse graph builder + MB subset with passing sanity checks + toy task run. No metrics beyond sanity checks are inspected before OSF registration.
