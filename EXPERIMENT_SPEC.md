# Experiment spec v1

## Environment
- molab (marimo) or local. Python >= 3.11. torch, torch_geometric, polars, pyarrow, scipy, networkx (analysis only), matplotlib.
- GPU optional (molab RTX Pro 6000 / 96GB). Scripts MUST be resumable from checkpoints (sessions cap at ~12h).

## Pipeline
1. DATA: get Codex API token (account page). Download dataset=fafb: connections product + consolidated_cell_types. Convert CSV.gz -> parquet (polars). Record sha256 in data/MANIFEST.json.
2. GRAPH: build directed weighted graph (weight = synapse count) as scipy.sparse CSR + torch_geometric Data. Node attrs: cell_type, predicted neurotransmitter -> edge sign (+/-), super class if available.
3. MB SUBSET: select cell types KC, PN (olfactory projection neurons), MBON, DAN (and their subtypes) via type labels; induce subgraph; keep threshold=5 version as primary; verify KC count ~2.2K (sanity vs Li et al. 2020) and known PN->KC convergence.
4. TASK ENGINE: implement odor patterns (sparse random KC drive, n_odors=8 held fixed), US->DAN pathways, three-factor plasticity at KC->MBON synapses (baseline biological readout), and MBON valence readout. Unit-test with toy 3-odor case.
5. MODELS: GCN and GAT on the fixed MB adjacency; node features: cell-type one-hot + input drive; trained end-to-end (gradient) to produce valence readout. Parameter budget matched across graphs.
6. NULLS: rewiring with preserved in/out degree sequences; variant B additionally preserves E/I sign per edge. Validate degree sequences after swap (assert equality).
7. RUNS: 30 seeds x {real, null-A x30, null-B x30} x {GCN, GAT}. One yaml config per run -> results/<run_id>.parquet + manifest JSON (git commit, config hash, seed).
8. ANALYSIS: per PREREGISTRATION.md only. Exploratory notebooks must be named exploratory_*.

## Sanity anchors (hard checks; fail loudly)
- Whole-brain neuron count ~139k (v783);
- KC count in MB subset ~2.2k;
- Graph is sparse; NEVER densify (140k x 140k dense ~ 78GB).

## Definition of done (week-1)
parquet data + sparse graph builder + MB subset with passing sanity checks + toy task run. No metrics beyond sanity checks are inspected before OSF registration.
