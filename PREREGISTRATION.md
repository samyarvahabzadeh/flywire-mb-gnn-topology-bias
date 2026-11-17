# Preregistration (template for OSF)

Status: DRAFT v0.1 - register on OSF (osf.io) before first training run. After registration, paste OSF DOI here.
OSF DOI: <TODO after registration>

## 1. Question
Does the real synaptic topology of the Drosophila mushroom body provide a measurable inductive bias for associative learning in GNNs, relative to degree-matched randomized topologies?

## 2. Hypotheses
- H1 (directional): GNNs on the real MB topology reach learning criterion in FEWER trials than on null topologies (matched degree sequences).
- H2 (secondary): the real-vs-null gap GROWS with input noise.
- H0: no difference beyond chance across matched seeds.

## 3. Frozen data
FlyWire FAFB snapshot v783; products: connections and consolidated_cell_types, downloaded via Codex API. Record: download date, file sha256. Primary synapse threshold = 5. Thresholds {1,3,10} = EXPLORATORY only.

## 4. Task
Simulated odor-shock associative conditioning. CS: sparse random activation patterns of KC inputs (fixed set of odors). US: DAN pathway drive. Readout: valence discrimination at MBON outputs. Plasticity: three-factor rule at KC->MBON synapses. Learning criterion: valence discrimination >= 0.8 for 20 consecutive trials.

## 5. Models and budget
Architectures: GCN, GAT. Hidden width 128, depth 2-3 (locked before first run; any change = logged deviation). Identical parameter budget across real and null graphs (+-5%). Seeds: 30 per condition, fixed list in configs/seeds.txt. No tuning on the real graph that is not also applied to nulls.

## 6. Null models
(a) Directed degree-preserving edge swap. (b) Same + excitatory/inhibitory sign preservation. >= 30 instances each; same seed list as condition (3).

## 7. Primary metric and statistics
PRIMARY: trials-to-criterion. Test: Wilcoxon signed-rank across matched seeds (real vs null), effect size rank-biserial r; FDR (Benjamini-Hochberg) across the two architectures. SECONDARY: learning-curve AUC; noise-robustness slope (H2 evaluated with repeated-measures comparison across noise levels {0, 0.1, 0.2, 0.3}).

## 8. Stopping, exclusion, reporting
- A run counts as failed (excluded) only if it does not reach criterion within 10x median null trials; all exclusions reported.
- Every preregistered analysis is reported regardless of outcome.
- Anything outside this plan goes in a section explicitly marked EXPLORATORY.
- Any deviation from this document is logged in DECISIONS.md with timestamp and justification before results are inspected.
