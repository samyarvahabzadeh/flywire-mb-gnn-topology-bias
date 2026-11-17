# Overnight status — 2026-11-18 (Asia/Tehran)

## Completed

- Frozen fafb/v783 sources downloaded and converted to parquet; full CSR/PyG graph built.
- Human-approved bilateral MB subset: KC 5177, PN 685, MBON 96, DAN 277;
  6235 nodes, 89769 directed edges at pair-total synapse threshold 5. Hard checks PASS.
- EXPERIMENT_SPEC KC anchor updated to bilateral [4400,5500], per explicit instruction.
- Task v1 parameters locked in configs/task_engine.yaml; seeds 0..29 saved.

## Completed phases

A. COMPLETE: locked v1 task engine, frozen 30-seed/eight-odor sets, MBON readout identities,
resumable weight/trial state and passing three-odor toy execution. No biological learning run.
B. COMPLETE: signed/magnitude-aware GCN/GAT, parameter-budget validation, exact directed
degree-preserving swap chains A/B, RNG/state checkpoints and synthetic invariant tests.
C. COMPLETE: one-command 30-seed runner, per-run YAML and parquet +
git/config/source/graph manifests; strict OSF gate, model/optimizer/RNG checkpoints,
partial-trial safety, immutable plans, session-yield and verified completed-result reuse.
Final plan-only generation complete: 3660 verified YAML files, active bundle
d418a415d07f03d1a019c95b2c952cf424bf327355eb5631bc5af38953d3d876.
Plan: plans/d418a415d07f/plan.parquet. Zero biological model execution and zero real null
graphs generated. Current code checkpoint: e31c469; final handoff docs committed separately.

## PROPOSED decisions

See DECISIONS W1-P06 through P09: odor IDs and balanced schedule, activity normalization,
shared-edge deterministic initialization, NT-based MBON readout surrogate and fixed
valence/discrimination normalization. The user may veto; no actual learning results
have been inspected. W2-P01..P06 additionally document model integration, E/I surrogate,
null mixing budget, the fully crossed run matrix and deterministic/session execution policy.

## Blocks / restrictions

- PREREGISTRATION is DRAFT: real biological learning and comparative runs remain disabled.
- No changes to PREREGISTRATION.md, dense adjacency, new datasets or account actions tonight.
- PN->KC distribution is reported; biological interpretation remains with the researcher.
- E/I and valence surrogate maps are explicitly PROPOSED and subject to weekly veto.

## Validation

Full suite: 38 passed. Real MB hard checks PASS. Task preparation created 30 frozen
eight-odor sets (259 active KCs/odor); PROPOSED readout has 23 avoidance, 68 approach
and 5 unassigned MBONs. Twelve MBONs have no KC inputs under threshold=5; retained
with the predeclared zero-input output rule. Three-odor task mechanics/checkpoint test PASS.
PREREGISTRATION remains unchanged; no biological model learning curves or scores inspected.
Phase B validated synthetic forward/backward, identical seeded initialization, <=5%
architecture budgets, exact nodewise signed degree and bitwise resumed null RNG trajectory.
No real null graph or real-vs-null model run was generated.
Runner tests also verified 3660 unique planned jobs, per-run YAML immutability, orphan
result recovery, result mismatch refusal, and a one-button OSF gate before graph loading.
Registration evidence is hash-bound to the exact task configuration and scientific/code
bundle; changed parameters cannot silently reuse a registration record.

## Handoff

Read reports/OVERNIGHT_REPORT.md for the full structural results and final DAN type table.
Safe next command: `.venv/Scripts/python src/run_experiment.py --plan`.
Actual execution remains intentionally blocked until the researcher completes OSF and
supplies exact registration/config/bundle hash evidence. See RUN_OVERNIGHT.md.
