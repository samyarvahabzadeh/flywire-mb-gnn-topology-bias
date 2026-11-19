# Overnight status — 2026-11-19 (Asia/Tehran)

## Authorized final sweep

The researcher released the frozen v1.2 matrix. Google Drive passed the
write/read-back preflight. Before the sandbox was reclaimed, the first Molab
session recorded all six benchmark exclusions, archived all 60 null graphs,
and saved 108/1,830 jobs in 18 verified batches. The rate at the last receipt
was 200.5 jobs/hour versus the prior 163/hour benchmark. Zero retries were
recorded. The second session restored those receipts and saved through
batch `0212`, bringing progress to 1,278/1,830 at 05:19 Tehran time on
2026-11-19. It added 1,170 jobs at 284.27 jobs/hour with zero retries
before the sandbox was reclaimed. After refreshed Drive authorization and
removing the implicit notebook console alias, session 3 recovered through
batch `0212` and saved through `0214`, bringing progress to 1,290/1,830.

## Final binary release pilot

Both final real-only GraphSAGE-max runs met the rolling-mean rule before
1,500. Seed 1337 stopped at trial 968 and seed 2026 at 879. AMENDMENT-02
replaces the historical Ridge method and is publicly frozen under
`v1.2-prereg-amendment`. The six-job engineering batch completed after the
tag, with exact early-stops and verified noise injection. See
`results/final_binary_rate/c3df5b2c7dfc/REPORT.md` for the measured and
full-cap Molab session estimates.

## Phase-4 binary pilot with real PN→KC connections

The six Molab runs completed and their manifests were verified. GraphSAGE-max
passed on both pilot seeds. GCN and GraphSAGE-mean missed the threshold on both.
See W3-S19 through W3-S21 and the local
`results/phase4_pilot/dac573617ca6/REPORT.md` for held-out curves and exact
per-trial KC counts. The main sweep awaits researcher approval.

## Phase-3 held-out pilot

The phase-3 pilot did not qualify a cap or either GraphSAGE aggregator. The
end-to-end GCN cap-finding run did not reach criterion within 5,000 updates.
See W3-S16/S17 and the local `results/phase3_pilot/65106f95c81b/REPORT.md`
for the cue-family design, held-out split audit, and all six exploratory jobs.
The v1.1 Ridge tag is historical. The qualifying gate for AMENDMENT-02 and
post-amendment rate measurements was unmet. The confirmatory sweep awaits
explicit researcher authorization.

## Phase-2 pilot decision

A failed on both pilot seeds for GCN and GraphSAGE mean/max. B passed that
pilot's gate at 20 readout fits following 500 warm-up updates. The researcher
subsequently rejected B for confirmation. C was not triggered. See
`results/phase2_pilot/0a368cfc455e/REPORT.md` and AMENDMENT-01.md. The main
sweep remains on hold.

## Paired-cue pilot

All four real-graph pilots completed for seeds 1337 and 2026 with GCN and
GraphSAGE. None met the unchanged criterion. The prospective amendment and
repeat throughput run were not triggered. See
`results/paired_pilot/ab08efe09953/REPORT.md`. The main sweep remains on hold.

## Seed 1337 diagnostic ladder

The first three predeclared gates passed on Molab. The condition for the fourth
gate was absent. See `results/diagnostic_ladder/fefe5ce31306/REPORT.md` for the
seed-1337 results and downstream readout localization. Engineering benchmarks use
GCN + GraphSAGE. The confirmatory protocol and main sweep hold stay in force.

## Learnability and batch-rate hold

The seed-0 probe diagnosis, single-graph seed-1337 pilot, and one benchmark of
K=8,16,32 all completed on Molab. The pilot selected no amendment. The measured
batch rates do not meet the 24-hour full-cap budget. The main sweep remains on hold
for researcher review. See the local reports linked from STATE.md.

## Current status — W3-P03

The public annotated tag `v1.0-prereg` supersedes the OSF gate described in the
historical status below. The tag, working SHA-256 and Git blob were verified before
all three two-trial GCN smoke jobs. All passed. See REGISTRATION_EVIDENCE.md,
reports/W3_SMOKE.md and RUN_OVERNIGHT.md. The updated plan has 3660 jobs under
`plans/4bfaf9e768c6/`. All 43 engineering tests passed. The full comparison has not run.

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
d26847e31f27f84e5435db36b61c5398957f5455688afea43e461180d54b9e4e.
Plan: plans/d26847e31f27/plan.parquet. Zero biological model execution and zero real null
graphs generated. Current code/review-packager checkpoint: 553b96f, pushed to origin;
generated plan/report metadata and handoff docs are committed separately.

## PROPOSED decisions

See DECISIONS W1-P06 through P09: odor IDs and balanced schedule, activity normalization,
shared-edge deterministic initialization, NT-based MBON readout surrogate and fixed
valence/discrimination normalization. The user may veto; no actual learning results
have been inspected. W2-P01..P06 additionally document model integration, E/I surrogate,
null mixing budget, the fully crossed run matrix and deterministic/session execution policy.

## Blocks / restrictions

- PREREGISTRATION is DRAFT: real biological learning and comparative runs remain disabled.
- No changes to PREREGISTRATION.md, dense adjacency, new datasets or account actions tonight.
- Superseding Git-only authorization (2026-11-18): branch pushed to origin; subsequent
  45-minute work checkpoints require both commit and push. Other account actions remain restricted.
- PN->KC distribution is reported; biological interpretation remains with the researcher.
- E/I and valence surrogate maps are explicitly PROPOSED and subject to weekly veto.

## Validation

Full suite: 43 passed. Real MB hard checks PASS. Task preparation created 30 frozen
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
Follow-up integration audit: the actual execute/results/cache path passed for temporary
synthetic GCN/GAT fixtures. Mutable bundles are re-hashed before planning or execution;
altering a model parameter while retaining an old hash now fails before model work.
Production FlyWire execution remains zero. The plan was regenerated for the code change.
The local DRAFT review packager passed synthetic file-scope and archive tests. It
excludes private files, raw/processed dataset payloads and live registration evidence.
The current archive was built and independently verified: 3725 project files, 2218442
bytes, ZIP CRC and every project-file SHA-256 PASS. No upload or registration occurred.
See reports/osf_review_package.json and reports/osf_review_verification.json.

## Handoff

Read reports/OVERNIGHT_REPORT.md for the full structural results and final DAN type table.
Safe next command: `.venv/Scripts/python src/run_experiment.py --plan`.
Actual execution remains intentionally blocked until the researcher completes OSF and
supplies exact registration/config/bundle hash evidence. See RUN_OVERNIGHT.md.
