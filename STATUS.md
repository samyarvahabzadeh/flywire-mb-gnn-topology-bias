# Research status — 2026-11-24 (Asia/Tehran)

## Exploratory differential conditioning on Molab

**Complete.** Molab computed all 3,120 conditions over 20 odor layouts.
The independent audit checked all 20 shards and receipts and recomputed
the paired contrasts from the full crossed inventory. Under local
prediction error, explicit exposure to a similar safe odor improved
novel-odor discrimination on all 20 layouts. The median gain was +1.527
percentage points on the model response scale for real wiring and +1.469
for the controls. The median paired real-minus-control effect was +0.006
points [IQR −0.059 to +0.034], positive in 10/20 layouts. The normalized
paired contrast was −0.028 points. The punishment-pulse control yielded
zero changes and the Adam reference gained +0.100 points with real
wiring. These are descriptive computational outcomes at one compartment
on one connectome. See `results/exploratory/differential_molab_v1/REPORT.md`.

## Exploratory aversive memory generalization

**Complete.** A reduced KCγ-to-MBON11 circuit compared local dopamine-gated
depression and Adam/BCE across 16 seeds and 12 capacity-matched input masks
plus real wiring. All 416 jobs matched conditioned-cue strength and the
audit verified 83,200 novel-cue evaluations. The real circuit transferred
32.0% of the conditioned response change to zero-overlap probes under local
plasticity and 44.9% under Adam. Near-probe transfer was 93.8% and 94.7%.
The matched-strength topology-by-rule interaction was -0.00292 in units of
conditioned-cue response, with five of 16 positive seed contrasts. The
topology controls retained 67.1–68.7% of real pairs. Read
`results/exploratory/memory_generalization_v1/REPORT.md`.

## Exploratory role-preserving layer intervention

**Complete.** Molab finished 65 jobs across five seeds, three paired donors,
and a 768-trial horizon. The median first-layer minus second-layer gap was
+2.96 percentage points with signed-degree swaps and +0.15 with additional
cell-role constraints. Median paired attenuation was +3.19 points and
positive in two of three donors. The restricted chain replaced 91.62–91.82%
of edges versus 94.18–94.23% for the signed-degree chain.
The independent audit verified all 49,920 trial rows. See
`results/exploratory/role_layers_v1/REPORT.md`.

## Molab replays and exploratory GraphSAGE layer intervention

**Complete.** The small Molab study finished all 35 fixed-horizon jobs.
Relative to real/real, a first-layer-only rewire increased mean held-out
discrimination by medians of 2.83 and 3.56 percentage points for A-00 and
B-00. Second-layer-only effects were close to zero in this two-donor study.
The accompanying PCA and decoder replays also completed and were archived.

**Crossed-bank replication complete within its wall budget.** The Molab
RTX PRO 6000 completed five two-seed panels in 101.1 minutes including its
numerical-check restart. The dataset contains 370 jobs with 768 trials each
and 12 independently generated fixed donor rewires. The first-layer-only
contrast was positive in 9 of 12 donor medians. The median donor effects
were +2.26 pp (A) and +2.47 pp (B). The second-layer-only contrast was
negative in 12 of 12 donor medians, with effects of −1.97 pp (A) and
−2.08 pp (B). The full rewire yielded smaller median effects of +0.39 pp
(A) and +0.87 pp (B). This computational pattern pertains to the tested
connectome, GraphSAGE-max, and the fixed cue-family layout.

The 65 wave receipts and six final ZIPs passed their Drive write/read-back
checks. The independent D-drive audit verified all 284,160 trial rows and
recomputed the recorded endpoints and paired contrasts. See
`results/exploratory/crossed_layer_bank_molab_v1/AUDIT.md` and its
`AUDIT_MANIFEST.json`.

## Exploratory decoder-by-topology diagnostic and PCA decomposition

**Complete.** The 30%-replacement interaction had a median of +1.04
percentage points, with two positive and two negative paired differences.
In the three-fresh-seed follow-up, graph-mean differences accounted for
99.35–99.88% of variance along the first two raw pooled PCA axes.
With local training centering and scaling in both arms, learning shared
axes from the within-graph covariance improved eight of nine graph–seed
cases. The real graph improved in two of three cases, with a median paired
gain of 10.42 percentage points. These estimates concern frozen features and
the sampled bank. See `results/exploratory/decoder_topology_v1/report/` and
`results/exploratory/pca_decomposition_v1/` for plots, summaries and manifests.

## Manuscript review package and five-seed pooling control

**Complete.** The real-only pooling control verified 15 matched readout fits
on seeds 4501–4505. Each pair of two-dimensional classifiers fit three
parameters on the same 91 frozen MBON scores. Train-only PCA and held-out
separation are documented in `results/exploratory/pooling_control_v1/`.
`PREPRINT-DRAFT/` contains the editable manuscript, visually reviewed
eight-page PDF, seven figures, source reports, and package manifest. The
manuscript identifies the registered and exploratory evidence. Author
information, declarations, and external archive access are pending review.

## Exploratory matched KCγ→MBON11 follow-up

**Complete.** Ten real runs and thirty input-set control runs produced 320
checkpoint rows. The local audit verified all four manifests, the frozen cue
hashes, equal input capacity and one train-derived numeric cut shared by both
rules within each graph, seed and checkpoint. See
`results/exploratory/matched_edge_rules_v1/summary/REPORT.md` for the eight
checkpoints, learning curves and CPU timings. The separate seed-4301 frozen
GNN readout diagnostic selected both heads by validation BCE before scoring
held-out. See `results/exploratory/matched_edge_rules_v1/gnn/REPORT.md`.

## Exploratory readability and real-edge control

**Complete.** Real-only readability seed 4301 used 512/256/512 presentations
with pairwise disjoint PN patterns and projected KC drives. The five feature
tables and train/validation/held-out scores passed the local hash and shape
audit. CPU wall time was 226.5 seconds. See
`results/exploratory/readability_positive_control_v1/B/readability.json`.

Seed 4302 passed all fixed-cue controls on 1,638 directed KCγ→MBON11 pairs.
The trained and swapped score gaps were +0.17129 and −0.15446 in fixed A+−A−
order. The unreinforced control changed zero weights. The gated probe scored
256 fresh cue instances without training. In-process CPU time was 0.34 seconds.
See `results/exploratory/readability_positive_control_v1/C/plasticity.json`.
The Adam implementation shares the edge vector and passed a toy inventory
test. The current null generators give no compartment- or plastic-mask-
preservation guarantee. See the dated PROPOSED entry in `DECISIONS.md`.

## Exploratory topology × learning rule screen

The 32-trial CUDA smoke passed. Held-out predictions matched at initialization,
all three readouts updated, B and C kept their bodies fixed, and C had zero drift
on an unreinforced block. Training and held-out PN patterns and KC drives had
zero exact overlap. The smoke wave took 1.3 seconds on an RTX PRO 6000 and peaked
at 2.53 GiB allocated VRAM. The archived smoke report and its inventories of
three jobs and 96 trial rows passed local SHA-256 checks.

**Complete.** Molab published all six waves, 105/105 jobs and their checkpoints.
The local audit independently re-derived TTC and AUC from all 210,000 trial
rows and verified the cue checksums, inventory and summary artifact hashes.
Wave training totaled 2,039.1 seconds. Peak allocated VRAM was 4.50 GiB.
See `results/exploratory/learning_rules/d9957e587455/screen/REPORT.md`.
The exploratory plan and run code ship in
`7dee80d399af05a41dd353a760a013e4e77a911c`. See `DECISIONS.md` for the locked
PROPOSED definitions and the outcome-independent null-index selection.

## Exploratory mechanism analysis

**Complete.** The analysis re-read the 305 frozen job archives and the 11
topology archives, and matched every job's graph-manifest SHA-256. The resulting
tables contain 61 graph rows and 1,830 job rows. The private Drive export passed
write and read-back checks. Its ZIP SHA-256 is
`6a050a34b181cd8b11e6e2c9d137dcf3f9afb65090137b2487b157bde0f56abb`.
The verified local report, manifest and figures are in
`results/exploratory/mechanism/`. `PREPRINT-DRAFT/` holds the working manuscript,
exploratory outputs, and byte-identical copies of the tagged confirmatory report
and four confirmatory figures. The confirmatory phase remains closed.

**PROPOSED decisions.** Use the sparse unweighted message-passing support, report
both full and component-level gaps, describe TTC associations at the topology
level, and summarize the stop-aligned final 100 observed scores. See
`DECISIONS.md` for definitions and the censoring rule.

**OSF link blocked.** The public registration exists at DOI
`10.17605/OSF.IO/H5GBV`. Archived links are read-only. Adding a Resource requires
a DOI for the linked results. The results currently have a GitHub tag.

## Confirmatory analysis

The single locked analysis verified all 1,830 confirmatory Parquets and all
305 archives. W3-S29/S30 document the v1.2 primary and the researcher-approved
ACE replacement in the secondary BH family. The results, numeric precision
and four figures are in `results/confirmatory/`.

## Authorized final sweep

**Complete.** The November 19 engineering audit verified 1,830/1,830 fresh
jobs, all 305 batch receipts and archive members, and the job manifest links
to the Parquet and checkpoint hashes. The result files and engineering
completion marker are stored outside Molab. The writer has exited.
See `results/final_binary_sweep/149776bcf90f/ENGINEERING_REPORT.md` for
session totals and the audit record.

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
batch `0212` and saved through `0285`, bringing progress to 1,716/1,830
at 11:01 Tehran time. It continued to the final audit with no recorded
job retries.

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

## 2026-11-24 · Exploratory cognitive follow-up prepared

- Kept the prior confirmatory dataset and analysis unchanged.
- Reviewed the existing differential-conditioning implementation and the Barth et al. behavioral result. The broad benefit of differential training is already known; the new exploratory question is whether model discrimination across held-out odor families varies jointly with CS+/CS− PN overlap, probe-to-trained-cue overlap, and learning rule, and whether any rule effect depends on the KCγ→MBON11 input mask.
- PROPOSED plan frozen in `configs/exploratory_similarity_gradient.json` and `DECISIONS.md`: 64 new synthetic cue layouts, five active-PN overlap levels, 20 probes per family and level, 48 paired cycles, reciprocal assignment, three learning rules, real mask plus 12 existing capacity-matched input masks. These layouts are repeated simulations on one connectome, not biological replicates; input-mask controls are not full graph nulls.
- Added CUDA-batched runner and input packager. Static compilation and lint are the remaining local checks; no run output has been inspected.
- Molab reports 4 CPU, 32 GiB host RAM and RTX PRO 6000 Blackwell Server Edition (101,975,851,008 bytes VRAM). CUDA is available, 3 MiB reported in use, and the prior study PIDs are no longer active. The input bundle downloaded from private Drive and matched SHA-256 `9df494d31bb4a47348260060b66de96fccd64b248ea8eabbb710ed391967f77d`.
- First GPU smoke attempt failed before saving a result: batched held-out probe readout had a 3D transpose shape error. A semantics-neutral elementwise batch dot product is now committed locally for the rerun; full sweep has not started.
- Corrected GPU smoke passed: 60 rows, 0.39 s on RTX PRO 6000 Blackwell; CUDA reported 101,975,851,008 total VRAM bytes.
- Full exploratory computation completed locally on Molab: 65/65 shards, 249,600 rows, runner manifest present. All 65 parquet shards and receipts were copied to private Drive and read back for byte equality; the supervisor reports transfer complete. Peak runner allocation in its shard logs was 13,839,872 bytes, which is appropriate for this small batched model; the 80+ GB device capacity was available but not needed.
- Completed integrity-checked descriptive analysis in `analysis/exploratory_similarity_gradient.py`; final report and tables/figures are in private Drive and the concise result is saved at `results/exploratory/similarity_gradient_v1/RESULTS.md` with `MANIFEST.json`.
- Exploratory result: prediction-error differential-minus-absolute held-out discrimination averaged 0.0629 at 0/40 PN overlap, peaked at 0.0667 at 10/40, then fell to 0.0157 at 36/40; input-mask control medians closely matched real. No confirmatory statistics were run and no confirmatory files changed.
- GPU/run engineering: 65/65 shards, 249,600 rows, 519.9 s through Drive readback, peak Torch allocation 13,839,872 bytes. The 80+ GB GPU was used via CUDA; this workload did not require large VRAM allocation.
- Local commits after pushed commit `45b340a` are not yet on origin. `git push origin codex/week1-night-build` failed because GitHub:443 was unreachable through the configured proxy; retry when network access returns.
- No confirmatory run, confirmatory file, registration document, or frozen preregistration was modified.
