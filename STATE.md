# Resume state — 2026-11-24 (Asia/Tehran)

Workspace: D:/icss/acc2/flywire-mb-gnn-topology-bias

## Manuscript ready for researcher review

Read `PREPRINT-DRAFT/MANUSCRIPT.md` or its reviewed PDF. The local package
manifest verifies all included reports and figures. The real-only pooling
control completed 15 fits. Review author details, declarations and external
archive access before any submission. There is no active compute run.

## Completed matched edge-rule follow-up

The 40-job exploratory comparison and the separate frozen-body GNN check are
complete. Read `results/exploratory/matched_edge_rules_v1/summary/REPORT.md`
and `results/exploratory/matched_edge_rules_v1/gnn/REPORT.md`. The latest
integrity check verified all 320 checkpoint rows, split and mask checks, and
all listed SHA-256 digests. There is no pending run.

## Completed exploratory learning-rule screen

Run commit `7dee80d399af05a41dd353a760a013e4e77a911c`. The 32-trial CUDA smoke
passed for all three arms and its Drive summary ZIP SHA-256 is
`ab389c2007b5692a8177da35ce92e5e95e9d47507ac04499d0b54df07c077dd4`.
The fixed-budget screen finished with `RULE_COMPLETE screen 105` in the Molab
notebook `nb_QuFeJ6u22LAUiouD4dTEBF`. The Drive subfolder
`exploratory-learning-rules-v1/screen` holds six immutable wave receipts,
per-job Parquets and model checkpoints. The summary archive SHA-256 is
`85c58512f684e84dccae2b2fd3b8ead7afcf4dd9e2d63f3599452276cb3e6652`.
The completed local audit verified all 105 jobs and 210,000 trial rows.
Open `results/exploratory/learning_rules/d9957e587455/screen/REPORT.md` for the
results and figures. This phase has no outstanding runs or transfers.

The one-time confirmatory analysis has completed from locked commit
`ef2eb2f95d73b118390aca13e9ad5f0a745cc2ae`. W3-S29/S30 freeze the
v1.2 primary and the researcher-defined two-sided ACE/three-outcome BH family
before opening data. All 1,830 inputs and 305 archives were verified.
The output archive SHA-256 is
`41b06a9081024a5277d2bed932d79866439aab2ca2e1ac43ba68b7403e6a43bb`.
The verified report, summaries and figures are in `results/confirmatory/`.
The [original result package](https://drive.google.com/file/d/1PRpXph0GuL7wTlhGp_sc4AxBtVK88sja/view?usp=drivesdk)
includes the one-time inference provenance.

**The v1.2 sweep and its engineering audit are complete.** At 11:48 Tehran
time on November 19, all 1,830 planned jobs and 305 Drive archives passed
inventory and hash checks. The third writer recorded a session summary and
released the writer lock. The local aggregate report is
`results/final_binary_sweep/149776bcf90f/ENGINEERING_REPORT.md`.
The [verified Drive completion manifest](https://drive.google.com/file/d/15qsM12zQ3rOfMQ85jSwcokR12ADeJvnk/view?usp=drivesdk)
is the durable completion marker. Analysis is a separate stage.

The researcher released the v1.2 confirmatory sweep. The frozen plan lives in
`configs/final_binary_sweep.json` (plan SHA-256
`149776bcf90f9d2a9ce5455d1c6e9b2cf2b97f8f9e1d56d76b7828121f6ee23f`).
The Google Drive read-back preflight passed at `2026-11-18T20:07:52Z`. Session
`1795032549415243511` is running on Molab from commit
`da7b9cb802f78b4eee5c598023c8dc350fd41250`. Six excluded-marked events were
verified in session 1. The durable Drive folder is
`root/FlyWireMB-v12-sweep/149776bcf90f`.

The first sandbox was reclaimed. It no longer has the prior writer or scratch
checkout. All 60 null graphs and 108 jobs in 18 batches survived on Drive.
The latest receipt, `batch-0017.receipt.json`, was read back with archive
SHA-256 `a75bb4a3a10c129d658f7d75d9c007631dc9760b612e936d2e3bc9327614a5cb`.
By the last receipt, session 1 had 1,938.91 measured seconds and 200.5 jobs/hour
versus the 163 benchmark. No retry or partial checkpoint receipt exists.

The notebook was relaunched on the RTX PRO 6000. The pinned checkout is
`/marimo/flywire-mb-v12-resume`, pinned commit `da7b9cb`, NumPy 2.5.5,
PyG 2.8.1, Torch 2.11.0+cu130 and the same Drive prefix
`root/FlyWireMB-v12-sweep`. Session 2 (`1795038123287148241`, PID 1172)
read 108 durable jobs at launch. It saved batch `0018`, bringing the confirmed
total to 114/1,830 at `2026-11-18T21:44:23Z` with zero retries.
Session 2 was also reclaimed. Drive has a contiguous 213-batch sequence
through `batch-0212.receipt.json`, or 1,278/1,830 jobs. Session 2 added 1,170
jobs through 05:19 Tehran time on November 19, with no recorded retries.
Its last durable time was 14,816.82 seconds and throughput 284.27 jobs/hour.
No final audit or partial receipt had been committed.

The third sandbox has the same pinned checkout at
`/marimo/flywire-mb-v12-resume-03` and the verified runtime packages. Its first
attempt exited during Google Drive authorization, before creating a writer lock
or training a job. The researcher renewed OAuth. The Drive cell now binds
`fs` without a notebook-global `mo` alias, and the separate writer preflight
passed. Session 3 (`1795067556671097903`, PID 2696) recovered 1,278 jobs and
safely archived the next batches. Latest verified count is 1,716/1,830
through batch `0285` at `2026-11-19T07:31:57Z`, zero session retries and
264.54 jobs/hour. Engineering ETA to the last batch was 0.43 hours.

Run log reference:
`/marimo/flywire-mb-v12-resume-03/final-binary-confirmatory-sweep-session3-resume.log`
and the durable Drive receipts. The final marker supersedes the incremental
checkpoint totals above.

The locked real-only pilot passed at trials 968 and 879. Raw pilot results
are under Molab `results/final_binary_pilot/fae650607b35/`. The public annotated
`v1.2-prereg-amendment` was verified before the six-job engineering run, which
is complete. Raw rate Parquets and manifests are in Molab
`results/final_binary_rate/c3df5b2c7dfc/` and its notebook log is
`final-binary-post-tag-rate.log`.

First review command from the workspace root:
`Get-Content -Encoding utf8 results/final_binary_rate/c3df5b2c7dfc/REPORT.md`.

Phase 4 is complete. The real PN→KC binary pilot on seeds 1337 and 2026
qualified GraphSAGE-max under the exploratory 2,000-trial rule. GCN and
GraphSAGE-mean missed threshold. W3-S19 through W3-S21 document the protocol
and decision. The main sweep remains on hold.

First review command from the workspace root:
`Get-Content -Encoding utf8 results/phase4_pilot/dac573617ca6/REPORT.md`.
The matching Molab run directory contains the original Parquets and manifests.
The launch log is `phase4-real-pn-pilot.log`.

Phase 3 is complete. The real-graph held-out pilot from commit `afed8a6`
exhausted 5,000 trials on all six jobs. W3-S18 records the failed gate.
The v1.1 Ridge tag remains historical after the researcher's rejection.
The cap and aggregator remain undecided. AMENDMENT-02 and its rate run
require a qualifying pilot. The main sweep remains on hold.

Phase-3 report:
`Get-Content results/phase3_pilot/65106f95c81b/REPORT.md`.
The matching Molab directory contains the original per-trial Parquets and
manifests, and its launch log is `phase3-heldout-pilot.log`.

Historical outcome: phase 2A failed and phase 2B passed its four-run core gate.
GraphSAGE-max passed B on both seeds as a supplemental check. Phase 2C was not
triggered. See `results/phase2_pilot/0a368cfc455e/REPORT.md`, AMENDMENT-01.md
and REGISTRATION_EVIDENCE.md. The public amendment tag records B's method.
Keep the main sweep on hold. No confirmatory implementation, timing grid, or noise
protocol was changed by the phase-2 pilot.

Current research hold: the main sweep requires explicit researcher release after review
of the seed-0 probe report and the three seed-1337 throughput measurements.
The checkpoint diagnosis and single-graph Adam pilot are recorded in the local
`results/molab_rate/d63676097993/LEARNABILITY.md` and
`results/learnability_pilot/0a1e8ed92f32/REPORT.md` reports. No pilot candidate met
criterion and no settings were changed. The only authorized GPU rate command is
`python scripts/molab_batch_rate.py --rate-grid` in the Molab checkout. It runs
the 200-trial benchmarks at K=8,16,32 and skips their verified completed manifests.
The one authorized grid completed. See `results/molab_batch_rate/cdeb76795db0/REPORT.md`.
The 24-hour target did not pass. Leave the main sweep on hold for researcher review.

W3-P03 update: the public GitHub tag `v1.0-prereg` is now the execution gate.
The OSF-dependent instructions below record the earlier state. Three GCN smoke
jobs at seed 0 passed. See REGISTRATION_EVIDENCE.md and reports/W3_SMOKE.md.
Current scientific bundle: `4bfaf9e768c6b4ade3874311652dc5aa928aaf30d03bce0eb0078f33ad8907ed`.
Active plan: `plans/4bfaf9e768c6/plan.parquet` (3660 jobs, public gate passed).
The full experiment is ready through `python src/run_experiment.py --execute`.
Smoke files live in separate directories and the full experiment starts fresh.

The real bilateral MB graph has been built and passed structural/count checks:
data/graphs/mb_bilateral_v783/MANIFEST.json; reports/mb_sanity.json.
KC=5177, PN=685, MBON=96, DAN=277, edges=89769.

Phase A complete: task_inputs.py, task_engine.py and task_toy.py implemented; configs/
odor_sets.json and mbon_readout.json frozen. Real input preparation and synthetic toy
command completed; 23 tests passed. Read reports/task_inputs.json and task_toy_sanity.json.
Phase B complete: models.py and nulls.py plus synthetic invariant/backward tests.
Phase C implementation now exists in runner.py, gnn_training.py and experiment_io.py;
the latest synthetic suite PASSED (43 tests). All
three authorized overnight phases are COMPLETE. Active plan is
plans/d26847e31f27/plan.parquet, with 3660 verified per-run YAMLs under
configs/runs/d26847e31f27/. The full current bundle hash is
d26847e31f27f84e5435db36b61c5398957f5455688afea43e461180d54b9e4e.
The plan was regenerated after the bundle-integrity code change. A local DRAFT OSF
review package was built with `python scripts/prepare_osf_review.py` and verified.
The new GCN/GAT execute/cache integration tests use temporary explicitly
synthetic fixtures only. No biological results or real nulls were generated.
Read reports/OVERNIGHT_REPORT.md and RUN_OVERNIGHT.md for the handoff.
Local ZIP: dist/osf_review/d26847e31f27/osf_review_d26847e31f27.zip
SHA-256: af2b32bc8cac3172e85b91a2ed4fc85190f645a0ed59b9dfae93ee62fce7f1bd
Size: 2218442 bytes. All 3725 project-file checksums and ZIP CRC PASS. No dataset
payloads, private files or live registration evidence are included; no upload occurred.
See reports/osf_review_package.json and reports/osf_review_verification.json.
Do not execute biological learning or real-vs-null
comparisons. User locked task v1 and authorized PROPOSED defaults (weekly human veto).

First resume command (safe plan-only verification; never trains models):

```powershell
Set-Location D:\icss\acc2\flywire-mb-gnn-topology-bias
.venv\Scripts\python src/run_experiment.py --plan
```

Read STATUS.md and latest DECISIONS.md before continuing. PREREGISTRATION.md must stay
unchanged. All work/data/cache/temp files stay on D. No new download or unrelated account
action is authorized. The researcher explicitly authorized Git origin pushes on
2026-11-18: commit AND push codex/week1-night-build at every 45-minute work interval.
The initial origin push and subsequent code/review-packager push through 553b96f
succeeded. The final generated metadata/docs checkpoint follows separately. Update this file at phase
boundaries or before any context/usage interruption. If continuing code changes, run
`python scripts/validate.py`, commit+push, regenerate --plan and rebuild the local review
ZIP. Do not run --execute
until genuine OSF registration evidence binds the exact preregistration hash, task config
hash and active bundle hash. No pre-OSF scientific execution remains authorized.
Raw credentials are in ignored
.local/; do not read, print, commit or bundle them for this overnight work.
