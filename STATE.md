# Resume state — 2026-11-18 (Asia/Tehran)

Workspace: D:/icss/acc2/flywire-mb-gnn-topology-bias

The researcher released the v1.2 confirmatory sweep. The frozen plan lives in
`configs/final_binary_sweep.json` (plan SHA-256
`149776bcf90f9d2a9ce5455d1c6e9b2cf2b97f8f9e1d56d76b7828121f6ee23f`).
The Google Drive read-back preflight passed at `2026-11-18T20:07:52Z`. Session
`1795032549415243511` is running on Molab from commit
`da7b9cb802f78b4eee5c598023c8dc350fd41250`. Six excluded-marked events were
verified in session 1. The durable Drive folder is
`root/FlyWireMB-v12-sweep/149776bcf90f`.

First resume action: inspect only `SWEEP` engineering events in Molab
`final-binary-confirmatory-sweep-session1.log` and check whether PID 42795 is
still the writer. If a new session is required, use the same sweep code, plan
and Drive prefix `root/FlyWireMB-v12-sweep` with the execution command in
RUN_OVERNIGHT.md. The runner skips verified batch receipts and restores the
latest durable partial state. Do not open Parquet trajectories during the sweep.

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
