# Resume state — 2026-11-18 (Asia/Tehran)

Workspace: D:/icss/acc2/flywire-mb-gnn-topology-bias

Latest decision: phase 2A failed and phase 2B passed its four-run core gate.
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
