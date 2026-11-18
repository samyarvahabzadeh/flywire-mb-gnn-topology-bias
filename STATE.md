# Resume state — 2026-11-18 (Asia/Tehran)

Workspace: D:/icss/acc2/flywire-mb-gnn-topology-bias

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
plans/d418a415d07f/plan.parquet, with 3660 verified per-run YAMLs under
configs/runs/d418a415d07f/. This plan is now superseded by the follow-up bundle-integrity
code change: regenerate --plan after the new code commit, then prepare the local OSF
review package with `python scripts/prepare_osf_review.py`. Its archive-scope tests pass.
The new GCN/GAT execute/cache integration tests use temporary explicitly
synthetic fixtures only. No biological results or real nulls were generated.
Read reports/OVERNIGHT_REPORT.md and RUN_OVERNIGHT.md for the handoff.
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
The initial origin push succeeded for 9ae6aae. Update this file at phase
boundaries or before any context/usage interruption. If continuing code changes, run
`python scripts/validate.py` and regenerate --plan after committing. Do not run --execute
until genuine OSF registration evidence binds the exact preregistration hash, task config
hash and active bundle hash. No pre-OSF scientific execution remains authorized.
Raw credentials are in ignored
.local/; do not read, print, commit or bundle them for this overnight work.
