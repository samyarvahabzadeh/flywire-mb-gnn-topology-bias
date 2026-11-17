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
the final synthetic suite PASSED (38 tests). Phase C was committed at 21c8a2a; a final
reporting/registration-hash audit is ready to commit. Then run ONLY
`python src/run_experiment.py --plan` on the real stored graph. Never --execute
before OSF. Verify no real results/null graphs were generated. Update STATUS and commit.
Do not execute biological learning or real-vs-null
comparisons. User locked task v1 and authorized PROPOSED defaults (weekly human veto).

First resume command (validates current code only):

```powershell
Set-Location D:\icss\acc2\flywire-mb-gnn-topology-bias
.venv\Scripts\python scripts/validate.py
```

Read STATUS.md and latest DECISIONS.md before continuing. PREREGISTRATION.md must stay
unchanged. All work/data/cache/temp files stay on D. No account action or new download
is authorized tonight. Commit locally every 45 minutes and update this file at phase
boundaries or before any context/usage interruption. Raw credentials are in ignored
.local/; do not read, print, commit or bundle them for this overnight work.
