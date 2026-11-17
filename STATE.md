# Resume state — 2026-11-18 (Asia/Tehran)

Workspace: D:/icss/acc2/flywire-mb-gnn-topology-bias

The real bilateral MB graph has been built and passed structural/count checks:
data/graphs/mb_bilateral_v783/MANIFEST.json; reports/mb_sanity.json.
KC=5177, PN=685, MBON=96, DAN=277, edges=89769.

Current work: task_engine.yaml and task_toy.yaml exist; scientific engine implementation
is not yet written. Implement sparse anti-Hebbian engine, deterministic frozen odors,
OSF gate and resumable toy execution first. Then GCN/GAT, two directed null variants,
and the 30-seed runner. Do not execute biological learning or real-vs-null comparisons.

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
