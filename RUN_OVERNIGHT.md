# Prepared experiment code — 2026-11-18

All commands run from `D:/icss/acc2/flywire-mb-gnn-topology-bias` or the equivalent
persistent molab project directory. No new download or account action is required.

## Safe checks before OSF

```powershell
.venv\Scripts\python scripts/validate.py
.venv\Scripts\python src/week1.py task-prepare
.venv\Scripts\python src/week1.py task-toy
.venv\Scripts\python src/run_experiment.py --plan
```

The plan creates 3660 immutable per-run YAML configurations under `configs/runs/<bundle>/`
and `plans/<bundle>/plan.parquet` with its JSON manifest. It does not generate real null
graphs, train biological models, produce learning curves or compare real/null results.
All scientific configuration, frozen inputs and implementation files contribute to the
bundle hash. Changed inputs produce a separate plan; mismatched artifacts fail loudly.

The full crossed plan is 30 task/model seeds times (one real + 30 null-A + 30 null-B
graphs) times GCN/GAT. Null graph instances are reused across seeds and architectures.
The graph, role/NT annotations, frozen odor identities, readout groups and initialization
policy have explicit fairness checks. Source graph files are never modified by training.

## After the researcher registers OSF

The runner cannot bypass the registration gate. The researcher must reconcile the DRAFT
with accepted DECISIONS (including v1 scalar DAN and trial-cap/censoring language), register
the exact plan, and update the registration metadata themselves. This overnight task
does not edit PREREGISTRATION.md or perform any account action.

The gate expects a REGISTERED status and a valid OSF registration DOI in that document,
plus `configs/osf_registration.json` containing the same `osf_doi`, `registered_at`,
the exact current file's `preregistration_sha256`, `task_config_sha256` from task_inputs.json,
and the accepted `bundle_sha256` from experiment_plan.json. The bundle also binds the
environment lock and implementation files. No such evidence is created tonight.

Once registration is complete, double-click `RUN_EXPERIMENT.cmd` on Windows, or execute:

```bash
python src/run_experiment.py --execute
```

In marimo, a single subprocess call to that `.py` entry point provides the same launch.
The Windows launcher currently exits through the OSF gate while the document is DRAFT.
The code does not expose a force-run or skip-registration option.

## Checkpoint and resume

Model checkpoints contain model/Adam state, Python/NumPy/Torch/CUDA RNG, trial cursor,
criterion streak and complete history. Checkpoints commit only completed trials; a crash
during an optimizer/probe operation replays from the last safe checkpoint. Null chains
save their current sparse edge targets, counters and exact PCG64 state. Progress is saved
at intervals <=900 seconds and at controlled boundaries. A long native operation may
finish/replay at the next completed-trial boundary; no partial trial is published.

The execution ledger locks device and library versions. A changed device, configuration,
input identity, graph or code signature cannot silently resume an old run. The session
budget yields at 11.5 hours for a safe restart before the 12-hour session limit.

Run the same command again to continue. Completed parquet/manifest pairs are verified
and skipped. Unfinished trials resume their checkpoints. Every scientific results parquet
has a `.manifest.json` with git commit, run YAML/config hash, scientific bundle hash,
source/code hashes, graph/input checksums, device/library metadata and registration evidence.
An orphan parquet is accepted only if it equals recovered checkpoint history.

## Explicit modeling proposals

Review DECISIONS W1-P06..P09 and W2-P01..P06 before registering. In particular, readout
NT groups are a coarse surrogate; E/I sign labels are a separate PROPOSED surrogate.
The frozen task anti-Hebbian baseline and gradient-trained GCN/GAT are distinct paths,
as described in EXPERIMENT_SPEC. GNNs receive cue features, never a direct US/target input.
The 2000-trial budget records right censoring; it does not implement automatic exclusions
or silently replace the preregistration's separate exclusion rule. Comparative statistics
and later noise robustness are outside tonight's code-preparation phases.
