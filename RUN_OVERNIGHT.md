# Prepared experiment code — 2026-11-18

All commands run from `D:/icss/acc2/flywire-mb-gnn-topology-bias` or the equivalent
persistent molab project directory. No new download or account action is required.

## Non-training checks

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

## Public tag and execution

W3-P03 replaced the OSF requirement with the public annotated GitHub tag.
See [REGISTRATION_EVIDENCE.md](REGISTRATION_EVIDENCE.md) for the tag and hashes.
The gate checks the public ref without credentials, the local tag target, and the
exact analysis plan bytes. A new Git checkout can fetch the tag with
`git fetch origin tag v1.0-prereg`.

For the full experiment, double-click `RUN_EXPERIMENT.cmd` on Windows, or execute:

```bash
python src/run_experiment.py --execute
```

In marimo, a single subprocess call to that `.py` entry point provides the same launch.
For an engineering smoke job on this checkout, run
`python scripts/smoke_experiment.py real` or substitute `A` or `B`.
Each job runs two GCN trials on CPU at seed 0. Its independent checkpoints and
resource manifest preserve the full experiment's starting state.

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

Review DECISIONS W1-P06..P09 and W2-P01..P06 before interpreting results. In particular, readout
NT groups are a coarse surrogate; E/I sign labels are a separate PROPOSED surrogate.
The frozen task anti-Hebbian baseline and gradient-trained GCN/GAT are distinct paths,
as described in EXPERIMENT_SPEC. GNNs receive cue features, never a direct US/target input.
The 2000-trial budget records right censoring; it does not implement automatic exclusions
or silently replace the preregistration's separate exclusion rule. Comparative statistics
and later noise robustness are outside tonight's code-preparation phases.
