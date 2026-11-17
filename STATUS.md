# Overnight status — 2026-11-18 (Asia/Tehran)

## Completed

- Frozen fafb/v783 sources downloaded and converted to parquet; full CSR/PyG graph built.
- Human-approved bilateral MB subset: KC 5177, PN 685, MBON 96, DAN 277;
  6235 nodes, 89769 directed edges at pair-total synapse threshold 5. Hard checks PASS.
- EXPERIMENT_SPEC KC anchor updated to bilateral [4400,5500], per explicit instruction.
- Task v1 parameters locked in configs/task_engine.yaml; seeds 0..29 saved.

## Active phase

A. Complete task engine, frozen odor sets/readout identities, resumable state and toy tests.
B. Implement GCN/GAT and directed degree-preserving null variants with invariant tests.
C. Implement one-command 30-seed runner, parquet + git/config manifests, strict OSF gate.

## PROPOSED decisions

See DECISIONS W1-P06 through P09: odor IDs and balanced schedule, activity normalization,
shared-edge deterministic initialization, NT-based MBON readout surrogate and fixed
valence/discrimination normalization. The user may veto; no actual learning results
have been inspected.

## Blocks / restrictions

- PREREGISTRATION is DRAFT: real biological learning and comparative runs remain disabled.
- No changes to PREREGISTRATION.md, dense adjacency, new datasets or account actions tonight.
- PN->KC distribution is reported; biological interpretation remains with the researcher.
- E/I mapping for the sign-preserving null will be an explicit PROPOSED policy, not an inferred fact.

## Validation

Previous full suite: 17 passed. New MB selection test passed; a real-data Polars Object
casting bug was fixed with a leading-unselected-node regression case (pending full rerun).
