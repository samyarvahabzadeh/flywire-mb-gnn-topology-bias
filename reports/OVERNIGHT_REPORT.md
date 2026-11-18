# Overnight handoff — 2026-11-18 (Asia/Tehran)

All three authorized code-preparation phases are complete. Week-one engineering sanity
and synthetic task tests pass. Scientific execution remains blocked by the DRAFT OSF
state. No real null graph, biological learning run or real-vs-null comparison ran tonight.

## MB subset and data

| Group | Selected neurons |
|---|---:|
| KC | 5177 |
| PN (ALPN) | 685 |
| MBON | 96 |
| DAN (PAM/PPL1 and predicted DA) | 277 |
| Total | 6235 |

Internal directed edges at pair-total synapse threshold 5: **89769**.
Bilateral KC hard check: **PASS**, allowed interval [4400,5500]. Whole-brain count
139255: **PASS**. Sparse CSR, positive thresholded weights and retention checks pass.
MB CSR buffers: 1102172 bytes. No dense adjacency is constructed.

PN->KC unique-source convergence: mean 4.307128, median
5.0, range 0..10; 427 zero-ALPN-input KCs at threshold 5
are retained. Numerical biological convergence acceptance bounds were not supplied;
this distribution is reported for human verification rather than fitted to a reference.

Raw compressed data total: 71972794 bytes. Parquet total: 54665983 bytes.
SHA-256/provenance: data/MANIFEST.json and each graph's MANIFEST.json.
The connection product is connections_princeton/fafb/783. Raw rows=5342446,
unique pre=137518, unique post=130183, raw rows >=5=3754052 (+43.66118% vs supplied
v630 anchor); pair-total >=5 edges in the full graph=3732460. No count was forced.

## Final selected DAN types

323 PAM/PPL1 prefix candidates; 46 with
blank predicted NT were excluded by the researcher's explicit DA filter. No NT confidence
cutoff was invented. Final selected types:

| Type | Count |
|---|---:|
| PAM01 | 28 |
| PAM02 | 13 |
| PAM03 | 1 |
| PAM04 | 23 |
| PAM05 | 20 |
| PAM06 | 30 |
| PAM07 | 18 |
| PAM08 | 43 |
| PAM09 | 9 |
| PAM10 | 15 |
| PAM11 | 16 |
| PAM12 | 23 |
| PAM13 | 9 |
| PAM14 | 10 |
| PAM15 | 3 |
| PPL101 | 2 |
| PPL102 | 2 |
| PPL103 | 2 |
| PPL104 | 2 |
| PPL105 | 2 |
| PPL106 | 2 |
| PPL107 | 2 |
| PPL108 | 2 |

## Phase A: locked task v1

Completed instantaneous anti-Hebbian KC->MBON depression, bounded unit weights,
global US scalar d, fixed valence readout and 20-consecutive-trial criterion logic.
The explicitly synthetic three-odor execution passed US targeting, no-US invariance,
weight bounds and exact resumed/uninterrupted-state equality. No biological task scores
or learning curves were computed.

configs/seeds.txt freezes 0..29; configs/odor_sets.json freezes eight cues per seed,
259 active KCs per cue (0.05002897).
configs/mbon_readout.json freezes the PROPOSED readout partition: 23 avoidance, 68 approach,
5 NT-unassigned excluded only from readout. All 96 MBONs stay in the graph.
Twelve MBONs have no KC inputs under threshold 5; their proposed output is zero.

## Phase B: models and nulls

GCN and GAT are implemented with sparse signed/magnitude-aware message passing,
width 128 and two message-passing layers. Counts: GCN=33921, GAT=35201,
relative difference 3.77347%, below the 5% budget check. Within an architecture,
matched conditions have identical learnable parameter counts and seeded initialization.

Null A preserves exact nodewise directed in/out degree. Null B additionally preserves
each node's in/out degree in every sign stratum. Loops/parallel edges are rejected;
weights remain with presynaptic slots. Both variants passed synthetic invariants and
bitwise interrupted/uninterrupted RNG-trajectory tests. Target 10 accepted swaps per edge
is PROPOSED, not a proof of perfect graph mixing. No real null was generated tonight.

## Phase C: ready runner, no execution

Planned jobs: **3660** = 30 seeds x (real + 30 A + 30 B) x GCN/GAT.
One frozen YAML exists per planned run, referencing the same immutable scientific bundle:
`d26847e31f27f84e5435db36b61c5398957f5455688afea43e461180d54b9e4e`.
Active plan: `plans/d26847e31f27/plan.parquet` with its JSON manifest.

RUN_EXPERIMENT.cmd is the Windows one-button launcher. Portable command:
`python src/run_experiment.py --execute`. While PREREGISTRATION is DRAFT this is blocked
before graph/model work; there is no force-run bypass. Planning uses --plan and runs no
models. Registration evidence must bind the exact preregistration hash, task config hash
and scientific/code bundle hash. The environment lock is included in the bundle.

Every scientific results parquet will have a JSON manifest containing git commit,
run YAML/config hash, scientific/source/code hashes, graph/input checksums, device/library
metadata and registration evidence. Model/Adam/all-RNG/trial/history checkpoints commit
at <=900-second completed-trial boundaries. Partial trial updates are never published.
Session safety yields after 11.5 hours; rerunning the same command resumes verified work.

## Validation and limits

Full synthetic suite: **43 tests**, failures/errors **0**; Ruff lint and
format checks pass. Tests cover model gradients, exact degree/sign invariants, same-seed
fairness, corrupt/mismatched checkpoints, result orphan recovery, 3660 unique immutable
plans, pre-OSF blocking and exact resume after interruption following an optimizer step.
Validation was CPU-only; CUDA/molab execution has not been exercised.

The coarse NT readout and E/I maps are explicitly PROPOSED. The readout inference is
motivated by tested MBON behavior in [Aso et al. 2014](https://cdn.elifesciences.org/articles/04580/elife-04580-v1.pdf),
not a validated behavior label for every current FlyWire MBON. See DECISIONS W1-P06..P09
and W2-P01..P06 for all defaults and scientific caveats, including trial-cap censoring.
No automatic scientific exclusions or comparative statistics were run.

PREREGISTRATION.md was not edited. Current SHA-256:
`3fc010a514841e6c1aef9e4f5dc016f94c974197fbdc91e3d307534f5e727be6`.
All active files/data/cache/temporary work are on D. No new dataset or account action
occurred during the original overnight phases. The researcher subsequently authorized
Git origin pushes: codex/week1-night-build is published and subsequent 45-minute work
checkpoints include both commit and push. The earlier Drive backup was not refreshed.

Follow-up tests exercise the actual execute/results/cache path using temporary synthetic
GCN/GAT fixtures and reject modified bundles that retain an old hash. The local DRAFT
review packager also passed archive-scope tests. It packages the current code/configuration
and planned jobs without dataset payloads or private files; it neither uploads nor
registers OSF. Run `python scripts/prepare_osf_review.py` after `--plan` to build it.

## Local Git checkpoints

```text
553b96f Prepare local OSF review archives with strict payload and credential exclusions
e56cd5d Audit synthetic runner execution and reject mutated scientific bundles
9ae6aae Record completed overnight phases, final 3660-run plan and safe OSF-blocked handoff
e31c469 Bind OSF evidence to exact task and code bundle and finalize bilateral status reporting
21c8a2a Prepare OSF-gated 3660-run launcher with immutable plans and transactional model resume
```

See STATUS.md for completion/blockers and STATE.md for the safe resume command. The
remaining execution block is intentional: the researcher must register OSF and reconcile
the DRAFT with accepted decisions. Future noise robustness and comparative statistical
analysis are outside tonight's authorized code-preparation phases.
