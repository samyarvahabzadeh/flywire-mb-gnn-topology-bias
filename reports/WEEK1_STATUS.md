# Week-one progress — 2026-11-17 (Asia/Tehran)

> Historical checkpoint, superseded on 2026-11-18 by the researcher's bilateral MB
> approval and the completed overnight code phases. Current report: OVERNIGHT_REPORT.md;
> current structural checks: mb_sanity.json; current progress: ../STATUS.md.

Status: **full-source ingestion and sparse graph complete; MB selection and scientific
acceptance remain incomplete pending human semantic decisions**. No model was trained,
no learning curve or performance metric was inspected, and no hypothesis test was run.

## Real connection schema, reported before processing

Source: `connections_princeton`, dataset `fafb`, snapshot `783` (explicit researcher
confirmation, superseding the unavailable legacy `connections` name).

Ordered header: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`.

| Observed quantity | Actual value |
|---|---:|
| Raw CSV rows | 5,342,446 |
| Unique pre_root_id | 137,518 |
| Unique post_root_id | 130,183 |
| Union of pre/post root IDs | 138,584 |
| Raw rows with syn_count >= 5 | 3,754,052 |
| Researcher-supplied v630 reference | 2,613,129 |
| Raw-row difference from that reference | +1,140,923 (+43.66118%) |
| Directed pairs after summing all neuropil rows and thresholding pair total >= 5 | 3,732,460 |
| Pair-count difference from supplied reference | +1,119,331 |

The raw sample contains syn_count=1, so neuropil-row counts and pair-total edge counts
are reported separately. No source rows, threshold or dataset version were adjusted to
make v783 match v630. No exact v783 acceptance tolerance was supplied; the comparison
is reported as an observation, not silently declared equal to the v630 anchor.

Connection gzip SHA-256:
`445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b`.
Full raw statistics: `connection_sanity.json`; ordered source headers: `schema_inventory.json`.

## Storage and sparse graph

| Artifact | Actual size / count |
|---|---:|
| connections_princeton gzip | 68,456,801 bytes |
| consolidated_cell_types gzip | 901,707 bytes |
| neurons gzip | 1,679,884 bytes |
| classification gzip | 934,402 bytes |
| Total raw gzip | 71,972,794 bytes |
| Total parquet shards | 54,665,983 bytes |
| Whole-brain source nodes | 139,255 |
| Directed pair-total >=5 edges | 3,732,460 |
| Isolated neurons retained | 671 |
| Self loops observed | 0 |
| CSR in-memory buffers | 45,346,544 bytes |
| Compressed full_csr.npz | 13,711,991 bytes |
| full_pyg.pt including raw annotation attributes | 125,058,971 bytes |

Files are on D under `data/`. `data/MANIFEST.json` records SHA-256, timestamps, sizes,
source products and snapshot metadata for raw/parquet artifacts. Graph files and their
SHA-256 are in `data/graphs/source_full/MANIFEST.json`.

The graph is directed, weighted by integer synapse counts, rows=pre and columns=post.
No dense adjacency was created. Every source neuron is retained, including isolates.
All raw annotation columns are preserved; `primary_type` and `additional_type(s)` remain
separate PyG attributes. Scientific MB roles and E/I edge signs are not invented.

## Environment, checks and reproducibility

Repository: `D:/icss/acc2/flywire-mb-gnn-topology-bias`.
Base commit: `dcbb90f4bc2546dd99d497aa40dc5f3734b79488`.
README, PREREGISTRATION and EXPERIMENT_SPEC were read in full and remain unchanged.

Python 3.11.15, polars 2.1.0, pyarrow 26.0.1, scipy 1.17.1, torch 2.15.0+cpu and
PyG 2.8.1 are installed on D; `uv.lock` pins the environment. GPU was not needed.
Runtime, caches and large temporary files are on D.

**17 synthetic engineering tests passed**, including high-precision root IDs, pair
aggregation before threshold, direction, isolates, explicit schema failures, unknown
sign handling, interruption recovery and a full unsigned source graph with raw labels.
Ruff lint/format checks passed. See `validation.json`, `engineering_tests.xml` and logs/.
The synthetic three-odor algebra/checkpoint fixture passed. It is not a biologically
validated conditioning task or a learned model result. Live molab execution was not run.

Atomic progress checkpoints and a background heartbeat are configured at <=900 seconds.
Downloads resume by validated HTTP ranges; conversion resumes verified parquet shards;
CSR construction checkpoints sparse matrices and reuses verified completed stages.
A heartbeat cannot serialize the internals of a native sort/serializer; an interrupted
native operation replays from the last committed stage. Scripts run as .py files.

## MB sanity: unresolved scope must not be forced to pass

Raw source annotation frequencies, before any approved MB role mapping:

| Source class label | Left | Right | Total |
|---|---:|---:|---:|
| Kenyon_Cell | 2,580 | 2,597 | 5,177 |
| MBON | 48 | 48 | 96 |
| DAN | 166 | 165 | 331 |
| ALPN | 341 | 344 | 685 |

Treating all Kenyon_Cell rows as the selected KC population would not reproduce the
initial ~2,200 anchor. No hemisphere was silently selected and the anchor was not
changed. These are raw source class frequencies, not a finalized MB membership rule.
The researcher has been asked to choose hemisphere/scope and exact KC/PN/MBON/DAN
membership. Full class/type/subclass frequencies are in `annotation_review.json`.

The MB command currently exits with loud FAIL for missing approved role lists. No MB
sanity pass is claimed. KC count bounds, PN->KC convergence bounds/statistic, final
annotation selection and NT signs remain open. Blank predicted NT values exist (19,658
rows); unknown/modulatory signs cannot be silently inferred.

## Decisions and source caveats

Implemented instructions: W1-001/002 (frozen source/threshold, sparse/no pre-OSF metrics),
W1-003 (human approval of auxiliary inspection downloads), W1-004 (human confirmation
of connections_princeton), W1-005 (v630 comparison), W1-006 (lossless ingestion and
full source mirror). Observations W1-O01 through O07 record actual files, metadata,
network diagnosis, counts and the MB scope issue. W1-P03/P04 and MB sign/policy choices
remain pending. Details and dates are in `DECISIONS.md`.

The API's legacy-name error was explicit HTTP 404 JSON: the resource `connections`
is unavailable for fafb. Approved downloads succeeded over the same route. Two active
SSH processes were found; no credential was displayed and no new tunnel was needed.
The catalogue returned HTTP 200 HTML without parseable product links.

Annotation resource Last-Modified dates include February 24, 2026 despite fafb/783 paths.
No claim is made that today's annotation files are byte-identical to October-2023 files;
historical annotation equivalence remains for human verification.

## Drive backup

Connected account identity was verified as `s.rasad.793@gmail.com`. The source/config/
review-report bundle is backed up in the project Drive folder:
https://drive.google.com/drive/folders/1hpGFHsg4HjV6WG1NMyN7KxFdm7V7QhCX .
Credentials, environments and raw dumps are excluded from the bundle. Raw and processed
data remain on D. Automatic review rejected raw gzip export because explicit raw-export
authorization was not established; the code/report backup is the safer completed scope.
No quota error was observed and no shared folder was left or removed.
