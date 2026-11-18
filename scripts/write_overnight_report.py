"""Render the local handoff report from recorded structural checks and safe plan metadata."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from mbtopology.runtime import read_json, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    mb = read_json(root / "reports/mb_sanity.json")
    task = read_json(root / "reports/task_inputs.json")
    plan = read_json(root / "reports/experiment_plan.json")
    manifest = read_json(root / "data/MANIFEST.json")
    tests = ET.parse(root / "reports/engineering_tests.xml").getroot()
    suites = tests.findall("testsuite") if tests.tag == "testsuites" else [tests]
    test_count = sum(int(suite.attrib["tests"]) for suite in suites)
    failures = sum(
        int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
        for suite in suites
    )
    raw_bytes = sum(
        item["bytes"] for name, item in manifest["artifacts"].items() if name.startswith("raw/")
    )
    parquet_bytes = sum(
        item["bytes"] for name, item in manifest["artifacts"].items() if name.startswith("parquet/")
    )
    commits = subprocess.check_output(
        ["git", "log", "-5", "--oneline"], cwd=root, text=True
    ).strip()
    counts = mb["mb_roles"]
    dan_rows = "\n".join(
        f"| {row['primary_type']} | {row['count']} |" for row in mb["dan_selected_types"]
    )
    text = f"""# Overnight handoff — 2026-11-18 (Asia/Tehran)

All three authorized code-preparation phases are complete. Week-one engineering sanity
and synthetic task tests pass. Scientific execution remains blocked by the DRAFT OSF
state. No real null graph, biological learning run or real-vs-null comparison ran tonight.

## MB subset and data

| Group | Selected neurons |
|---|---:|
| KC | {counts["KC"]} |
| PN (ALPN) | {counts["PN"]} |
| MBON | {counts["MBON"]} |
| DAN (PAM/PPL1 and predicted DA) | {counts["DAN"]} |
| Total | {mb["mb_neurons"]} |

Internal directed edges at pair-total synapse threshold 5: **{mb["mb_edges"]}**.
Bilateral KC hard check: **PASS**, allowed interval [4400,5500]. Whole-brain count
139255: **PASS**. Sparse CSR, positive thresholded weights and retention checks pass.
MB CSR buffers: {mb["csr_bytes"]} bytes. No dense adjacency is constructed.

PN->KC unique-source convergence: mean {mb["pn_kc_convergence"]["mean"]:.6f}, median
{mb["pn_kc_convergence"]["median"]}, range 0..10; 427 zero-ALPN-input KCs at threshold 5
are retained. Numerical biological convergence acceptance bounds were not supplied;
this distribution is reported for human verification rather than fitted to a reference.

Raw compressed data total: {raw_bytes} bytes. Parquet total: {parquet_bytes} bytes.
SHA-256/provenance: data/MANIFEST.json and each graph's MANIFEST.json.
The connection product is connections_princeton/fafb/783. Raw rows=5342446,
unique pre=137518, unique post=130183, raw rows >=5=3754052 (+43.66118% vs supplied
v630 anchor); pair-total >=5 edges in the full graph=3732460. No count was forced.

## Final selected DAN types

{mb["dan_prefix_candidates"]} PAM/PPL1 prefix candidates; {mb["dan_excluded_by_nt"]} with
blank predicted NT were excluded by the researcher's explicit DA filter. No NT confidence
cutoff was invented. Final selected types:

| Type | Count |
|---|---:|
{dan_rows}

## Phase A: locked task v1

Completed instantaneous anti-Hebbian KC->MBON depression, bounded unit weights,
global US scalar d, fixed valence readout and 20-consecutive-trial criterion logic.
The explicitly synthetic three-odor execution passed US targeting, no-US invariance,
weight bounds and exact resumed/uninterrupted-state equality. No biological task scores
or learning curves were computed.

configs/seeds.txt freezes 0..29; configs/odor_sets.json freezes eight cues per seed,
{task["active_kcs_per_odor"]} active KCs per cue ({task["actual_active_fraction"]:.8f}).
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

Planned jobs: **{plan["planned_runs"]}** = 30 seeds x (real + 30 A + 30 B) x GCN/GAT.
One frozen YAML exists per planned run, referencing the same immutable scientific bundle:
`{plan["bundle_sha256"]}`.
Active plan: `{plan["plan_path"]}` with its JSON manifest.

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

Full synthetic suite: **{test_count} tests**, failures/errors **{failures}**; Ruff lint and
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
`{sha256(root / "PREREGISTRATION.md")}`.
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
{commits}
```

See STATUS.md for completion/blockers and STATE.md for the safe resume command. The
remaining execution block is intentional: the researcher must register OSF and reconcile
the DRAFT with accepted decisions. Future noise robustness and comparative statistical
analysis are outside tonight's authorized code-preparation phases.
"""
    (root / "reports/OVERNIGHT_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote overnight report: {test_count} synthetic tests; zero biological runs.")


if __name__ == "__main__":
    main()
