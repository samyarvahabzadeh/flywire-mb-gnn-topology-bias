# Public preregistration evidence

- [GitHub annotated tag v1.0-prereg](https://github.com/samyarvahabzadeh/flywire-mb-gnn-topology-bias/tree/v1.0-prereg)
- Tagged commit: `5e2c0b384252c1415f4e0fd61fd973aa28fcdb0a`
- Annotated tag object: `92b818d2ccd9c3c2c5f4a3bc74408d279659f05e`
- Tagger timestamp: `2026-11-18T11:51:10+03:30` (`2026-11-18T08:21:10Z`)
- Public ref verified without credentials at `2026-11-18T08:22:46Z`.
- SHA-256 of `PREREGISTRATION.md` in the Windows checkout (CRLF): `3fc010a514841e6c1aef9e4f5dc016f94c974197fbdc91e3d307534f5e727be6`
- SHA-256 of the tagged Git blob (LF): `9ec100c10124c483855513ff119d154a70dd712bfd6fa7c0a0f69d0e61942d82`

The plan first appears in commit `dcbb90f`, dated November 17, 2026. Its Windows copy
is byte-for-byte the tagged blob with each LF expanded to CRLF. The runner checks the
public annotated ref, the pinned local tag and commit, and the exact document bytes
before scientific execution. Fetch the tag in a new Git checkout with
`git fetch origin tag v1.0-prereg`.

Decision W3-P03 uses the public GitHub tag as the timestamped record. An OSF DOI is
not a prerequisite under this decision.

## Phase-2B amendment evidence

- [GitHub annotated tag v1.1-prereg-amendment](https://github.com/samyarvahabzadeh/flywire-mb-gnn-topology-bias/tree/v1.1-prereg-amendment)
- Tagged commit: `48e8f07cf6c2ee7702f8a4f535ff609c7e0e6dcd`
- Annotated tag object: `c1f90a1961878982ef9e1334916b948e372d9573`
- Tagger timestamp: `2026-11-18T21:25:23+03:30` (`2026-11-18T17:55:23Z`)
- Public ref verified without credentials at `2026-11-18T17:56:06Z`.
- SHA-256 of the tagged `AMENDMENT-01.md` Git blob (LF): `16ce4683a7664df610e84bca273010acb547f01aae71721b121a3e2c92b1dad9`

The tag records the intervention selected by the phase-2B pilot. The original
v1.0 bytes and the confirmatory runner and configurations remain in place.
The main sweep is on hold pending separate researcher authorization.

**Subsequent decision W3-S15:** the researcher rejected phase 2B for
confirmatory use. The v1.1 evidence is historical. Phase 3 evaluates an
end-to-end held-out cue protocol before any new amendment or execution release.

## Final binary protocol amendment evidence

- [Public annotated tag v1.2-prereg-amendment](https://github.com/samyarvahabzadeh/flywire-mb-gnn-topology-bias/tree/v1.2-prereg-amendment)
- Tagged commit: `515e65506616c7ad3c4ef17e6def1f06463f26e4`
- Annotated tag object: `50fcd6000781c27cdb96ce7f6258b75193f62c22`
- Tagger timestamp: `2026-11-18T22:49:20+03:30` (`2026-11-18T19:19:20Z`)
- Public ref and peeled commit verified without credentials at `2026-11-18T19:19:51Z`.
- SHA-256 of the tagged `AMENDMENT-02.md` blob: `045f34049ff9b6f747c054f325360bf0af88ea26f94f2da2ccfb6517349b2e9b`
- SHA-256 of the tagged `configs/final_binary.yaml` blob: `4a721b186fd1c8cd1a21fe2c6ea2d622ed083c79338f34c18dc4f905d15ef673`

Both real-only pilot runs passed W3-S22 and AMENDMENT-02 supersedes the
historical Ridge protocol. This tag precedes any null job under the final binary
protocol. The six-job engineering rate batch is declared in W3-S24. The
1,830-job sweep requires separate explicit researcher approval.
