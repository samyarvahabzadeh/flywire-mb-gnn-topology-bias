# W3 smoke jobs — 2026-11-18

All three jobs completed two GCN trials on CPU at seed 0 with four PyTorch threads.
Each job ran in a fresh process. Peak RAM is the Windows peak process working set.
Wall time includes startup, the public tag check, graph preparation, two training
and probe trials, and checkpoints.

| Graph | Wall time (s) | Graph preparation (s) | Peak RAM (MiB) | Outcome |
|---|---:|---:|---:|---|
| real | 23.02 | <0.01 | 500.37 | PASS |
| null-A, instance 0 | 27.77 | 11.91 | 500.20 | PASS |
| null-B, instance 0 | 29.47 | 13.45 | 496.79 | PASS |

Both null generators accepted 897,690 swaps on 89,769 edges and passed exact
nodewise in/out degree checks. Variant B also passed exact signed degree checks.
Both null graph payloads and all three trial-2 checkpoints matched their manifests.

Execution commit: `16a03cff5579a802b2cae780c98f56ae87caae40`.
Scientific bundle: `4bfaf9e768c6b4ade3874311652dc5aa928aaf30d03bce0eb0078f33ad8907ed`.
Manifest directory: `reports/smoke/4bfaf9e768c6/`.
Public preregistration evidence: [REGISTRATION_EVIDENCE.md](../REGISTRATION_EVIDENCE.md).
These timings describe the engineering smoke jobs. The full comparison has not run.
