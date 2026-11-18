# AMENDMENT-01 — frozen body and Ridge readout

Date: 2026-11-18. Evidence: exploratory phase 2B. This record does not authorize
starting the confirmatory sweep. The original preregistration remains the document
at `v1.0-prereg`.

## Rationale and selection

At real-GCN initialization for seed 1337, the cosine between the mean CS+
and CS− BCE gradients was −0.999779. Averaging across opposite labels exposed a
small balanced gradient compared with the per-cue gradient norms. The initial
paired pilot did not reach criterion. In phase 2, averaging BCE over all four
pairs for each update (A) also failed in all six real-graph pilot runs, including
the GraphSAGE-max check. A enters this amendment only as the fixed warm-up for
the qualifying frozen-body Ridge procedure (B).

## Procedure selected from phase 2B

1. For each matched run, warm up its model on its assigned topology for exactly
   500 updates. Each update uses one existing shuffled eight-cue block grouped
   into the four same-family CS+/CS− pairs. Average the original BCE over the
   eight cue presentations, take one Adam step at 0.001 with the frozen optimizer
   settings, and run the original probe. Freeze the resulting model including
   its existing nodewise output projection.
2. From the frozen model collect the vector of 96 nodewise sigmoid outputs at
   the same ordered MBON identities for each of the eight fixed cues. Keep the
   body, projection and topology fixed. Each readout trial contributes one
   complete eight-cue block to the cumulative fit, with CS+ target 1 and CS−
   target 0. Fit the 96 coefficients and an unpenalized intercept by minimizing
   the sum of squared errors plus 0.001 times the squared L2 norm of the
   coefficients. Do not standardize features.
3. Clip scalar Ridge predictions to [0,1]. Probe the same eight fixed cues after
   each cumulative fit and compute the existing balanced discrimination formula,
   0.5 × (mean score for CS+ + 1 − mean score for CS−). Stop at 20 consecutive
   readout trials strictly above 0.8, or cap at 500 readout trials. Report both
   readout trials to criterion and total updates including the 500-update
   warm-up. Keep nonreaching runs censored at the cap. Do not compare 2B's
   readout-only trial count numerically with the original one-cue training count.
4. Use the same warm-up budget, cue blocks, MBON order, fitting procedure, alpha,
   cutoffs and reporting in every matched topology condition within an
   architecture. Each run fits its own coefficients. Topology is the varying
   factor within a matched contrast. This amendment does not select GraphSAGE
   over GAT for the confirmatory matrix.

## Pilot evidence and its limits

The A and B pilots used the real graph and two out-of-matrix task seeds, 1337 and
2026. The four prespecified GCN/GraphSAGE-mean B runs and the two separately
reported GraphSAGE-max B runs all reached criterion at the earliest possible
readout trial, 20. The total in each case was 520 updates, or 4,160 training cue
presentations. Final in-sample Ridge RMSE ranged from 6.38e-7 to 7.22e-6. B's
fixed cue identities were used for fitting and probing. This measures the
readability of known cue representations. It cannot demonstrate transfer to
novel cues or a real-versus-null advantage. The observed minimum-time ties may
also limit the sensitivity of a trials-to-criterion comparison.

GraphSAGE-max improved A's discrimination over GraphSAGE-mean on both seeds but
did not reach criterion. Both aggregations hit the same 20-readout-trial floor
under B. The GAT transfer and the behavior on nulls were outside this pilot.
Phase C was not triggered. No new throughput grid, noise-policy change, or
confirmatory execution was authorized by this pilot.

Pilot code commit `f2b219e905470c56e7b6b59895aa8dab958ba0cf`.
Verified run directory in Molab: `results/phase2_pilot/0a368cfc455e/`.
