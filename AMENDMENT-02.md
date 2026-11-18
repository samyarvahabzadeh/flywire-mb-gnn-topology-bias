# AMENDMENT-02 — Binary held-out conditioning with GraphSAGE-max

Date: 2026-11-18 (Asia/Tehran). This amendment supersedes AMENDMENT-01's frozen-body Ridge protocol. Repeated training and evaluation on the same cue instances allowed memorization. The recorded phase-2B TTCs also tied at the minimum 20-fit window, which made the endpoint insensitive to learning speed. The replacement trains end-to-end and evaluates freshly sampled cue instances.

The original PREREGISTRATION.md remains the historical record under `v1.0-prereg`. The hypothesis remains that real mushroom-body connectivity improves learning speed relative to matched rewired graphs. The treatment is the message-passing topology of the GNN under a common, fixed sensory projection.

## Pilot evidence and architecture

Use GraphSAGE with `aggregation="max"`, hidden width 128, two directed message-passing layers, its standard root transform and the unchanged MBON valence readout. Retire GCN and GraphSAGE-mean from the primary design. Both missed the relevant learning criterion in the phase-2A, phase-3 and phase-4 real-graph pilots with non-comparative seeds 1337 and 2026. GraphSAGE-max met the phase-4 held-out threshold and qualified under the new early-stop criterion on both seeds.

The architecture, cue design and final criterion were selected using real-only pilots, without viewing any null output for those selection decisions. Historical exploratory null smoke, diagnosis and rate work belongs to earlier protocols and supplied no selection evidence for this amendment. The post-amendment benchmark is limited to engineering estimates, with null instances and task seeds outside the comparison set.

The final release test was frozen in commit `88024159c2c516c0cea6d09f8767b3cdb7fb7ae5` (`configs/final_binary.yaml`). Both real-only GraphSAGE-max runs used the specified stop rule and were required to finish strictly before trial 1,500.

| Non-comparative seed | Observed TTC | Trailing 20-probe mean at stop |
| ---: | ---: | ---: |
| 1337 | 968 | 0.75099749 |
| 2026 | 879 | 0.75196483 |

## Data, binary task and held-out evaluation

Use the bilateral FlyWire FAFB v783 mushroom-body graph built from `connections_princeton` and the approved annotations, at a directed-pair threshold of 5 synapses. Select the 285 olfactory PNs with at least one outgoing edge to a KC in this fixed real graph. A layout seeded with 1337 assigns 16 shared core PNs and two disjoint private cores of 16 PNs each for A+ and A−. The remaining 237 connected PNs form one shared variable pool. Draw 8 from this pool without replacement for each presentation. Each cue activates exactly 40 PNs. Use task-seed-keyed PCG64 streams 404 for training and 405 for held-out. Verify disjoint exact PN patterns and exact projected KC drives between splits.

Apply the same sparse, fixed real PN→KC projection in every condition. Each KC receives the sum of real synapse counts from its active cue PNs divided by its total real PN synapse count, with zero when it has no input. Use the same projection, labels, sampled cue instances, initialization rule, loss and optimizer across each matched seed. Rewiring changes only the GNN message-passing adjacency. This design tests an advantage conditional on the shared real sensory projection.

One trial presents four independently sampled A+/A− pairs from the train split (eight cues). Train end-to-end with one Adam update at learning rate 0.001 and zero weight decay on the mean BCE with A+ target 1 and A− target 0. The binary CS labels are supervision targets. There is no separate Ridge fit or explicit DAN/US update in this version. Evaluate four fresh held-out A+/A− pairs after each update with the model's own MBON valence readout. Held-out discrimination is (mean A+ valence + 1 − mean A− valence) / 2.

## Outcomes, stopping and noise

- **Primary — TTC.** Let the criterion be the first full trailing arithmetic mean of 20 consecutive trial-level held-out discrimination scores at least 0.75. Early-stop that job at the end of the first qualifying window. The cap is 2,000 Adam updates. A job without a qualifying window through trial 2,000 is right-censored, retained and reported. An observed crossing at trial 2,000 remains a hit. This replaces the original 10× median-null exclusion rule for TTC. No incomplete or corrupt job enters analysis before it is repaired or explicitly accounted for.
- **Secondary — first single crossing.** Record the first observed trial-level held-out discrimination ≥0.75. A missing single crossing through the cap is right-censored.
- **Secondary — AUC through cap.** Define normalized discrete area as the sum of observed clean held-out scores plus the terminal 20-trial rolling mean carried forward to trial 2,000 after early-stop, divided by 2,000. Fully observed runs use all 2,000 scores. The extension is fixed for analysis and adds no training.
- **Secondary — noise robustness.** Freeze the model at its stopping point or cap. Use an independent, seed-keyed probe stream (406) with 20 blocks of four fresh balanced pairs. At levels 0, 0.1, 0.2 and 0.3, exchange respectively 0, 4, 8 and 12 active PNs with the same number of inactive PNs sampled from the 285 eligible PNs. Apply nested replacements to the same clean cue at each level, use the fixed real projection, and use identical probes across topology conditions. The injector is tested end-to-end for zero perturbation at level 0 and positive projected perturbation at positive levels. The outcome is the OLS slope of mean held-out discrimination versus noise level. Clean training and the TTC probe use zero injected noise.

## Design and analysis

The final design has 30 fixed seeds from `configs/seeds.txt`, 1 real graph plus 30 null-A and 30 null-B instances, and one architecture, for **1,830 jobs**. Null-A preserves each node's directed in-/out-degree. Null-B additionally preserves each node's directed in-/out-degree by excitatory/inhibitory sign stratum, including the unassigned stratum. Each null instance is generated once and is reused across the 30 task seeds.

The unit of paired inference is the task seed. For each seed compare the real endpoint with the median over its 60 matched null jobs, giving null-A and null-B equal representation. Report both strata descriptively. For TTC and first single crossing, the rank analysis records a capped miss as 2,001 to order it after a hit at 2,000. Report censoring counts separately and interpret the rank comparison as a bounded follow-up analysis. Use the one-sided Wilcoxon signed-rank test of real advantage and signed rank-biserial `r = (W+ − W−)/(W+ + W−)`, where positive paired differences favor the real graph. Drop zero differences. If all differences are zero, report `p=1` and `r=0`. Smaller TTC and first-crossing values, larger AUC and a larger noise slope favor the real graph. The primary family is one comparison, so no BH-FDR correction applies to TTC. Apply BH-FDR across the three secondary tests, including the H2 robustness slope.

Report every prespecified analysis, all censoring and discrepancies, and mark work outside the plan as EXPLORATORY. Freeze this document and final configuration with the public annotated tag `v1.2-prereg-amendment` before the engineering null benchmark or any confirmatory comparison. The tag timestamps the plan. Starting the full sweep requires separate explicit researcher approval.
