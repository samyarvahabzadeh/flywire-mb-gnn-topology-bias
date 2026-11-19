# Exploratory topology × learning rule screen

## Computational map

- **Variable topology.** The real or rewired directed adjacency of the entire induced KC/PN/MBON/DAN subgraph supplies the neighbors in both GraphSAGE layers. Both layers use unweighted incoming-neighbor max aggregation and a shared learned root transform. They ignore explicit self-loop edges, synapse counts and neurotransmitter sign when passing messages. Null-B still uses sign strata during rewiring.
- **Fixed sensory drive.** `scripts/phase4_pilot.py::load_real_cues` slices PN rows and KC columns from the real FlyWire CSR once per task seed. `real_cues.project_cues` divides the synapse count from active PNs by the count from all PNs to each KC. The real slice has 22,298 PN→KC edges and 366,946 synapses. The drive becomes the fifth node feature alongside four neuron-role indicators. The real PN→KC projection is shared across all conditions. PN→KC arcs can separately appear as rewired neighbors in the message-passing adjacency.
- **Output and weights.** A 5→128 encoder and two width-128 GraphSAGE-max layers produce node embeddings. A shared affine 128→1 head with sigmoid produces node scores. Fixed indices select MBONs and compute `(mean(avoidance) − mean(approach) + 1) / 2`. Adam/BCE in the baseline updates the encoder, both convolution layers and head, including their biases. MBON membership and the arithmetic pooling are fixed.
- **KC→MBON identity.** In the real graph, 26,937 physical KC→MBON pairs appear as directed adjacency entries attached to annotated neuron IDs. The nulls preserve node IDs and rewire adjacency. A shared GraphSAGE transform acts at these edges, so there is no learned per-synapse KC→MBON weight, compartment-specific dopamine or independently plastic KC→MBON transmission. The older input builder creates per-edge magnitudes, which GraphSAGE-max never reads.

## Arms

- **A.** The existing GraphSAGE-max `binary_step`, with Adam (0.001), mean BCE over all eight training cues, and the standard held-out probe. Continue training through the fixed horizon after a first criterion hit.
- **B.** Initialize the same model, freeze the encoder and both graph layers, and train the original shared affine head (128 weights and bias) with the same Adam/BCE. The fixed MBON pooling is shared with A.
- **C. Reward-modulated local readout.** Use the same frozen embeddings and the same initial scores as A/B. Keep the original affine head and bias fixed, and add a 128-coordinate offset `δ` to the approach-MBON head only. The fixed output valence is the same as in A/B. `R=1` denotes the external US following a train A+ cue and `R=0` denotes absence of US after a train A− cue. With `h_qm` the frozen embedding of approach node `m` for cue `q`, the forward-pairing depression rule is

  `δ[t+1] = clip(δ[t] − 0.001 × mean_{q,m}(R_q h_qm), −0.05, 0)`

  The mean covers eight cues and the fixed approach MBONs, including zero contributions from A−. The offset initializes at zero. C applies no backward pass. The perturbation cap uses the numerical 0.05 weight scale of Jiang et al. as an exploratory bound on the offset, in different units. The step size uses the baseline's numerical scale. Both are fixed before the smoke and shared by all topologies. Approach depression is the opponent mapping for this benchmark's positive-coded **avoidance** target. The same motif occurs in the published model, where a US of one valence preferentially activates DANs in compartments with an opponent-valence MBON.

  In Jiang & Litwin-Kumar (2021), Eq. 4 is `dw_ij/dt = trace(r_DAN_i) r_KC_j − trace(r_KC_j) r_DAN_i`, with separate low-pass eligibility traces and a lagged bounded synaptic weight at biological KC→MBON contacts. C uses the forward-pairing depression term in a cue-discrete approximation with transient within-cue eligibility. Plasticity is in tied feature-to-logit coefficients at approach MBONs. The original model learns recurrent DAN/MBON/feedback dynamics, uses both timing terms and an induction time constant, and applies plasticity at the synapses while keeping its readout fixed. The screen supplies a scalar external US, with discrete updates and no recurrent circuitry. B–C therefore compares two readout prescriptions, including the opponent-specific adaptation and the fixed bias in C.

## Frozen exploratory design

Task seeds 4100–4104 and engineering smoke seed 4199 are outside the previous sets. Null selection ranks IDs 00–29 by SHA-256 of `lr-screen-v1:<variant>:<id>` inside each variant, taking the first three. The selected archived graphs are A-14, A-17, A-24, B-11, B-19 and B-21. Selection uses graph identifiers only. Each seed reuses identical cue tables and initial parameters across graphs and arms. The binary task uses a common 2,000 updates/pair blocks, 16,000 training cues and 16,000 held-out cues per job. Each block contains four balanced pairs. Held-out scores are recorded after the update. The 32-trial smoke uses separate seed 4199 on real. TTC is the first full 20-trial rolling mean ≥0.75, with a flag for a miss by 2,000. AUC is the mean observed discrimination across the full common horizon. The reports retain all traces and show descriptive real-minus-matched-null contrasts by arm and variant.

## Sources

- Caron et al. (2013) describes the measured convergence of olfactory glomerular inputs to KCs, the sensory projection held fixed in this screen. [Nature](https://doi.org/10.1038/nature12063)
- Li et al. (2020) describes actual synaptic connectivity and feedback around MBONs and DANs. [eLife](https://doi.org/10.7554/eLife.62576)
- Aso et al. (2014) describes the anatomical MBON/DAN compartments and KC convergence. [eLife](https://doi.org/10.7554/eLife.04577)
- Jiang & Litwin-Kumar (2021), Eq. 4 and Figure 5, supplies the timing-dependent dopamine-gated rule and opponent mapping that motivate arm C. [PLOS Computational Biology](https://doi.org/10.1371/journal.pcbi.1009205)
