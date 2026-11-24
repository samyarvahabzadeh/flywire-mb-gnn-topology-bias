# Exploratory graded-cue-similarity result

**Date:** 2026-11-24  
**Status:** exploratory, descriptive; not part of the closed confirmatory study.

## Question

In the reduced KCγ→MBON11 model, does explicit experience with an unreinforced safe cue alter held-out family discrimination differently across learning rules and modeled cue similarity? Does that schedule effect depend on the real KC partner mask compared with fixed-capacity input-mask controls?

## Run and integrity

- Molab device: NVIDIA RTX PRO 6000 Blackwell Server Edition; CUDA available; total VRAM 101,975,851,008 bytes.
- Complete run: 64 cue-layout seeds (8301–8364), 13 masks (real plus 12 fixed-capacity input-mask controls), 5 training overlap levels, 3 learning rules, 2 training schedules, both reciprocal assignments, and 5 held-out probe overlap levels.
- Total verified rows: 249,600 in 65 Parquet shards. Each shard was SHA-256 checked against the run manifest and read back byte-for-byte from private Drive.
- End-to-end Molab time through Drive readback: 519.9 s (8.7 min). Summed per-shard GPU compute time: 64.11 s; mean shard time: 0.986 s. Peak Torch GPU allocation: 13,839,872 bytes. The card had ample VRAM, but this compact model did not need to fill it.
- Training used 48 paired cycles. Held-out probes used 20 fresh cue patterns per class and overlap level. PN-set overlap is a model manipulation, not a measured odor distance.
- Run commit: `9894ac032cb310797eb0eff7d0b5e6a9dda4654b`.
- Config SHA-256: `771aa0f6fd7e1d2551a115e0b5a76a2a55890f6d2eb82f2340779bccd12fb058`.
- Runner SHA-256: `8765760f1ae6111274f0f64df58e9f0d404b4f1df8226d05a69715c299ea60fb`.
- Analysis script SHA-256: `69ed65ce1d8fc06f5df02f422c6b722f1ecbf178d948e5b6953da9bc17fbbd98`.
- Input run manifest SHA-256: `d431b7dc9da2398c5035ea2742edd1cda4145f887143a8884e6a552795883c4f`.
- Analysis manifest SHA-256: `2187dbeba80820eee4574ffc35c794d7befb1dbdb23a7adab215acb65e093d6c`.
- Final analysis archive SHA-256: `c4bc485d7222faff9cbb22b7bd4c57a01c2c7ae5bf789f65b397e54fc895a033`.

## Main descriptive pattern

Outcome is the differential-minus-absolute change in held-out discrimination, averaged over five held-out probe-overlap levels within each cue-layout seed. The intervals below are 5,000-draw percentile bootstrap intervals over the 64 simulated cue layouts; they do not represent biological replication.

| CS+/CS− training overlap | Real mask: prediction-error schedule effect [95% layout interval] | Median control-mask effect |
|---:|---:|---:|
| 0/40 | 0.0629 [0.0609, 0.0649] | 0.0635 |
| 10/40 | 0.0667 [0.0646, 0.0689] | 0.0669 |
| 20/40 | 0.0582 [0.0564, 0.0601] | 0.0582 |
| 30/40 | 0.0367 [0.0348, 0.0386] | 0.0364 |
| 36/40 | 0.0157 [0.0146, 0.0169] | 0.0157 |

At the same overlap levels, the largest absolute Adam/BCE effect was 0.0017, while the pulse-only schedule effect was exactly 0 at every cell (required software sanity check). Prediction-error minus pulse and prediction-error minus Adam/BCE paired layout intervals are in the linked CSV.

The rule pattern is large relative to the real-mask-versus-control differences. Across the 75 rule × training-overlap × probe-overlap cell means, real-minus-control values ranged from −0.000966 to +0.000393. The lower and upper endpoints are post-run descriptive extremes, not selected hypotheses. The real mask therefore did not show a practically distinct benefit in this assay.

## Cognitive interpretation and limits

The reduced model predicts that an omission-sensitive update can increase held-out discrimination after differential rather than absolute training, but the modeled effect is greatest at low-to-moderate PN overlap (peak at 10/40) and declines substantially at high overlap. It is not monotonic over the first 0-to-10 step. All twelve fixed-capacity input masks show nearly the same learning-rule pattern, so this does **not** support the claim that real KCγ→MBON11 partner identity causes the effect or is optimal.

The broad behavioral benefit of differential conditioning with similar odors is already in the literature. This computation does not test flies, and PN identity overlap is not calibrated chemical similarity. The model also omits antennal-lobe inhibition and is a reduced local KC→MBON11 readout rather than a complete mushroom-body circuit. Treat the result as a computational candidate mechanism and a constraint for follow-up: calibrate cue similarity against measured olfactory representations and test whether adding upstream inhibitory decorrelation changes the predicted generalization gradient.

No confirmatory p-values were calculated. No confirmatory data or frozen preregistration files were changed.

## Artifacts

- [Private Drive results folder](https://drive.google.com/drive/folders/1y6QuKjF91dvwuXuBuR4cb7if9xjAk6Yp)
- [Private Drive analysis folder](https://drive.google.com/drive/folders/1vm0mdE1GhxwttoFOv2Gn0okd7aHAoOj1)
- [Analysis archive ZIP](https://drive.google.com/file/d/15-s3BY0-HFcbHQzgfcIXbfjOVCIleErx/view?usp=drivesdk)
- [Similarity surface figure](https://drive.google.com/file/d/1_g9a9UO5O4sN9UVxNqn-eidn_m0ZUCpi/view?usp=drivesdk)
- [Rule-by-similarity figure](https://drive.google.com/file/d/1vpqOprOZNE0zk52CV9uF3CZd-M9Sz_Pq/view?usp=drivesdk)
- [Probe-averaged rule summary CSV](https://drive.google.com/file/d/1aSQXVrgItNGEVPj6OtizFmEQZ5iPDWxH/view?usp=drivesdk)
- [Paired rule differences CSV](https://drive.google.com/file/d/1Nen3JVKo7-TW-tU2e_yA2ndfry41-QGU/view?usp=drivesdk)
