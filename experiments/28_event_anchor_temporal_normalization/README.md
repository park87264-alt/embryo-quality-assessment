# Event-anchor temporal normalization

This experiment separates validity gating from piecewise time normalization at tSB, tB, and tEB. The formal three-seed run uses 360 Nantes embryos with all anchors, embryo-level 90/10 splits, and evaluates every variant on the same original test frames after interpolating logits back from normalized time.

| Variant | Accuracy | Balanced accuracy | Macro-F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Uniform raw | 0.5935 | 0.4778 | 0.4413 | Baseline |
| Uniform GT gate | 0.6317 | 0.5096 | 0.4727 | Oracle validity gate |
| GT anchor raw | 0.6803 | 0.5452 | 0.5111 | Oracle time alignment |
| GT anchor + gate | 0.6609 | 0.5438 | 0.5050 | Oracle combination |
| Predicted anchor + gate | 0.5625 | 0.4644 | 0.4158 | Deployable variant, below baseline |

Means across seeds 42, 43, 44 are from the completed `formal_v2` run. The GT rows use true test events and **cannot** be cited as deployable gains. Test anchor error of roughly 3.5-5.6 hours is a current bottleneck. The two later-defined variants, predicted gate without alignment and predicted alignment without gating, do not yet have completed multi-seed results.

Run `src/event_anchor_temporal_normalization.py --help` for the original entry point. Data, private manifests, per-embryo predictions, and checkpoints are excluded from this public snapshot.
