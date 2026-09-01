# Experiment registry

This registry maps the original experiment chronology to the public code snapshot. Metrics are three-seed means unless stated otherwise. Gardner Macro-F1 values are listed as `Expansion / ICM / TE`.

Status meanings:

- **Entry present**: the primary experiment script is included; external data/features/checkpoints are still required.
- **Incomplete archive**: the local snapshot is missing helper or training files and is retained only for provenance.
- **Results only**: aggregate evidence is present but the corresponding source was not available locally.

| ID | Experiment and question | Primary entry point | Headline result | Status |
|---|---|---|---|---|
| 06 | ResNet18 F0 and multifocal Nantes baseline | `06_baseline_open_backbone/src/train_baseline.py` | Phase Macro-F1 approximately 0.83; weak ICM/TE auxiliary heads | Incomplete archive: `dataset.py` missing |
| 07 | Nantes stage + Kromp Gardner dual experts | `07_multitask_experts/src/train_dual_expert.py` | Phase F1 0.751; Gardner F1 0.601 / 0.438 / 0.392 | Incomplete archive: dataset/metric helpers missing |
| 08 | SFU ICM/TE/ZP segmentation expert | `08_sfu_segmentation_expert/src/train_segmentation.py` | Mean Dice 0.763; ICM 0.767, TE 0.707, ZP 0.817 | Incomplete archive: dataset/loss helpers missing |
| 09 | Segmentation-mask-assisted Gardner grading | `09_gardner_segassist/src/model.py` | Gardner F1 0.627 / 0.393 / 0.457 | Incomplete archive: training entry missing |
| 10 | ROI pooling with segmentation assistance | `10_gardner_roi_segassist/src/model.py` | Gardner F1 0.671 / 0.444 / 0.468 | Incomplete archive: training entry missing |
| 12 | Structure-only Gardner diagnostic | None in local snapshot | RF F1 0.386 / 0.311 / 0.297; geometry alone is insufficient | Results only |
| 13 | MedSAM-LoRA structure segmentation | `13_medsam_lora_structure/src/medsam_lora_sfu.py` | Mean Dice 0.903; structure-only grading remained weak | Entry present; requires MedSAM |
| 14 | MedSAM image + structure fusion | `14_gardner_medsam_fusion/src/train_gardner_medsam_fusion.py` | No consistent improvement over the image-only comparison | Entry present |
| 15 | Task-aware ROI gating | `15_task_aware_roi_gating/src/train_task_aware_roi_gating_v4.py` | Gating/concatenation gains were task-dependent, not uniformly positive | Entry present; ROI features required |
| 16 | Global/ROI/structure multibranch Gardner model | `16_full_multibranch_gardner/src/train_full_gardner_multibranch_v1.py` | ROI concat raised Expansion F1 from 0.442 to 0.510; temporal branch inactive | Entry present |
| 17 | Nantes pretraining transferred to Gardner | `17_nantes_pretrain_transfer/src/train_nantes_pretrain_transfer_gardner_v2.py` | Nantes-global F1 0.671 / 0.410 / 0.497; Expansion and TE improved over ImageNet | Entry present; current main backbone result |
| 18 | Nantes 32-point structure-curve pretraining | `18_nantes_structure_curve/src/nantes_structure_curve_pretrain.py` | 16-stage Accuracy 0.486, Balanced Acc 0.408, Macro-F1 0.382 | Entry present |
| 19 | Structure-curve encoder transferred to Kromp | `19_temporal_curve_transfer/src/train_temporal_curve_gardner_transfer.py` | Transfer evaluated, but Kromp token is static compatibility input rather than a trajectory | Entry present |
| 20 | Event-aligned structure curves | `20_event_aligned_curve/src/event_aligned_curve_experiment.py` | ICM/TE suppressed before blastocyst events; later audit found unfair checkpoint/flag effects | Entry present; superseded by 22/23 |
| 21 | Image/curve/structure task-gated MoE | `21_task_gated_moe/src/task_gated_moe_weak_nantes.py` | Event MoE F1 0.648 / 0.396 / 0.469; no uniform gain across heads | Entry present |
| 22 | Fair event-alignment ablation and checkpoint repair | `22_event_alignment_fair_ablation/src/fair_event_alignment_ablation.py` | Compared raw, mask-only, mask+GT flag, and predicted soft gate with matched splits/init | Entry and focused tests present |
| 23 | Nantes embryo-level 90/10 event experiment | `23_nantes_90_10_event_alignment/src/nantes_90_10_event_alignment.py` | Raw Acc/F1 0.387/0.294; oracle mask 0.509/0.397; soft gate 0.406/0.311 | Entry and focused tests present |
| 24 | Independent ICM texture and TE boundary experts + constrained MoE | `24_local_experts_constrained_moe/src/train_local_experts_constrained_moe.py` | ICM global/local/MoE F1 0.439/0.411/0.413; TE 0.417/0.401/0.409 | Entry present; negative result |
| 25 | Class-imbalance loss ablation | `25_class_imbalance_ablation/src/class_imbalance_ablation.py` | Balanced Softmax F1 0.336 vs CE 0.293, but Accuracy 0.464 vs 0.511 | Entry and focused tests present |
| 26 | Balanced Softmax calibration and cRT | `26_imbalance_calibration_crt/src/tune_imbalance_calibration.py` | Tuned BS Acc/F1 0.507/0.311; cRT 0.509/0.326 | Entry and focused tests present |
| 27 | Ten-sample Nantes Gardner inference utility | `27_nantes_gardner_inference/src/predict_nantes_gardner_samples.py` | Utility only; real images and per-embryo predictions excluded | Entry present; de-identified code only |

## Main paper narrative

The strongest coherent sequence is:

1. establish image and segmentation baselines (06-13);
2. test whether segmentation-derived structure and ROI features improve Gardner grading (09-16);
3. transfer embryo-domain pretraining from Nantes (17);
4. encode structural development curves and correct their biological timing with event alignment (18-23);
5. test independent local experts and MoE routing (21, 24);
6. address the remaining long-tail error with calibrated loss and two-stage classifier retraining (25-26).

Negative results are retained because they constrain the claims: good Dice does not guarantee good Gardner grading, Kromp does not provide temporal supervision, oracle flags overestimate deployable event gains, and weak local experts cannot be rescued by a forced MoE gate.

## Metric interpretation

- **Accuracy**: fraction of all samples classified correctly; dominated by common classes under imbalance.
- **Balanced accuracy**: mean recall across classes; each class contributes equally.
- **Macro-F1**: unweighted mean of per-class F1; sensitive to minority-class precision and recall.
- **Weighted-F1**: per-class F1 weighted by support; tends to track accuracy on imbalanced data.
- **Macro-AUC OvR**: mean one-vs-rest ranking AUC; can improve even when argmax predictions and F1 do not.
- **Dice / IoU**: overlap metrics for segmentation masks; they are not Gardner grading metrics.
