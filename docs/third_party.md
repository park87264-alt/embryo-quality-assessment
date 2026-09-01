# Third-party provenance

This repository does not vendor third-party repositories or weights. Their licenses and access terms must be checked independently.

| Resource | Use in this project | Recorded upstream revision |
|---|---|---|
| [Blastocyst Dataset](https://github.com/software-competence-center-hagenberg/Blastocyst-Dataset) | Kromp/Gardner data source and label format | `d2acd672e708bdf6a2e52bf527e6e9419a01ee6a` |
| [MedSAM](https://github.com/bowang-lab/MedSAM) | Medical SAM encoder and LoRA segmentation experiments | `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488` |
| [SAM 2](https://github.com/facebookresearch/sam2) | Investigated segmentation foundation-model alternative | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| [CoSTeM](https://github.com/RIL-Lab/CoSTeM) | Morphology/temporal fusion and expert-routing inspiration | `06e7d0fadc19ff8c394a4690a49c0c1c87b282fe` |
| [EmbryoDiff](https://github.com/RIL-Lab/EmbryoDiff) | Multifocal and temporal-prior inspiration | `112644f9961c67f51932076b64dd753ce0969454` |
| FEMI, *A foundational model for in vitro fertilization* | Domain-pretraining and downstream-task design inspiration | Paper only |

Important distinctions:

- FEMI weights were not obtained or used; results must not be described as FEMI-based.
- CoSTeM and EmbryoDiff were consulted for method ideas; their full training code was not copied into these experiments.
- MedSAM/SAM2 checkpoints are not included.
- The project's task heads, event-alignment ablations, constrained MoE, and imbalance calibration scripts are local implementations.

Long-tail experiments additionally follow these published methods:

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019.
- Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*, NeurIPS 2020.
