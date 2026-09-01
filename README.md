# Embryo Quality Assessment

Research code for embryo development-stage recognition, blastocyst structure segmentation, Gardner grading, event-aligned structure curves, task-gated mixture-of-experts (MoE), and long-tailed class calibration.

本仓库整理了胚胎质量评估项目从基础模型到事件对齐、MoE 和类别不平衡校准的完整实验链。代码仅用于科研，不构成临床诊断工具。

## Research pipeline

```text
Nantes stage/multifocal pretraining
              |
              v
SFU ICM/TE/ZP segmentation -> structure and ROI features
              |                         |
              +-------------------------+
                                        v
Kromp Gardner Expansion/ICM/TE grading
              |
              v
32-point structure curves -> event alignment -> task-gated MoE
              |
              v
class-imbalance ablation -> Balanced Softmax calibration / cRT
```

The datasets serve different roles. Nantes provides continuous developmental trajectories and stage supervision. SFU provides ICM/TE/ZP segmentation supervision. Kromp provides static blastocyst images with Gardner labels; it is not treated as a temporal dataset.

## Repository layout

```text
experiments/   Numbered experiment code, tests, and aggregate results
docs/          Experiment registry, data policy, and third-party provenance
results/       Cross-experiment publication figures and tables
```

Start with [the experiment registry](docs/experiment_registry.md). It records the question, entry point, result, and reproducibility status of every retained experiment. Early experiments with missing helper files are kept for provenance and are explicitly marked incomplete.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

MedSAM experiments additionally require an upstream MedSAM checkout and an authorized checkpoint. Data and checkpoints are not distributed. Replace `/path/to/...` defaults or pass the corresponding CLI arguments explicitly.

## Reproducibility notes

- Splits are embryo-level wherever embryo identifiers are available.
- Formal comparisons use seeds 42, 43, and 44 unless noted otherwise.
- Macro-F1 and balanced accuracy are primary under class imbalance; raw accuracy is reported alongside them.
- Ground-truth event flags are oracle inputs and must not be presented as deployable performance.
- Kromp static samples are never claimed to provide continuous temporal trajectories.

## Privacy and release status

No embryo images, patient identifiers, private manifests, prediction contact sheets, server credentials, or model checkpoints are included. See [data access](docs/data_access.md) before adding any local file.

This is a research snapshot. No open-source license has been granted yet; third-party components retain their own licenses.
