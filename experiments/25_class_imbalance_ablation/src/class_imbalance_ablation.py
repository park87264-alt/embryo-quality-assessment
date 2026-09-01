from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


LOSS_VARIANTS = [
    "ce",
    "inverse_freq_ce",
    "effective_num_ce",
    "class_balanced_focal",
    "balanced_softmax",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def class_counts(labels: np.ndarray, masks: np.ndarray, indices: np.ndarray, n_classes: int) -> np.ndarray:
    active = labels[indices][masks[indices] > 0]
    return np.bincount(active, minlength=n_classes).astype(np.float64)


def inverse_frequency_weights(counts: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    weights[present] = counts[present].sum() / counts[present]
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def effective_number_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Class-balanced weights from Cui et al., CVPR 2019."""
    weights = np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    effective_number = 1.0 - np.power(beta, counts[present])
    weights[present] = (1.0 - beta) / np.maximum(effective_number, 1e-12)
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = (1.0 - pt).pow(gamma) * ce
    if alpha is not None:
        loss = loss * alpha[targets]
    return loss.mean()


def imbalance_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    variant: str,
    counts: torch.Tensor,
    inverse_weights: torch.Tensor,
    effective_weights: torch.Tensor,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if variant == "ce":
        return F.cross_entropy(logits, targets)
    if variant == "inverse_freq_ce":
        return F.cross_entropy(logits, targets, weight=inverse_weights)
    if variant == "effective_num_ce":
        return F.cross_entropy(logits, targets, weight=effective_weights)
    if variant == "class_balanced_focal":
        return focal_loss(logits, targets, effective_weights, focal_gamma)
    if variant == "balanced_softmax":
        # Balanced Softmax (Ren et al., NeurIPS 2020): training-time prior correction.
        adjusted_logits = logits + counts.clamp_min(1.0).log().unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets)
    raise ValueError(f"Unknown loss variant: {variant}")


def support_groups(counts: np.ndarray) -> dict[str, list[int]]:
    """Split present classes into head/medium/tail thirds by training support."""
    present = np.flatnonzero(counts > 0)
    ordered = present[np.argsort(counts[present])]
    chunks = np.array_split(ordered, 3)
    return {
        "tail": [int(x) for x in chunks[0]],
        "medium": [int(x) for x in chunks[1]],
        "head": [int(x) for x in chunks[2]],
    }


def mean_for_supported(values: np.ndarray, support: np.ndarray, ids: list[int]) -> float | None:
    valid = [idx for idx in ids if support[idx] > 0]
    return float(np.mean(values[valid])) if valid else None


def evaluate(model, loader, device: torch.device, phases: list[str], train_counts: np.ndarray):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, gate, y, mask in loader:
            pred = model(x.to(device), gate.to(device)).argmax(-1).cpu()
            active = mask > 0
            y_true.extend(y[active].numpy().tolist())
            y_pred.extend(pred[active].numpy().tolist())

    labels = list(range(len(phases)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    supported = support > 0
    groups = support_groups(train_counts)
    metrics = {
        "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall[supported].mean()) if supported.any() else 0.0,
        "macro_f1_all_16": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1_supported": float(per_f1[supported].mean()) if supported.any() else 0.0,
        "tail_recall": mean_for_supported(recall, support, groups["tail"]),
        "tail_f1": mean_for_supported(per_f1, support, groups["tail"]),
        "medium_recall": mean_for_supported(recall, support, groups["medium"]),
        "head_recall": mean_for_supported(recall, support, groups["head"]),
        "n_frames": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "class_id": idx,
                "phase": phases[idx],
                "train_support": int(train_counts[idx]),
                "test_support": int(support[idx]),
                "correct": int(cm[idx, idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(per_f1[idx]),
            }
            for idx in labels
        ],
        "support_groups": groups,
    }
    predictions = pd.DataFrame({"true": y_true, "pred": y_pred})
    return metrics, predictions


def plot_distribution(counts: np.ndarray, phases: list[str], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(phases, counts, color="#4472C4")
    ax.set_ylabel("Training frames")
    ax.set_title("Nantes 16-phase training distribution")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, phases: list[str], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(phases)), phases, rotation=45, ha="right")
    ax.set_yticks(range(len(phases)), phases)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.set_title("Row-normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1_all_16",
        "macro_f1_supported",
        "tail_recall",
        "tail_f1",
        "medium_recall",
        "head_recall",
    ]
    for variant in LOSS_VARIANTS:
        selected = [run for run in runs if run["loss_variant"] == variant]
        result: dict[str, Any] = {}
        for metric in metric_names:
            values = np.asarray(
                [run[metric] for run in selected if run.get(metric) is not None], dtype=np.float64
            )
            result[metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "n": int(len(values)),
            }
        result["summed_confusion_matrix"] = np.asarray(
            [run["confusion_matrix"] for run in selected], dtype=np.int64
        ).sum(axis=0).tolist()
        output[variant] = result
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair class-imbalance ablation for Nantes 16-phase event-aligned curves"
    )
    parser.add_argument("--fair-script", type=Path, required=True)
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--loss-variants", nargs="+", choices=LOSS_VARIANTS, default=LOSS_VARIANTS)
    parser.add_argument("--input-variant", choices=["raw_curve", "mask_only", "mask_gt_flag", "soft_pred_gate"], default="soft_pred_gate")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames-per-embryo", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--effective-beta", type=float, default=0.999)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fair = load_module(args.fair_script, "fair_event_alignment")
    base, labels, masks, embryo_ids, feature_names = fair.load_nantes_sequences(
        args.frame_features, args.frames_per_embryo
    )
    all_idx = np.arange(len(base))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        train_idx, test_idx = train_test_split(all_idx, test_size=0.10, random_state=seed)
        counts_np = class_counts(labels, masks, train_idx, len(fair.PHASES))
        inverse_np = inverse_frequency_weights(counts_np)
        effective_np = effective_number_weights(counts_np, args.effective_beta)
        groups = support_groups(counts_np)
        distribution = {
            "seed": seed,
            "train_embryos": int(len(train_idx)),
            "test_embryos": int(len(test_idx)),
            "class_counts": {fair.PHASES[i]: int(counts_np[i]) for i in range(len(fair.PHASES))},
            "inverse_frequency_weights": {fair.PHASES[i]: float(inverse_np[i]) for i in range(len(fair.PHASES))},
            "effective_number_weights": {fair.PHASES[i]: float(effective_np[i]) for i in range(len(fair.PHASES))},
            "support_groups": {name: [fair.PHASES[i] for i in ids] for name, ids in groups.items()},
        }
        write_json(args.out_dir / "splits" / f"seed_{seed}_distribution.json", distribution)
        plot_distribution(counts_np, list(fair.PHASES), args.out_dir / "splits" / f"seed_{seed}_distribution.png")

        soft_gate, gate_state, gate_audit = fair.make_oof_soft_gates(
            base, labels, masks, embryo_ids, train_idx, test_idx, seed
        )
        write_json(args.out_dir / "splits" / f"seed_{seed}_soft_gate.json", gate_state)
        write_json(args.out_dir / "splits" / f"seed_{seed}_soft_gate_audit.json", gate_audit)
        x, event_scalar = fair.variant_inputs(base, labels, masks, soft_gate, args.input_variant)

        fair.seed_all(seed)
        template = fair.CurvePhasePretrainer(71, args.hidden, args.dropout)
        common_state = clone_state(template)
        common_hash = state_hash(common_state)
        counts = torch.from_numpy(counts_np.astype(np.float32)).to(device)
        inverse = torch.from_numpy(inverse_np).to(device)
        effective = torch.from_numpy(effective_np).to(device)

        for loss_variant in args.loss_variants:
            fair.seed_all(seed)
            model = fair.CurvePhasePretrainer(71, args.hidden, args.dropout).to(device)
            model.load_state_dict(common_state, strict=True)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                fair.CurveDataset(x, event_scalar, labels, masks, train_idx),
                batch_size=args.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
            test_loader = DataLoader(
                fair.CurveDataset(x, event_scalar, labels, masks, test_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            history: list[dict[str, float]] = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses: list[float] = []
                for batch_x, batch_gate, batch_y, batch_mask in train_loader:
                    logits = model(batch_x.to(device), batch_gate.to(device))
                    target = batch_y.to(device)
                    active = batch_mask.to(device) > 0
                    loss = imbalance_loss(
                        logits[active],
                        target[active],
                        loss_variant,
                        counts,
                        inverse,
                        effective,
                        args.focal_gamma,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})

            metrics, predictions = evaluate(
                model, test_loader, device, list(fair.PHASES), counts_np
            )
            run_dir = args.out_dir / "runs" / loss_variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            predictions.to_csv(run_dir / "test_predictions.csv", index=False)
            write_json(run_dir / "test_metrics.json", metrics)
            plot_confusion(
                np.asarray(metrics["confusion_matrix"]),
                list(fair.PHASES),
                run_dir / "confusion_normalized.png",
            )
            torch.save(
                {
                    "model": clone_state(model),
                    "seed": seed,
                    "loss_variant": loss_variant,
                    "input_variant": args.input_variant,
                    "common_initialization_hash": common_hash,
                    "class_counts": counts_np,
                    "feature_names": feature_names,
                },
                run_dir / "final.pt",
            )
            run = {
                "seed": seed,
                "loss_variant": loss_variant,
                "input_variant": args.input_variant,
                "common_initialization_hash": common_hash,
                **metrics,
            }
            runs.append(run)
            print(
                f"seed={seed} loss={loss_variant} acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1_all_16']:.4f} "
                f"tail_recall={metrics['tail_recall']}",
                flush=True,
            )

    summary = {
        "task": "Nantes 16-phase class-imbalance fair ablation",
        "input_variant": args.input_variant,
        "split": "embryo-level 90/10; fixed epochs; held-out test evaluated once",
        "fairness_controls": [
            "same embryo split within each seed",
            "same model initialization within each seed",
            "same data order within each seed",
            "only the training loss changes",
        ],
        "loss_variants": {
            "ce": "ordinary cross entropy without class correction",
            "inverse_freq_ce": "existing project approach: inverse-frequency weighted cross entropy",
            "effective_num_ce": "class-balanced loss using effective sample number (Cui et al., 2019)",
            "class_balanced_focal": "effective-number alpha plus focal hard-example modulation",
            "balanced_softmax": "training-time class-prior correction (Ren et al., 2020)",
        },
        "selection_rule": (
            "Prefer a method that improves macro-F1, balanced accuracy and tail recall without a material "
            "accuracy decrease; do not select by test performance in a final paper protocol."
        ),
        "aggregate": aggregate(runs),
        "per_seed": runs,
    }
    write_json(args.out_dir / "class_imbalance_summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
