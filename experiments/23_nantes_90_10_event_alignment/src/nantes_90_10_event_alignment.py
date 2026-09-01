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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


VARIANTS = ["raw_curve", "mask_only", "mask_gt_flag", "soft_pred_gate"]


def load_fair_module(path: Path):
    spec = importlib.util.spec_from_file_location("fair_event_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate(model, loader, device, phases: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            "class_id": class_id,
            "phase": phases[class_id],
            "support": int(support[class_id]),
            "correct": int(cm[class_id, class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
        }
        for class_id in labels
    ]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n_frames": len(y_true),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    predictions = [{"true": int(t), "pred": int(p)} for t, p in zip(y_true, y_pred)]
    return metrics, predictions


def plot_confusion(cm: np.ndarray, phases: list[str], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(phases)), phases, rotation=45, ha="right")
    ax.set_yticks(range(len(phases)), phases)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        item: dict[str, Any] = {}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            values = np.asarray([run[metric] for run in selected], dtype=np.float64)
            item[metric] = {"mean": float(values.mean()), "std": float(values.std()), "n": len(values)}
        cm = np.asarray([run["confusion_matrix"] for run in selected], dtype=np.int64).sum(axis=0)
        item["summed_confusion_matrix"] = cm.tolist()
        item["correct"] = int(np.trace(cm))
        item["total"] = int(cm.sum())
        item["per_class"] = []
        for class_id in range(cm.shape[0]):
            support = int(cm[class_id].sum())
            item["per_class"].append(
                {
                    "class_id": class_id,
                    "support": support,
                    "correct": int(cm[class_id, class_id]),
                    "recall": float(cm[class_id, class_id] / support) if support else None,
                }
            )
        result[variant] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair-script", type=Path, required=True)
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames-per-embryo", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    fair = load_fair_module(args.fair_script)
    base, labels, masks, embryo_ids, feature_names = fair.load_nantes_sequences(
        args.frame_features, args.frames_per_embryo
    )
    all_idx = np.arange(len(base))
    runs: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        train_idx, test_idx = train_test_split(all_idx, test_size=0.10, random_state=seed)
        train_ids = [embryo_ids[i] for i in train_idx]
        test_ids = [embryo_ids[i] for i in test_idx]
        split = {
            "seed": seed,
            "split_unit": "embryo",
            "train_fraction": len(train_idx) / len(all_idx),
            "test_fraction": len(test_idx) / len(all_idx),
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_hash": ids_hash(train_ids),
            "test_hash": ids_hash(test_ids),
        }
        write_json(args.out_dir / "splits" / f"seed_{seed}.json", split)

        soft_gate, fitted_gate, gate_audit = fair.make_oof_soft_gates(
            base, labels, masks, embryo_ids, train_idx, test_idx, seed
        )
        write_json(args.out_dir / "splits" / f"seed_{seed}_soft_gate.json", fitted_gate)
        write_json(args.out_dir / "splits" / f"seed_{seed}_soft_gate_audit.json", gate_audit)

        fair.seed_all(seed)
        template = fair.CurvePhasePretrainer(71, args.hidden, args.dropout)
        common_state = clone_state(template)
        common_hash = state_hash(common_state)

        active_y = labels[train_idx][masks[train_idx] > 0]
        counts = np.bincount(active_y, minlength=len(fair.PHASES)).astype(np.float32)
        class_weight = counts.sum() / np.maximum(counts, 1.0)
        class_weight = torch.from_numpy((class_weight / class_weight.mean()).astype(np.float32)).to(args.device)

        for variant in VARIANTS:
            fair.seed_all(seed)
            x, event_scalar = fair.variant_inputs(base, labels, masks, soft_gate, variant)
            model = fair.CurvePhasePretrainer(71, args.hidden, args.dropout).to(args.device)
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch_x, batch_gate, batch_y, batch_mask in train_loader:
                    logits = model(batch_x.to(args.device), batch_gate.to(args.device))
                    active = batch_mask.to(args.device) > 0
                    loss = F.cross_entropy(logits[active], batch_y.to(args.device)[active], weight=class_weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})

            metrics, predictions = evaluate(model, test_loader, args.device, list(fair.PHASES))
            run_dir = args.out_dir / "runs" / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            pd.DataFrame(predictions).to_csv(run_dir / "test_predictions.csv", index=False)
            write_json(run_dir / "test_metrics.json", metrics)
            plot_confusion(np.asarray(metrics["confusion_matrix"]), list(fair.PHASES), run_dir / "confusion.png")
            torch.save(
                {
                    "model": clone_state(model),
                    "variant": variant,
                    "seed": seed,
                    "epochs": args.epochs,
                    "common_initialization_hash": common_hash,
                    "split": split,
                    "feature_names": feature_names,
                },
                run_dir / "final.pt",
            )
            run = {
                "seed": seed,
                "variant": variant,
                "common_initialization_hash": common_hash,
                "train_embryos": len(train_idx),
                "test_embryos": len(test_idx),
                **metrics,
            }
            runs.append(run)
            print(
                f"seed={seed} variant={variant} acc={metrics['accuracy']:.4f} "
                f"balanced_acc={metrics['balanced_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )

    summary = {
        "task": "Nantes 16-phase event-alignment ablation with embryo-level 90/10 split",
        "n_embryos": len(embryo_ids),
        "train_test_protocol": "90% train, fixed epochs, one final evaluation on 10% held-out test",
        "checkpoint_selection": "final epoch; test set is not used for model selection",
        "frames_per_embryo": args.frames_per_embryo,
        "epochs": args.epochs,
        "variants": {
            "raw_curve": "unaligned baseline",
            "mask_only": "ground-truth-stage-derived biological mask; oracle alignment",
            "mask_gt_flag": "oracle mask plus ground-truth event flag; leakage upper bound",
            "soft_pred_gate": "deployable predicted-probability soft gate",
        },
        "aggregate": aggregate(runs),
        "per_seed": runs,
    }
    write_json(args.out_dir / "nantes_90_10_summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
