from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, accuracy_score

from dataset import NantesEmbryoDataset, PHASES
from models import OpenBackboneMultiTask


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def masked_ce(logits, targets, mask):
    valid = mask > 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[valid], targets[valid])


def masked_bce(logits, targets, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def batch_basic_counts(out, phase_targets, phase_mask, te_target, te_mask, icm_target, icm_mask):
    probs = torch.sigmoid(out["phase_logits"])
    pred = (probs >= 0.5).float()
    valid = phase_mask > 0
    phase_correct = ((pred == phase_targets) * valid).sum().item()
    phase_total = valid.sum().item()

    def cls_counts(logits, targets, mask):
        valid_cls = mask > 0
        if valid_cls.sum() == 0:
            return 0, 0
        pred_cls = logits.argmax(dim=1)
        return (pred_cls[valid_cls] == targets[valid_cls]).sum().item(), valid_cls.sum().item()

    te_correct, te_total = cls_counts(out["te_logits"], te_target, te_mask)
    icm_correct, icm_total = cls_counts(out["icm_logits"], icm_target, icm_mask)
    return phase_correct, phase_total, te_correct, te_total, icm_correct, icm_total


def update_prediction_store(store, batch, out):
    store["embryo_id"].extend(batch["embryo_id"])
    store["phase_prob"].append(torch.sigmoid(out["phase_logits"]).detach().cpu())
    store["phase_target"].append(batch["phase_targets"].detach().cpu())
    store["phase_mask"].append(batch["phase_mask"].detach().cpu())
    store["te_prob"].append(torch.softmax(out["te_logits"], dim=1).detach().cpu())
    store["te_target"].append(batch["te_target"].detach().cpu())
    store["te_mask"].append(batch["te_mask"].detach().cpu())
    store["icm_prob"].append(torch.softmax(out["icm_logits"], dim=1).detach().cpu())
    store["icm_target"].append(batch["icm_target"].detach().cpu())
    store["icm_mask"].append(batch["icm_mask"].detach().cpu())


def safe_auc(y_true, y_score):
    try:
        if len(set(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def safe_multiclass_auc(y_true, prob, n_classes=3):
    try:
        if len(set(y_true)) < 2:
            return None
        labels = list(range(n_classes))
        return float(roc_auc_score(y_true, prob, labels=labels, multi_class="ovr", average="macro"))
    except Exception:
        return None


def compute_detailed_metrics(store):
    phase_prob = torch.cat(store["phase_prob"], dim=0).numpy()
    phase_target = torch.cat(store["phase_target"], dim=0).numpy()
    phase_mask = torch.cat(store["phase_mask"], dim=0).numpy()
    phase_pred = (phase_prob >= 0.5).astype(np.int64)
    phase_target_i = phase_target.astype(np.int64)

    per_phase = {}
    f1_values = []
    auc_values = []
    for i, phase in enumerate(PHASES):
        valid = phase_mask[:, i] > 0
        y = phase_target_i[valid, i]
        p = phase_pred[valid, i]
        s = phase_prob[valid, i]
        if valid.sum() == 0:
            per_phase[phase] = {"n": 0, "positive": 0, "accuracy": None, "f1": None, "auc": None}
            continue
        f1 = float(f1_score(y, p, zero_division=0))
        auc = safe_auc(y.tolist(), s.tolist())
        acc = float(accuracy_score(y, p))
        per_phase[phase] = {"n": int(valid.sum()), "positive": int(y.sum()), "accuracy": acc, "f1": f1, "auc": auc}
        f1_values.append(f1)
        if auc is not None:
            auc_values.append(auc)

    def cls_metrics(prefix):
        prob = torch.cat(store[f"{prefix}_prob"], dim=0).numpy()
        target = torch.cat(store[f"{prefix}_target"], dim=0).numpy()
        mask = torch.cat(store[f"{prefix}_mask"], dim=0).numpy() > 0
        if mask.sum() == 0:
            return {"n": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None, "macro_auc_ovr": None}
        y = target[mask]
        pred = prob[mask].argmax(axis=1)
        return {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "macro_auc_ovr": safe_multiclass_auc(y.tolist(), prob[mask], n_classes=3),
        }

    return {
        "phase_macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "phase_macro_auc": float(np.mean(auc_values)) if auc_values else None,
        "per_phase": per_phase,
        "te": cls_metrics("te"),
        "icm": cls_metrics("icm"),
    }


def save_predictions(store, out_path: Path):
    phase_prob = torch.cat(store["phase_prob"], dim=0).numpy()
    phase_target = torch.cat(store["phase_target"], dim=0).numpy()
    te_prob = torch.cat(store["te_prob"], dim=0).numpy()
    te_target = torch.cat(store["te_target"], dim=0).numpy()
    te_mask = torch.cat(store["te_mask"], dim=0).numpy()
    icm_prob = torch.cat(store["icm_prob"], dim=0).numpy()
    icm_target = torch.cat(store["icm_target"], dim=0).numpy()
    icm_mask = torch.cat(store["icm_mask"], dim=0).numpy()

    header = ["embryo_id"]
    for phase in PHASES:
        header.extend([f"{phase}_target", f"{phase}_prob"])
    header.extend(["te_target", "te_mask", "te_prob_A", "te_prob_B", "te_prob_C", "icm_target", "icm_mask", "icm_prob_A", "icm_prob_B", "icm_prob_C"])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, eid in enumerate(store["embryo_id"]):
            row = [eid]
            for j in range(len(PHASES)):
                row.extend([int(phase_target[i, j]), float(phase_prob[i, j])])
            row.extend([int(te_target[i]), float(te_mask[i]), *[float(x) for x in te_prob[i]], int(icm_target[i]), float(icm_mask[i]), *[float(x) for x in icm_prob[i]]])
            writer.writerow(row)


def run_epoch(model, loader, optimizer, device, train=True, collect_predictions=False, phase_weight=1.0, te_weight=0.3, icm_weight=0.3):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    stats = {"phase_correct": 0, "phase_total": 0, "te_correct": 0, "te_total": 0, "icm_correct": 0, "icm_total": 0}
    store = {"embryo_id": [], "phase_prob": [], "phase_target": [], "phase_mask": [], "te_prob": [], "te_target": [], "te_mask": [], "icm_prob": [], "icm_target": [], "icm_mask": []}

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        images = batch["images"].to(device)
        phase_targets = batch["phase_targets"].to(device)
        phase_mask = batch["phase_mask"].to(device)
        te_target = batch["te_target"].to(device)
        te_mask = batch["te_mask"].to(device)
        icm_target = batch["icm_target"].to(device)
        icm_mask = batch["icm_mask"].to(device)

        with torch.set_grad_enabled(train):
            out = model(images)
            phase_loss = masked_bce(out["phase_logits"], phase_targets, phase_mask)
            te_loss = masked_ce(out["te_logits"], te_target, te_mask)
            icm_loss = masked_ce(out["icm_logits"], icm_target, icm_mask)
            loss = phase_weight * phase_loss + te_weight * te_loss + icm_weight * icm_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1
        pc, pt, tc, tt, ic, it = batch_basic_counts(out, phase_targets, phase_mask, te_target, te_mask, icm_target, icm_mask)
        stats["phase_correct"] += pc; stats["phase_total"] += pt
        stats["te_correct"] += tc; stats["te_total"] += tt
        stats["icm_correct"] += ic; stats["icm_total"] += it
        if collect_predictions:
            cpu_batch = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
            update_prediction_store(store, cpu_batch, out)
        iterator.set_description(f"loss={total_loss/max(1,n_batches):.4f}")

    metrics = {
        "loss": total_loss / max(1, n_batches),
        "phase_acc": stats["phase_correct"] / max(1, stats["phase_total"]),
        "te_acc": stats["te_correct"] / max(1, stats["te_total"]),
        "icm_acc": stats["icm_correct"] / max(1, stats["icm_total"]),
        "te_n": stats["te_total"],
        "icm_n": stats["icm_total"],
    }
    return metrics, store if collect_predictions else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/path/to/embryo_data/05_Metadata_Manifests/nantes_16phase_manifest.csv")
    p.add_argument("--splits-dir", default="/path/to/embryo_data/06_Baseline_OpenBackbone/splits")
    p.add_argument("--output-dir", default="outputs/06_baseline_open_backbone/run")
    p.add_argument("--focals", default="F0")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    focals = [x.strip() for x in args.focals.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {}
    loaders = {}
    for split, train in [("train", True), ("val", False), ("test", False)]:
        ds = NantesEmbryoDataset(
            manifest_path=args.manifest,
            split_file=Path(args.splits_dir) / f"{split}.txt",
            focal_keys=focals,
            processed=True,
            image_size=args.image_size,
            train=train,
            frame_mode="random" if train else "middle",
            max_samples=args.max_samples if split == "train" else None,
        )
        datasets[split] = ds
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers, pin_memory=True)

    model = OpenBackboneMultiTask(args.backbone, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config = vars(args).copy()
    config.update({"device": str(device), "focals_list": focals, "dataset_sizes": {k: len(v) for k, v in datasets.items()}})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_val = math.inf
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, loaders["train"], optimizer, device, train=True)
        val_metrics, _ = run_epoch(model, loaders["val"], optimizer, device, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val": val_metrics}, out_dir / "best.pt")

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_store = run_epoch(model, loaders["test"], optimizer, device, train=False, collect_predictions=True)
    detailed = compute_detailed_metrics(test_store)
    result = {"best_epoch": best_epoch, "best_val_loss": best_val, **test_metrics, **detailed}
    (out_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_predictions(test_store, out_dir / "test_predictions.csv")

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    print("TEST", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
