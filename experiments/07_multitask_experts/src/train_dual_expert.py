from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from datasets import KrompGardnerDataset, NantesStageDataset, PHASES
from metrics import class_metrics, masked_bce_loss, masked_ce_loss, phase_metrics
from models import DualExpertEmbryoModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def read_yaml(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_phase_pos_weight(dataset: NantesStageDataset) -> torch.Tensor:
    positives = torch.zeros(len(PHASES), dtype=torch.float32)
    valid = torch.zeros(len(PHASES), dtype=torch.float32)
    for row in dataset.rows:
        for i, phase in enumerate(PHASES):
            value = row.get(f"{phase}_present", "")
            if value != "":
                valid[i] += 1
                positives[i] += float(value)
    negatives = valid - positives
    return negatives / positives.clamp_min(1.0)


def collect_class_weights(dataset: KrompGardnerDataset, key: str, n_classes: int) -> torch.Tensor:
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for row in dataset.rows:
        value = (row.get(key, "") or "").strip()
        if value not in {"", "ND", "NA", "nan"}:
            idx = int(value)
            if 0 <= idx < n_classes:
                counts[idx] += 1
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-6)


def make_loaders(cfg, seed: int):
    data = cfg["data"]
    train_cfg = cfg["train"]
    stage_train = NantesStageDataset(
        data["nantes_manifest"], data["nantes_train_split"],
        focal_keys=data["focal_keys"], image_size=data["image_size"], train=True,
        max_samples=train_cfg.get("max_train_samples"),
    )
    stage_val = NantesStageDataset(
        data["nantes_manifest"], data["nantes_val_split"],
        focal_keys=data["focal_keys"], image_size=data["image_size"], train=False,
        max_samples=train_cfg.get("max_val_samples"),
    )
    gardner_train = KrompGardnerDataset(
        data["kromp_train_csv"], data["kromp_image_dir"], "silver",
        image_size=data["image_size"], train=True,
        max_samples=train_cfg.get("max_train_samples"),
    )
    gardner_val = KrompGardnerDataset(
        data["kromp_gold_csv"], data["kromp_image_dir"], "gold",
        image_size=data["image_size"], train=False,
        max_samples=train_cfg.get("max_val_samples"),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    stage_train_loader = DataLoader(
        stage_train, batch_size=train_cfg["batch_size_stage"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, generator=generator,
    )
    gardner_train_loader = DataLoader(
        gardner_train, batch_size=train_cfg["batch_size_gardner"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, generator=generator,
    )
    stage_val_loader = DataLoader(
        stage_val, batch_size=train_cfg["batch_size_stage"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )
    gardner_val_loader = DataLoader(
        gardner_val, batch_size=train_cfg["batch_size_gardner"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )
    return {
        "stage_train": stage_train_loader,
        "gardner_train": gardner_train_loader,
        "stage_val": stage_val_loader,
        "gardner_val": gardner_val_loader,
        "stage_train_ds": stage_train,
        "gardner_train_ds": gardner_train,
    }


def evaluate(model, stage_loader, gardner_loader, device):
    model.eval()
    stage_logits, stage_targets, stage_masks = [], [], []
    gardner = {k: [] for k in [
        "exp_logits", "exp_targets", "exp_masks",
        "icm_logits", "icm_targets", "icm_masks",
        "te_logits", "te_targets", "te_masks",
    ]}
    with torch.no_grad():
        for batch in stage_loader:
            batch = to_device(batch, device)
            out = model(batch["images"], "stage")
            stage_logits.append(out["phase_logits"].cpu().numpy())
            stage_targets.append(batch["phase_targets"].cpu().numpy())
            stage_masks.append(batch["phase_mask"].cpu().numpy())
        for batch in gardner_loader:
            batch = to_device(batch, device)
            out = model(batch["images"], "gardner")
            for head in ["exp", "icm", "te"]:
                gardner[f"{head}_logits"].append(out[f"{head}_logits"].cpu().numpy())
                gardner[f"{head}_targets"].append(batch[f"{head}_target"].cpu().numpy())
                gardner[f"{head}_masks"].append(batch[f"{head}_mask"].cpu().numpy())

    metrics = {}
    metrics.update(phase_metrics(
        np.concatenate(stage_logits), np.concatenate(stage_targets),
        np.concatenate(stage_masks), PHASES,
    ))
    for head in ["exp", "icm", "te"]:
        metrics.update(class_metrics(
            np.concatenate(gardner[f"{head}_logits"]),
            np.concatenate(gardner[f"{head}_targets"]),
            np.concatenate(gardner[f"{head}_masks"]),
            f"gardner_{head}",
        ))
    return metrics


def train_one(config_path: str, output_dir: str, seed: int):
    cfg = read_yaml(config_path)
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(cfg, seed)
    model = DualExpertEmbryoModel(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    stage_pos_weight = None
    exp_weights = icm_weights = te_weights = None
    if cfg["train"].get("use_class_weights", True):
        stage_pos_weight = collect_phase_pos_weight(loaders["stage_train_ds"]).to(device)
        exp_weights = collect_class_weights(loaders["gardner_train_ds"], "EXP_silver", 5).to(device)
        icm_weights = collect_class_weights(loaders["gardner_train_ds"], "ICM_silver", 3).to(device)
        te_weights = collect_class_weights(loaders["gardner_train_ds"], "TE_silver", 3).to(device)

    history_path = out_dir / "history.csv"
    best_score = -math.inf
    task_weights = cfg["train"].get("task_loss_weights", {"stage": 1.0, "gardner": 1.0})
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "stage_loss", "gardner_loss",
            "phase_macro_f1", "phase_macro_auc", "gardner_exp_macro_f1",
            "gardner_icm_macro_f1", "gardner_te_macro_f1", "lr",
        ])
        writer.writeheader()
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            model.train()
            total_loss = total_stage = total_gardner = 0.0
            steps = max(len(loaders["stage_train"]), len(loaders["gardner_train"]))
            pbar = tqdm(
                zip(cycle(loaders["stage_train"]), cycle(loaders["gardner_train"])),
                total=steps, desc=f"epoch {epoch}", leave=False,
            )
            for step, (stage_batch, gardner_batch) in enumerate(pbar, start=1):
                if step > steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                stage_batch = to_device(stage_batch, device)
                gardner_batch = to_device(gardner_batch, device)
                stage_out = model(stage_batch["images"], "stage")
                stage_loss = masked_bce_loss(
                    stage_out["phase_logits"], stage_batch["phase_targets"],
                    stage_batch["phase_mask"], pos_weight=stage_pos_weight,
                )
                gardner_out = model(gardner_batch["images"], "gardner")
                exp_loss = masked_ce_loss(
                    gardner_out["exp_logits"], gardner_batch["exp_target"],
                    gardner_batch["exp_mask"], exp_weights,
                )
                icm_loss = masked_ce_loss(
                    gardner_out["icm_logits"], gardner_batch["icm_target"],
                    gardner_batch["icm_mask"], icm_weights,
                )
                te_loss = masked_ce_loss(
                    gardner_out["te_logits"], gardner_batch["te_target"],
                    gardner_batch["te_mask"], te_weights,
                )
                gardner_loss = exp_loss + icm_loss + te_loss
                loss = task_weights["stage"] * stage_loss + task_weights["gardner"] * gardner_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
                optimizer.step()
                total_loss += float(loss.item())
                total_stage += float(stage_loss.item())
                total_gardner += float(gardner_loss.item())
                pbar.set_postfix(loss=total_loss / step)
            scheduler.step()

            val_metrics = evaluate(model, loaders["stage_val"], loaders["gardner_val"], device)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / steps,
                "stage_loss": total_stage / steps,
                "gardner_loss": total_gardner / steps,
                "phase_macro_f1": val_metrics.get("phase_macro_f1"),
                "phase_macro_auc": val_metrics.get("phase_macro_auc"),
                "gardner_exp_macro_f1": val_metrics.get("gardner_exp_macro_f1"),
                "gardner_icm_macro_f1": val_metrics.get("gardner_icm_macro_f1"),
                "gardner_te_macro_f1": val_metrics.get("gardner_te_macro_f1"),
                "lr": scheduler.get_last_lr()[0],
            }
            writer.writerow(row)
            f.flush()
            score_parts = [
                val_metrics.get("phase_macro_f1") or 0.0,
                val_metrics.get("gardner_exp_macro_f1") or 0.0,
                val_metrics.get("gardner_icm_macro_f1") or 0.0,
                val_metrics.get("gardner_te_macro_f1") or 0.0,
            ]
            score = float(np.mean(score_parts))
            with open(out_dir / "latest_metrics.json", "w", encoding="utf-8") as mf:
                json.dump(val_metrics, mf, indent=2, ensure_ascii=False)
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, out_dir / "best.pt")
                with open(out_dir / "best_metrics.json", "w", encoding="utf-8") as bf:
                    json.dump(val_metrics, bf, indent=2, ensure_ascii=False)

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "best_score": best_score, "output_dir": str(out_dir)}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_one(args.config, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
