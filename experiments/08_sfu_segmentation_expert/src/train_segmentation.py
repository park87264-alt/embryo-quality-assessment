from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import SFUSegmentationDataset, STRUCTURES
from losses import bce_dice_loss, dice_iou_metrics
from model import ResNet18UNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    model.eval()
    dice_values, iou_values = [], []
    loss_values = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss_values.append(float(bce_dice_loss(logits, masks).item()))
            dice, iou = dice_iou_metrics(logits, masks)
            dice_values.append(dice.numpy())
            iou_values.append(iou.numpy())
    dice_arr = np.stack(dice_values, axis=0)
    iou_arr = np.stack(iou_values, axis=0)
    metrics = {"loss": float(np.mean(loss_values))}
    per_class = {}
    for idx, name in enumerate(STRUCTURES):
        per_class[name] = {
            "dice": float(dice_arr[:, idx].mean()),
            "iou": float(iou_arr[:, idx].mean()),
        }
    metrics["per_class"] = per_class
    metrics["mean_dice"] = float(np.mean([per_class[k]["dice"] for k in STRUCTURES]))
    metrics["mean_iou"] = float(np.mean([per_class[k]["iou"] for k in STRUCTURES]))
    return metrics


def train_one(config_path: str, output_dir: str, seed: int):
    cfg = load_yaml(config_path)
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    data = cfg["data"]
    train_cfg = cfg["train"]
    train_ds = SFUSegmentationDataset(data["sfu_root"], data["train_split"], data["image_size"], train=True)
    val_ds = SFUSegmentationDataset(data["sfu_root"], data["val_split"], data["image_size"], train=False)
    test_ds = SFUSegmentationDataset(data["sfu_root"], data["test_split"], data["image_size"], train=False)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18UNet(out_channels=3, pretrained=cfg["model"].get("pretrained", True)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    history_path = out_dir / "history.csv"
    best_dice = -1.0
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "val_loss", "val_mean_dice", "val_mean_iou",
            "val_icm_dice", "val_te_dice", "val_zp_dice", "lr",
        ])
        writer.writeheader()
        for epoch in range(1, train_cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
            for batch in pbar:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = bce_dice_loss(logits, masks, bce_weight=train_cfg.get("bce_weight", 0.5))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
                optimizer.step()
                total_loss += float(loss.item())
                pbar.set_postfix(loss=total_loss / max(1, len(pbar)))
            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / len(train_loader),
                "val_loss": val_metrics["loss"],
                "val_mean_dice": val_metrics["mean_dice"],
                "val_mean_iou": val_metrics["mean_iou"],
                "val_icm_dice": val_metrics["per_class"]["ICM"]["dice"],
                "val_te_dice": val_metrics["per_class"]["TE"]["dice"],
                "val_zp_dice": val_metrics["per_class"]["ZP"]["dice"],
                "lr": scheduler.get_last_lr()[0],
            }
            writer.writerow(row)
            f.flush()
            with open(out_dir / "latest_val_metrics.json", "w", encoding="utf-8") as mf:
                json.dump(val_metrics, mf, indent=2, ensure_ascii=False)
            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "mean_dice": best_dice}, out_dir / "best.pt")
                with open(out_dir / "best_val_metrics.json", "w", encoding="utf-8") as bf:
                    json.dump(val_metrics, bf, indent=2, ensure_ascii=False)

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "best_val_mean_dice": float(checkpoint["mean_dice"]),
            "test_mean_dice": test_metrics["mean_dice"],
            "test_mean_iou": test_metrics["mean_iou"],
            "output_dir": str(out_dir),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_one(args.config, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
