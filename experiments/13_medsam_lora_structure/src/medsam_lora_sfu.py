from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


MEDSAM_CODE = Path("/path/to/MedSAM")
if str(MEDSAM_CODE) not in sys.path:
    sys.path.insert(0, str(MEDSAM_CODE))

from segment_anything import sam_model_registry  # noqa: E402


STRUCTURES = ["ICM", "TE", "ZP"]
STRUCTURE_TO_INDEX = {name: i for i, name in enumerate(STRUCTURES)}
FEATURE_COLUMNS = [
    "icm_area",
    "icm_bbox_area",
    "icm_fill_ratio",
    "icm_cx",
    "icm_cy",
    "icm_aspect",
    "te_area",
    "te_bbox_area",
    "te_fill_ratio",
    "te_cx",
    "te_cy",
    "te_aspect",
    "zp_area",
    "zp_bbox_area",
    "zp_fill_ratio",
    "zp_cx",
    "zp_cy",
    "zp_aspect",
    "icm_te_area_ratio",
    "icm_zp_area_ratio",
    "te_zp_area_ratio",
    "structure_total_area",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def image_path_for(root: Path, sample_id: str) -> Path:
    for suffix in [".BMP", ".bmp", ".png", ".jpg", ".jpeg"]:
        p = root / "Images" / f"{sample_id}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(sample_id)


def mask_path_for(root: Path, sample_id: str, structure: str) -> Path:
    p = root / f"GT_{structure}" / f"{sample_id} {structure}_Mask.bmp"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def load_rgb_1024(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask_1024(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((1024, 1024), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.float32)


def bbox_from_mask(mask: np.ndarray, shift: int = 20) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.asarray([0, 0, mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if shift > 0:
        x0 = max(0, x0 - random.randint(0, shift))
        y0 = max(0, y0 - random.randint(0, shift))
        x1 = min(mask.shape[1] - 1, x1 + random.randint(0, shift))
        y1 = min(mask.shape[0] - 1, y1 + random.randint(0, shift))
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


class SFUMedSAMDataset(Dataset):
    def __init__(self, root: str | Path, split_file: str | Path, train: bool = True, structures: list[str] | None = None):
        self.root = Path(root)
        self.sample_ids = read_split(split_file)
        self.train = train
        self.items: list[tuple[str, str]] = []
        for sid in self.sample_ids:
            for structure in structures or STRUCTURES:
                mask = load_mask_1024(mask_path_for(self.root, sid, structure))
                if mask.sum() > 0:
                    self.items.append((sid, structure))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sid, structure = self.items[idx]
        image = load_rgb_1024(image_path_for(self.root, sid))
        mask = load_mask_1024(mask_path_for(self.root, sid, structure))
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        bbox = bbox_from_mask(mask, shift=20 if self.train else 0)
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask[None]).float(),
            "box": torch.from_numpy(bbox).float(),
            "sample_id": sid,
            "structure": structure,
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / max(r, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def inject_lora_to_image_encoder(sam_model, r: int, alpha: float, dropout: float) -> int:
    count = 0
    for block in sam_model.image_encoder.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r=r, alpha=alpha, dropout=dropout)
        count += 1
    return count


class MedSAMLoRA(nn.Module):
    def __init__(self, checkpoint: str, lora_r: int = 4, lora_alpha: float = 8.0, lora_dropout: float = 0.0):
        super().__init__()
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        for p in sam_model.parameters():
            p.requires_grad = False
        self.lora_layers = inject_lora_to_image_encoder(sam_model, lora_r, lora_alpha, lora_dropout)
        for p in sam_model.mask_decoder.parameters():
            p.requires_grad = True
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            if box.ndim == 2:
                box = box[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=box, masks=None)
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)

    def encode_features(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.image_encoder(image)
        return feat.mean(dim=(2, 3))


def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    tgt_sum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
    union = pred_sum + tgt_sum - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice.mean().item()), float(iou.mean().item())


def train(args):
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_ds = SFUMedSAMDataset(args.sfu_root, args.train_split, train=True)
    val_ds = SFUMedSAMDataset(args.sfu_root, args.val_split, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model = MedSAMLoRA(args.checkpoint, args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history = []
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            box = batch["box"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(image, box)
                prob = torch.sigmoid(logits)
                inter = (prob * mask).sum(dim=(1, 2, 3))
                dice_loss = 1 - ((2 * inter + 1e-6) / (prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1e-6)).mean()
                loss = bce(logits, mask) + dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_metrics = evaluate_model(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["mean_dsc"] > best_val:
            best_val = val_metrics["mean_dsc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "args": vars(args),
                    "trainable_params": trainable,
                    "total_params": total,
                },
                out_dir / "medsam_lora_sfu_best.pth",
            )
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, out_dir / "medsam_lora_sfu_latest.pth")
    summary = {"trainable_params": trainable, "total_params": total, "lora_layers": model.lora_layers, "history": history, "best_val_mean_dsc": best_val}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    by_structure = {s: {"dice": [], "iou": []} for s in STRUCTURES}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            box = batch["box"].to(device)
            logits = model(image, box)
            dice, iou = dice_iou_from_logits(logits, mask)
            structure = batch["structure"][0]
            by_structure[structure]["dice"].append(dice)
            by_structure[structure]["iou"].append(iou)
    metrics = {}
    all_dice, all_iou = [], []
    for s, values in by_structure.items():
        d = values["dice"]
        i = values["iou"]
        metrics[f"{s}_dsc"] = float(np.mean(d)) if d else None
        metrics[f"{s}_iou"] = float(np.mean(i)) if i else None
        all_dice.extend(d)
        all_iou.extend(i)
    metrics["mean_dsc"] = float(np.mean(all_dice)) if all_dice else None
    metrics["mean_iou"] = float(np.mean(all_iou)) if all_iou else None
    return metrics


def eval_checkpoint(args):
    device = torch.device(args.device)
    ds = SFUMedSAMDataset(args.sfu_root, args.test_split, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = MedSAMLoRA(args.checkpoint, args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    ckpt = torch.load(args.lora_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    metrics = evaluate_model(model, loader, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": args.lora_checkpoint, "test_items": len(ds), "metrics": metrics}
    (out_dir / "sfu_test_dsc_iou.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def safe_auc(y_true, proba, classes):
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] < 2:
            return None
        return float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
    except Exception:
        return None


def train_quality_rf(args):
    import pandas as pd

    train = pd.read_csv(args.train_features)
    test = pd.read_csv(args.test_features)
    targets = {"Expansion": "exp", "ICM": "icm", "TE": "te"}
    result = {"note": "Diagnostic Gardner classification using segmentation-derived structure features. It measures quality assessment separately from segmentation DSC/IoU.", "heads": {}}
    for display, col in targets.items():
        tr = train[pd.to_numeric(train[col], errors="coerce").fillna(-1) >= 0].copy()
        te = test[pd.to_numeric(test[col], errors="coerce").fillna(-1) >= 0].copy()
        ytr = tr[col].astype(int).astype(str).to_numpy()
        yte = te[col].astype(int).astype(str).to_numpy()
        clf = RandomForestClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1)
        clf.fit(tr[FEATURE_COLUMNS].fillna(0).to_numpy(), ytr)
        pred = clf.predict(te[FEATURE_COLUMNS].fillna(0).to_numpy())
        proba = clf.predict_proba(te[FEATURE_COLUMNS].fillna(0).to_numpy())
        classes = clf.classes_
        result["heads"][display] = {
            "support": int(len(yte)),
            "acc": float(accuracy_score(yte, pred)),
            "balanced_acc": float(balanced_accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yte, pred, average="macro", zero_division=0)),
            "macro_auc_ovr": safe_auc(yte, proba, classes),
        }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gardner_structure_quality_rf.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "quality_rf"])
    p.add_argument("--sfu_root", default="/path/to/embryo_data/01_Static_Datasets/SFU_1024_SAM2_Ready")
    p.add_argument("--train_split", default="/path/to/embryo_data/08_SFU_Segmentation_Expert/splits/train.txt")
    p.add_argument("--val_split", default="/path/to/embryo_data/08_SFU_Segmentation_Expert/splits/val.txt")
    p.add_argument("--test_split", default="/path/to/embryo_data/08_SFU_Segmentation_Expert/splits/test.txt")
    p.add_argument("--checkpoint", default="/path/to/weights/medsam_vit_b.pth")
    p.add_argument("--lora_checkpoint", default="")
    p.add_argument("--out_dir", default="outputs/13_medsam_lora_structure/pilot_seed42")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    p.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    args = p.parse_args()
    os.environ.setdefault("PYTHONPATH", str(MEDSAM_CODE))
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_checkpoint(args)
    else:
        train_quality_rf(args)


if __name__ == "__main__":
    main()
