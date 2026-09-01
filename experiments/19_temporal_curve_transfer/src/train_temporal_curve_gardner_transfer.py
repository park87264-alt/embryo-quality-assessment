from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


TASKS = {"exp": ("Expansion", 5), "icm": ("ICM", 3), "te": ("TE", 3)}
STRUCTURE_COLS = [
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_transform(train: bool, image_size: int):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def safe_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def wh_from_bbox_area_aspect(area: float, aspect: float) -> tuple[float, float]:
    area = max(0.0, float(area))
    aspect = max(1e-3, float(aspect))
    width = float(np.sqrt(area * aspect))
    height = float(np.sqrt(area / aspect))
    return min(width, 1.0), min(height, 1.0)


def kromp_static_to_curve_sequence(row: dict, seq_len: int = 32, in_dim: int = 71) -> np.ndarray:
    # Match the Nantes curve feature order: 35 static structural features,
    # 35 temporal deltas, and one normalized time coordinate.
    features: list[float] = []
    areas = {}
    for prefix in ["icm", "te", "zp"]:
        area = safe_float(row, f"{prefix}_area")
        bbox_area = safe_float(row, f"{prefix}_bbox_area")
        fill = safe_float(row, f"{prefix}_fill_ratio")
        cx = safe_float(row, f"{prefix}_cx")
        cy = safe_float(row, f"{prefix}_cy")
        aspect = safe_float(row, f"{prefix}_aspect", 1.0)
        bw, bh = wh_from_bbox_area_aspect(bbox_area, aspect)
        areas[prefix] = max(area, 0.0)
        # The static Kromp table does not have intensity/perimeter statistics, so
        # these slots are set to 0. The model learns whether to use them.
        features.extend([area, area, cx, cy, bw, bh, bbox_area, 0.0, fill, 0.0])
    total = areas["icm"] + areas["te"] + areas["zp"] + 1e-6
    features.extend(
        [
            areas["icm"] / total,
            areas["te"] / total,
            areas["zp"] / total,
            (areas["icm"] + areas["te"]) / (areas["zp"] + 1e-6),
            safe_float(row, "structure_total_area"),
        ]
    )
    static = np.asarray(features, dtype=np.float32)
    delta = np.zeros_like(static)
    seq = []
    for t in range(seq_len):
        time_norm = t / max(seq_len - 1, 1)
        seq.append(np.concatenate([static, delta, np.asarray([time_norm], dtype=np.float32)]))
    out = np.stack(seq, axis=0)
    if out.shape[1] != in_dim:
        raise ValueError(f"Curve dim mismatch: got {out.shape[1]}, expected {in_dim}")
    return out


class KrompCurveDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path, train: bool, image_size: int, seq_len: int, curve_dim: int):
        self.rows = rows
        self.image_dir = image_dir
        self.tf = image_transform(train, image_size)
        self.seq_len = seq_len
        self.curve_dim = curve_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(self.image_dir / row["image"]).convert("RGB")
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(row[task]))
            if y < 0:
                labels.append(0)
                masks.append(0.0)
            else:
                labels.append(y)
                masks.append(1.0)
        curve = kromp_static_to_curve_sequence(row, self.seq_len, self.curve_dim)
        return {
            "image": self.tf(img),
            "curve": torch.from_numpy(curve).float(),
            "labels": torch.tensor(labels).long(),
            "masks": torch.tensor(masks).float(),
            "name": row["image"],
        }


class TemporalCurveEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z, _ = self.gru(z)
        return z.mean(dim=1)


class GardnerImageCurveModel(nn.Module):
    def __init__(
        self,
        curve_in_dim: int,
        curve_hidden: int,
        dropout: float,
        use_curve: bool,
        freeze_curve: bool,
        pretrained_image: bool = True,
    ):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained_image else None
        backbone = tv_models.resnet18(weights=weights)
        img_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.use_curve = use_curve
        self.curve = TemporalCurveEncoder(curve_in_dim, curve_hidden, dropout)
        curve_dim = curve_hidden * 2
        fused_dim = img_dim + (curve_dim if use_curve else 0)
        self.fuse = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(dropout))
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(fused_dim, img_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(img_dim // 2, n),
                )
                for task, (_, n) in TASKS.items()
            }
        )
        if freeze_curve:
            for p in self.curve.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor, curve: torch.Tensor):
        feat = self.backbone(image)
        if self.use_curve:
            cfeat = self.curve(curve)
            feat = torch.cat([feat, cfeat], dim=1)
        feat = self.fuse(feat)
        return {task: head(feat) for task, head in self.heads.items()}


def load_nantes_backbone(model: GardnerImageCurveModel, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    backbone_sd = {k.replace("backbone.", "", 1): v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
    return {"source": str(ckpt_path), "loaded_keys": len(backbone_sd), "missing": list(missing), "unexpected": list(unexpected)}


def load_curve_encoder(model: GardnerImageCurveModel, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"]
    curve_sd = {k: v for k, v in sd.items() if k.startswith("proj.") or k.startswith("gru.")}
    missing, unexpected = model.curve.load_state_dict(curve_sd, strict=False)
    return {
        "source": str(ckpt_path),
        "loaded_keys": len(curve_sd),
        "best_metrics": ck.get("best_metrics"),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def class_weights(rows: list[dict], task: str, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float32)
    for r in rows:
        y = int(float(r[task]))
        if y >= 0:
            counts[y] += 1
    w = counts.sum() / np.maximum(counts, 1)
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32))


def sample_weights(rows: list[dict]) -> torch.DoubleTensor:
    counts = {task: {} for task in ["exp", "icm", "te"]}
    for task in counts:
        for r in rows:
            y = int(float(r[task]))
            if y >= 0:
                counts[task][y] = counts[task].get(y, 0) + 1
    weights = []
    for r in rows:
        vals = []
        for task in ["exp", "icm", "te"]:
            y = int(float(r[task]))
            if y >= 0:
                vals.append(1.0 / max(1, counts[task].get(y, 1)))
        weights.append(float(np.mean(vals)) if vals else 1.0)
    return torch.DoubleTensor(weights)


def focal_ce(logits: torch.Tensor, target: torch.Tensor, weight=None, gamma: float = 1.5) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def loss_fn(outputs, labels, masks, weights, device, focal: bool) -> torch.Tensor:
    total = 0.0
    used = 0
    for i, task in enumerate(["exp", "icm", "te"]):
        active = masks[:, i] > 0
        if active.any():
            weight = weights[task].to(device)
            if focal:
                loss = focal_ce(outputs[task][active], labels[:, i][active], weight=weight)
            else:
                loss = F.cross_entropy(outputs[task][active], labels[:, i][active], weight=weight)
            total = total + loss
            used += 1
    return total / max(used, 1)


def safe_auc(y_true, prob, n_classes: int):
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.shape[1] < 2:
            return None
        return float(roc_auc_score(y_bin, np.asarray(prob), average="macro", multi_class="ovr"))
    except Exception:
        return None


def compute_metrics(y_true, y_pred, y_prob):
    out = {}
    for task, (display, n_classes) in TASKS.items():
        yt, yp, prob = y_true[task], y_pred[task], y_prob[task]
        out[display] = {
            "support": len(yt),
            "acc": float(accuracy_score(yt, yp)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "macro_auc_ovr": safe_auc(yt, prob, n_classes),
        }
    return out


def predict(model, loader, device):
    y_true = {t: [] for t in TASKS}
    y_pred = {t: [] for t in TASKS}
    y_prob = {t: [] for t in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            curve = batch["curve"].to(device)
            labels = batch["labels"]
            masks = batch["masks"]
            out = model(image, curve)
            for i, task in enumerate(["exp", "icm", "te"]):
                active = masks[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[task].detach().cpu(), dim=1).numpy()
                    pred = prob.argmax(axis=1)
                    idx = active.numpy().astype(bool)
                    y_true[task].extend(labels[:, i].numpy()[idx].tolist())
                    y_pred[task].extend(pred[idx].tolist())
                    y_prob[task].extend(prob[idx].tolist())
    return compute_metrics(y_true, y_pred, y_prob)


def train_variant(args, seed: int, variant: str):
    seed_all(seed)
    train_rows = read_rows(Path(args.train_features))
    test_rows = read_rows(Path(args.test_features))
    tr_rows, val_rows = train_test_split(train_rows, test_size=0.15, random_state=seed)
    use_curve = "curve" in variant
    freeze_curve = "frozen" in variant
    use_nantes = variant.startswith("nantes")
    use_focal = "focal" in variant
    use_sampler = "balanced" in variant
    out_dir = Path(args.out_dir) / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = KrompCurveDataset(tr_rows, Path(args.image_dir), True, args.image_size, args.seq_len, args.curve_dim)
    val = KrompCurveDataset(val_rows, Path(args.image_dir), False, args.image_size, args.seq_len, args.curve_dim)
    te = KrompCurveDataset(test_rows, Path(args.image_dir), False, args.image_size, args.seq_len, args.curve_dim)
    if use_sampler:
        sampler = WeightedRandomSampler(sample_weights(tr_rows), num_samples=len(tr_rows), replacement=True)
        tr_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = GardnerImageCurveModel(args.curve_dim, args.curve_hidden, args.dropout, use_curve, freeze_curve).to(args.device)
    load_info = {"image": None, "curve": None}
    if use_nantes:
        ckpt = Path(args.nantes_ckpt_template.format(seed=seed))
        if not ckpt.exists():
            ckpt = Path(args.nantes_ckpt_template.format(seed=42))
        load_info["image"] = load_nantes_backbone(model, ckpt)
    if use_curve:
        ckpt = Path(args.curve_ckpt_template.format(seed=seed))
        if not ckpt.exists():
            ckpt = Path(args.curve_ckpt_template.format(seed=42))
        load_info["curve"] = load_curve_encoder(model, ckpt)
    weights = {task: class_weights(tr_rows, task, n) for task, (_, n) in TASKS.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            image = batch["image"].to(args.device)
            curve = batch["curve"].to(args.device)
            labels = batch["labels"].to(args.device)
            masks = batch["masks"].to(args.device)
            opt.zero_grad(set_to_none=True)
            out = model(image, curve)
            loss = loss_fn(out, labels, masks, weights, args.device, use_focal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        val_metrics = predict(model, val_loader, args.device)
        val_score = float(np.mean([val_metrics[h]["macro_f1"] for h in ["Expansion", "ICM", "TE"]]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mean_macro_f1": val_score})
        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=args.device))
    test_metrics = predict(model, te_loader, args.device)
    summary = {
        "seed": seed,
        "variant": variant,
        "best_val_mean_macro_f1": best,
        "test": test_metrics,
        "load_info": load_info,
        "history": history,
        "note": "Temporal curve branch uses Nantes-pretrained structure curve encoder. Kromp static structure features are converted to a compatible curve token; no Nantes embryo IDs are matched to Kromp labels.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def summarize(args, runs):
    final = {"variants": {}, "notes": []}
    for variant in args.variants:
        vruns = [r for r in runs if r["variant"] == variant]
        final["variants"][variant] = {}
        for head in ["Expansion", "ICM", "TE"]:
            final["variants"][variant][head] = {}
            for key in vruns[0]["test"][head].keys():
                vals = [r["test"][head][key] for r in vruns if r["test"][head][key] is not None]
                if key == "support":
                    final["variants"][variant][head][key] = vals[0]
                else:
                    final["variants"][variant][head][key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals),
                    }
    final["notes"].append("This experiment evaluates whether Nantes temporal-structure pretraining improves Kromp/Gardner classification.")
    final["notes"].append("Because Kromp is static and Nantes has no Gardner scores, this is weight/representation transfer, not direct feature concatenation by embryo ID.")
    final["notes"].append("The curve branch maps Kromp static structure features into the same 71-dimensional curve format and loads the Nantes temporal curve encoder weights.")
    out = Path(args.out_dir) / "temporal_curve_gardner_transfer_summary.json"
    out.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    p.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    p.add_argument("--image_dir", default="/path/to/embryo_data/03_Kromp2344_Gardner/Blastocyst_Dataset/Images")
    p.add_argument("--nantes_ckpt_template", default="/path/to/checkpoints/nantes_seed_{seed}.pt")
    p.add_argument("--curve_ckpt_template", default="/path/to/checkpoints/curve_encoder_seed_{seed}.pt")
    p.add_argument("--out_dir", default="outputs/19_temporal_curve_transfer/formal")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", default=["imagenet_global", "nantes_global", "nantes_curve_fusion", "nantes_curve_fusion_focal_balanced"])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--curve_dim", type=int, default=71)
    p.add_argument("--curve_hidden", type=int, default=128)
    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        for variant in args.variants:
            print(f"RUN seed={seed} variant={variant}", flush=True)
            runs.append(train_variant(args, seed, variant))
    summarize(args, runs)


if __name__ == "__main__":
    main()
