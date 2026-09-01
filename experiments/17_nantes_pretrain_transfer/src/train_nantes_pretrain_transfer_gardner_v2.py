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
MASK_REGIONS = ["ICM", "TE", "ZP"]


def seed_all(seed: int):
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


def mask_transform(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    mask = mask.float().unsqueeze(0) if mask.ndim == 2 else mask.float()
    mask = F.interpolate(mask.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    return mask.clamp(0, 1)


class KrompGardnerImageDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path, mask_dir: Path, train: bool, image_size: int, use_roi: bool):
        self.rows = rows
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_size = image_size
        self.use_roi = use_roi
        self.tf = image_transform(train, image_size)
        self.raw_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rows)

    def _load_roi_views(self, image_name: str):
        img = Image.open(self.image_dir / image_name).convert("RGB")
        base = self.raw_tf(img)
        views = [self.norm(base)]
        mask_path = self.mask_dir / (Path(image_name).stem + ".pt")
        if mask_path.exists():
            masks = torch.load(mask_path, map_location="cpu").float()
            masks = mask_transform(masks, self.image_size)
            for i in range(3):
                m = masks[i : i + 1]
                views.append(self.norm(base * m))
        else:
            views.extend([self.norm(base * 0.0) for _ in range(3)])
        return torch.stack(views, dim=0)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_name = row["image"]
        if self.use_roi:
            images = self._load_roi_views(image_name)
        else:
            img = Image.open(self.image_dir / image_name).convert("RGB")
            images = self.tf(img).unsqueeze(0)
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(row[task]))
            if y < 0:
                labels.append(0)
                masks.append(0.0)
            else:
                labels.append(y)
                masks.append(1.0)
        return {"images": images, "labels": torch.tensor(labels).long(), "masks": torch.tensor(masks).float(), "image": image_name}


class TaskAttentionGardner(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.25, use_attention: bool = False):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.use_attention = use_attention
        self.attn = nn.ModuleDict({task: nn.Linear(dim, 1) for task in TASKS})
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim // 2, n),
                )
                for task, (_, n) in TASKS.items()
            }
        )

    def forward(self, images):
        b, v, c, h, w = images.shape
        feat = self.backbone(images.reshape(b * v, c, h, w)).reshape(b, v, -1)
        out, attn = {}, {}
        for task in TASKS:
            if self.use_attention and v > 1:
                score = self.attn[task](feat).squeeze(-1)
                weights = torch.softmax(score, dim=1)
                fused = (feat * weights.unsqueeze(-1)).sum(dim=1)
                attn[task] = weights
            else:
                fused = feat.mean(dim=1)
                attn[task] = torch.ones(b, v, device=images.device) / v
            out[task] = self.heads[task](fused)
        return out, attn


def load_nantes_backbone(model: TaskAttentionGardner, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    backbone_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            backbone_sd[k.replace("backbone.", "", 1)] = v
    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
    return {"loaded_keys": len(backbone_sd), "missing": missing, "unexpected": unexpected, "source": str(ckpt_path)}


def class_weights(rows, task, n_classes):
    counts = np.zeros(n_classes, dtype=np.float32)
    for r in rows:
        y = int(float(r[task]))
        if y >= 0:
            counts[y] += 1
    w = counts.sum() / np.maximum(counts, 1)
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32))


def focal_ce(logits, target, weight=None, gamma=1.5):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def loss_fn(outputs, labels, masks, weights, device, focal: bool):
    total = 0.0
    used = 0
    for i, task in enumerate(["exp", "icm", "te"]):
        active = masks[:, i] > 0
        if active.any():
            w = weights[task].to(device)
            if focal:
                loss = focal_ce(outputs[task][active], labels[:, i][active], weight=w)
            else:
                loss = F.cross_entropy(outputs[task][active], labels[:, i][active], weight=w)
            total = total + loss
            used += 1
    return total / max(used, 1)


def sample_weights(rows):
    # Emphasize rows with rare ICM/TE labels because these heads are the bottleneck.
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


def safe_auc(y_true, prob, n_classes):
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
    attn_store = {t: [] for t in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"]
            masks = batch["masks"]
            out, attn = model(images)
            for task in TASKS:
                attn_store[task].append(attn[task].detach().cpu().numpy())
            for i, task in enumerate(["exp", "icm", "te"]):
                active = masks[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[task].detach().cpu(), dim=1).numpy()
                    pred = prob.argmax(axis=1)
                    idx = active.numpy().astype(bool)
                    y_true[task].extend(labels[:, i].numpy()[idx].tolist())
                    y_pred[task].extend(pred[idx].tolist())
                    y_prob[task].extend(prob[idx].tolist())
    metrics = compute_metrics(y_true, y_pred, y_prob)
    attn_summary = {}
    names = ["global", "icm_roi", "te_roi", "zp_roi"]
    for task, vals in attn_store.items():
        arr = np.concatenate(vals, axis=0)
        attn_summary[TASKS[task][0]] = dict(zip(names[: arr.shape[1]], arr.mean(axis=0).astype(float).tolist()))
    return metrics, attn_summary


def train_variant(args, seed: int, variant: str):
    seed_all(seed)
    train_rows = read_rows(Path(args.train_features))
    test_rows = read_rows(Path(args.test_features))
    tr_rows, val_rows = train_test_split(train_rows, test_size=0.15, random_state=seed)
    use_roi = "roi" in variant
    use_attention = "attn" in variant
    use_focal = "focal" in variant
    use_sampler = "balanced" in variant
    out_dir = Path(args.out_dir) / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = KrompGardnerImageDataset(tr_rows, Path(args.image_dir), Path(args.mask_dir), True, args.image_size, use_roi)
    val = KrompGardnerImageDataset(val_rows, Path(args.image_dir), Path(args.mask_dir), False, args.image_size, use_roi)
    te = KrompGardnerImageDataset(test_rows, Path(args.image_dir), Path(args.mask_dir), False, args.image_size, use_roi)
    if use_sampler:
        sampler = WeightedRandomSampler(sample_weights(tr_rows), num_samples=len(tr_rows), replacement=True)
        tr_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = TaskAttentionGardner(pretrained=True, dropout=args.dropout, use_attention=use_attention).to(args.device)
    load_info = None
    if variant.startswith("nantes"):
        ckpt = Path(args.nantes_ckpt_template.format(seed=seed))
        if not ckpt.exists():
            ckpt = Path(args.nantes_ckpt_template.format(seed=42))
        load_info = load_nantes_backbone(model, ckpt)
    weights = {task: class_weights(tr_rows, task, n) for task, (_, n) in TASKS.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tr_loader:
            images = batch["images"].to(args.device)
            labels = batch["labels"].to(args.device)
            masks = batch["masks"].to(args.device)
            opt.zero_grad(set_to_none=True)
            out, _ = model(images)
            loss = loss_fn(out, labels, masks, weights, args.device, use_focal)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_metrics, _ = predict(model, val_loader, args.device)
        val_score = float(np.mean([val_metrics[h]["macro_f1"] for h in ["Expansion", "ICM", "TE"]]))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mean_macro_f1": val_score})
        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=args.device))
    test_metrics, attn_summary = predict(model, te_loader, args.device)
    summary = {"seed": seed, "variant": variant, "best_val_mean_macro_f1": best, "test": test_metrics, "attention": attn_summary, "nantes_load": load_info, "history": history}
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
                    final["variants"][variant][head][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        final["variants"][variant]["attention"] = {}
        for head in ["Expansion", "ICM", "TE"]:
            final["variants"][variant]["attention"][head] = {}
            keys = vruns[0]["attention"][head].keys()
            for k in keys:
                vals = [r["attention"][head][k] for r in vruns]
                final["variants"][variant]["attention"][head][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    final["notes"].append("Nantes-pretrained variants initialize the Kromp Gardner ResNet18 backbone from the Nantes multi-focal 16-phase checkpoint, then fine-tune on Kromp.")
    final["notes"].append("ROI variants use four views: whole image, ICM-masked image, TE-masked image and ZP-masked image.")
    out = Path(args.out_dir) / "nantes_pretrain_transfer_summary.json"
    out.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    p.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    p.add_argument("--image_dir", default="/path/to/embryo_data/03_Kromp2344_Gardner/Blastocyst_Dataset/Images")
    p.add_argument("--mask_dir", default="/path/to/embryo_data/09_Gardner_SegAssist/mask_cache/kromp_sfu_seed42")
    p.add_argument("--nantes_ckpt_template", default="/path/to/checkpoints/nantes_seed_{seed}.pt")
    p.add_argument("--out_dir", default="outputs/17_nantes_pretrain_transfer/formal")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", default=["imagenet_global", "nantes_global", "nantes_roi_attn", "nantes_roi_attn_focal_balanced"])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        for variant in args.variants:
            runs.append(train_variant(args, seed, variant))
    summarize(args, runs)


if __name__ == "__main__":
    main()
