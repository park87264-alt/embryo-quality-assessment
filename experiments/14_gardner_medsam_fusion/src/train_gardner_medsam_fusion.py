from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


V2_SRC = Path(__file__).resolve().parents[2] / "13_medsam_lora_structure" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))
from medsam_lora_sfu import MedSAMLoRA  # noqa: E402


FEATURE_COLUMNS = [
    "icm_area", "icm_bbox_area", "icm_fill_ratio", "icm_cx", "icm_cy", "icm_aspect",
    "te_area", "te_bbox_area", "te_fill_ratio", "te_cx", "te_cy", "te_aspect",
    "zp_area", "zp_bbox_area", "zp_fill_ratio", "zp_cx", "zp_cy", "zp_aspect",
    "icm_te_area_ratio", "icm_zp_area_ratio", "te_zp_area_ratio", "structure_total_area",
]
TASKS = {
    "exp": {"classes": 5, "display": "Expansion"},
    "icm": {"classes": 3, "display": "ICM"},
    "te": {"classes": 3, "display": "TE"},
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_feature_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


class ImageListDataset(Dataset):
    def __init__(self, rows: list[dict], image_dir: Path):
        self.rows = rows
        self.image_dir = image_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return image_tensor(self.image_dir / row["image"]), row["image"]


def extract_embeddings(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_feature_rows(Path(args.train_features)) + read_feature_rows(Path(args.test_features))
    seen = {}
    unique_rows = []
    for r in rows:
        if r["image"] not in seen:
            seen[r["image"]] = True
            unique_rows.append(r)
    device = torch.device(args.device)
    model = MedSAMLoRA(args.medsam_checkpoint, args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    ckpt = torch.load(args.medsam_lora_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    ds = ImageListDataset(unique_rows, Path(args.image_dir))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    names, feats = [], []
    with torch.no_grad():
        for image, name in loader:
            image = image.to(device)
            feat = model.encode_features(image).detach().cpu().numpy()
            feats.append(feat)
            names.extend(list(name))
    feat_arr = np.concatenate(feats, axis=0).astype(np.float32)
    np.savez_compressed(out_dir / "kromp_medsam_lora_image_embeddings.npz", images=np.array(names), features=feat_arr)
    meta = {
        "embedding_file": str(out_dir / "kromp_medsam_lora_image_embeddings.npz"),
        "image_count": len(names),
        "feature_dim": int(feat_arr.shape[1]),
        "source_checkpoint": args.medsam_lora_checkpoint,
    }
    (out_dir / "embedding_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


class GardnerFeatureDataset(Dataset):
    def __init__(self, rows, embedding_map, use_structure: bool, structure_mean=None, structure_std=None):
        self.rows = rows
        self.embedding_map = embedding_map
        self.use_structure = use_structure
        structs = np.array([[float(r.get(c, 0) or 0) for c in FEATURE_COLUMNS] for r in rows], dtype=np.float32)
        if structure_mean is None:
            structure_mean = structs.mean(axis=0)
        if structure_std is None:
            structure_std = structs.std(axis=0)
        self.structure_mean = structure_mean.astype(np.float32)
        self.structure_std = np.maximum(structure_std.astype(np.float32), 1e-6)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        image_feat = self.embedding_map[r["image"]]
        struct = np.array([float(r.get(c, 0) or 0) for c in FEATURE_COLUMNS], dtype=np.float32)
        struct = (struct - self.structure_mean) / self.structure_std
        if self.use_structure:
            x = np.concatenate([image_feat, struct], axis=0)
        else:
            x = image_feat
        labels, masks = [], []
        for t in ["exp", "icm", "te"]:
            y = int(float(r[t]))
            if y < 0:
                labels.append(0)
                masks.append(0.0)
            else:
                labels.append(y)
                masks.append(1.0)
        return torch.from_numpy(x).float(), torch.tensor(labels).long(), torch.tensor(masks).float()


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, cfg["classes"]) for t, cfg in TASKS.items()})

    def forward(self, x):
        h = self.trunk(x)
        return {t: head(h) for t, head in self.heads.items()}


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(name): feat.astype(np.float32) for name, feat in zip(data["images"], data["features"])}


def class_weights(rows, task: str, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float32)
    for r in rows:
        y = int(float(r[task]))
        if y >= 0:
            counts[y] += 1
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def multitask_loss(logits, labels, masks, weights, device):
    total = 0.0
    used = 0
    for i, t in enumerate(["exp", "icm", "te"]):
        active = masks[:, i] > 0
        if active.any():
            ce = nn.CrossEntropyLoss(weight=weights[t].to(device))
            total = total + ce(logits[t][active], labels[:, i][active])
            used += 1
    return total / max(used, 1)


def predict(model, loader, device):
    model.eval()
    y_true = {t: [] for t in TASKS}
    y_pred = {t: [] for t in TASKS}
    y_prob = {t: [] for t in TASKS}
    with torch.no_grad():
        for x, labels, masks in loader:
            x = x.to(device)
            out = model(x)
            for i, t in enumerate(["exp", "icm", "te"]):
                active = masks[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[t].detach().cpu(), dim=1).numpy()
                    pred = prob.argmax(axis=1)
                    idx = active.numpy().astype(bool)
                    y_true[t].extend(labels[:, i].numpy()[idx].tolist())
                    y_pred[t].extend(pred[idx].tolist())
                    y_prob[t].extend(prob[idx].tolist())
    return y_true, y_pred, y_prob


def safe_auc(y_true, y_prob, classes):
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] < 2:
            return None
        return float(roc_auc_score(y_bin, np.array(y_prob), average="macro", multi_class="ovr"))
    except Exception:
        return None


def metrics_from_pred(y_true, y_pred, y_prob):
    result = {}
    for t, cfg in TASKS.items():
        yt, yp, ypb = y_true[t], y_pred[t], y_prob[t]
        result[cfg["display"]] = {
            "support": len(yt),
            "acc": float(accuracy_score(yt, yp)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "macro_auc_ovr": safe_auc(yt, ypb, list(range(cfg["classes"]))),
        }
    return result


def train_one(args, seed: int, use_structure: bool):
    seed_all(seed)
    out_dir = Path(args.out_dir) / ("image_structure" if use_structure else "image_only") / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_feature_rows(Path(args.train_features))
    test_rows = read_feature_rows(Path(args.test_features))
    tr_rows, val_rows = train_test_split(train_rows, test_size=0.15, random_state=seed)
    emb = load_embeddings(Path(args.embedding_file))
    train_struct = np.array([[float(r.get(c, 0) or 0) for c in FEATURE_COLUMNS] for r in tr_rows], dtype=np.float32)
    sm, ss = train_struct.mean(axis=0), train_struct.std(axis=0)
    tr_ds = GardnerFeatureDataset(tr_rows, emb, use_structure, sm, ss)
    val_ds = GardnerFeatureDataset(val_rows, emb, use_structure, sm, ss)
    te_ds = GardnerFeatureDataset(test_rows, emb, use_structure, sm, ss)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    input_dim = 256 + (len(FEATURE_COLUMNS) if use_structure else 0)
    device = torch.device(args.device)
    model = MultiTaskMLP(input_dim, args.hidden, args.dropout).to(device)
    weights = {t: class_weights(tr_rows, t, cfg["classes"]) for t, cfg in TASKS.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, labels, masks in tr_loader:
            x, labels, masks = x.to(device), labels.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            loss = multitask_loss(model(x), labels, masks, weights, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        y_true, y_pred, y_prob = predict(model, val_loader, device)
        val_metrics = metrics_from_pred(y_true, y_pred, y_prob)
        val_score = np.mean([v["macro_f1"] for v in val_metrics.values()])
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_mean_macro_f1": float(val_score), "val": val_metrics}
        history.append(rec)
        if val_score > best_val:
            best_val = val_score
            torch.save({"model": model.state_dict(), "seed": seed, "use_structure": use_structure, "input_dim": input_dim}, out_dir / "best.pt")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    y_true, y_pred, y_prob = predict(model, te_loader, device)
    test_metrics = metrics_from_pred(y_true, y_pred, y_prob)
    summary = {"seed": seed, "use_structure": use_structure, "best_val_mean_macro_f1": best_val, "test": test_metrics, "history": history}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"seed": seed, "use_structure": use_structure, "test": test_metrics}, ensure_ascii=False))
    return summary


def summarize_runs(args):
    base = Path(args.out_dir)
    final = {}
    for variant in ["image_only", "image_structure"]:
        runs = [json.loads((base / variant / f"seed_{s}" / "summary.json").read_text(encoding="utf-8")) for s in args.seeds]
        final[variant] = {}
        for head in ["Expansion", "ICM", "TE"]:
            final[variant][head] = {}
            keys = runs[0]["test"][head].keys()
            for k in keys:
                vals = [r["test"][head][k] for r in runs if r["test"][head][k] is not None]
                if k == "support":
                    final[variant][head][k] = vals[0]
                else:
                    final[variant][head][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    final["interpretation"] = "Image+structure tests whether MedSAM-derived structural priors improve Gardner grading over image-only MedSAM features."
    (base / "fusion_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def train_fusion(args):
    for seed in args.seeds:
        train_one(args, seed, use_structure=False)
        train_one(args, seed, use_structure=True)
    summarize_runs(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["extract", "train"])
    p.add_argument("--image_dir", default="/path/to/embryo_data/03_Kromp2344_Gardner/Blastocyst_Dataset/Images")
    p.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    p.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    p.add_argument("--medsam_checkpoint", default="/path/to/weights/medsam_vit_b.pth")
    p.add_argument("--medsam_lora_checkpoint", default="/path/to/weights/medsam_lora_sfu_best.pth")
    p.add_argument("--out_dir", default="outputs/14_gardner_medsam_fusion/formal")
    p.add_argument("--embedding_file", default="outputs/14_gardner_medsam_fusion/formal/kromp_medsam_lora_image_embeddings.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    args = p.parse_args()
    if args.mode == "extract":
        extract_embeddings(args)
    else:
        train_fusion(args)


if __name__ == "__main__":
    main()
