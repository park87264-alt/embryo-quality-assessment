from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PHASES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9plus", "tM", "tSB", "tB", "tEB", "tHB"]
BLASTOCYST_PHASES = {"tSB", "tB", "tEB", "tHB"}
TASKS = {"exp": ("Expansion", 5), "icm": ("ICM", 3), "te": ("TE", 3)}
BASE_FEATURES = [
    "ICM_area_bin", "ICM_area_soft", "ICM_cx", "ICM_cy", "ICM_bbox_w", "ICM_bbox_h", "ICM_bbox_area", "ICM_perimeter_proxy", "ICM_int_mean", "ICM_int_std",
    "TE_area_bin", "TE_area_soft", "TE_cx", "TE_cy", "TE_bbox_w", "TE_bbox_h", "TE_bbox_area", "TE_perimeter_proxy", "TE_int_mean", "TE_int_std",
    "ZP_area_bin", "ZP_area_soft", "ZP_cx", "ZP_cy", "ZP_bbox_w", "ZP_bbox_h", "ZP_bbox_area", "ZP_perimeter_proxy", "ZP_int_mean", "ZP_int_std",
    "ICM_area_share", "TE_area_share", "ZP_area_share", "ICM_TE_to_ZP_ratio", "union_area",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def event_align_features(frame_df: pd.DataFrame, frames_per_embryo: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs, ys, masks, embryo_ids = [], [], [], []
    for embryo_id, group in frame_df.groupby("embryo_id", sort=True):
        group = group.sort_values("sample_index")
        seq, labels, valid = [], [], []
        for _, row in group.iterrows():
            phase = str(row["phase"])
            feat = row[BASE_FEATURES].astype(float).to_numpy(dtype=np.float32)
            icm_te_valid = 1.0 if phase in BLASTOCYST_PHASES else 0.0
            blastocyst_valid = icm_te_valid
            if not icm_te_valid:
                feat[0:20] = 0.0
                feat[30] = 0.0
                feat[31] = 0.0
                feat[33] = 0.0
                # Keep ZP and union_area as generic embryo/outer-structure information.
            label = PHASES.index(phase) if phase in PHASES else 0
            seq.append(feat)
            labels.append(label)
            valid.append(1 if phase in PHASES else 0)
        while len(seq) < frames_per_embryo:
            seq.append(np.zeros(len(BASE_FEATURES), dtype=np.float32))
            labels.append(0)
            valid.append(0)
        seq = np.stack(seq[:frames_per_embryo], axis=0)
        delta = np.zeros_like(seq)
        delta[1:] = seq[1:] - seq[:-1]
        time = np.linspace(0, 1, frames_per_embryo, dtype=np.float32)[:, None]
        phase_flags = np.asarray([[1.0 if PHASES[l] in BLASTOCYST_PHASES else 0.0, 1.0 if PHASES[l] in BLASTOCYST_PHASES else 0.0] for l in labels[:frames_per_embryo]], dtype=np.float32)
        out = np.concatenate([seq, delta, time, phase_flags], axis=1)
        xs.append(out)
        ys.append(np.asarray(labels[:frames_per_embryo], dtype=np.int64))
        masks.append(np.asarray(valid[:frames_per_embryo], dtype=np.float32))
        embryo_ids.append(str(embryo_id))
    feature_names = BASE_FEATURES + [f"delta_{x}" for x in BASE_FEATURES] + ["time_norm", "blastocyst_phase_flag", "icm_te_valid_flag"]
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), np.asarray(masks), np.asarray(embryo_ids), feature_names


class CurveDataset(Dataset):
    def __init__(self, x, y, m, idx):
        self.x = torch.from_numpy(x[idx]).float()
        self.y = torch.from_numpy(y[idx]).long()
        self.m = torch.from_numpy(m[idx]).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]


class TemporalCurveEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.25, n_classes: int = 16):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

    def encode(self, x):
        z = self.proj(x)
        z, _ = self.gru(z)
        return z.mean(dim=1)

    def forward(self, x):
        z = self.proj(x)
        z, _ = self.gru(z)
        return self.head(z)


def eval_curve(model, loader, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y, m in loader:
            pred = model(x.to(device)).detach().cpu().argmax(-1)
            active = m > 0
            yt.extend(y[active].numpy().tolist())
            yp.extend(pred[active].numpy().tolist())
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "n_frames": int(len(yt)),
    }


def train_event_pretrain(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_df = pd.read_csv(args.frame_features)
    x, y, m, embryo_ids, feature_names = event_align_features(frame_df, args.frames_per_embryo)
    np.savez_compressed(out / "nantes_event_aligned_curve_sequences.npz", X=x, y=y, mask=m, embryo_ids=embryo_ids, feature_names=np.asarray(feature_names), phases=np.asarray(PHASES))
    all_idx = np.arange(len(x))
    results = []
    for seed in args.seeds:
        seed_all(seed)
        train_idx, test_idx = train_test_split(all_idx, test_size=0.3, random_state=seed)
        train_loader = DataLoader(CurveDataset(x, y, m, train_idx), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(CurveDataset(x, y, m, test_idx), batch_size=args.batch_size, shuffle=False)
        device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        model = TemporalCurveEncoder(x.shape[-1], args.hidden, args.dropout, len(PHASES)).to(device)
        yy = y[train_idx][m[train_idx] > 0].reshape(-1)
        counts = np.bincount(yy, minlength=len(PHASES)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = torch.from_numpy((weights / weights.mean()).astype(np.float32)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state, history = {"macro_f1": -1}, None, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for bx, by, bm in train_loader:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                logits = model(bx)
                active = bm > 0
                loss = F.cross_entropy(logits[active], by[active], weight=weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
            metrics = eval_curve(model, test_loader, device)
            metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
            history.append(metrics)
            if metrics["macro_f1"] > best["macro_f1"]:
                best = metrics
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.save({"model": best_state, "in_dim": x.shape[-1], "hidden": args.hidden, "phases": PHASES, "best_metrics": best, "feature_names": feature_names}, seed_dir / "event_aligned_temporal_curve_encoder_best.pt")
        pd.DataFrame(history).to_csv(seed_dir / "history.csv", index=False)
        (seed_dir / "best_metrics.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append({"seed": seed, **best, "checkpoint": str(seed_dir / "event_aligned_temporal_curve_encoder_best.pt")})
    summary = {"task": "Nantes 16-phase prediction with event-aligned structure curves", "n_embryos": int(len(x)), "frames_per_embryo": args.frames_per_embryo, "feature_dim": int(x.shape[-1]), "per_seed": results, "aggregate": {}}
    for key in ["accuracy", "balanced_accuracy", "macro_f1"]:
        vals = [r[key] for r in results]
        summary["aggregate"][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["interpretation"] = "ICM/TE pseudo-mask features are suppressed before blastocyst-related phases to avoid biologically invalid curve interpretation."
    (out / "event_aligned_pretrain_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_transform(train: bool, image_size: int):
    if train:
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(8), transforms.ColorJitter(brightness=0.12, contrast=0.12), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def sf(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def wh(area, aspect):
    area, aspect = max(0.0, float(area)), max(1e-3, float(aspect))
    return min(float(np.sqrt(area * aspect)), 1.0), min(float(np.sqrt(area / aspect)), 1.0)


def kromp_to_event_curve(row, seq_len=32, in_dim=73):
    features = []
    areas = {}
    for prefix, up in [("icm", "ICM"), ("te", "TE"), ("zp", "ZP")]:
        area = sf(row, f"{prefix}_area")
        bbox = sf(row, f"{prefix}_bbox_area")
        fill = sf(row, f"{prefix}_fill_ratio")
        cx = sf(row, f"{prefix}_cx")
        cy = sf(row, f"{prefix}_cy")
        aspect = sf(row, f"{prefix}_aspect", 1.0)
        bw, bh = wh(bbox, aspect)
        areas[prefix] = area
        features.extend([area, area, cx, cy, bw, bh, bbox, 0.0, fill, 0.0])
    total = areas["icm"] + areas["te"] + areas["zp"] + 1e-6
    features.extend([areas["icm"] / total, areas["te"] / total, areas["zp"] / total, (areas["icm"] + areas["te"]) / (areas["zp"] + 1e-6), sf(row, "structure_total_area")])
    static = np.asarray(features, dtype=np.float32)
    delta = np.zeros_like(static)
    seq = []
    for t in range(seq_len):
        seq.append(np.concatenate([static, delta, np.asarray([t / max(seq_len - 1, 1), 1.0, 1.0], dtype=np.float32)]))
    arr = np.stack(seq)
    if arr.shape[1] != in_dim:
        raise ValueError((arr.shape, in_dim))
    return arr


class KrompDataset(Dataset):
    def __init__(self, rows, image_dir, train, image_size, seq_len, curve_dim):
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.tf = image_transform(train, image_size)
        self.seq_len = seq_len
        self.curve_dim = curve_dim

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = self.tf(Image.open(self.image_dir / row["image"]).convert("RGB"))
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(row[task]))
            labels.append(0 if y < 0 else y)
            masks.append(0.0 if y < 0 else 1.0)
        curve = kromp_to_event_curve(row, self.seq_len, self.curve_dim)
        return {"image": image, "curve": torch.from_numpy(curve).float(), "labels": torch.tensor(labels).long(), "masks": torch.tensor(masks).float()}


class GardnerModel(nn.Module):
    def __init__(self, curve_dim, hidden, dropout, use_curve):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.use_curve = use_curve
        self.curve = TemporalCurveEncoder(curve_dim, hidden, dropout, len(PHASES))
        fused = dim + (hidden * 2 if use_curve else 0)
        self.norm = nn.Sequential(nn.LayerNorm(fused), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({task: nn.Sequential(nn.Linear(fused, dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, n)) for task, (_, n) in TASKS.items()})

    def forward(self, image, curve):
        feat = self.backbone(image)
        if self.use_curve:
            feat = torch.cat([feat, self.curve.encode(curve)], dim=1)
        feat = self.norm(feat)
        return {task: head(feat) for task, head in self.heads.items()}


def load_nantes(model, template, seed):
    ckpt = Path(template.format(seed=seed))
    if not ckpt.exists():
        ckpt = Path(template.format(seed=42))
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model", ck)
    bsd = {k.replace("backbone.", "", 1): v for k, v in sd.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(bsd, strict=False)
    return str(ckpt)


def load_event_curve(model, template, seed):
    ckpt = Path(template.format(seed=seed))
    if not ckpt.exists():
        ckpt = Path(template.format(seed=42))
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck["model"]
    csd = {k: v for k, v in sd.items() if k.startswith("proj.") or k.startswith("gru.")}
    model.curve.load_state_dict(csd, strict=False)
    return {"source": str(ckpt), "best_metrics": ck.get("best_metrics")}


def class_weights(rows, task, n):
    counts = np.zeros(n, dtype=np.float32)
    for r in rows:
        y = int(float(r[task]))
        if y >= 0:
            counts[y] += 1
    w = counts.sum() / np.maximum(counts, 1)
    return torch.from_numpy((w / w.mean()).astype(np.float32))


def loss_fn(out, labels, masks, weights, device):
    total, used = 0.0, 0
    for i, task in enumerate(["exp", "icm", "te"]):
        active = masks[:, i] > 0
        if active.any():
            total = total + F.cross_entropy(out[task][active], labels[:, i][active], weight=weights[task].to(device))
            used += 1
    return total / max(used, 1)


def safe_auc(y_true, prob, n):
    try:
        yb = label_binarize(y_true, classes=list(range(n)))
        return float(roc_auc_score(yb, np.asarray(prob), average="macro", multi_class="ovr")) if yb.shape[1] >= 2 else None
    except Exception:
        return None


def predict(model, loader, device):
    yt, yp, yprob = {t: [] for t in TASKS}, {t: [] for t in TASKS}, {t: [] for t in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device), batch["curve"].to(device))
            labels, masks = batch["labels"], batch["masks"]
            for i, task in enumerate(["exp", "icm", "te"]):
                active = masks[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[task].detach().cpu(), dim=1).numpy()
                    pred = prob.argmax(1)
                    idx = active.numpy().astype(bool)
                    yt[task].extend(labels[:, i].numpy()[idx].tolist())
                    yp[task].extend(pred[idx].tolist())
                    yprob[task].extend(prob[idx].tolist())
    res = {}
    for task, (name, n) in TASKS.items():
        res[name] = {"support": len(yt[task]), "acc": float(accuracy_score(yt[task], yp[task])), "balanced_acc": float(balanced_accuracy_score(yt[task], yp[task])), "macro_f1": float(f1_score(yt[task], yp[task], average="macro", zero_division=0)), "macro_recall": float(recall_score(yt[task], yp[task], average="macro", zero_division=0)), "macro_auc_ovr": safe_auc(yt[task], yprob[task], n)}
    return res


def train_transfer(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        seed_all(seed)
        rows = read_rows(Path(args.train_features))
        test_rows = read_rows(Path(args.test_features))
        tr_rows, val_rows = train_test_split(rows, test_size=0.15, random_state=seed)
        for variant in args.variants:
            print(f"RUN seed={seed} variant={variant}", flush=True)
            use_curve = "event_curve" in variant
            model = GardnerModel(args.curve_dim, args.hidden, args.dropout, use_curve).to(args.device)
            load_info = {"nantes": load_nantes(model, args.nantes_ckpt_template, seed) if variant.startswith("nantes") else None, "event_curve": load_event_curve(model, args.event_curve_ckpt_template, seed) if use_curve else None}
            tr = KrompDataset(tr_rows, args.image_dir, True, args.image_size, args.seq_len, args.curve_dim)
            val = KrompDataset(val_rows, args.image_dir, False, args.image_size, args.seq_len, args.curve_dim)
            te = KrompDataset(test_rows, args.image_dir, False, args.image_size, args.seq_len, args.curve_dim)
            tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            weights = {task: class_weights(tr_rows, task, n) for task, (_, n) in TASKS.items()}
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best, history = -1, []
            od = out / variant / f"seed_{seed}"
            od.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch in tr_loader:
                    image, curve = batch["image"].to(args.device), batch["curve"].to(args.device)
                    labels, masks = batch["labels"].to(args.device), batch["masks"].to(args.device)
                    loss = loss_fn(model(image, curve), labels, masks, weights, args.device)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss.item()))
                vm = predict(model, val_loader, args.device)
                score = float(np.mean([vm[h]["macro_f1"] for h in ["Expansion", "ICM", "TE"]]))
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mean_macro_f1": score})
                if score > best:
                    best = score
                    torch.save(model.state_dict(), od / "best.pt")
            model.load_state_dict(torch.load(od / "best.pt", map_location=args.device))
            tm = predict(model, te_loader, args.device)
            summary = {"seed": seed, "variant": variant, "best_val_mean_macro_f1": best, "test": tm, "load_info": load_info, "history": history}
            (od / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            runs.append(summary)
    final = {"variants": {}, "notes": ["Event-aligned curve suppresses ICM/TE pseudo-mask features before tSB/tB/tEB/tHB to avoid hallucinated biological interpretation."]}
    for variant in args.variants:
        vr = [r for r in runs if r["variant"] == variant]
        final["variants"][variant] = {}
        for head in ["Expansion", "ICM", "TE"]:
            final["variants"][variant][head] = {}
            for key in vr[0]["test"][head]:
                vals = [r["test"][head][key] for r in vr if r["test"][head][key] is not None]
                final["variants"][variant][head][key] = vals[0] if key == "support" else {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    (out / "event_aligned_gardner_transfer_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pretrain")
    a.add_argument("--frame_features", default="/path/to/embryo_data/18_Nantes_StructureCurve_Pretrain/outputs/formal/nantes_structure_curve_frame_features.csv")
    a.add_argument("--out_dir", default="outputs/20_event_aligned_curve/pretrain")
    a.add_argument("--frames_per_embryo", type=int, default=32)
    a.add_argument("--epochs", type=int, default=35)
    a.add_argument("--batch_size", type=int, default=64)
    a.add_argument("--hidden", type=int, default=128)
    a.add_argument("--dropout", type=float, default=0.25)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--weight_decay", type=float, default=1e-4)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    b = sub.add_parser("transfer")
    b.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    b.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    b.add_argument("--image_dir", default="/path/to/embryo_data/03_Kromp2344_Gardner/Blastocyst_Dataset/Images")
    b.add_argument("--nantes_ckpt_template", default="/path/to/checkpoints/nantes_seed_{seed}.pt")
    b.add_argument("--event_curve_ckpt_template", default="/path/to/checkpoints/event_curve_seed_{seed}.pt")
    b.add_argument("--out_dir", default="outputs/20_event_aligned_curve/transfer")
    b.add_argument("--variants", nargs="+", default=["nantes_global", "nantes_event_curve_fusion"])
    b.add_argument("--epochs", type=int, default=35)
    b.add_argument("--batch_size", type=int, default=48)
    b.add_argument("--image_size", type=int, default=224)
    b.add_argument("--num_workers", type=int, default=4)
    b.add_argument("--hidden", type=int, default=128)
    b.add_argument("--dropout", type=float, default=0.25)
    b.add_argument("--lr", type=float, default=1e-4)
    b.add_argument("--weight_decay", type=float, default=1e-4)
    b.add_argument("--seq_len", type=int, default=32)
    b.add_argument("--curve_dim", type=int, default=73)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if args.cmd == "pretrain":
        train_event_pretrain(args)
    else:
        train_transfer(args)


if __name__ == "__main__":
    main()
