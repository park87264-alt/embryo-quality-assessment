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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


TASKS = {"exp": ("Expansion", 5), "icm": ("ICM", 3), "te": ("TE", 3)}
ABC = {"A": 0, "B": 1, "C": 2}
STRUCT_COLS = [
    "icm_area", "icm_bbox_area", "icm_fill_ratio", "icm_cx", "icm_cy", "icm_aspect",
    "te_area", "te_bbox_area", "te_fill_ratio", "te_cx", "te_cy", "te_aspect",
    "zp_area", "zp_bbox_area", "zp_fill_ratio", "zp_cx", "zp_cy", "zp_aspect",
    "icm_te_area_ratio", "icm_zp_area_ratio", "te_zp_area_ratio", "structure_total_area",
]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class TemporalCurveEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        z = self.proj(x)
        z, _ = self.gru(z)
        return z.mean(dim=1)


class NantesWeakDataset(Dataset):
    def __init__(self, x, labels, masks, idx):
        self.x = torch.from_numpy(x[idx]).float()
        self.y = torch.from_numpy(labels[idx]).long()
        self.m = torch.from_numpy(masks[idx]).float()

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i], self.m[i]


class WeakICMTEModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.encoder = TemporalCurveEncoder(in_dim, hidden, dropout)
        self.heads = nn.ModuleDict({
            "ICM": nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, 3)),
            "TE": nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, 3)),
        })

    def forward(self, x):
        z = self.encoder(x)
        return {k: h(z) for k, h in self.heads.items()}


def safe_auc(y_true, prob, n=3):
    try:
        yb = label_binarize(y_true, classes=list(range(n)))
        return float(roc_auc_score(yb, np.asarray(prob), average="macro", multi_class="ovr")) if yb.shape[1] >= 2 else None
    except Exception:
        return None


def eval_weak(model, loader, device):
    model.eval()
    store = {k: {"y": [], "p": [], "prob": []} for k in ["ICM", "TE"]}
    with torch.no_grad():
        for x, y, m in loader:
            out = model(x.to(device))
            for i, task in enumerate(["ICM", "TE"]):
                active = m[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[task].detach().cpu(), 1).numpy()
                    pred = prob.argmax(1)
                    idx = active.numpy().astype(bool)
                    store[task]["y"].extend(y[:, i].numpy()[idx].tolist())
                    store[task]["p"].extend(pred[idx].tolist())
                    store[task]["prob"].extend(prob[idx].tolist())
    res = {}
    for task, s in store.items():
        res[task] = {
            "support": len(s["y"]),
            "acc": float(accuracy_score(s["y"], s["p"])) if s["y"] else None,
            "balanced_acc": float(balanced_accuracy_score(s["y"], s["p"])) if s["y"] else None,
            "macro_f1": float(f1_score(s["y"], s["p"], average="macro", zero_division=0)) if s["y"] else None,
            "macro_auc_ovr": safe_auc(s["y"], s["prob"]) if s["y"] else None,
        }
    return res


def train_weak(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seq = np.load(args.curve_npz, allow_pickle=True)
    x = seq["X"].astype(np.float32)
    embryo_ids = [str(e) for e in seq["embryo_ids"]]
    grades = pd.read_csv(args.nantes_manifest).set_index("embryo_id")
    y = np.zeros((len(embryo_ids), 2), dtype=np.int64)
    m = np.zeros((len(embryo_ids), 2), dtype=np.float32)
    for i, eid in enumerate(embryo_ids):
        if eid in grades.index:
            for j, col in enumerate(["ICM", "TE"]):
                val = grades.loc[eid, col]
                if isinstance(val, str) and val in ABC:
                    y[i, j] = ABC[val]; m[i, j] = 1.0
    all_idx = np.where(m.sum(axis=1) > 0)[0]
    results = []
    for seed in args.seeds:
        seed_all(seed)
        train_idx, test_idx = train_test_split(all_idx, test_size=0.3, random_state=seed)
        train_loader = DataLoader(NantesWeakDataset(x, y, m, train_idx), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(NantesWeakDataset(x, y, m, test_idx), batch_size=args.batch_size, shuffle=False)
        device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        model = WeakICMTEModel(x.shape[-1], args.hidden, args.dropout).to(device)
        # Start from event-aligned phase encoder if available.
        ck = Path(args.event_curve_ckpt_template.format(seed=seed))
        if not ck.exists(): ck = Path(args.event_curve_ckpt_template.format(seed=42))
        obj = torch.load(ck, map_location="cpu")
        enc_sd = {k.replace("model.", ""): v for k, v in obj.get("model", {}).items()}
        model.encoder.load_state_dict({k: v for k, v in enc_sd.items() if k.startswith("proj.") or k.startswith("gru.")}, strict=False)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, best_state = -1, None
        hist = []
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []
            for bx, by, bm in train_loader:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                out = model(bx); loss = 0; used = 0
                for j, task in enumerate(["ICM", "TE"]):
                    active = bm[:, j] > 0
                    if active.any():
                        loss = loss + F.cross_entropy(out[task][active], by[:, j][active])
                        used += 1
                loss = loss / max(used, 1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(float(loss.item()))
            metrics = eval_weak(model, test_loader, device)
            score = np.mean([metrics[t]["macro_f1"] for t in ["ICM", "TE"] if metrics[t]["macro_f1"] is not None])
            hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "mean_macro_f1": float(score), "metrics": metrics})
            if score > best:
                best = score; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        seed_dir = out_dir / f"weak_seed_{seed}"; seed_dir.mkdir(exist_ok=True)
        torch.save({"model": best_state, "in_dim": x.shape[-1], "hidden": args.hidden, "source_event_curve": str(ck)}, seed_dir / "weak_icm_te_curve_encoder_best.pt")
        (seed_dir / "summary.json").write_text(json.dumps({"seed": seed, "best_mean_macro_f1": best, "history": hist, "support_total": int(len(all_idx)), "label_support": {"ICM": int(m[:,0].sum()), "TE": int(m[:,1].sum())}}, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append({"seed": seed, "best_mean_macro_f1": float(best), "best_metrics": hist[int(np.argmax([h["mean_macro_f1"] for h in hist]))]["metrics"], "checkpoint": str(seed_dir / "weak_icm_te_curve_encoder_best.pt")})
    final = {"task": "Nantes weak ICM/TE supervision on event-aligned curve encoder", "label_support": {"ICM": int(m[:,0].sum()), "TE": int(m[:,1].sum()), "any": int(len(all_idx))}, "per_seed": results, "aggregate": {}}
    for task in ["ICM", "TE"]:
        final["aggregate"][task] = {}
        for key in ["macro_f1", "macro_auc_ovr", "balanced_acc"]:
            vals = [r["best_metrics"][task][key] for r in results if r["best_metrics"][task][key] is not None]
            final["aggregate"][task][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    (out_dir / "weak_nantes_icm_te_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def image_tf(train, size):
    aug = [transforms.Resize((size, size))]
    if train:
        aug += [transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(8), transforms.ColorJitter(brightness=0.12, contrast=0.12)]
    aug += [transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])]
    return transforms.Compose(aug)


def sf(row, k, d=0.0):
    try:
        v = row.get(k, d)
        if v in ("", None): return d
        v = float(v)
        return v if np.isfinite(v) else d
    except Exception: return d


def kromp_curve(row, seq_len, dim):
    vals = []
    areas = {}
    for p in ["icm", "te", "zp"]:
        area, bbox, fill = sf(row, f"{p}_area"), sf(row, f"{p}_bbox_area"), sf(row, f"{p}_fill_ratio")
        cx, cy, aspect = sf(row, f"{p}_cx"), sf(row, f"{p}_cy"), max(sf(row, f"{p}_aspect", 1.0), 1e-3)
        bw, bh = min(np.sqrt(bbox * aspect), 1.0), min(np.sqrt(bbox / aspect), 1.0)
        areas[p] = area
        vals += [area, area, cx, cy, bw, bh, bbox, 0.0, fill, 0.0]
    total = areas["icm"] + areas["te"] + areas["zp"] + 1e-6
    vals += [areas["icm"]/total, areas["te"]/total, areas["zp"]/total, (areas["icm"]+areas["te"])/(areas["zp"]+1e-6), sf(row, "structure_total_area")]
    static = np.asarray(vals, dtype=np.float32)
    delta = np.zeros_like(static)
    arr = np.stack([np.concatenate([static, delta, np.asarray([i/max(seq_len-1,1), 1.0, 1.0], dtype=np.float32)]) for i in range(seq_len)])
    if arr.shape[1] != dim: raise ValueError((arr.shape, dim))
    return arr


class KrompDataset(Dataset):
    def __init__(self, rows, image_dir, train, image_size, seq_len, curve_dim):
        self.rows = rows; self.image_dir = Path(image_dir); self.tf = image_tf(train, image_size); self.seq_len = seq_len; self.curve_dim = curve_dim
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(r[task])); labels.append(max(y, 0)); masks.append(0.0 if y < 0 else 1.0)
        struct = np.asarray([sf(r, c) for c in STRUCT_COLS], dtype=np.float32)
        return {"image": self.tf(Image.open(self.image_dir / r["image"]).convert("RGB")), "curve": torch.from_numpy(kromp_curve(r, self.seq_len, self.curve_dim)).float(), "structure": torch.from_numpy(struct).float(), "labels": torch.tensor(labels).long(), "masks": torch.tensor(masks).float()}


class TaskGatedMoE(nn.Module):
    def __init__(self, curve_dim, hidden, dropout):
        super().__init__()
        b = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT); img_dim = b.fc.in_features; b.fc = nn.Identity()
        self.backbone = b
        self.curve = TemporalCurveEncoder(curve_dim, hidden, dropout)
        self.struct = nn.Sequential(nn.LayerNorm(len(STRUCT_COLS)), nn.Linear(len(STRUCT_COLS), 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, hidden * 2))
        self.img_proj = nn.Linear(img_dim, hidden * 2)
        expert_dim = hidden * 2
        self.gates = nn.ModuleDict({t: nn.Linear(expert_dim * 3, 3) for t in TASKS})
        self.heads = nn.ModuleDict({t: nn.Sequential(nn.LayerNorm(expert_dim), nn.Dropout(dropout), nn.Linear(expert_dim, expert_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(expert_dim // 2, n)) for t, (_, n) in TASKS.items()})

    def forward(self, image, curve, structure):
        img = self.img_proj(self.backbone(image))
        cur = self.curve(curve)
        st = self.struct(structure)
        cat = torch.cat([img, cur, st], dim=1)
        out, gates = {}, {}
        stack = torch.stack([img, cur, st], dim=1)
        for task in TASKS:
            w = torch.softmax(self.gates[task](cat), dim=1)
            fused = (stack * w.unsqueeze(-1)).sum(dim=1)
            out[task] = self.heads[task](fused)
            gates[task] = w
        return out, gates


def load_nantes_backbone(model, template, seed):
    ck = Path(template.format(seed=seed))
    if not ck.exists(): ck = Path(template.format(seed=42))
    obj = torch.load(ck, map_location="cpu"); sd = obj.get("model", obj)
    model.backbone.load_state_dict({k.replace("backbone.","",1): v for k,v in sd.items() if k.startswith("backbone.")}, strict=False)
    return str(ck)


def load_weak_curve(model, template, seed):
    ck = Path(template.format(seed=seed))
    if not ck.exists(): ck = Path(template.format(seed=42))
    obj = torch.load(ck, map_location="cpu"); sd = obj["model"]
    model.curve.load_state_dict({k.replace("encoder.","",1): v for k,v in sd.items() if k.startswith("encoder.proj") or k.startswith("encoder.gru")}, strict=False)
    return str(ck)


def class_weights(rows, task, n):
    counts = np.zeros(n, dtype=np.float32)
    for r in rows:
        y = int(float(r[task]))
        if y >= 0: counts[y] += 1
    w = counts.sum() / np.maximum(counts, 1)
    return torch.from_numpy((w / w.mean()).astype(np.float32))


def predict(model, loader, device):
    data = {t: {"y": [], "p": [], "prob": []} for t in TASKS}; gate_vals = {t: [] for t in TASKS}
    model.eval()
    with torch.no_grad():
        for b in loader:
            out, gates = model(b["image"].to(device), b["curve"].to(device), b["structure"].to(device))
            labels, masks = b["labels"], b["masks"]
            for t in TASKS: gate_vals[t].append(gates[t].detach().cpu().numpy())
            for i, t in enumerate(["exp", "icm", "te"]):
                active = masks[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[t].detach().cpu(), 1).numpy(); pred = prob.argmax(1); idx = active.numpy().astype(bool)
                    data[t]["y"] += labels[:, i].numpy()[idx].tolist(); data[t]["p"] += pred[idx].tolist(); data[t]["prob"] += prob[idx].tolist()
    res = {}
    for t, (name, n) in TASKS.items():
        s = data[t]
        res[name] = {"support": len(s["y"]), "acc": float(accuracy_score(s["y"], s["p"])), "balanced_acc": float(balanced_accuracy_score(s["y"], s["p"])), "macro_f1": float(f1_score(s["y"], s["p"], average="macro", zero_division=0)), "macro_auc_ovr": safe_auc(s["y"], s["prob"], n)}
    gate_summary = {}
    for t, vals in gate_vals.items():
        arr = np.concatenate(vals, 0)
        gate_summary[TASKS[t][0]] = dict(zip(["image", "curve", "structure"], arr.mean(0).astype(float).tolist()))
    return res, gate_summary


def train_transfer(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True); runs = []
    for seed in args.seeds:
        seed_all(seed)
        rows = read_rows(Path(args.train_features)); test_rows = read_rows(Path(args.test_features))
        tr_rows, val_rows = train_test_split(rows, test_size=0.15, random_state=seed)
        for variant in args.variants:
            print(f"RUN seed={seed} variant={variant}", flush=True)
            device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
            model = TaskGatedMoE(args.curve_dim, args.hidden, args.dropout).to(device)
            load_info = {"nantes": load_nantes_backbone(model, args.nantes_ckpt_template, seed), "weak_curve": load_weak_curve(model, args.weak_ckpt_template, seed) if "weak" in variant else None}
            tr = KrompDataset(tr_rows, args.image_dir, True, args.image_size, args.seq_len, args.curve_dim)
            val = KrompDataset(val_rows, args.image_dir, False, args.image_size, args.seq_len, args.curve_dim)
            te = KrompDataset(test_rows, args.image_dir, False, args.image_size, args.seq_len, args.curve_dim)
            tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            weights = {t: class_weights(tr_rows, t, n) for t, (_, n) in TASKS.items()}
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best = -1; hist = []; od = out_dir / variant / f"seed_{seed}"; od.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, args.epochs + 1):
                model.train(); losses = []
                for b in tr_loader:
                    out, _ = model(b["image"].to(device), b["curve"].to(device), b["structure"].to(device))
                    labels, masks = b["labels"].to(device), b["masks"].to(device)
                    loss = 0; used = 0
                    for i, t in enumerate(["exp", "icm", "te"]):
                        active = masks[:, i] > 0
                        if active.any():
                            loss = loss + F.cross_entropy(out[t][active], labels[:, i][active], weight=weights[t].to(device)); used += 1
                    loss = loss / max(used, 1)
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    losses.append(float(loss.item()))
                vm, _ = predict(model, val_loader, device)
                score = float(np.mean([vm[h]["macro_f1"] for h in ["Expansion", "ICM", "TE"]]))
                hist.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mean_macro_f1": score})
                if score > best:
                    best = score; torch.save(model.state_dict(), od / "best.pt")
            model.load_state_dict(torch.load(od / "best.pt", map_location=device))
            tm, gates = predict(model, te_loader, device)
            summary = {"seed": seed, "variant": variant, "best_val_mean_macro_f1": best, "test": tm, "gates": gates, "load_info": load_info, "history": hist}
            (od / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            runs.append(summary)
    final = {"variants": {}, "notes": ["Task-gated MoE has image, temporal-curve and static-structure experts with separate gates for Expansion, ICM and TE."]}
    for variant in args.variants:
        vr = [r for r in runs if r["variant"] == variant]; final["variants"][variant] = {}
        for head in ["Expansion", "ICM", "TE"]:
            final["variants"][variant][head] = {}
            for key in vr[0]["test"][head]:
                vals = [r["test"][head][key] for r in vr if r["test"][head][key] is not None]
                final["variants"][variant][head][key] = vals[0] if key == "support" else {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            final["variants"][variant][head]["gates"] = {}
            for g in ["image", "curve", "structure"]:
                vals = [r["gates"][head][g] for r in vr]
                final["variants"][variant][head]["gates"][g] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    (out_dir / "task_gated_moe_gardner_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("weak")
    a.add_argument("--curve_npz", default="/path/to/embryo_data/20_EventAligned_StructureCurve/outputs/formal/nantes_event_aligned_curve_sequences.npz")
    a.add_argument("--nantes_manifest", default="/path/to/embryo_data/05_Metadata_Manifests/nantes_16phase_manifest.csv")
    a.add_argument("--event_curve_ckpt_template", default="/path/to/checkpoints/event_curve_seed_{seed}.pt")
    a.add_argument("--out_dir", default="outputs/21_task_gated_moe/weak_pretrain")
    a.add_argument("--epochs", type=int, default=60); a.add_argument("--batch_size", type=int, default=64); a.add_argument("--hidden", type=int, default=128); a.add_argument("--dropout", type=float, default=0.25); a.add_argument("--lr", type=float, default=1e-3); a.add_argument("--weight_decay", type=float, default=1e-4); a.add_argument("--device", default="cuda:0"); a.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    b = sub.add_parser("transfer")
    b.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    b.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    b.add_argument("--image_dir", default="/path/to/embryo_data/03_Kromp2344_Gardner/Blastocyst_Dataset/Images")
    b.add_argument("--nantes_ckpt_template", default="/path/to/checkpoints/nantes_seed_{seed}.pt")
    b.add_argument("--weak_ckpt_template", default="/path/to/checkpoints/weak_curve_seed_{seed}.pt")
    b.add_argument("--out_dir", default="outputs/21_task_gated_moe/gardner_transfer")
    b.add_argument("--variants", nargs="+", default=["task_gated_moe_event", "task_gated_moe_weak"])
    b.add_argument("--epochs", type=int, default=35); b.add_argument("--batch_size", type=int, default=48); b.add_argument("--image_size", type=int, default=224); b.add_argument("--num_workers", type=int, default=4); b.add_argument("--hidden", type=int, default=128); b.add_argument("--dropout", type=float, default=0.25); b.add_argument("--lr", type=float, default=1e-4); b.add_argument("--weight_decay", type=float, default=1e-4); b.add_argument("--seq_len", type=int, default=32); b.add_argument("--curve_dim", type=int, default=73); b.add_argument("--device", default="cuda:0"); b.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    args = p.parse_args()
    if args.cmd == "weak": train_weak(args)
    else: train_transfer(args)


if __name__ == "__main__":
    main()
