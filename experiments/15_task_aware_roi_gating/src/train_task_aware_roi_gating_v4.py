from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


TASKS = {"exp": ("Expansion", 5), "icm": ("ICM", 3), "te": ("TE", 3)}
REGIONS = ["global", "icm", "te", "zp"]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_roi_features(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(name): feat.astype(np.float32) for name, feat in zip(data["images"], data["features"])}


class GardnerROIDataset(Dataset):
    def __init__(self, rows: list[dict], feature_map: dict[str, np.ndarray], mode: str):
        self.rows = rows
        self.feature_map = feature_map
        self.mode = mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        feat = self.feature_map[r["image"]]  # 4, 256
        if self.mode == "global":
            x = feat[0]
        elif self.mode == "concat":
            x = feat.reshape(-1)
        else:
            x = feat
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(r[task]))
            if y < 0:
                labels.append(0)
                masks.append(0.0)
            else:
                labels.append(y)
                masks.append(1.0)
        return torch.from_numpy(x).float(), torch.tensor(labels).long(), torch.tensor(masks).float()


class MLPHeads(nn.Module):
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
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, (_, n) in TASKS.items()})

    def forward(self, x):
        h = self.trunk(x)
        return {task: head(h) for task, head in self.heads.items()}


class TaskAwareGatedHeads(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.region_proj = nn.Sequential(nn.Linear(feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gates = nn.ModuleDict({task: nn.Linear(feature_dim, 1) for task in TASKS})
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n),
                )
                for task, (_, n) in TASKS.items()
            }
        )

    def forward(self, x):
        # x: B,4,256
        region_h = self.region_proj(x)
        out = {}
        gate_weights = {}
        for task in TASKS:
            score = self.gates[task](x).squeeze(-1)
            weights = torch.softmax(score, dim=1)
            fused = (region_h * weights.unsqueeze(-1)).sum(dim=1)
            out[task] = self.heads[task](fused)
            gate_weights[task] = weights
        return out, gate_weights


def class_weights(rows, task, n_classes):
    counts = np.zeros(n_classes, dtype=np.float32)
    for r in rows:
        y = int(float(r[task]))
        if y >= 0:
            counts[y] += 1
    w = counts.sum() / np.maximum(counts, 1)
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32))


def loss_fn(outputs, labels, masks, weights, device):
    total = 0.0
    used = 0
    for i, task in enumerate(["exp", "icm", "te"]):
        active = masks[:, i] > 0
        if active.any():
            ce = nn.CrossEntropyLoss(weight=weights[task].to(device))
            total = total + ce(outputs[task][active], labels[:, i][active])
            used += 1
    return total / max(used, 1)


def forward_model(model, x, variant):
    if variant == "gated":
        return model(x)[0]
    return model(x)


def collect_gate_weights(model, loader, device):
    if not isinstance(model, TaskAwareGatedHeads):
        return None
    sums = {task: [] for task in TASKS}
    model.eval()
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            _, gates = model(x)
            for task, w in gates.items():
                sums[task].append(w.detach().cpu().numpy())
    return {TASKS[t][0]: dict(zip(REGIONS, np.concatenate(v, axis=0).mean(axis=0).astype(float).tolist())) for t, v in sums.items()}


def predict(model, loader, device, variant):
    y_true = {t: [] for t in TASKS}
    y_pred = {t: [] for t in TASKS}
    y_prob = {t: [] for t in TASKS}
    model.eval()
    with torch.no_grad():
        for x, labels, masks in loader:
            x = x.to(device)
            out = forward_model(model, x, variant)
            for i, task in enumerate(["exp", "icm", "te"]):
                active = masks[:, i] > 0
                if active.any():
                    prob = torch.softmax(out[task].detach().cpu(), dim=1).numpy()
                    pred = prob.argmax(axis=1)
                    idx = active.numpy().astype(bool)
                    y_true[task].extend(labels[:, i].numpy()[idx].tolist())
                    y_pred[task].extend(pred[idx].tolist())
                    y_prob[task].extend(prob[idx].tolist())
    return y_true, y_pred, y_prob


def safe_auc(y_true, prob, n_classes):
    try:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.shape[1] < 2:
            return None
        return float(roc_auc_score(y_bin, np.asarray(prob), average="macro", multi_class="ovr"))
    except Exception:
        return None


def metrics(y_true, y_pred, y_prob):
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


def train_variant(args, seed: int, variant: str):
    seed_all(seed)
    mode = "global" if variant == "global" else ("concat" if variant == "concat" else "gated")
    out_dir = Path(args.out_dir) / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_map = load_roi_features(Path(args.roi_feature_file))
    train_rows = read_rows(Path(args.train_features))
    test_rows = read_rows(Path(args.test_features))
    tr_rows, val_rows = train_test_split(train_rows, test_size=0.15, random_state=seed)
    tr = GardnerROIDataset(tr_rows, feature_map, mode)
    val = GardnerROIDataset(val_rows, feature_map, mode)
    te = GardnerROIDataset(test_rows, feature_map, mode)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    te_loader = DataLoader(te, batch_size=256, shuffle=False)
    device = torch.device(args.device)
    if variant == "global":
        model = MLPHeads(256, args.hidden, args.dropout).to(device)
    elif variant == "concat":
        model = MLPHeads(4 * 256, args.hidden, args.dropout).to(device)
    else:
        model = TaskAwareGatedHeads(256, args.hidden, args.dropout).to(device)
    weights = {task: class_weights(tr_rows, task, n) for task, (_, n) in TASKS.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, labels, masks in tr_loader:
            x, labels, masks = x.to(device), labels.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            outputs = forward_model(model, x, variant)
            loss = loss_fn(outputs, labels, masks, weights, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        yt, yp, ypb = predict(model, val_loader, device, variant)
        val_metrics = metrics(yt, yp, ypb)
        val_score = float(np.mean([v["macro_f1"] for v in val_metrics.values()]))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mean_macro_f1": val_score})
        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    yt, yp, ypb = predict(model, te_loader, device, variant)
    test_metrics = metrics(yt, yp, ypb)
    gate_summary = collect_gate_weights(model, te_loader, device)
    summary = {"seed": seed, "variant": variant, "best_val_mean_macro_f1": best, "test": test_metrics, "gate_summary": gate_summary, "history": history}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"seed": seed, "variant": variant, "test": test_metrics, "gate_summary": gate_summary}, ensure_ascii=False))


def summarize(args):
    base = Path(args.out_dir)
    final = {}
    for variant in args.variants:
        runs = [json.loads((base / variant / f"seed_{seed}" / "summary.json").read_text(encoding="utf-8")) for seed in args.seeds]
        final[variant] = {}
        for head in ["Expansion", "ICM", "TE"]:
            final[variant][head] = {}
            for key in runs[0]["test"][head].keys():
                vals = [r["test"][head][key] for r in runs if r["test"][head][key] is not None]
                if key == "support":
                    final[variant][head][key] = vals[0]
                else:
                    final[variant][head][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        if variant == "gated":
            final[variant]["gate_summary"] = {}
            for head in ["Expansion", "ICM", "TE"]:
                final[variant]["gate_summary"][head] = {}
                for region in REGIONS:
                    vals = [r["gate_summary"][head][region] for r in runs]
                    final[variant]["gate_summary"][head][region] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    final["interpretation"] = "Task-aware gating evaluates whether Gardner heads can select global/ICM/TE/ZP ROI features instead of using naive concatenation."
    (base / "task_aware_roi_gating_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roi_feature_file", default="/path/to/embryo_data/15_TaskAware_ROI_Gating/outputs/formal/kromp_medsam_roi_region_features.npz")
    p.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    p.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    p.add_argument("--out_dir", default="outputs/15_task_aware_roi_gating/formal")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", default=["global", "concat", "gated"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    args = p.parse_args()
    for seed in args.seeds:
        for variant in args.variants:
            train_variant(args, seed, variant)
    summarize(args)


if __name__ == "__main__":
    main()
