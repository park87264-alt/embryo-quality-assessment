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
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset


TASKS = {"exp": ("Expansion", 5), "icm": ("ICM", 3), "te": ("TE", 3)}
REGIONS = ["global", "icm", "te", "zp"]
STRUCTURE_COLUMNS = [
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


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_roi_features(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(name): feat.astype(np.float32) for name, feat in zip(data["images"], data["features"])}


def safe_float(row: dict, key: str) -> float:
    try:
        val = row.get(key, "")
        if val in ("", "NA", "nan", None):
            return 0.0
        out = float(val)
        if not np.isfinite(out):
            return 0.0
        return out
    except Exception:
        return 0.0


def fit_structure_scaler(rows: list[dict]) -> StandardScaler:
    arr = np.asarray([[safe_float(r, c) for c in STRUCTURE_COLUMNS] for r in rows], dtype=np.float32)
    return StandardScaler().fit(arr)


def structure_array(rows: list[dict], scaler: StandardScaler) -> dict[str, np.ndarray]:
    out = {}
    for r in rows:
        arr = np.asarray([[safe_float(r, c) for c in STRUCTURE_COLUMNS]], dtype=np.float32)
        out[r["image"]] = scaler.transform(arr)[0].astype(np.float32)
    return out


class GardnerFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], roi_features: dict[str, np.ndarray], structure_features: dict[str, np.ndarray], variant: str):
        self.rows = rows
        self.roi_features = roi_features
        self.structure_features = structure_features
        self.variant = variant

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = row["image"]
        roi = self.roi_features[image]  # 4, 256
        structure = self.structure_features[image]
        temporal = np.zeros(1, dtype=np.float32)
        multifocal = np.zeros(1, dtype=np.float32)
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(row[task]))
            if y < 0:
                labels.append(0)
                masks.append(0.0)
            else:
                labels.append(y)
                masks.append(1.0)
        return {
            "roi": torch.from_numpy(roi).float(),
            "structure": torch.from_numpy(structure).float(),
            "temporal": torch.from_numpy(temporal).float(),
            "multifocal": torch.from_numpy(multifocal).float(),
            "labels": torch.tensor(labels).long(),
            "masks": torch.tensor(masks).float(),
        }


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

    def forward(self, batch):
        roi = batch["roi"]
        structure = batch["structure"]
        if self.mode == "global":
            x = roi[:, 0]
        elif self.mode == "roi_concat":
            x = roi.reshape(roi.shape[0], -1)
        elif self.mode == "global_structure":
            x = torch.cat([roi[:, 0], structure], dim=1)
        elif self.mode == "roi_structure":
            x = torch.cat([roi.reshape(roi.shape[0], -1), structure], dim=1)
        else:
            raise ValueError(self.mode)
        h = self.trunk(x)
        return {task: head(h) for task, head in self.heads.items()}


class FullTaskAwareExperts(nn.Module):
    def __init__(self, structure_dim: int, hidden: int = 256, dropout: float = 0.25, use_roi: bool = True, use_structure: bool = True):
        super().__init__()
        self.use_roi = use_roi
        self.use_structure = use_structure
        self.region_proj = nn.Sequential(nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.structure_proj = nn.Sequential(nn.Linear(structure_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal_proj = nn.Sequential(nn.Linear(1, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.multifocal_proj = nn.Sequential(nn.Linear(1, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gates = nn.ModuleDict({task: nn.Linear(hidden, 1) for task in TASKS})
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

    def forward(self, batch):
        roi = batch["roi"]
        region_h = self.region_proj(roi)
        global_h = region_h[:, 0:1]
        roi_h = region_h[:, 1:] if self.use_roi else region_h[:, 1:] * 0.0
        structure_h = self.structure_proj(batch["structure"]).unsqueeze(1) if self.use_structure else self.structure_proj(batch["structure"]).unsqueeze(1) * 0.0
        temporal_h = self.temporal_proj(batch["temporal"]).unsqueeze(1) * 0.0
        multifocal_h = self.multifocal_proj(batch["multifocal"]).unsqueeze(1) * 0.0
        tokens = torch.cat([global_h, roi_h, structure_h, temporal_h, multifocal_h], dim=1)
        available = torch.ones(tokens.shape[:2], device=tokens.device)
        if not self.use_roi:
            available[:, 1:4] = 0
        if not self.use_structure:
            available[:, 4] = 0
        available[:, 5] = 0  # temporal branch has no aligned Kromp values in current data.
        available[:, 6] = 0  # multifocal branch has no aligned Kromp values in current data.
        outputs, gate_weights = {}, {}
        for task in TASKS:
            scores = self.gates[task](tokens).squeeze(-1)
            scores = scores.masked_fill(available <= 0, -1e4)
            weights = torch.softmax(scores, dim=1)
            fused = (tokens * weights.unsqueeze(-1)).sum(dim=1)
            outputs[task] = self.heads[task](fused)
            gate_weights[task] = weights
        return outputs, gate_weights


def make_model(variant: str, hidden: int, dropout: float):
    if variant in {"global", "roi_concat", "global_structure", "roi_structure"}:
        dims = {"global": 256, "roi_concat": 1024, "global_structure": 256 + len(STRUCTURE_COLUMNS), "roi_structure": 1024 + len(STRUCTURE_COLUMNS)}
        model = MLPHeads(dims[variant], hidden, dropout)
        model.mode = variant
        return model
    if variant == "full_expert":
        return FullTaskAwareExperts(len(STRUCTURE_COLUMNS), hidden, dropout, use_roi=True, use_structure=True)
    if variant == "no_roi_expert":
        return FullTaskAwareExperts(len(STRUCTURE_COLUMNS), hidden, dropout, use_roi=False, use_structure=True)
    if variant == "no_structure_expert":
        return FullTaskAwareExperts(len(STRUCTURE_COLUMNS), hidden, dropout, use_roi=True, use_structure=False)
    raise ValueError(variant)


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


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_model(model, batch):
    out = model(batch)
    if isinstance(out, tuple):
        return out[0]
    return out


def predict(model, loader, device):
    y_true = {t: [] for t in TASKS}
    y_pred = {t: [] for t in TASKS}
    y_prob = {t: [] for t in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = forward_model(model, batch)
            labels = batch["labels"].detach().cpu()
            masks = batch["masks"].detach().cpu()
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


def collect_gate_weights(model, loader, device):
    if not isinstance(model, FullTaskAwareExperts):
        return None
    token_names = ["global", "icm_roi", "te_roi", "zp_roi", "structure", "temporal", "multifocal"]
    sums = {task: [] for task in TASKS}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            _, gates = model(batch)
            for task, w in gates.items():
                sums[task].append(w.detach().cpu().numpy())
    return {TASKS[t][0]: dict(zip(token_names, np.concatenate(v, axis=0).mean(axis=0).astype(float).tolist())) for t, v in sums.items()}


def audit_alignment(args, rows):
    nantes = Path(args.nantes_manifest)
    nantes_ids = set()
    if nantes.exists():
        with nantes.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nantes_ids.add(str(r.get("embryo_id", "")))
    kromp_stems = {Path(r["image"]).stem for r in rows}
    overlap = sorted(kromp_stems & nantes_ids)
    return {
        "kromp_images": len(kromp_stems),
        "nantes_embryos": len(nantes_ids),
        "kromp_nantes_id_overlap": len(overlap),
        "overlap_examples": overlap[:10],
        "temporal_branch_status": "implemented as model interface, not activated for Gardner training because Kromp/Gardner samples do not align with Nantes temporal embryo IDs.",
        "multifocal_branch_status": "implemented as model interface, not activated for Gardner training because Kromp/Gardner samples do not have paired F0/F±15/F±30/F±45 focal planes.",
    }


def train_one(args, seed: int, variant: str):
    seed_all(seed)
    out_dir = Path(args.out_dir) / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(Path(args.train_features))
    test_rows = read_rows(Path(args.test_features))
    tr_rows, val_rows = train_test_split(train_rows, test_size=0.15, random_state=seed)
    roi_features = load_roi_features(Path(args.roi_feature_file))
    scaler = fit_structure_scaler(tr_rows)
    structure_features = structure_array(train_rows + test_rows, scaler)
    tr = GardnerFeatureDataset(tr_rows, roi_features, structure_features, variant)
    val = GardnerFeatureDataset(val_rows, roi_features, structure_features, variant)
    te = GardnerFeatureDataset(test_rows, roi_features, structure_features, variant)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    te_loader = DataLoader(te, batch_size=256, shuffle=False)
    device = torch.device(args.device)
    model = make_model(variant, args.hidden, args.dropout).to(device)
    weights = {task: class_weights(tr_rows, task, n) for task, (_, n) in TASKS.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tr_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            outputs = forward_model(model, batch)
            loss = loss_fn(outputs, batch["labels"], batch["masks"], weights, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        yt, yp, ypb = predict(model, val_loader, device)
        val_metrics = metrics(yt, yp, ypb)
        val_score = float(np.mean([v["macro_f1"] for v in val_metrics.values()]))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mean_macro_f1": val_score})
        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    yt, yp, ypb = predict(model, te_loader, device)
    test_metrics = metrics(yt, yp, ypb)
    gate_summary = collect_gate_weights(model, te_loader, device)
    summary = {"seed": seed, "variant": variant, "best_val_mean_macro_f1": best, "test": test_metrics, "gate_summary": gate_summary, "history": history}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def summarize(args, runs, alignment):
    final = {"alignment_audit": alignment, "variants": {}, "notes": []}
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
        gates = [r["gate_summary"] for r in vruns if r["gate_summary"] is not None]
        if gates:
            final["variants"][variant]["gate_summary"] = {}
            for head in ["Expansion", "ICM", "TE"]:
                final["variants"][variant]["gate_summary"][head] = {}
                for token in gates[0][head]:
                    vals = [g[head][token] for g in gates]
                    final["variants"][variant]["gate_summary"][head][token] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    final["notes"].append("Full expert code contains global MedSAM-LoRA, ROI, structure, temporal and multifocal branches.")
    final["notes"].append("Temporal and multifocal branches are disabled in this Gardner experiment because no sample-level Kromp/Gardner alignment is available; this avoids invalid cross-dataset feature leakage.")
    out_path = Path(args.out_dir) / "full_multibranch_ablation_summary.json"
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roi_feature_file", default="/path/to/embryo_data/15_TaskAware_ROI_Gating/outputs/formal/kromp_medsam_roi_region_features.npz")
    p.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    p.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    p.add_argument("--nantes_manifest", default="/path/to/embryo_data/05_Metadata_Manifests/nantes_16phase_manifest.csv")
    p.add_argument("--out_dir", default="outputs/16_full_multibranch_gardner/formal")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--variants", nargs="+", default=["global", "roi_concat", "global_structure", "roi_structure", "full_expert", "no_roi_expert", "no_structure_expert"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_rows = read_rows(Path(args.train_features)) + read_rows(Path(args.test_features))
    alignment = audit_alignment(args, all_rows)
    (Path(args.out_dir) / "data_alignment_audit.json").write_text(json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8")
    runs = []
    for seed in args.seeds:
        for variant in args.variants:
            runs.append(train_one(args, seed, variant))
    summarize(args, runs, alignment)


if __name__ == "__main__":
    main()
