from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PHASES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9plus", "tM", "tSB", "tB", "tEB", "tHB"]
BLASTOCYST_START = PHASES.index("tSB")
TASKS = {"exp": ("Expansion", 5), "icm": ("ICM", 3), "te": ("TE", 3)}
VARIANTS = ["raw_curve", "mask_only", "mask_gt_flag", "soft_pred_gate"]

# 35 structure features: ICM 0:10, TE 10:20, ZP 20:30, derived 30:35.
# These positions depend on ICM/TE and are suppressed before blastocyst formation.
EVENT_DEPENDENT_FEATURES = list(range(0, 20)) + [30, 31, 33]


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_checksum(state: dict[str, torch.Tensor], prefixes: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if prefixes and not key.startswith(prefixes):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def ids_checksum(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def clone_cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def phase_index(value: str) -> int:
    try:
        return PHASES.index(str(value))
    except ValueError:
        return -1


def load_nantes_sequences(frame_features: Path, frames_per_embryo: int = 32):
    df = pd.read_csv(frame_features)
    required = {"embryo_id", "sample_index", "phase"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    feature_names = [c for c in df.columns if c not in {"embryo_id", "sample_index", "frame", "phase"}]
    if len(feature_names) != 35:
        raise ValueError(f"Expected 35 structure features, found {len(feature_names)}")
    sequences, labels, masks, embryo_ids = [], [], [], []
    for embryo_id, group in df.groupby("embryo_id", sort=True):
        group = group.sort_values("sample_index").head(frames_per_embryo)
        x = group[feature_names].to_numpy(np.float32)
        y = np.asarray([phase_index(p) for p in group["phase"]], dtype=np.int64)
        m = (y >= 0).astype(np.float32)
        y[y < 0] = 0
        if len(x) < frames_per_embryo:
            pad = frames_per_embryo - len(x)
            x = np.pad(x, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad))
            m = np.pad(m, (0, pad))
        sequences.append(x)
        labels.append(y)
        masks.append(m)
        embryo_ids.append(str(embryo_id))
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(masks, dtype=np.float32),
        embryo_ids,
        feature_names,
    )


def augment_curve(base: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(base)
    delta[:, 1:] = base[:, 1:] - base[:, :-1]
    time = np.linspace(0.0, 1.0, base.shape[1], dtype=np.float32)[None, :, None]
    time = np.repeat(time, base.shape[0], axis=0)
    return np.concatenate([base, delta, time], axis=-1).astype(np.float32)


def apply_soft_mask(base: np.ndarray, gate: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[:, :, EVENT_DEPENDENT_FEATURES] *= gate[:, :, None]
    return out


def fit_logistic_gate(x: np.ndarray, y: np.ndarray, seed: int):
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    classifier.fit(xs, y)
    return scaler, classifier


def predict_logistic_gate(x: np.ndarray, scaler: StandardScaler, classifier: LogisticRegression) -> np.ndarray:
    return classifier.predict_proba(scaler.transform(x))[:, 1].astype(np.float32)


def gate_state(scaler: StandardScaler, classifier: LogisticRegression) -> dict[str, Any]:
    return {
        "type": "standard_scaler_logistic_regression",
        "feature_dim": int(len(scaler.mean_)),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "classes": classifier.classes_.tolist(),
    }


def gate_from_state(x: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(state["mean"], dtype=np.float32)
    scale = np.asarray(state["scale"], dtype=np.float32)
    coef = np.asarray(state["coef"], dtype=np.float32)
    z = ((x - mean) / np.maximum(scale, 1e-12)) @ coef + float(state["intercept"])
    return (1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))).astype(np.float32)


def make_oof_soft_gates(
    base: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    embryo_ids: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
):
    gate = np.zeros(labels.shape, dtype=np.float32)
    frame_x = base[train_idx].reshape(-1, base.shape[-1])
    frame_y = (labels[train_idx].reshape(-1) >= BLASTOCYST_START).astype(np.int64)
    frame_m = masks[train_idx].reshape(-1) > 0
    groups = np.repeat(np.asarray(embryo_ids)[train_idx], base.shape[1])
    valid_positions = np.where(frame_m)[0]
    x_valid, y_valid, groups_valid = frame_x[frame_m], frame_y[frame_m], groups[frame_m]
    oof = np.zeros(len(x_valid), dtype=np.float32)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups_valid))))
    for fold_train, fold_valid in splitter.split(x_valid, y_valid, groups_valid):
        scaler, classifier = fit_logistic_gate(x_valid[fold_train], y_valid[fold_train], seed)
        oof[fold_valid] = predict_logistic_gate(x_valid[fold_valid], scaler, classifier)
    train_flat_gate = np.zeros(len(frame_x), dtype=np.float32)
    train_flat_gate[valid_positions] = oof
    gate[train_idx] = train_flat_gate.reshape(len(train_idx), base.shape[1])

    final_scaler, final_classifier = fit_logistic_gate(x_valid, y_valid, seed)
    test_x = base[test_idx].reshape(-1, base.shape[-1])
    test_m = masks[test_idx].reshape(-1) > 0
    test_gate = np.zeros(len(test_x), dtype=np.float32)
    test_gate[test_m] = predict_logistic_gate(test_x[test_m], final_scaler, final_classifier)
    gate[test_idx] = test_gate.reshape(len(test_idx), base.shape[1])
    state = gate_state(final_scaler, final_classifier)
    audit = {
        "train_gate_mode": "5-fold group-out-of-fold by embryo",
        "test_gate_mode": "model fitted on all training embryos",
        "train_valid_frames": int(frame_m.sum()),
        "test_valid_frames": int(test_m.sum()),
        "oof_accuracy_at_0_5": float(accuracy_score(y_valid, oof >= 0.5)),
        "oof_auc": float(roc_auc_score(y_valid, oof)),
        "oof_probability_mean": float(oof.mean()),
    }
    return gate, state, audit


def variant_inputs(base: np.ndarray, labels: np.ndarray, masks: np.ndarray, soft_gate: np.ndarray, variant: str):
    hard_gate = ((labels >= BLASTOCYST_START) & (masks > 0)).astype(np.float32)
    if variant == "raw_curve":
        transformed = base
        scalar = np.zeros_like(hard_gate)
    elif variant == "mask_only":
        transformed = apply_soft_mask(base, hard_gate)
        scalar = np.zeros_like(hard_gate)
    elif variant == "mask_gt_flag":
        transformed = apply_soft_mask(base, hard_gate)
        scalar = hard_gate
    elif variant == "soft_pred_gate":
        transformed = apply_soft_mask(base, soft_gate)
        scalar = soft_gate
    else:
        raise ValueError(variant)
    scalar = (scalar * masks).astype(np.float32)
    return augment_curve(transformed), scalar


class CurveDataset(Dataset):
    def __init__(self, x: np.ndarray, gate: np.ndarray, y: np.ndarray, mask: np.ndarray, indices: np.ndarray):
        self.x = torch.from_numpy(x[indices]).float()
        self.gate = torch.from_numpy(gate[indices]).float()
        self.y = torch.from_numpy(y[indices]).long()
        self.mask = torch.from_numpy(mask[indices]).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.gate[index], self.y[index], self.mask[index]


class TemporalCurveEncoder(nn.Module):
    def __init__(self, in_dim: int = 71, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z, _ = self.gru(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(x).mean(dim=1)


class CurvePhasePretrainer(nn.Module):
    def __init__(self, in_dim: int = 71, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.encoder = TemporalCurveEncoder(in_dim, hidden, dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2 + 1), nn.Dropout(dropout), nn.Linear(hidden * 2 + 1, len(PHASES)))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode_sequence(x)
        return self.head(torch.cat([z, gate.unsqueeze(-1)], dim=-1))


def phase_metrics(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, gate, y, mask in loader:
            pred = model(x.to(device), gate.to(device)).argmax(-1).cpu()
            active = mask > 0
            y_true.extend(y[active].numpy().tolist())
            y_pred.extend(pred[active].numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(PHASES))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_frames": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
    }


def pretrain_variants(args) -> dict[str, Any]:
    out_root = Path(args.out_dir) / "pretrain"
    base, labels, masks, embryo_ids, feature_names = load_nantes_sequences(Path(args.frame_features), args.frames_per_embryo)
    all_idx = np.arange(len(base))
    runs = []
    for seed in args.seeds:
        train_idx, test_idx = train_test_split(all_idx, test_size=0.30, random_state=seed)
        soft_gate, fitted_gate, gate_audit = make_oof_soft_gates(base, labels, masks, embryo_ids, train_idx, test_idx, seed)
        split_dir = Path(args.out_dir) / "splits" / f"seed_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            split_dir / "nantes_pretrain_split.json",
            {
                "seed": seed,
                "train_ids": [embryo_ids[i] for i in train_idx],
                "test_ids": [embryo_ids[i] for i in test_idx],
                "train_hash": ids_checksum([embryo_ids[i] for i in train_idx]),
                "test_hash": ids_checksum([embryo_ids[i] for i in test_idx]),
            },
        )
        write_json(split_dir / "soft_gate_model.json", fitted_gate)
        write_json(split_dir / "soft_gate_audit.json", gate_audit)

        seed_all(seed)
        template = CurvePhasePretrainer(71, args.curve_hidden, args.dropout)
        common_state = clone_cpu_state(template)
        common_hash = tensor_checksum(common_state)
        for variant in args.variants:
            seed_all(seed)
            x, event_scalar = variant_inputs(base, labels, masks, soft_gate, variant)
            model = CurvePhasePretrainer(71, args.curve_hidden, args.dropout).to(args.device)
            model.load_state_dict(common_state, strict=True)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                CurveDataset(x, event_scalar, labels, masks, train_idx),
                batch_size=args.pretrain_batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
            test_loader = DataLoader(
                CurveDataset(x, event_scalar, labels, masks, test_idx),
                batch_size=args.pretrain_batch_size,
                shuffle=False,
                num_workers=0,
            )
            active_y = labels[train_idx][masks[train_idx] > 0]
            counts = np.bincount(active_y, minlength=len(PHASES)).astype(np.float32)
            class_weight = counts.sum() / np.maximum(counts, 1.0)
            class_weight = torch.from_numpy((class_weight / class_weight.mean()).astype(np.float32)).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
            best, best_state, history = {"macro_f1": -1.0}, None, []
            for epoch in range(1, args.pretrain_epochs + 1):
                model.train()
                losses = []
                for batch_x, batch_gate, batch_y, batch_mask in train_loader:
                    batch_x = batch_x.to(args.device)
                    batch_gate = batch_gate.to(args.device)
                    batch_y = batch_y.to(args.device)
                    batch_mask = batch_mask.to(args.device)
                    logits = model(batch_x, batch_gate)
                    active = batch_mask > 0
                    loss = F.cross_entropy(logits[active], batch_y[active], weight=class_weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                metrics = phase_metrics(model, test_loader, args.device)
                metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
                history.append({k: v for k, v in metrics.items() if k != "confusion_matrix"})
                if metrics["macro_f1"] > best["macro_f1"]:
                    best = metrics
                    best_state = clone_cpu_state(model)
            if best_state is None:
                raise RuntimeError("No pretraining checkpoint selected")
            model.load_state_dict(best_state, strict=True)
            variant_dir = out_root / variant / f"seed_{seed}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "encoder": clone_cpu_state(model.encoder),
                "model": best_state,
                "in_dim": 71,
                "hidden": args.curve_hidden,
                "variant": variant,
                "seed": seed,
                "phases": PHASES,
                "feature_names": feature_names + [f"delta_{name}" for name in feature_names] + ["time_norm"],
                "best_metrics": best,
                "common_initialization_hash": common_hash,
            }
            torch.save(checkpoint, variant_dir / "curve_encoder_best.pt")
            pd.DataFrame(history).to_csv(variant_dir / "history.csv", index=False)
            write_json(variant_dir / "best_metrics.json", best)
            runs.append({"seed": seed, "variant": variant, "common_initialization_hash": common_hash, **{k: v for k, v in best.items() if k != "confusion_matrix"}})
            print(f"PRETRAIN seed={seed} variant={variant} macro_f1={best['macro_f1']:.4f}", flush=True)
    summary = aggregate_simple_runs(runs, ["accuracy", "balanced_accuracy", "macro_f1"])
    summary.update({"task": "Nantes 16-phase fair event-alignment ablation", "n_embryos": len(embryo_ids), "feature_dim": 71, "per_seed": runs})
    write_json(out_root / "pretrain_summary.json", summary)
    return summary


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        result = float(value)
        return result if np.isfinite(result) else default
    except Exception:
        return default


def bbox_wh(area: float, aspect: float) -> tuple[float, float]:
    area, aspect = max(0.0, area), max(1e-3, aspect)
    return min(math.sqrt(area * aspect), 1.0), min(math.sqrt(area / aspect), 1.0)


def kromp_raw_features(row: dict[str, str]) -> np.ndarray:
    features, areas = [], {}
    for prefix in ["icm", "te", "zp"]:
        area = safe_float(row, f"{prefix}_area")
        bbox_area = safe_float(row, f"{prefix}_bbox_area")
        fill = safe_float(row, f"{prefix}_fill_ratio")
        cx = safe_float(row, f"{prefix}_cx")
        cy = safe_float(row, f"{prefix}_cy")
        aspect = safe_float(row, f"{prefix}_aspect", 1.0)
        width, height = bbox_wh(bbox_area, aspect)
        areas[prefix] = max(area, 0.0)
        features.extend([area, area, cx, cy, width, height, bbox_area, 0.0, fill, 0.0])
    total = sum(areas.values()) + 1e-6
    features.extend(
        [
            areas["icm"] / total,
            areas["te"] / total,
            areas["zp"] / total,
            (areas["icm"] + areas["te"]) / (areas["zp"] + 1e-6),
            safe_float(row, "structure_total_area"),
        ]
    )
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (35,):
        raise ValueError(result.shape)
    return result


def kromp_variant_curve(row: dict[str, str], variant: str, gate_model: dict[str, Any], seq_len: int):
    raw = kromp_raw_features(row)
    if variant == "soft_pred_gate":
        probability = float(gate_from_state(raw[None], gate_model)[0])
        base = raw.copy()
        base[EVENT_DEPENDENT_FEATURES] *= probability
        scalar = probability
    elif variant == "mask_gt_flag":
        base, scalar = raw, 1.0
    elif variant in {"raw_curve", "mask_only"}:
        base, scalar = raw, 0.0
    else:
        raise ValueError(variant)
    sequence = np.repeat(base[None], seq_len, axis=0)
    delta = np.zeros_like(sequence)
    time = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)[:, None]
    return np.concatenate([sequence, delta, time], axis=1).astype(np.float32), np.float32(scalar)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_transform(train: bool, image_size: int):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class KrompDataset(Dataset):
    def __init__(self, rows, image_dir: Path, train: bool, image_size: int, seq_len: int, variant: str, gate_model: dict[str, Any]):
        self.rows = rows
        self.image_dir = image_dir
        self.transform = image_transform(train, image_size)
        self.seq_len = seq_len
        self.variant = variant
        self.gate_model = gate_model

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = self.transform(Image.open(self.image_dir / row["image"]).convert("RGB"))
        labels, masks = [], []
        for task in ["exp", "icm", "te"]:
            y = int(float(row[task]))
            labels.append(max(y, 0))
            masks.append(float(y >= 0))
        curve, gate = kromp_variant_curve(row, self.variant, self.gate_model, self.seq_len)
        return {
            "image": image,
            "curve": torch.from_numpy(curve),
            "gate": torch.tensor(gate),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.tensor(masks, dtype=torch.float32),
            "name": row["image"],
        }


class GardnerModel(nn.Module):
    def __init__(self, curve_hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.curve = TemporalCurveEncoder(71, curve_hidden, dropout)
        fused_dim = image_dim + curve_hidden * 2 + 1
        self.norm = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(dropout))
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(nn.Linear(fused_dim, image_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(image_dim // 2, n_classes))
                for task, (_, n_classes) in TASKS.items()
            }
        )

    def forward(self, image: torch.Tensor, curve: torch.Tensor, gate: torch.Tensor):
        image_feature = self.backbone(image)
        curve_feature = self.curve(curve)
        fused = self.norm(torch.cat([image_feature, curve_feature, gate[:, None]], dim=1))
        return {task: head(fused) for task, head in self.heads.items()}


def verified_load_backbone(model: GardnerModel, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    target = model.backbone.state_dict()
    candidates = {key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")}
    matched = {key: value for key, value in candidates.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
    shape_mismatch = {key: {"source": list(value.shape), "target": list(target[key].shape)} for key, value in candidates.items() if key in target and tuple(value.shape) != tuple(target[key].shape)}
    if not matched:
        raise RuntimeError(f"Zero backbone keys loaded from {checkpoint_path}")
    before = clone_cpu_state(model.backbone)
    missing, unexpected = model.backbone.load_state_dict(matched, strict=False)
    after = clone_cpu_state(model.backbone)
    max_change = max(float((after[key] - before[key]).abs().max()) for key in matched)
    if max_change == 0.0:
        raise RuntimeError(f"Backbone load made no parameter change: {checkpoint_path}")
    return {
        "source": str(checkpoint_path),
        "loaded_key_count": len(matched),
        "loaded_keys": sorted(matched),
        "shape_mismatch": shape_mismatch,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "checksum_before": tensor_checksum(before),
        "checksum_after": tensor_checksum(after),
        "max_abs_parameter_change": max_change,
    }


def extract_encoder_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint.get("encoder"), dict):
        return checkpoint["encoder"]
    state = checkpoint.get("model", checkpoint)
    direct = {key: value for key, value in state.items() if key.startswith(("proj.", "gru."))}
    if direct:
        return direct
    for prefix in ["encoder.", "curve."]:
        mapped = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix) and key.removeprefix(prefix).startswith(("proj.", "gru."))}
        if mapped:
            return mapped
    return {}


def verified_load_curve(model: GardnerModel, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = extract_encoder_state(checkpoint)
    target = model.curve.state_dict()
    matched = {key: value for key, value in source.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
    mismatched = {key: {"source": list(value.shape), "target": list(target[key].shape)} for key, value in source.items() if key in target and tuple(value.shape) != tuple(target[key].shape)}
    if not matched:
        raise RuntimeError(f"Zero curve encoder keys loaded from {checkpoint_path}")
    if len(matched) != len(target):
        raise RuntimeError(f"Incomplete curve encoder load from {checkpoint_path}: {len(matched)}/{len(target)}, mismatched={mismatched}")
    before = clone_cpu_state(model.curve)
    missing, unexpected = model.curve.load_state_dict(matched, strict=True)
    after = clone_cpu_state(model.curve)
    max_change = max(float((after[key] - before[key]).abs().max()) for key in matched)
    if max_change == 0.0:
        raise RuntimeError(f"Curve load made no parameter change: {checkpoint_path}")
    return {
        "source": str(checkpoint_path),
        "checkpoint_variant": checkpoint.get("variant"),
        "loaded_key_count": len(matched),
        "target_key_count": len(target),
        "loaded_keys": sorted(matched),
        "shape_mismatch": mismatched,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "checksum_before": tensor_checksum(before),
        "checksum_after": tensor_checksum(after),
        "max_abs_parameter_change": max_change,
    }


def audit_legacy_event_checkpoint(path: Path, hidden: int, dropout: float) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    source = extract_encoder_state(checkpoint)
    in_dim = int(checkpoint.get("in_dim", 73))
    model = TemporalCurveEncoder(in_dim, hidden, dropout)
    target = model.state_dict()
    matched = {key: value for key, value in source.items() if key in target and tuple(value.shape) == tuple(target[key].shape)}
    if len(matched) != len(target):
        raise RuntimeError(f"Legacy event checkpoint load audit failed: {len(matched)}/{len(target)}")
    before = clone_cpu_state(model)
    model.load_state_dict(matched, strict=True)
    after = clone_cpu_state(model)
    return {
        "source": str(path),
        "checkpoint_top_level_keys": list(checkpoint),
        "detected_encoder_layout": "direct proj./gru. keys" if any(key.startswith("proj.") for key in checkpoint.get("model", {})) else "prefixed",
        "in_dim": in_dim,
        "loaded_key_count": len(matched),
        "target_key_count": len(target),
        "loaded_keys": sorted(matched),
        "checksum_before": tensor_checksum(before),
        "checksum_after": tensor_checksum(after),
        "max_abs_parameter_change": max(float((after[key] - before[key]).abs().max()) for key in matched),
        "status": "verified_nonzero_complete_load",
    }


def class_weights(rows: list[dict[str, str]], task: str, n_classes: int, exponent: float = 0.5) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float32)
    for row in rows:
        y = int(float(row[task]))
        if y >= 0:
            counts[y] += 1
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** exponent
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 2.0)
    return torch.from_numpy(weights.astype(np.float32))


def multitask_loss(outputs, labels, masks, weights, device):
    losses = []
    for index, task in enumerate(["exp", "icm", "te"]):
        active = masks[:, index] > 0
        if active.any():
            losses.append(F.cross_entropy(outputs[task][active], labels[:, index][active], weight=weights[task].to(device)))
    return torch.stack(losses).mean()


def task_metrics(y_true: list[int], y_pred: list[int], y_prob: list[list[float]], n_classes: int) -> dict[str, Any]:
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    per_class = []
    probabilities = np.asarray(y_prob, dtype=np.float64)
    for class_id in labels:
        tp = int(cm[class_id, class_id])
        fn = int(cm[class_id, :].sum() - tp)
        fp = int(cm[:, class_id].sum() - tp)
        tn = total - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        binary = np.asarray(y_true) == class_id
        auc = float(roc_auc_score(binary, probabilities[:, class_id])) if binary.any() and (~binary).any() else None
        per_class.append(
            {
                "class_id": class_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "f1": f1,
                "auc_ovr": auc,
                "support": int(cm[class_id, :].sum()),
            }
        )
    try:
        y_binary = label_binarize(y_true, classes=labels)
        macro_auc = float(roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr"))
        weighted_auc = float(roc_auc_score(y_binary, probabilities, average="weighted", multi_class="ovr"))
    except ValueError:
        macro_auc = weighted_auc = None
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)
    return {
        "support": total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_specificity": float(np.mean([item["specificity"] for item in per_class])),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_auc_ovr": macro_auc,
        "weighted_auc_ovr": weighted_auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": normalized.tolist(),
    }


def predict_gardner(model: GardnerModel, loader: DataLoader, device: torch.device):
    truth = {task: [] for task in TASKS}
    prediction = {task: [] for task in TASKS}
    probability = {task: [] for task in TASKS}
    records = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device), batch["curve"].to(device), batch["gate"].to(device))
            labels, masks = batch["labels"], batch["masks"]
            batch_probabilities = {task: torch.softmax(outputs[task], dim=1).cpu().numpy() for task in TASKS}
            for sample_index, name in enumerate(batch["name"]):
                record = {"image": name, "event_gate": float(batch["gate"][sample_index])}
                for task_index, task in enumerate(["exp", "icm", "te"]):
                    if masks[sample_index, task_index] <= 0:
                        record[f"{task}_true"] = -1
                        record[f"{task}_pred"] = -1
                        continue
                    true = int(labels[sample_index, task_index])
                    probs = batch_probabilities[task][sample_index]
                    pred = int(probs.argmax())
                    truth[task].append(true)
                    prediction[task].append(pred)
                    probability[task].append(probs.tolist())
                    record[f"{task}_true"] = true
                    record[f"{task}_pred"] = pred
                    for class_id, value in enumerate(probs):
                        record[f"{task}_prob_{class_id}"] = float(value)
                records.append(record)
    metrics = {display: task_metrics(truth[task], prediction[task], probability[task], n_classes) for task, (display, n_classes) in TASKS.items()}
    return metrics, records


def save_confusion_outputs(metrics: dict[str, Any], output_dir: Path, prefix: str = "test") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for display, values in metrics.items():
        stem = display.lower()
        cm = np.asarray(values["confusion_matrix"], dtype=np.int64)
        normalized = np.asarray(values["confusion_matrix_row_normalized"], dtype=np.float64)
        labels = [str(i) for i in range(len(cm))]
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{prefix}_{stem}_confusion_counts.csv")
        pd.DataFrame(normalized, index=labels, columns=labels).to_csv(output_dir / f"{prefix}_{stem}_confusion_row_normalized.csv")
        write_json(
            output_dir / f"{prefix}_{stem}_confusion.json",
            {"class_labels": labels, "counts": cm, "row_normalized": normalized, "per_class": values["per_class"]},
        )
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
        for axis, matrix, title, fmt in [
            (axes[0], cm, f"{display} counts", "d"),
            (axes[1], normalized, f"{display} row-normalized", ".2f"),
        ]:
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=(matrix.max() if fmt == "d" else 1.0))
            axis.set_xticks(range(len(labels)), labels)
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            axis.set_title(title)
            threshold = float(matrix.max()) / 2 if fmt == "d" else 0.5
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix[row, column]
                    axis.text(column, row, format(int(value), fmt) if fmt == "d" else format(value, fmt), ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.savefig(output_dir / f"{prefix}_{stem}_confusion.png", dpi=180)
        plt.close(fig)


def selection_score(metrics: dict[str, Any]) -> float:
    return float(np.mean([0.5 * values["accuracy"] + 0.5 * values["macro_f1"] for values in metrics.values()]))


def aggregate_simple_runs(runs: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    aggregate = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        if not selected:
            continue
        aggregate[variant] = {}
        for metric in metric_names:
            values = [float(run[metric]) for run in selected]
            aggregate[variant][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
    return {"aggregate": aggregate}


def train_gardner_variants(args) -> dict[str, Any]:
    out_root = Path(args.out_dir) / "transfer"
    train_rows = read_rows(Path(args.train_features))
    test_rows = read_rows(Path(args.test_features))
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_test_rows:
        test_rows = test_rows[: args.max_test_rows]
    all_runs, fairness = [], {"seeds": {}, "global_settings": {}}
    for seed in args.seeds:
        train_subset, val_rows = train_test_split(train_rows, test_size=0.15, random_state=seed)
        train_ids = [row["image"] for row in train_subset]
        val_ids = [row["image"] for row in val_rows]
        test_ids = [row["image"] for row in test_rows]
        split_dir = Path(args.out_dir) / "splits" / f"seed_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_record = {
            "seed": seed,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "train_hash": ids_checksum(train_ids),
            "val_hash": ids_checksum(val_ids),
            "test_hash": ids_checksum(test_ids),
        }
        write_json(split_dir / "kromp_split.json", split_record)
        gate_model = json.loads((split_dir / "soft_gate_model.json").read_text(encoding="utf-8"))

        seed_all(seed)
        template = GardnerModel(args.curve_hidden, args.dropout)
        random_initialization_hash = tensor_checksum(clone_cpu_state(template))
        nantes_path = Path(args.nantes_ckpt_template.format(seed=seed))
        if not nantes_path.exists():
            raise FileNotFoundError(nantes_path)
        backbone_audit = verified_load_backbone(template, nantes_path)
        common_state = clone_cpu_state(template)
        common_hash = tensor_checksum(common_state)
        common_head_hash = tensor_checksum(common_state, prefixes=("norm.", "heads."))
        seed_audit = {
            "split_hashes": {key: split_record[key] for key in ["train_hash", "val_hash", "test_hash"]},
            "random_initialization_hash": random_initialization_hash,
            "common_state_after_nantes_hash": common_hash,
            "common_head_hash": common_head_hash,
            "backbone_load": backbone_audit,
            "variants": {},
        }

        for variant in args.variants:
            print(f"TRANSFER seed={seed} variant={variant}", flush=True)
            seed_all(seed)
            model = GardnerModel(args.curve_hidden, args.dropout)
            model.load_state_dict(common_state, strict=True)
            checkpoint_path = Path(args.out_dir) / "pretrain" / variant / f"seed_{seed}" / "curve_encoder_best.pt"
            curve_audit = verified_load_curve(model, checkpoint_path)
            head_hash_after_curve_load = tensor_checksum(clone_cpu_state(model), prefixes=("norm.", "heads."))
            if head_hash_after_curve_load != common_head_hash:
                raise RuntimeError("Curve loading changed Gardner fusion/head initialization")
            model = model.to(args.device)

            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                KrompDataset(train_subset, Path(args.image_dir), True, args.image_size, args.seq_len, variant, gate_model),
                batch_size=args.transfer_batch_size,
                shuffle=True,
                generator=generator,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            val_loader = DataLoader(
                KrompDataset(val_rows, Path(args.image_dir), False, args.image_size, args.seq_len, variant, gate_model),
                batch_size=args.transfer_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            test_loader = DataLoader(
                KrompDataset(test_rows, Path(args.image_dir), False, args.image_size, args.seq_len, variant, gate_model),
                batch_size=args.transfer_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            weights = {task: class_weights(train_subset, task, n_classes, args.class_weight_exponent) for task, (_, n_classes) in TASKS.items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.transfer_lr, weight_decay=args.weight_decay)
            variant_dir = out_root / variant / f"seed_{seed}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            best_score, best_epoch, history = -1.0, -1, []
            for epoch in range(1, args.transfer_epochs + 1):
                model.train()
                losses = []
                for batch in train_loader:
                    outputs = model(batch["image"].to(args.device), batch["curve"].to(args.device), batch["gate"].to(args.device))
                    loss = multitask_loss(outputs, batch["labels"].to(args.device), batch["masks"].to(args.device), weights, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                val_metrics, _ = predict_gardner(model, val_loader, args.device)
                score = selection_score(val_metrics)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "val_selection_score": score,
                        **{f"val_{display}_{metric}": values[metric] for display, values in val_metrics.items() for metric in ["accuracy", "balanced_accuracy", "macro_f1"]},
                    }
                )
                if score > best_score:
                    best_score, best_epoch = score, epoch
                    torch.save({"model": clone_cpu_state(model), "epoch": epoch, "selection_score": score, "val_metrics": val_metrics}, variant_dir / "best.pt")
            best_checkpoint = torch.load(variant_dir / "best.pt", map_location="cpu")
            model.load_state_dict(best_checkpoint["model"], strict=True)
            model = model.to(args.device)
            test_metrics, predictions = predict_gardner(model, test_loader, args.device)
            pd.DataFrame(history).to_csv(variant_dir / "history.csv", index=False)
            pd.DataFrame(predictions).to_csv(variant_dir / "test_predictions.csv", index=False)
            save_confusion_outputs(test_metrics, variant_dir)
            run_summary = {
                "seed": seed,
                "variant": variant,
                "best_epoch": best_epoch,
                "best_val_selection_score": best_score,
                "selection_rule": "mean across Gardner heads of 0.5*Accuracy + 0.5*macro-F1",
                "test": test_metrics,
                "checkpoint_load": {"backbone": backbone_audit, "curve": curve_audit},
                "fairness": {
                    "split_hashes": seed_audit["split_hashes"],
                    "common_state_hash": common_hash,
                    "common_head_hash_before_curve_load": common_head_hash,
                    "head_hash_after_curve_load": head_hash_after_curve_load,
                },
            }
            write_json(variant_dir / "summary.json", run_summary)
            all_runs.append(run_summary)
            seed_audit["variants"][variant] = {
                "curve_load": curve_audit,
                "head_hash_after_curve_load": head_hash_after_curve_load,
                "split_hashes": split_record,
            }
            print(
                "RESULT " + " ".join(
                    [
                        f"seed={seed}",
                        f"variant={variant}",
                        *[f"{display}_acc={values['accuracy']:.4f} {display}_f1={values['macro_f1']:.4f}" for display, values in test_metrics.items()],
                    ]
                ),
                flush=True,
            )
        fairness["seeds"][str(seed)] = seed_audit

    summary = aggregate_transfer_runs(all_runs, out_root)
    summary["per_seed"] = all_runs
    write_json(out_root / "fair_ablation_summary.json", summary)
    fairness["global_settings"] = {
        "variants": args.variants,
        "seeds": args.seeds,
        "optimizer": "AdamW",
        "transfer_lr": args.transfer_lr,
        "weight_decay": args.weight_decay,
        "class_weight_formula": f"clipped normalized inverse-frequency**{args.class_weight_exponent}",
        "class_weight_clip": [0.5, 2.0],
        "batch_size": args.transfer_batch_size,
        "epochs": args.transfer_epochs,
        "num_workers": args.num_workers,
        "selection_rule": "mean across heads of 0.5*Accuracy + 0.5*macro-F1",
        "test_set_used_for_selection": False,
    }
    write_json(Path(args.out_dir) / "fairness_audit.json", fairness)
    return summary


def aggregate_transfer_runs(runs: list[dict[str, Any]], out_root: Path) -> dict[str, Any]:
    summary = {"variants": {}, "comparison": {}}
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_specificity",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "macro_auc_ovr",
        "weighted_auc_ovr",
    ]
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        if not selected:
            continue
        summary["variants"][variant] = {}
        aggregate_dir = out_root / variant / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        aggregate_metrics = {}
        for display, (_, n_classes) in zip(["Expansion", "ICM", "TE"], TASKS.values()):
            aggregate_metrics[display] = {}
            for metric in metrics:
                values = [run["test"][display][metric] for run in selected if run["test"][display][metric] is not None]
                aggregate_metrics[display][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)} if values else None
            summed = np.sum([np.asarray(run["test"][display]["confusion_matrix"], dtype=np.int64) for run in selected], axis=0)
            row_sums = summed.sum(axis=1, keepdims=True)
            normalized = np.divide(summed, row_sums, out=np.zeros_like(summed, dtype=np.float64), where=row_sums != 0)
            aggregate_metrics[display]["summed_confusion_matrix"] = summed.tolist()
            aggregate_metrics[display]["summed_confusion_matrix_row_normalized"] = normalized.tolist()
            fake = {display: {"confusion_matrix": summed.tolist(), "confusion_matrix_row_normalized": normalized.tolist(), "per_class": []}}
            save_confusion_outputs(fake, aggregate_dir, prefix="summed")
        summary["variants"][variant] = aggregate_metrics
    baseline = summary["variants"].get("raw_curve")
    for variant, values in summary["variants"].items():
        if variant == "raw_curve" or baseline is None:
            continue
        summary["comparison"][f"{variant}_minus_raw_curve"] = {
            display: {
                metric: values[display][metric]["mean"] - baseline[display][metric]["mean"]
                for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"]
                if values[display][metric] is not None and baseline[display][metric] is not None
            }
            for display in ["Expansion", "ICM", "TE"]
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pretrain", "transfer", "all", "audit_legacy"])
    parser.add_argument("--frame_features", default="/path/to/embryo_data/18_Nantes_StructureCurve_Pretrain/outputs/formal/nantes_structure_curve_frame_features.csv")
    parser.add_argument("--train_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_train_structure_features.csv")
    parser.add_argument("--test_features", default="/path/to/embryo_data/12_StructureTemporal_Quality/outputs/kromp_gold_structure_features.csv")
    parser.add_argument("--image_dir", default="/path/to/embryo_data/03_Kromp2344_Gardner/Blastocyst_Dataset/Images")
    parser.add_argument("--nantes_ckpt_template", default="/path/to/checkpoints/nantes_seed_{seed}.pt")
    parser.add_argument("--legacy_event_ckpt_template", default="/path/to/checkpoints/legacy_event_curve_seed_{seed}.pt")
    parser.add_argument("--out_dir", default="outputs/22_event_alignment_fair_ablation/formal")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--frames_per_embryo", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--curve_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=35)
    parser.add_argument("--pretrain_batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--transfer_epochs", type=int, default=35)
    parser.add_argument("--transfer_batch_size", type=int, default=48)
    parser.add_argument("--transfer_lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--class_weight_exponent", type=float, default=0.5)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_test_rows", type=int, default=0)
    args = parser.parse_args()
    if not set(args.variants).issubset(VARIANTS):
        raise ValueError(args.variants)
    args.device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    return args


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    legacy_audits = []
    for seed in args.seeds:
        path = Path(args.legacy_event_ckpt_template.format(seed=seed))
        if path.exists():
            legacy_audits.append({"seed": seed, **audit_legacy_event_checkpoint(path, args.curve_hidden, args.dropout)})
    write_json(Path(args.out_dir) / "legacy_event_checkpoint_audit.json", {"audits": legacy_audits})
    if args.command == "audit_legacy":
        print(json.dumps(legacy_audits, indent=2, ensure_ascii=False))
        return
    if args.command in {"pretrain", "all"}:
        pretrain_variants(args)
    if args.command in {"transfer", "all"}:
        train_gardner_variants(args)


if __name__ == "__main__":
    main()
