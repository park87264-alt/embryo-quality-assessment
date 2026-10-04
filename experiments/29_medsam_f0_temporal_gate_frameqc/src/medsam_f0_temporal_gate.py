from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset


PHASES = [
    "tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6",
    "t7", "t8", "t9plus", "tM", "tSB", "tB", "tEB", "tHB",
]
BLASTOCYST_START = PHASES.index("tSB")
EVENT_DEPENDENT_FEATURES = list(range(0, 20)) + [30, 31, 33]
VARIANTS = [
    "structure_event_gate", "medsam_frame", "medsam_temporal",
    "medsam_temporal_gate", "medsam_event_gate",
]
RUN_RE = re.compile(r"RUN(\d+)")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def frame_number(path_or_name: str | Path) -> int:
    match = RUN_RE.search(Path(path_or_name).name)
    return int(match.group(1)) if match else -1


def phase_index(value: str) -> int:
    try:
        return PHASES.index(str(value))
    except ValueError:
        return -1


def load_medsam_class(source_file: Path):
    spec = importlib.util.spec_from_file_location("medsam_lora_sfu", source_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import MedSAM source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MedSAMLoRA


def select_sequence_rows(frame_features: Path, frames_per_embryo: int, max_embryos: int = 0) -> pd.DataFrame:
    table = pd.read_csv(frame_features)
    required = {"embryo_id", "sample_index", "frame", "phase"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Frame feature table lacks columns: {sorted(missing)}")
    embryo_ids = sorted(table["embryo_id"].astype(str).unique())
    if max_embryos:
        embryo_ids = embryo_ids[:max_embryos]
    table = table[table["embryo_id"].astype(str).isin(embryo_ids)].copy()
    selected_groups = []
    for _, group in table.sort_values(["embryo_id", "sample_index"]).groupby("embryo_id", sort=True):
        if len(group) > frames_per_embryo:
            positions = np.linspace(0, len(group) - 1, frames_per_embryo).round().astype(int)
            group = group.iloc[positions]
        selected_groups.append(group)
    return pd.concat(selected_groups, ignore_index=True)


def resolve_image_paths(rows: pd.DataFrame, manifest_path: Path) -> list[Path]:
    manifest = pd.read_csv(manifest_path)
    if not {"embryo_id", "processed_F0_dir"}.issubset(manifest.columns):
        raise ValueError("Manifest must contain embryo_id and processed_F0_dir")
    directory_by_id = {
        str(row.embryo_id): Path(str(row.processed_F0_dir))
        for row in manifest[["embryo_id", "processed_F0_dir"]].itertuples(index=False)
    }
    paths: list[Path] = []
    for row in rows.itertuples(index=False):
        embryo_id = str(row.embryo_id)
        frame = int(row.frame)
        directory = directory_by_id.get(embryo_id)
        if directory is None:
            raise KeyError(f"Missing embryo in manifest: {embryo_id}")
        matches = list(directory.glob(f"*RUN{frame}.jpeg"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one F0 image for {embryo_id} RUN{frame}, found {len(matches)}")
        paths.append(matches[0])
    return paths


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1)).float(), index


def extract(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")

    rows = select_sequence_rows(Path(args.frame_features), args.frames_per_embryo, args.max_embryos)
    structure_columns = [
        column for column in rows.columns
        if column not in {"embryo_id", "sample_index", "frame", "phase"}
    ]
    if len(structure_columns) != 35:
        raise ValueError(f"Expected 35 structural features, found {len(structure_columns)}")
    image_paths = resolve_image_paths(rows, Path(args.manifest))

    device = torch.device(args.device)
    MedSAMLoRA = load_medsam_class(Path(args.medsam_source))
    model = MedSAMLoRA(args.medsam_base, args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    checkpoint = torch.load(args.medsam_lora, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    loader = DataLoader(
        ImageDataset(image_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings = np.full((len(rows), 256), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            features = model.encode_features(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
            embeddings[indices.numpy()] = features
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                print(f"embedding batch {batch_index}/{len(loader)}", flush=True)
    if np.isnan(embeddings).any():
        raise RuntimeError("Some MedSAM embeddings were not written")

    embryo_ids = sorted(rows["embryo_id"].astype(str).unique())
    n, time_steps = len(embryo_ids), args.frames_per_embryo
    image_x = np.zeros((n, time_steps, embeddings.shape[1]), dtype=np.float32)
    structure_x = np.zeros((n, time_steps, len(structure_columns)), dtype=np.float32)
    labels = np.zeros((n, time_steps), dtype=np.int64)
    mask = np.zeros((n, time_steps), dtype=np.float32)
    frames = np.full((n, time_steps), -1, dtype=np.int64)
    paths = np.full((n, time_steps), "", dtype=object)

    row_position = {index: pos for pos, index in enumerate(rows.index)}
    for embryo_pos, embryo_id in enumerate(embryo_ids):
        group = rows[rows["embryo_id"].astype(str) == embryo_id].sort_values("sample_index")
        for time_pos, (row_index, row) in enumerate(group.iterrows()):
            if time_pos >= time_steps:
                break
            flat_pos = row_position[row_index]
            label = phase_index(row["phase"])
            image_x[embryo_pos, time_pos] = embeddings[flat_pos]
            structure_x[embryo_pos, time_pos] = row[structure_columns].to_numpy(np.float32)
            labels[embryo_pos, time_pos] = max(label, 0)
            mask[embryo_pos, time_pos] = float(label >= 0)
            frames[embryo_pos, time_pos] = int(row["frame"])
            paths[embryo_pos, time_pos] = str(image_paths[flat_pos])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_x=image_x,
        structure_x=structure_x,
        labels=labels,
        mask=mask,
        frames=frames,
        paths=paths,
        embryo_ids=np.asarray(embryo_ids),
        phases=np.asarray(PHASES),
        structure_feature_names=np.asarray(structure_columns),
    )
    summary = {
        "task": "Nantes F0 MedSAM-LoRA embedding extraction",
        "output": str(output),
        "embryos": n,
        "frames_per_embryo": time_steps,
        "valid_frames": int(mask.sum()),
        "image_feature_dim": int(image_x.shape[-1]),
        "structure_feature_dim": int(structure_x.shape[-1]),
        "preprocessing": "Existing processed_F0_dir images; 32 points sampled uniformly across each full trajectory; RGB resize 1024; exact legacy MedSAM input scaling [0,1]",
        "medsam_role": "Frozen image encoder only; no segmentation decoder and no GT-derived prompt box",
        "medsam_base": args.medsam_base,
        "medsam_lora": args.medsam_lora,
        "medsam_lora_sha256": sha256_file(Path(args.medsam_lora)),
        "source_frame_features": args.frame_features,
        "source_frame_features_sha256": sha256_file(Path(args.frame_features)),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def load_qc_probabilities(path: Path, embryo_ids: np.ndarray, frames: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path)
    probability_column = "qc_probability_invalid"
    frame_column = "frame_index" if "frame_index" in table.columns else "frame"
    required = {"embryo_id", frame_column, probability_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"QC predictions lack columns: {sorted(missing)}")
    if table.duplicated(["embryo_id", frame_column]).any():
        raise ValueError("QC predictions contain duplicate embryo/frame rows")
    mapping = {
        (str(row["embryo_id"]), int(row[frame_column])): float(row[probability_column])
        for _, row in table.iterrows()
    }
    probabilities = np.full(frames.shape, np.nan, dtype=np.float32)
    for embryo_pos, embryo_id in enumerate(embryo_ids.astype(str)):
        for time_pos, frame in enumerate(frames[embryo_pos]):
            if frame < 0:
                continue
            probabilities[embryo_pos, time_pos] = mapping.get((embryo_id, int(frame)), np.nan)
    active = frames >= 0
    if np.isnan(probabilities[active]).any():
        missing_count = int(np.isnan(probabilities[active]).sum())
        raise ValueError(f"QC predictions do not cover {missing_count} selected frames")
    if not np.isfinite(probabilities[active]).all() or ((probabilities[active] < 0) | (probabilities[active] > 1)).any():
        raise ValueError("QC invalid probabilities must be finite values in [0, 1]")
    probabilities[~active] = 1.0
    return probabilities


def fit_soft_event_gate(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    classifier.fit(scaler.transform(x), y)
    return scaler, classifier


def predict_soft_event_gate(structure_x: np.ndarray, mask: np.ndarray, scaler, classifier) -> np.ndarray:
    flat = structure_x.reshape(-1, structure_x.shape[-1])
    active = mask.reshape(-1) > 0
    gate = np.zeros(len(flat), dtype=np.float32)
    gate[active] = classifier.predict_proba(scaler.transform(flat[active]))[:, 1].astype(np.float32)
    return gate.reshape(mask.shape)


def make_oof_event_gate(
    structure_x: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    embryo_ids: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    gate = np.zeros(mask.shape, dtype=np.float32)
    train_x = structure_x[train_idx].reshape(-1, structure_x.shape[-1])
    train_y = (labels[train_idx].reshape(-1) >= BLASTOCYST_START).astype(np.int64)
    train_active = mask[train_idx].reshape(-1) > 0
    groups = np.repeat(embryo_ids[train_idx], structure_x.shape[1])
    valid_x = train_x[train_active]
    valid_y = train_y[train_active]
    valid_groups = groups[train_active]
    oof = np.zeros(len(valid_y), dtype=np.float32)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(valid_groups))))
    for fold_train, fold_valid in splitter.split(valid_x, valid_y, valid_groups):
        scaler, classifier = fit_soft_event_gate(valid_x[fold_train], valid_y[fold_train], seed)
        oof[fold_valid] = classifier.predict_proba(scaler.transform(valid_x[fold_valid]))[:, 1]
    train_flat = np.zeros(len(train_x), dtype=np.float32)
    train_flat[np.where(train_active)[0]] = oof
    gate[train_idx] = train_flat.reshape(len(train_idx), structure_x.shape[1])

    final_scaler, final_classifier = fit_soft_event_gate(valid_x, valid_y, seed)
    remaining_idx = np.setdiff1d(np.arange(len(structure_x)), train_idx)
    gate[remaining_idx] = predict_soft_event_gate(
        structure_x[remaining_idx], mask[remaining_idx], final_scaler, final_classifier
    )
    audit = {
        "train_gate": "5-fold group-out-of-fold predictions by embryo",
        "validation_test_gate": "single classifier fitted only on training embryos",
        "oof_accuracy_at_0_5": float(accuracy_score(valid_y, oof >= 0.5)),
        "oof_auc": float(roc_auc_score(valid_y, oof)),
        "train_valid_frames": int(train_active.sum()),
    }
    return gate, audit


def apply_event_gate(structure_x: np.ndarray, event_gate: np.ndarray) -> np.ndarray:
    gated = structure_x.copy()
    gated[:, :, EVENT_DEPENDENT_FEATURES] *= event_gate[:, :, None]
    delta = np.zeros_like(gated)
    delta[:, 1:] = gated[:, 1:] - gated[:, :-1]
    time = np.linspace(0, 1, gated.shape[1], dtype=np.float32)[None, :, None]
    time = np.repeat(time, gated.shape[0], axis=0)
    return np.concatenate([gated, delta, time, event_gate[:, :, None]], axis=-1)


def standardize(train: np.ndarray, all_values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    active = train[train_mask > 0]
    mean = active.mean(axis=0)
    std = np.maximum(active.std(axis=0), 1e-6)
    result = (all_values - mean) / std
    return result.astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


class SequenceDataset(Dataset):
    def __init__(self, image_x, structure_x, labels, mask, reliability, indices):
        self.image_x = torch.from_numpy(image_x[indices]).float()
        self.structure_x = torch.from_numpy(structure_x[indices]).float()
        self.labels = torch.from_numpy(labels[indices]).long()
        self.mask = torch.from_numpy(mask[indices]).float()
        self.reliability = torch.from_numpy(reliability[indices]).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.image_x[index], self.structure_x[index], self.labels[index],
            self.mask[index], self.reliability[index],
        )


class MedSAMStageModel(nn.Module):
    def __init__(self, variant: str, image_dim: int, structure_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.variant = variant
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.structure_proj = nn.Sequential(
            nn.LayerNorm(structure_dim), nn.Linear(structure_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        if variant != "medsam_frame":
            self.temporal = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            self.context_proj = nn.Linear(hidden * 2, hidden)
        if variant in {"structure_event_gate", "medsam_temporal_gate", "medsam_event_gate"}:
            self.context_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, len(PHASES)))

    def encode(self, image_x, structure_x, reliability):
        if self.variant == "structure_event_gate":
            frame = self.structure_proj(structure_x)
        else:
            frame = self.image_proj(image_x)
        if self.variant == "medsam_event_gate":
            frame = frame + self.structure_proj(structure_x)
        frame = frame * reliability.unsqueeze(-1)
        if self.variant == "medsam_frame":
            fused = frame
        else:
            context, _ = self.temporal(frame)
            context = self.context_proj(context)
            if self.variant in {"structure_event_gate", "medsam_temporal_gate", "medsam_event_gate"}:
                gate = self.context_gate(torch.cat([frame, context], dim=-1))
                fused = frame + gate * context
            else:
                fused = context
        return fused

    def forward(self, image_x, structure_x, reliability):
        return self.head(self.encode(image_x, structure_x, reliability))


def metrics_from_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels = list(range(len(PHASES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n_frames": len(y_true),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "class_id": i,
                "phase": PHASES[i],
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in labels
        ],
    }


@torch.inference_mode()
def evaluate(model, loader, device, retained_only: bool = False):
    model.eval()
    true, predicted = [], []
    for image_x, structure_x, labels, mask, reliability in loader:
        logits = model(image_x.to(device), structure_x.to(device), reliability.to(device)).cpu()
        active = mask > 0
        if retained_only:
            active &= reliability >= 0.5
        pred = logits.argmax(-1)
        true.extend(labels[active].numpy().tolist())
        predicted.extend(pred[active].numpy().tolist())
    return metrics_from_predictions(true, predicted)


def train(args: argparse.Namespace) -> None:
    data = np.load(args.features, allow_pickle=True)
    image_x = data["image_x"].astype(np.float32)
    structure_raw = data["structure_x"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    base_mask = data["mask"].astype(np.float32)
    frames = data["frames"].astype(np.int64)
    embryo_ids = data["embryo_ids"].astype(str)
    all_idx = np.arange(len(embryo_ids))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.qc_mode == "none":
        probability_invalid = np.zeros(base_mask.shape, dtype=np.float32)
    else:
        if not args.qc_predictions:
            raise ValueError("--qc-predictions is required for qc-mode soft or hard")
        probability_invalid = load_qc_probabilities(Path(args.qc_predictions), embryo_ids, frames)
    reliability = (1.0 - probability_invalid).astype(np.float32)
    if args.qc_mode == "none":
        reliability[:] = 1.0
    elif args.qc_mode == "hard":
        reliability = (probability_invalid < args.qc_threshold).astype(np.float32)

    all_runs = []
    for seed in args.seeds:
        seed_all(seed)
        train_val_idx, test_idx = train_test_split(all_idx, test_size=0.10, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=1 / 9, random_state=seed)
        split = {
            "seed": seed,
            "train_ids": embryo_ids[train_idx].tolist(),
            "val_ids": embryo_ids[val_idx].tolist(),
            "test_ids": embryo_ids[test_idx].tolist(),
            "train_hash": ids_hash(embryo_ids[train_idx].tolist()),
            "val_hash": ids_hash(embryo_ids[val_idx].tolist()),
            "test_hash": ids_hash(embryo_ids[test_idx].tolist()),
        }
        write_json(output / "splits" / f"seed_{seed}.json", split)

        event_gate, event_gate_audit = make_oof_event_gate(
            structure_raw, labels, base_mask, embryo_ids, train_idx, seed
        )
        write_json(output / "splits" / f"seed_{seed}_event_gate_audit.json", event_gate_audit)
        structure_gated = apply_event_gate(structure_raw, event_gate)
        image_norm, image_stats = standardize(image_x[train_idx], image_x, base_mask[train_idx])
        structure_norm, structure_stats = standardize(
            structure_gated[train_idx], structure_gated, base_mask[train_idx]
        )
        write_json(
            output / "splits" / f"seed_{seed}_normalization.json",
            {"image": image_stats, "structure": structure_stats},
        )

        train_loss_mask = base_mask.copy()
        if args.qc_mode == "hard":
            train_loss_mask *= reliability
        active_train_labels = labels[train_idx][train_loss_mask[train_idx] > 0]
        counts = np.bincount(active_train_labels, minlength=len(PHASES)).astype(np.float32)
        class_weights = counts.sum() / np.maximum(counts, 1.0)
        class_weights = torch.from_numpy((class_weights / class_weights.mean()).astype(np.float32)).to(args.device)

        for variant in args.variants:
            seed_all(seed)
            model = MedSAMStageModel(
                variant, image_norm.shape[-1], structure_norm.shape[-1], args.hidden, args.dropout
            ).to(args.device)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                SequenceDataset(image_norm, structure_norm, labels, train_loss_mask, reliability, train_idx),
                batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0,
            )
            val_loader = DataLoader(
                SequenceDataset(image_norm, structure_norm, labels, train_loss_mask, reliability, val_idx),
                batch_size=args.batch_size, shuffle=False, num_workers=0,
            )
            test_loader = DataLoader(
                SequenceDataset(image_norm, structure_norm, labels, base_mask, reliability, test_idx),
                batch_size=args.batch_size, shuffle=False, num_workers=0,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_score = -1.0
            best_state = None
            history = []
            patience_left = args.patience
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch_image, batch_structure, batch_labels, batch_mask, batch_reliability in train_loader:
                    logits = model(
                        batch_image.to(args.device), batch_structure.to(args.device), batch_reliability.to(args.device)
                    )
                    active = batch_mask.to(args.device) > 0
                    loss = F.cross_entropy(logits[active], batch_labels.to(args.device)[active], weight=class_weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                val_metrics = evaluate(model, val_loader, args.device)
                history.append({
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                })
                if val_metrics["macro_f1"] > best_score + 1e-6:
                    best_score = val_metrics["macro_f1"]
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    patience_left = args.patience
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        break
            if best_state is None:
                raise RuntimeError("No validation checkpoint selected")
            model.load_state_dict(best_state)
            test_all = evaluate(model, test_loader, args.device)
            test_retained = evaluate(model, test_loader, args.device, retained_only=True)
            run_dir = output / args.qc_mode / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            torch.save(
                {"model": best_state, "variant": variant, "seed": seed, "best_val_macro_f1": best_score},
                run_dir / "best.pt",
            )
            run = {
                "seed": seed,
                "variant": variant,
                "qc_mode": args.qc_mode,
                "best_val_macro_f1": best_score,
                "test_all_original_frames": test_all,
                "test_retained_frames": test_retained,
                "test_invalid_probability_mean": float(probability_invalid[test_idx][base_mask[test_idx] > 0].mean()),
                "test_retained_fraction": float(
                    (reliability[test_idx][base_mask[test_idx] > 0] >= 0.5).mean()
                ),
            }
            write_json(run_dir / "metrics.json", run)
            all_runs.append(run)
            print(
                f"seed={seed} variant={variant} qc={args.qc_mode} "
                f"acc={test_all['accuracy']:.4f} bacc={test_all['balanced_accuracy']:.4f} "
                f"macro_f1={test_all['macro_f1']:.4f}",
                flush=True,
            )

    aggregate: dict[str, Any] = {}
    for variant in args.variants:
        selected = [run for run in all_runs if run["variant"] == variant]
        aggregate[variant] = {}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            values = np.asarray([run["test_all_original_frames"][metric] for run in selected])
            aggregate[variant][metric] = {
                "mean": float(values.mean()), "std": float(values.std()), "n": len(values)
            }
    summary = {
        "task": "Nantes F0 16-stage classification with frozen MedSAM-LoRA embeddings",
        "endpoint_warning": "This is developmental-stage classification, not Gardner Expansion/ICM/TE grading.",
        "split": "Embryo-level 80/10/10 train/validation/test; validation selects epoch; test evaluated once.",
        "variants": {
            "structure_event_gate": "Structural curves with predicted post-tSB soft masking and a learned temporal-context gate; no MedSAM image embedding.",
            "medsam_frame": "Per-frame MedSAM embedding; no temporal context.",
            "medsam_temporal": "Bidirectional GRU temporal context; no learned residual gate.",
            "medsam_temporal_gate": "Predicted feature-dependent gate controls temporal context; no GT event input.",
            "medsam_event_gate": "MedSAM temporal gate plus structural features softly masked by a train-fitted predicted post-tSB probability.",
        },
        "qc_mode": args.qc_mode,
        "qc_predictions": args.qc_predictions,
        "qc_threshold": args.qc_threshold,
        "aggregate": aggregate,
        "per_seed": all_runs,
    }
    write_json(output / f"summary_{args.qc_mode}.json", summary)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


QUALITY_TASKS = {"ICM": "ICM", "TE": "TE"}
QUALITY_CLASS_TO_INDEX = {"A": 0, "B": 1, "C": 2}


def quality_labels(manifest_path: Path, embryo_ids: np.ndarray):
    manifest = pd.read_csv(manifest_path)
    manifest["embryo_id"] = manifest["embryo_id"].astype(str)
    by_id = manifest.set_index("embryo_id")
    labels = np.zeros((len(embryo_ids), len(QUALITY_TASKS)), dtype=np.int64)
    masks = np.zeros_like(labels, dtype=np.float32)
    for embryo_pos, embryo_id in enumerate(embryo_ids.astype(str)):
        if embryo_id not in by_id.index:
            continue
        row = by_id.loc[embryo_id]
        for task_pos, column in enumerate(QUALITY_TASKS.values()):
            value = str(row[column]).strip().upper()
            if value in QUALITY_CLASS_TO_INDEX:
                labels[embryo_pos, task_pos] = QUALITY_CLASS_TO_INDEX[value]
                masks[embryo_pos, task_pos] = 1.0
    return labels, masks


def quality_split(
    candidate_idx: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray,
    seed: int,
    holdout_fraction: float = 0.15,
):
    # A single predeclared split avoids selecting a favorable test set by inspecting its labels.
    train_val_idx, test_idx = train_test_split(candidate_idx, test_size=holdout_fraction, random_state=seed)
    relative_val = holdout_fraction / (1.0 - holdout_fraction)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val, random_state=seed + 1)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


class QualityDataset(Dataset):
    def __init__(
        self, image_x, structure_x, sequence_mask, reliability, event_gate,
        labels, label_mask, indices,
    ):
        self.image_x = torch.from_numpy(image_x[indices]).float()
        self.structure_x = torch.from_numpy(structure_x[indices]).float()
        self.sequence_mask = torch.from_numpy(sequence_mask[indices]).float()
        self.reliability = torch.from_numpy(reliability[indices]).float()
        self.event_gate = torch.from_numpy(event_gate[indices]).float()
        self.labels = torch.from_numpy(labels[indices]).long()
        self.label_mask = torch.from_numpy(label_mask[indices]).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.image_x[index], self.structure_x[index], self.sequence_mask[index],
            self.reliability[index], self.event_gate[index], self.labels[index], self.label_mask[index],
        )


class WeakGardnerModel(nn.Module):
    def __init__(self, backbone: MedSAMStageModel, hidden: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.pool_score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.heads = nn.ModuleDict({
            task: nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 3))
            for task in QUALITY_TASKS
        })

    def forward(self, image_x, structure_x, sequence_mask, reliability, event_gate):
        features = self.backbone.encode(image_x, structure_x, reliability)
        score = self.pool_score(features).squeeze(-1)
        prior = torch.clamp(event_gate, min=1e-4)
        score = score + torch.log(prior)
        score = score.masked_fill(sequence_mask <= 0, -1e4)
        weights = torch.softmax(score, dim=1)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return {task: head(pooled) for task, head in self.heads.items()}


def quality_metrics(true_by_task, pred_by_task):
    result = {}
    for task in QUALITY_TASKS:
        result[task] = metrics_from_predictions_3class(true_by_task[task], pred_by_task[task])
    return result


def metrics_from_predictions_3class(y_true: list[int], y_pred: list[int]):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "support": len(y_true),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "class": ["A", "B", "C"][i], "support": int(support[i]),
                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            }
            for i in labels
        ],
    }


@torch.inference_mode()
def evaluate_quality(model, loader, device):
    model.eval()
    true = {task: [] for task in QUALITY_TASKS}
    predicted = {task: [] for task in QUALITY_TASKS}
    for image_x, structure_x, sequence_mask, reliability, event_gate, labels, label_mask in loader:
        outputs = model(
            image_x.to(device), structure_x.to(device), sequence_mask.to(device),
            reliability.to(device), event_gate.to(device),
        )
        for task_pos, task in enumerate(QUALITY_TASKS):
            active = label_mask[:, task_pos] > 0
            true[task].extend(labels[:, task_pos][active].numpy().tolist())
            predicted[task].extend(outputs[task].argmax(-1).cpu()[active].numpy().tolist())
    return quality_metrics(true, predicted)


def train_quality(args: argparse.Namespace) -> None:
    data = np.load(args.features, allow_pickle=True)
    image_x = data["image_x"].astype(np.float32)
    structure_raw = data["structure_x"].astype(np.float32)
    stage_labels = data["labels"].astype(np.int64)
    sequence_mask = data["mask"].astype(np.float32)
    frames = data["frames"].astype(np.int64)
    embryo_ids = data["embryo_ids"].astype(str)
    labels, label_mask = quality_labels(Path(args.manifest), embryo_ids)
    candidate_idx = np.where(label_mask.sum(axis=1) > 0)[0]
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.qc_mode == "none":
        probability_invalid = np.zeros(sequence_mask.shape, dtype=np.float32)
    else:
        if not args.qc_predictions:
            raise ValueError("--qc-predictions is required for quality QC experiments")
        probability_invalid = load_qc_probabilities(Path(args.qc_predictions), embryo_ids, frames)
    reliability = 1.0 - probability_invalid
    if args.qc_mode == "hard":
        reliability = (probability_invalid < args.qc_threshold).astype(np.float32)

    runs = []
    for seed in args.seeds:
        train_idx, val_idx, test_idx = quality_split(
            candidate_idx, labels, label_mask, seed, args.holdout_fraction
        )
        split = {
            "seed": seed,
            "protocol": "Fixed embryo-level 70/15/15 split among embryos with at least one ICM/TE label; no test-label-driven retries",
            "train_ids": embryo_ids[train_idx].tolist(),
            "val_ids": embryo_ids[val_idx].tolist(),
            "test_ids": embryo_ids[test_idx].tolist(),
            "class_support": {
                part: {
                    task: np.bincount(
                        labels[indices, pos][label_mask[indices, pos] > 0], minlength=3
                    ).tolist()
                    for pos, task in enumerate(QUALITY_TASKS)
                }
                for part, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx))
            },
        }
        write_json(output / "quality_splits" / f"seed_{seed}.json", split)
        event_gate, gate_audit = make_oof_event_gate(
            structure_raw, stage_labels, sequence_mask, embryo_ids, train_idx, seed
        )
        write_json(output / "quality_splits" / f"seed_{seed}_event_gate_audit.json", gate_audit)
        image_norm, _ = standardize(image_x[train_idx], image_x, sequence_mask[train_idx])
        structure_gated = apply_event_gate(structure_raw, event_gate)
        structure_norm, _ = standardize(
            structure_gated[train_idx], structure_gated, sequence_mask[train_idx]
        )
        class_weights = {}
        for task_pos, task in enumerate(QUALITY_TASKS):
            active = label_mask[train_idx, task_pos] > 0
            counts = np.bincount(labels[train_idx, task_pos][active], minlength=3).astype(np.float32)
            weights = counts.sum() / np.maximum(counts, 1.0)
            class_weights[task] = torch.from_numpy((weights / weights.mean()).astype(np.float32)).to(args.device)

        for variant in args.variants:
            seed_all(seed)
            backbone = MedSAMStageModel(
                variant, image_norm.shape[-1], structure_norm.shape[-1], args.hidden, args.dropout
            )
            stage_checkpoint = Path(args.stage_dir) / "none" / variant / f"seed_{seed}" / "best.pt"
            pretrained = False
            if args.use_stage_pretrain and stage_checkpoint.exists():
                checkpoint = torch.load(stage_checkpoint, map_location="cpu")
                backbone.load_state_dict(checkpoint["model"], strict=True)
                pretrained = True
            model = WeakGardnerModel(backbone, args.hidden, args.dropout).to(args.device)
            generator = torch.Generator().manual_seed(seed)
            dataset_args = (
                image_norm, structure_norm, sequence_mask, reliability, event_gate, labels, label_mask
            )
            train_loader = DataLoader(
                QualityDataset(*dataset_args, train_idx), batch_size=args.batch_size,
                shuffle=True, generator=generator, num_workers=0,
            )
            val_loader = DataLoader(
                QualityDataset(*dataset_args, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0,
            )
            test_loader = DataLoader(
                QualityDataset(*dataset_args, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_score, best_state, patience_left = -1.0, None, args.patience
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch in train_loader:
                    image_batch, structure_batch, seq_mask_batch, reliability_batch, event_batch, y_batch, y_mask = batch
                    outputs = model(
                        image_batch.to(args.device), structure_batch.to(args.device), seq_mask_batch.to(args.device),
                        reliability_batch.to(args.device), event_batch.to(args.device),
                    )
                    loss, used = 0.0, 0
                    for task_pos, task in enumerate(QUALITY_TASKS):
                        active = y_mask[:, task_pos].to(args.device) > 0
                        if active.any():
                            loss = loss + F.cross_entropy(
                                outputs[task][active], y_batch[:, task_pos].to(args.device)[active],
                                weight=class_weights[task],
                            )
                            used += 1
                    loss = loss / max(used, 1)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                val_metrics = evaluate_quality(model, val_loader, args.device)
                score = float(np.mean([val_metrics[task]["macro_f1"] for task in QUALITY_TASKS]))
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_macro_f1": score})
                if score > best_score + 1e-6:
                    best_score = score
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    patience_left = args.patience
                else:
                    patience_left -= 1
                    if patience_left == 0:
                        break
            if best_state is None:
                raise RuntimeError("No quality checkpoint selected")
            model.load_state_dict(best_state)
            test_metrics = evaluate_quality(model, test_loader, args.device)
            run_dir = output / "quality" / args.qc_mode / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            run = {
                "seed": seed, "variant": variant, "pretrained_from_stage": pretrained,
                "best_val_mean_macro_f1": best_score, "test": test_metrics,
            }
            torch.save({"model": best_state, **run}, run_dir / "best.pt")
            write_json(run_dir / "metrics.json", run)
            runs.append(run)
            print(
                f"quality seed={seed} variant={variant} "
                f"ICM_acc={test_metrics['ICM']['accuracy']:.4f} ICM_f1={test_metrics['ICM']['macro_f1']:.4f} "
                f"TE_acc={test_metrics['TE']['accuracy']:.4f} TE_f1={test_metrics['TE']['macro_f1']:.4f}",
                flush=True,
            )

    aggregate = {}
    for variant in args.variants:
        selected = [run for run in runs if run["variant"] == variant]
        aggregate[variant] = {}
        for task in QUALITY_TASKS:
            aggregate[variant][task] = {}
            for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
                values = np.asarray([run["test"][task][metric] for run in selected])
                aggregate[variant][task][metric] = {
                    "mean": float(values.mean()), "std": float(values.std()), "n": len(values)
                }
    summary = {
        "task": "Exploratory Nantes F0 embryo-level ICM/TE grading",
        "label_scope": {
            "TE": int(label_mask[:, 1].sum()), "ICM": int(label_mask[:, 0].sum()),
            "Expansion": 0,
        },
        "warning": "These sparse legacy Nantes labels are auxiliary weak supervision, not the pending complete expert Gardner gold standard.",
        "qc_mode": args.qc_mode,
        "aggregate": aggregate,
        "per_seed": runs,
    }
    write_json(output / f"quality_summary_{args.qc_mode}.json", summary)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


def make_qc_index(args: argparse.Namespace) -> None:
    data = np.load(args.features, allow_pickle=True)
    embryo_ids = data["embryo_ids"].astype(str)
    frames = data["frames"].astype(np.int64)
    paths = data["paths"].astype(str)
    mask = data["mask"].astype(np.float32)
    records = []
    for embryo_pos, embryo_id in enumerate(embryo_ids):
        for time_pos, frame in enumerate(frames[embryo_pos]):
            if mask[embryo_pos, time_pos] <= 0 or frame < 0:
                continue
            records.append({
                "embryo_id": embryo_id,
                "frame_index": int(frame),
                "split": "inference",
                "image_path": paths[embryo_pos, time_pos],
            })
    table = pd.DataFrame(records)
    if table.duplicated(["embryo_id", "frame_index"]).any():
        raise RuntimeError("Generated QC index has duplicate embryo/frame rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    summary = {
        "output": str(output), "rows": len(table),
        "embryos": int(table["embryo_id"].nunique()),
        "note": "Paired QC inference index for the exact F0 frames used by the downstream experiment.",
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--frame-features", required=True)
    extract_parser.add_argument("--manifest", required=True)
    extract_parser.add_argument("--medsam-source", required=True)
    extract_parser.add_argument("--medsam-base", required=True)
    extract_parser.add_argument("--medsam-lora", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--device", default="cuda:1")
    extract_parser.add_argument("--batch-size", type=int, default=2)
    extract_parser.add_argument("--num-workers", type=int, default=2)
    extract_parser.add_argument("--frames-per-embryo", type=int, default=32)
    extract_parser.add_argument("--max-embryos", type=int, default=0)
    extract_parser.add_argument("--lora-r", type=int, default=4)
    extract_parser.add_argument("--lora-alpha", type=float, default=8.0)
    extract_parser.add_argument("--lora-dropout", type=float, default=0.05)
    extract_parser.add_argument("--seed", type=int, default=42)
    extract_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--features", required=True)
    train_parser.add_argument("--out-dir", required=True)
    train_parser.add_argument("--device", default="cuda:1")
    train_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    train_parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--patience", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--hidden", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.25)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--qc-mode", choices=["none", "soft", "hard"], default="none")
    train_parser.add_argument("--qc-predictions", default="")
    train_parser.add_argument("--qc-threshold", type=float, default=0.391845703125)

    quality_parser = subparsers.add_parser("train-quality")
    quality_parser.add_argument("--features", required=True)
    quality_parser.add_argument("--manifest", required=True)
    quality_parser.add_argument("--stage-dir", required=True)
    quality_parser.add_argument("--out-dir", required=True)
    quality_parser.add_argument("--device", default="cuda:1")
    quality_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    quality_parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    quality_parser.add_argument("--epochs", type=int, default=100)
    quality_parser.add_argument("--patience", type=int, default=15)
    quality_parser.add_argument("--batch-size", type=int, default=32)
    quality_parser.add_argument("--hidden", type=int, default=128)
    quality_parser.add_argument("--dropout", type=float, default=0.30)
    quality_parser.add_argument("--lr", type=float, default=3e-4)
    quality_parser.add_argument("--weight-decay", type=float, default=1e-4)
    quality_parser.add_argument("--holdout-fraction", type=float, default=0.15)
    quality_parser.add_argument("--qc-mode", choices=["none", "soft", "hard"], default="none")
    quality_parser.add_argument("--qc-predictions", default="")
    quality_parser.add_argument("--qc-threshold", type=float, default=0.391845703125)
    quality_parser.add_argument(
        "--use-stage-pretrain", action="store_true",
        help="Opt in only when the stage-pretraining split excludes the quality validation/test embryos.",
    )

    qc_index_parser = subparsers.add_parser("make-qc-index")
    qc_index_parser.add_argument("--features", required=True)
    qc_index_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    elif args.command == "train":
        train(args)
    elif args.command == "train-quality":
        train_quality(args)
    else:
        make_qc_index(args)


if __name__ == "__main__":
    main()
