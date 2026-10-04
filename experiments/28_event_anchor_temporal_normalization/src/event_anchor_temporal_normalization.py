from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, train_test_split


PHASES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9plus", "tM", "tSB", "tB", "tEB", "tHB"]
ANCHORS = ["tSB", "tB", "tEB"]
VARIANTS = [
    "uniform_raw",
    "uniform_gt_gate",
    "uniform_pred_gate",
    "anchor_gt_raw",
    "anchor_gt_gate",
    "anchor_pred_raw",
    "anchor_pred_gate",
]
EVENT_DEPENDENT_FEATURES = list(range(0, 20)) + [30, 31, 33]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def enforce_anchor_order(relative: np.ndarray, minimum_gap: float = 0.015) -> np.ndarray:
    values = np.asarray(relative, dtype=np.float64).copy()
    values = np.clip(values, 0.02, 0.98)
    for row in values:
        row.sort()
        row[1] = max(row[1], row[0] + minimum_gap)
        row[2] = max(row[2], row[1] + minimum_gap)
        if row[2] > 0.98:
            row -= row[2] - 0.98
        if row[0] < 0.02:
            row += 0.02 - row[0]
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def sequence_descriptor(features: np.ndarray, frames: np.ndarray) -> np.ndarray:
    position = (frames - frames[0]) / max(frames[-1] - frames[0], 1.0)
    pieces = [
        features.mean(0),
        features.std(0),
        features.min(0),
        features.max(0),
        features[-1] - features[0],
    ]
    slopes = []
    centered = position - position.mean()
    denom = float(np.square(centered).sum()) + 1e-8
    for column in features.T:
        slopes.append(float((centered * (column - column.mean())).sum() / denom))
    pieces.append(np.asarray(slopes, dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def interpolate_features(source_frames: np.ndarray, source_x: np.ndarray, target_frames: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_frames, source_frames, source_x[:, column]) for column in range(source_x.shape[1])],
        axis=1,
    ).astype(np.float32)


def nearest_labels(source_frames: np.ndarray, source_y: np.ndarray, target_frames: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_frames, target_frames, side="left")
    right = np.clip(right, 0, len(source_frames) - 1)
    left = np.clip(right - 1, 0, len(source_frames) - 1)
    choose_right = np.abs(source_frames[right] - target_frames) < np.abs(source_frames[left] - target_frames)
    indices = np.where(choose_right, right, left)
    return source_y[indices].astype(np.int64)


def inverse_piecewise_warp(
    canonical_grid: np.ndarray,
    canonical_anchors: np.ndarray,
    actual_start: float,
    actual_anchors: np.ndarray,
    actual_end: float,
) -> np.ndarray:
    canonical_knots = np.concatenate([[0.0], canonical_anchors, [1.0]])
    actual_knots = np.concatenate([[actual_start], actual_anchors, [actual_end]])
    if not np.all(np.diff(actual_knots) > 0):
        raise ValueError(f"Non-increasing actual knots: {actual_knots}")
    return np.interp(canonical_grid, canonical_knots, actual_knots).astype(np.float32)


def load_complete_sequences(frame_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    frames = pd.read_csv(frame_path)
    manifest = pd.read_csv(manifest_path).set_index("embryo_id")
    excluded = {"embryo_id", "sample_index", "frame", "phase"}
    feature_names = [column for column in frames.columns if column not in excluded]
    if len(feature_names) != 35:
        raise ValueError(f"Expected 35 structure features, got {len(feature_names)}")
    sequences: list[dict[str, Any]] = []
    audit = {"frame_embryos": int(frames.embryo_id.nunique()), "missing_manifest": 0, "missing_anchor": 0, "bad_order": 0, "outside_observed_window": 0}
    for embryo_id, group in frames.groupby("embryo_id", sort=True):
        embryo_id = str(embryo_id)
        if embryo_id not in manifest.index:
            audit["missing_manifest"] += 1
            continue
        row = manifest.loc[embryo_id]
        anchors = np.asarray([row.get(f"{name}_start_frame", np.nan) for name in ANCHORS], dtype=np.float32)
        if not np.isfinite(anchors).all():
            audit["missing_anchor"] += 1
            continue
        if not np.all(np.diff(anchors) > 0):
            audit["bad_order"] += 1
            continue
        group = group.sort_values("frame").drop_duplicates("frame")
        observed_frames = group.frame.to_numpy(np.float32)
        if len(group) < 8 or observed_frames[0] > anchors[0] or observed_frames[-1] < anchors[-1]:
            audit["outside_observed_window"] += 1
            continue
        phase_ids = np.asarray([PHASES.index(str(value)) if str(value) in PHASES else -1 for value in group.phase], dtype=np.int64)
        valid = phase_ids >= 0
        if valid.sum() < 8:
            continue
        group = group.iloc[np.flatnonzero(valid)]
        observed_frames = group.frame.to_numpy(np.float32)
        phase_ids = phase_ids[valid]
        x = group[feature_names].to_numpy(np.float32)
        relative = (anchors - observed_frames[0]) / max(observed_frames[-1] - observed_frames[0], 1.0)
        time_rows = float(row.get("time_rows", np.nan))
        first_time = float(row.get("first_time", np.nan))
        last_time = float(row.get("last_time", np.nan))
        hours_per_frame = (last_time - first_time) / (time_rows - 1.0) if np.isfinite([time_rows, first_time, last_time]).all() and time_rows > 1 else np.nan
        sequences.append(
            {
                "embryo_id": embryo_id,
                "frames": observed_frames,
                "x": x,
                "y": phase_ids,
                "anchors": anchors,
                "anchor_relative": relative.astype(np.float32),
                "descriptor": sequence_descriptor(x, observed_frames),
                "hours_per_frame": hours_per_frame,
            }
        )
    audit["usable_complete_anchor_embryos"] = len(sequences)
    return sequences, feature_names, audit


def fit_predict_anchors(
    sequences: list[dict[str, Any]], train_idx: np.ndarray, test_idx: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    descriptors = np.stack([item["descriptor"] for item in sequences])
    targets = np.stack([item["anchor_relative"] for item in sequences])
    regression_predicted = np.zeros_like(targets, dtype=np.float32)
    n_splits = min(5, len(train_idx))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_train, fold_valid in kfold.split(train_idx):
        fitted = train_idx[fold_train]
        held_out = train_idx[fold_valid]
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=-1)
        model.fit(descriptors[fitted], targets[fitted])
        regression_predicted[held_out] = model.predict(descriptors[held_out])
    final_model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.7, random_state=seed, n_jobs=-1)
    final_model.fit(descriptors[train_idx], targets[train_idx])
    regression_predicted[test_idx] = final_model.predict(descriptors[test_idx])
    regression_predicted = enforce_anchor_order(regression_predicted)

    classification_predicted = np.zeros_like(targets, dtype=np.float32)

    def frame_matrix(indices: np.ndarray, anchor_id: int) -> tuple[np.ndarray, np.ndarray]:
        feature_rows, labels = [], []
        for index in indices:
            item = sequences[index]
            position = ((item["frames"] - item["frames"][0]) / max(item["frames"][-1] - item["frames"][0], 1.0))[:, None]
            delta = np.zeros_like(item["x"])
            delta[1:] = item["x"][1:] - item["x"][:-1]
            feature_rows.append(np.concatenate([item["x"], delta, position], axis=1))
            labels.append((item["frames"] >= item["anchors"][anchor_id]).astype(np.int64))
        return np.concatenate(feature_rows), np.concatenate(labels)

    def probability_to_anchor(item: dict[str, Any], probabilities: np.ndarray) -> float:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        smooth = np.convolve(np.pad(probabilities, (1, 1), mode="edge"), kernel, mode="valid")
        monotonic = np.maximum.accumulate(smooth)
        position = (item["frames"] - item["frames"][0]) / max(item["frames"][-1] - item["frames"][0], 1.0)
        above = np.flatnonzero(monotonic >= 0.5)
        if not len(above):
            return float(position[np.argmax(monotonic)])
        right = int(above[0])
        if right == 0:
            return float(position[0])
        left = right - 1
        fraction = float((0.5 - monotonic[left]) / max(monotonic[right] - monotonic[left], 1e-6))
        return float(position[left] + fraction * (position[right] - position[left]))

    def predict_classifier(model: ExtraTreesClassifier, indices: np.ndarray, anchor_id: int) -> None:
        for index in indices:
            item = sequences[index]
            position = ((item["frames"] - item["frames"][0]) / max(item["frames"][-1] - item["frames"][0], 1.0))[:, None]
            delta = np.zeros_like(item["x"])
            delta[1:] = item["x"][1:] - item["x"][:-1]
            matrix = np.concatenate([item["x"], delta, position], axis=1)
            probabilities = model.predict_proba(matrix)[:, list(model.classes_).index(1)]
            classification_predicted[index, anchor_id] = probability_to_anchor(item, probabilities)

    for anchor_id in range(len(ANCHORS)):
        for fold_train, fold_valid in kfold.split(train_idx):
            fitted = train_idx[fold_train]
            held_out = train_idx[fold_valid]
            matrix, binary = frame_matrix(fitted, anchor_id)
            classifier = ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=4, max_features=0.7, class_weight="balanced", random_state=seed, n_jobs=-1
            )
            classifier.fit(matrix, binary)
            predict_classifier(classifier, held_out, anchor_id)
        matrix, binary = frame_matrix(train_idx, anchor_id)
        classifier = ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=4, max_features=0.7, class_weight="balanced", random_state=seed, n_jobs=-1
        )
        classifier.fit(matrix, binary)
        predict_classifier(classifier, test_idx, anchor_id)
    classification_predicted = enforce_anchor_order(classification_predicted)

    reg_oof_mae = np.abs(regression_predicted[train_idx] - targets[train_idx]).mean(0)
    cls_oof_mae = np.abs(classification_predicted[train_idx] - targets[train_idx]).mean(0)
    selected_methods = np.where(cls_oof_mae < reg_oof_mae, "framewise_binary", "sequence_regression")
    predicted = regression_predicted.copy()
    for anchor_id, method in enumerate(selected_methods):
        if method == "framewise_binary":
            predicted[:, anchor_id] = classification_predicted[:, anchor_id]
    predicted = enforce_anchor_order(predicted)

    records = []
    for split, indices in [("train_oof", train_idx), ("test", test_idx)]:
        true_frame, pred_frame, hour_errors = [], [], []
        for index in indices:
            item = sequences[index]
            span = item["frames"][-1] - item["frames"][0]
            pf = item["frames"][0] + predicted[index] * span
            true_frame.append(item["anchors"])
            pred_frame.append(pf)
            if np.isfinite(item["hours_per_frame"]):
                hour_errors.append(np.abs(pf - item["anchors"]) * item["hours_per_frame"])
        true_frame = np.stack(true_frame)
        pred_frame = np.stack(pred_frame)
        mae_frame = np.abs(pred_frame - true_frame).mean(0)
        mae_hours = np.stack(hour_errors).mean(0) if hour_errors else np.full(3, np.nan)
        records.append({"split": split, "n": len(indices), "mae_frames": dict(zip(ANCHORS, mae_frame.tolist())), "approx_mae_hours": dict(zip(ANCHORS, mae_hours.tolist()))})
    return predicted, {
        "model_selection": "Per-anchor method selected by training-only 5-fold OOF normalized MAE",
        "candidates": ["sequence_regression", "framewise_binary"],
        "selected_method": dict(zip(ANCHORS, selected_methods.tolist())),
        "train_oof_normalized_mae": {
            "sequence_regression": dict(zip(ANCHORS, reg_oof_mae.tolist())),
            "framewise_binary": dict(zip(ANCHORS, cls_oof_mae.tolist())),
        },
        "test_predictions": "Selected candidate refit on all training embryos; test labels are not used for selection",
        "metrics": records,
    }


def build_variant(
    sequences: list[dict[str, Any]], predicted_relative: np.ndarray, canonical_anchors: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical_grid = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    all_x, all_y, aligned_frames, original_frames = [], [], [], []
    for index, item in enumerate(sequences):
        frames, source_x, source_y = item["frames"], item["x"], item["y"]
        predicted_anchors = frames[0] + predicted_relative[index] * (frames[-1] - frames[0])
        if variant.startswith("uniform"):
            target_frames = np.linspace(frames[0], frames[-1], 32, dtype=np.float32)
            gate_anchor = predicted_anchors[0] if variant == "uniform_pred_gate" else item["anchors"][0]
        else:
            relative = item["anchor_relative"] if "_gt_" in variant else predicted_relative[index]
            actual_anchors = frames[0] + relative * (frames[-1] - frames[0])
            target_frames = inverse_piecewise_warp(canonical_grid, canonical_anchors, frames[0], actual_anchors, frames[-1])
            gate_anchor = actual_anchors[0]
        x = interpolate_features(frames, source_x, target_frames)
        if variant in {"uniform_gt_gate", "uniform_pred_gate", "anchor_gt_gate", "anchor_pred_gate"}:
            early = target_frames < gate_anchor
            x[np.ix_(early, EVENT_DEPENDENT_FEATURES)] = 0.0
        delta = np.zeros_like(x)
        delta[1:] = x[1:] - x[:-1]
        absolute_time = (target_frames / 1000.0)[:, None].astype(np.float32)
        augmented = np.concatenate([x, delta, canonical_grid[:, None], absolute_time], axis=1)
        all_x.append(augmented)
        all_y.append(nearest_labels(frames, source_y, target_frames))
        aligned_frames.append(target_frames)
        original_frames.append(frames)
    return np.stack(all_x), np.stack(all_y), np.stack(aligned_frames), np.stack(original_frames)


class TemporalModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(dropout), nn.Linear(hidden * 2, len(PHASES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z, _ = self.gru(z)
        return self.head(z)


def evaluate_same_original_frames(
    model: TemporalModel,
    x: np.ndarray,
    aligned_frames: np.ndarray,
    sequences: list[dict[str, Any]],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    truth, prediction = [], []
    per_embryo = []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            batch_indices = indices[start : start + 64]
            logits = model(torch.from_numpy(x[batch_indices]).float().to(device)).cpu().numpy()
            for local, index in enumerate(batch_indices):
                item = sequences[index]
                source_axis = aligned_frames[index]
                target_axis = item["frames"]
                mapped = np.stack([np.interp(target_axis, source_axis, logits[local, :, class_id]) for class_id in range(len(PHASES))], axis=1)
                pred = mapped.argmax(1)
                truth.extend(item["y"].tolist())
                prediction.extend(pred.tolist())
                per_embryo.append({"embryo_id": item["embryo_id"], "n_frames": len(target_axis), "accuracy": float((pred == item["y"]).mean())})
    labels = list(range(len(PHASES)))
    cm = confusion_matrix(truth, prediction, labels=labels)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "n_evaluation_frames": len(truth),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "phase": PHASES[class_id],
                "support": int(cm[class_id].sum()),
                "correct": int(cm[class_id, class_id]),
                "recall": float(cm[class_id, class_id] / cm[class_id].sum()) if cm[class_id].sum() else None,
            }
            for class_id in labels
        ],
        "per_embryo": per_embryo,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        result[variant] = {}
        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            values = np.asarray([run[metric] for run in selected], dtype=np.float64)
            result[variant][metric] = {"mean": float(values.mean()), "std": float(values.std()), "n": len(values)}
        cm = np.sum([np.asarray(run["confusion_matrix"], dtype=np.int64) for run in selected], axis=0)
        result[variant]["summed_confusion_matrix"] = cm.tolist()
        result[variant]["per_class"] = [
            {"phase": PHASES[i], "support": int(cm[i].sum()), "recall": float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None}
            for i in range(len(PHASES))
        ]
    baseline = result["uniform_raw"]
    for variant in VARIANTS[1:]:
        result[variant]["delta_vs_uniform_raw"] = {
            metric: result[variant][metric]["mean"] - baseline[metric]["mean"]
            for metric in ["accuracy", "balanced_accuracy", "macro_f1"]
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    sequences, feature_names, dataset_audit = load_complete_sequences(args.frame_features, args.manifest)
    all_indices = np.arange(len(sequences))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    anchor_audits = []
    for seed in args.seeds:
        train_idx, test_idx = train_test_split(all_indices, test_size=0.10, random_state=seed)
        train_ids = [sequences[index]["embryo_id"] for index in train_idx]
        test_ids = [sequences[index]["embryo_id"] for index in test_idx]
        split = {"seed": seed, "train_ids": train_ids, "test_ids": test_ids, "train_hash": ids_hash(train_ids), "test_hash": ids_hash(test_ids)}
        write_json(args.out_dir / "splits" / f"seed_{seed}.json", split)
        predicted_relative, predictor_audit = fit_predict_anchors(sequences, train_idx, test_idx, seed)
        canonical_anchors = np.median(np.stack([sequences[index]["anchor_relative"] for index in train_idx]), axis=0).astype(np.float32)
        predictor_audit.update({"seed": seed, "canonical_anchor_coordinates": dict(zip(ANCHORS, canonical_anchors.tolist()))})
        anchor_audits.append(predictor_audit)
        write_json(args.out_dir / "splits" / f"seed_{seed}_anchor_predictor_audit.json", predictor_audit)

        seed_all(seed)
        template = TemporalModel(72, args.hidden, args.dropout)
        common_state = clone_state(template)
        common_hash = state_hash(common_state)
        original_train_y = np.concatenate([sequences[index]["y"] for index in train_idx])
        counts = np.bincount(original_train_y, minlength=len(PHASES)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = torch.from_numpy((weights / weights.mean()).astype(np.float32)).to(device)

        for variant in VARIANTS:
            seed_all(seed)
            x, aligned_y, aligned_frames, _ = build_variant(sequences, predicted_relative, canonical_anchors, variant)
            model = TemporalModel(x.shape[-1], args.hidden, args.dropout).to(device)
            model.load_state_dict(common_state, strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            generator = torch.Generator().manual_seed(seed)
            order_loader = torch.utils.data.DataLoader(train_idx.tolist(), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                losses = []
                for batch_indices in order_loader:
                    batch_np = batch_indices.numpy()
                    bx = torch.from_numpy(x[batch_np]).float().to(device)
                    by = torch.from_numpy(aligned_y[batch_np]).long().to(device)
                    loss = F.cross_entropy(model(bx).reshape(-1, len(PHASES)), by.reshape(-1), weight=weights)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
            metrics = evaluate_same_original_frames(model, x, aligned_frames, sequences, test_idx, device)
            run_dir = args.out_dir / "runs" / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            pd.DataFrame(metrics.pop("per_embryo")).to_csv(run_dir / "per_embryo_metrics.csv", index=False)
            pd.DataFrame(metrics["confusion_matrix"], index=PHASES, columns=PHASES).to_csv(run_dir / "confusion_matrix.csv")
            write_json(run_dir / "test_metrics.json", metrics)
            torch.save({"model": clone_state(model), "variant": variant, "seed": seed, "common_initialization_hash": common_hash, "feature_names": feature_names}, run_dir / "final.pt")
            run = {"seed": seed, "variant": variant, "common_initialization_hash": common_hash, **metrics}
            runs.append(run)
            print(f"seed={seed} variant={variant} accuracy={metrics['accuracy']:.4f} balanced={metrics['balanced_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)

    summary = {
        "task": "Nantes 16-phase prediction with true event-anchor temporal normalization",
        "dataset_audit": dataset_audit,
        "n_embryos": len(sequences),
        "evaluation_protocol": "All variants are evaluated after logit interpolation on the identical original 32 test frames.",
        "split_protocol": "Embryo-level 90/10; same split, initialization, epochs, optimizer, batches, and class weights within each seed.",
        "anchor_definitions": {"tSB": "start of blastulation", "tB": "blastocyst", "tEB": "expanded blastocyst"},
        "variant_definitions": {
            "uniform_raw": "uniform 32-point baseline without validity gating",
            "uniform_gt_gate": "uniform baseline with GT tSB validity gate; oracle",
            "uniform_pred_gate": "uniform baseline with training-only predicted tSB validity gate; deployable gate-only",
            "anchor_gt_raw": "GT tSB/tB/tEB piecewise temporal normalization without gating; oracle alignment-only",
            "anchor_gt_gate": "GT anchor normalization plus GT tSB validity gate; oracle combined upper bound",
            "anchor_pred_raw": "OOF/train-only predicted anchor normalization without gating; deployable alignment-only",
            "anchor_pred_gate": "OOF/train-only predicted anchors plus predicted tSB gate; deployable experiment",
        },
        "anchor_predictor_audit": anchor_audits,
        "aggregate": aggregate(runs),
        "per_seed": runs,
    }
    write_json(args.out_dir / "event_anchor_temporal_normalization_summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
