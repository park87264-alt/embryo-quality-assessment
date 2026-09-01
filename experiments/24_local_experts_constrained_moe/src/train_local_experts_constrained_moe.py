from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


TASK_CLASSES = {"icm": 3, "te": 3}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_rows(rows: list[dict[str, str]], task: str) -> list[dict[str, str]]:
    return [row for row in rows if int(float(row[task])) >= 0]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_checksum(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def normalize_image(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def mild_color(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    return ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))


def load_mask(mask_dir: Path, image_name: str) -> torch.Tensor:
    mask = torch.load(mask_dir / f"{Path(image_name).stem}.pt", map_location="cpu").float()
    if mask.shape != (3, 224, 224):
        mask = F.interpolate(mask[None], size=(224, 224), mode="bilinear", align_corners=False)[0]
    return mask.clamp(0, 1)


def binary_mask(mask: torch.Tensor) -> torch.Tensor:
    out = mask >= 0.5
    if int(out.sum()) < 16:
        threshold = torch.quantile(mask.flatten(), 0.85)
        out = mask >= threshold
    return out


def bbox_from_mask(mask: torch.Tensor, margin: float = 0.15) -> tuple[float, float, float, float]:
    points = torch.nonzero(mask, as_tuple=False)
    if len(points) == 0:
        return 0.0, 0.0, 1.0, 1.0
    y0, x0 = points.min(dim=0).values.tolist()
    y1, x1 = points.max(dim=0).values.tolist()
    width = max(1.0, x1 - x0 + 1.0)
    height = max(1.0, y1 - y0 + 1.0)
    x0 -= width * margin
    x1 += width * margin
    y0 -= height * margin
    y1 += height * margin
    return max(0.0, x0 / 224), max(0.0, y0 / 224), min(1.0, (x1 + 1) / 224), min(1.0, (y1 + 1) / 224)


def crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    return image.crop((int(x0 * width), int(y0 * height), max(int(x1 * width), 1), max(int(y1 * height), 1)))


def mask_geometry(mask: torch.Tensor, image: Image.Image, boundary: bool = False) -> torch.Tensor:
    mask = mask.bool()
    points = torch.nonzero(mask, as_tuple=False)
    if len(points) == 0:
        return torch.zeros(10, dtype=torch.float32)
    ys = points[:, 0].float()
    xs = points[:, 1].float()
    area = float(mask.float().mean())
    y0, x0 = points.min(dim=0).values.float()
    y1, x1 = points.max(dim=0).values.float()
    bw = float((x1 - x0 + 1) / 224)
    bh = float((y1 - y0 + 1) / 224)
    fill = area / max(bw * bh, 1e-6)
    cx, cy = float(xs.mean() / 223), float(ys.mean() / 223)
    aspect = bw / max(bh, 1e-6)
    radius = torch.sqrt((xs - xs.mean()).square() + (ys - ys.mean()).square())
    radial_cv = float(radius.std() / radius.mean().clamp_min(1e-6)) if len(radius) > 1 else 0.0
    gray = TF.to_tensor(image.convert("L").resize((224, 224), Image.Resampling.BILINEAR))[0]
    values = gray[mask]
    intensity_mean = float(values.mean()) if len(values) else 0.0
    intensity_std = float(values.std()) if len(values) > 1 else 0.0
    angles = torch.atan2(ys - ys.mean(), xs - xs.mean())
    bins = torch.clamp(((angles + math.pi) / (2 * math.pi) * 24).long(), 0, 23)
    angular_coverage = float(torch.unique(bins).numel() / 24)
    boundary_flag = 1.0 if boundary else 0.0
    return torch.tensor(
        [area, fill, cx, cy, aspect, radial_cv, intensity_mean, intensity_std, angular_coverage, boundary_flag],
        dtype=torch.float32,
    )


def te_boundary_mask(mask: torch.Tensor, radius: int = 4) -> torch.Tensor:
    x = mask.float()[None, None]
    kernel = radius * 2 + 1
    dilated = F.max_pool2d(x, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-x, kernel, stride=1, padding=radius)
    return ((dilated - eroded)[0, 0] > 0.05)


def boundary_patch_boxes(boundary: torch.Tensor, count: int, patch_fraction: float) -> list[tuple[float, float, float, float]]:
    points = torch.nonzero(boundary, as_tuple=False)
    if len(points) == 0:
        return [(0.0, 0.0, 1.0, 1.0)] * count
    center = points.float().mean(dim=0)
    angles = torch.atan2(points[:, 0].float() - center[0], points[:, 1].float() - center[1])
    order = torch.argsort(angles)
    ordered = points[order]
    indices = torch.linspace(0, len(ordered) - 1, count).round().long()
    half = patch_fraction / 2
    boxes = []
    for y, x in ordered[indices].tolist():
        cx, cy = (x + 0.5) / 224, (y + 0.5) / 224
        boxes.append((max(0.0, cx - half), max(0.0, cy - half), min(1.0, cx + half), min(1.0, cy + half)))
    return boxes


class GlobalDataset(Dataset):
    def __init__(self, rows, image_dir: Path, task: str, train: bool, image_size: int):
        self.rows = rows
        self.image_dir = image_dir
        self.task = task
        self.train = train
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
            image = mild_color(image)
        return normalize_image(image, self.image_size), int(float(row[self.task])), row["image"]


class ICMDataset(Dataset):
    def __init__(self, rows, image_dir: Path, mask_dir: Path, train: bool, image_size: int):
        self.rows = rows
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        mask = load_mask(self.mask_dir, row["image"])[0]
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = torch.flip(mask, dims=[1])
        if self.train:
            image = mild_color(image)
        hard = binary_mask(mask)
        patch = crop_fraction(image, bbox_from_mask(hard, margin=0.18))
        geometry = mask_geometry(hard, image)
        return normalize_image(patch, self.image_size), geometry, int(float(row["icm"])), row["image"]


class TEDataset(Dataset):
    def __init__(self, rows, image_dir: Path, mask_dir: Path, train: bool, patch_size: int, patch_count: int):
        self.rows = rows
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.patch_size = patch_size
        self.patch_count = patch_count

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        mask = load_mask(self.mask_dir, row["image"])[1]
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = torch.flip(mask, dims=[1])
        if self.train:
            image = mild_color(image)
        hard = binary_mask(mask)
        boundary = te_boundary_mask(hard)
        boxes = boundary_patch_boxes(boundary, self.patch_count, patch_fraction=0.24)
        patches = torch.stack([normalize_image(crop_fraction(image, box), self.patch_size) for box in boxes])
        geometry = mask_geometry(boundary, image, boundary=True)
        return patches, geometry, int(float(row["te"])), row["image"]


class AllExpertDataset(Dataset):
    def __init__(self, rows, image_dir: Path, mask_dir: Path, train: bool, image_size: int, patch_size: int, patch_count: int):
        self.rows = rows
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_count = patch_count

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.image_dir / row["image"]).convert("RGB")
        masks = load_mask(self.mask_dir, row["image"])
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            masks = torch.flip(masks, dims=[2])
        if self.train:
            image = mild_color(image)
        global_image = normalize_image(image, self.image_size)
        icm_mask = binary_mask(masks[0])
        icm_patch = normalize_image(crop_fraction(image, bbox_from_mask(icm_mask, 0.18)), self.image_size)
        icm_geometry = mask_geometry(icm_mask, image)
        te_mask = binary_mask(masks[1])
        boundary = te_boundary_mask(te_mask)
        boxes = boundary_patch_boxes(boundary, self.patch_count, 0.24)
        te_patches = torch.stack([normalize_image(crop_fraction(image, box), self.patch_size) for box in boxes])
        te_geometry = mask_geometry(boundary, image, boundary=True)
        labels = torch.tensor([max(int(float(row["icm"])), 0), max(int(float(row["te"])), 0)]).long()
        valid = torch.tensor([int(float(row["icm"])) >= 0, int(float(row["te"])) >= 0]).float()
        return {
            "global": global_image,
            "icm": icm_patch,
            "icm_geometry": icm_geometry,
            "te": te_patches,
            "te_geometry": te_geometry,
            "labels": labels,
            "valid": valid,
            "name": row["image"],
        }


def resnet_encoder() -> nn.Module:
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model


def keep_batch_norm_eval(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class GlobalExpert(nn.Module):
    feature_dim = 512

    def __init__(self, classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.encoder = resnet_encoder()
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(dropout), nn.Linear(512, classes))

    def forward(self, image):
        feature = self.encoder(image)
        return feature, self.head(feature)


class ICMTextureExpert(nn.Module):
    feature_dim = 576

    def __init__(self, classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.encoder = resnet_encoder()
        self.geometry = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, 64), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(576), nn.Dropout(dropout), nn.Linear(576, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, classes))

    def forward(self, image, geometry):
        feature = torch.cat([self.encoder(image), self.geometry(geometry)], dim=1)
        return feature, self.head(feature)


class TEBoundaryExpert(nn.Module):
    feature_dim = 320

    def __init__(self, classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.encoder = resnet_encoder()
        self.sequence = nn.GRU(512, 128, batch_first=True, bidirectional=True)
        self.geometry = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, 64), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(320), nn.Dropout(dropout), nn.Linear(320, 192), nn.GELU(), nn.Dropout(dropout), nn.Linear(192, classes))

    def forward(self, patches, geometry):
        batch, count, channels, height, width = patches.shape
        encoded = self.encoder(patches.reshape(batch * count, channels, height, width)).reshape(batch, count, 512)
        sequence, _ = self.sequence(encoded)
        feature = torch.cat([sequence.mean(dim=1), self.geometry(geometry)], dim=1)
        return feature, self.head(feature)


def configure_encoder(model: nn.Module, unfreeze_layer4: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    if unfreeze_layer4:
        for parameter in model.encoder.layer4.parameters():
            parameter.requires_grad = True


def sampler_for(rows: list[dict[str, str]], task: str, seed: int) -> WeightedRandomSampler:
    labels = np.asarray([int(float(row[task])) for row in rows], dtype=np.int64)
    counts = np.bincount(labels, minlength=TASK_CLASSES[task]).astype(np.float32)
    maximum = counts.max()
    class_scale = np.minimum(np.sqrt(maximum / np.maximum(counts, 1.0)), 3.0)
    weights = torch.from_numpy(class_scale[labels]).double()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def safe_auc(y_true: list[int], probabilities: list[list[float]], classes: int) -> float | None:
    try:
        binary = label_binarize(y_true, classes=list(range(classes)))
        return float(roc_auc_score(binary, np.asarray(probabilities), average="macro", multi_class="ovr"))
    except ValueError:
        return None


def classification_metrics(y_true, y_pred, probabilities, classes=3) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(classes)))
    result = {
        "support": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(classes)), average="macro", zero_division=0)),
        "macro_auc_ovr": safe_auc(y_true, probabilities, classes),
        "confusion_matrix": cm.tolist(),
        "per_class": [],
    }
    for class_id in range(classes):
        support = int(cm[class_id].sum())
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["per_class"].append({"class_id": class_id, "support": support, "correct": tp, "precision": precision, "recall": recall, "f1": f1})
    return result


def forward_single(model, batch, mode: str, device):
    if mode == "global":
        return model(batch[0].to(device))
    return model(batch[0].to(device), batch[1].to(device))


def labels_single(batch, mode: str):
    return batch[1] if mode == "global" else batch[2]


def names_single(batch, mode: str):
    return batch[2] if mode == "global" else batch[3]


def evaluate_single(model, loader, mode: str, device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    y_true, y_pred, probabilities, records = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            _, logits = forward_single(model, batch, mode, device)
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            labels = labels_single(batch, mode).numpy()
            predictions = probability.argmax(axis=1)
            names = names_single(batch, mode)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
            probabilities.extend(probability.tolist())
            for name, true, pred, prob in zip(names, labels, predictions, probability):
                records.append({"image": name, "true": int(true), "pred": int(pred), **{f"prob_{i}": float(p) for i, p in enumerate(prob)}})
    return classification_metrics(y_true, y_pred, probabilities), records


def make_single_dataset(mode, rows, args, train):
    if mode == "global":
        raise ValueError("Global dataset requires task")
    if mode == "icm_texture":
        return ICMDataset(rows, args.image_dir, args.mask_dir, train, args.image_size)
    if mode == "te_boundary":
        return TEDataset(rows, args.image_dir, args.mask_dir, train, args.patch_size, args.patch_count)
    raise ValueError(mode)


def train_single(task: str, mode: str, train_rows, val_rows, test_rows, seed: int, args, out_dir: Path):
    seed_all(seed)
    if mode == "global":
        model = GlobalExpert().to(args.device)
        train_ds = GlobalDataset(train_rows, args.image_dir, task, True, args.image_size)
        val_ds = GlobalDataset(val_rows, args.image_dir, task, False, args.image_size)
        test_ds = GlobalDataset(test_rows, args.image_dir, task, False, args.image_size)
    elif mode == "icm_texture":
        model = ICMTextureExpert().to(args.device)
        train_ds = make_single_dataset(mode, train_rows, args, True)
        val_ds = make_single_dataset(mode, val_rows, args, False)
        test_ds = make_single_dataset(mode, test_rows, args, False)
    else:
        model = TEBoundaryExpert().to(args.device)
        train_ds = make_single_dataset(mode, train_rows, args, True)
        val_ds = make_single_dataset(mode, val_rows, args, False)
        test_ds = make_single_dataset(mode, test_rows, args, False)

    configure_encoder(model, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler_for(train_rows, task, seed), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score, best_state, history = -1.0, None, []
    for epoch in range(1, args.expert_epochs + 1):
        if epoch == args.head_epochs + 1:
            configure_encoder(model, True)
        model.train()
        keep_batch_norm_eval(model)
        losses = []
        for batch in train_loader:
            _, logits = forward_single(model, batch, "global" if mode == "global" else mode, args.device)
            labels = labels_single(batch, "global" if mode == "global" else mode).to(args.device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics, _ = evaluate_single(model, val_loader, "global" if mode == "global" else mode, args.device)
        score = 0.5 * val_metrics["balanced_accuracy"] + 0.5 * val_metrics["macro_f1"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "score": score, **{k: v for k, v in val_metrics.items() if k not in {"confusion_matrix", "per_class"}}})
        if score > best_score:
            best_score = score
            best_state = clone_state(model)
    if best_state is None:
        raise RuntimeError("No expert checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    test_metrics, records = evaluate_single(model, test_loader, "global" if mode == "global" else mode, args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "task": task, "mode": mode, "seed": seed, "best_val_score": best_score}, out_dir / "best.pt")
    write_json(out_dir / "test_metrics.json", test_metrics)
    write_json(out_dir / "history.json", history)
    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return model, {"seed": seed, "task": task, "mode": mode, "best_val_score": best_score, **test_metrics}


class ConstrainedMoE(nn.Module):
    def __init__(self, global_icm, local_icm, global_te, local_te):
        super().__init__()
        self.global_icm = global_icm
        self.local_icm = local_icm
        self.global_te = global_te
        self.local_te = local_te
        self.icm_gate = nn.Sequential(nn.LayerNorm(512 + 576), nn.Linear(512 + 576, 128), nn.GELU(), nn.Linear(128, 1))
        self.te_gate = nn.Sequential(nn.LayerNorm(512 + 320), nn.Linear(512 + 320, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, batch, device):
        gi_feat, gi_logits = self.global_icm(batch["global"].to(device))
        li_feat, li_logits = self.local_icm(batch["icm"].to(device), batch["icm_geometry"].to(device))
        gt_feat, gt_logits = self.global_te(batch["global"].to(device))
        lt_feat, lt_logits = self.local_te(batch["te"].to(device), batch["te_geometry"].to(device))
        icm_local_weight = 0.6 + 0.4 * torch.sigmoid(self.icm_gate(torch.cat([gi_feat, li_feat], dim=1)))
        te_local_weight = 0.6 + 0.4 * torch.sigmoid(self.te_gate(torch.cat([gt_feat, lt_feat], dim=1)))
        outputs = {
            "icm": icm_local_weight * li_logits + (1 - icm_local_weight) * gi_logits,
            "te": te_local_weight * lt_logits + (1 - te_local_weight) * gt_logits,
        }
        auxiliary = {"global_icm": gi_logits, "local_icm": li_logits, "global_te": gt_logits, "local_te": lt_logits}
        gates = {"icm_local": icm_local_weight.squeeze(1), "te_local": te_local_weight.squeeze(1)}
        return outputs, auxiliary, gates


def set_moe_stage(model: ConstrainedMoE, fine_tune: bool) -> None:
    for expert in [model.global_icm, model.local_icm, model.global_te, model.local_te]:
        for parameter in expert.parameters():
            parameter.requires_grad = fine_tune
        configure_encoder(expert, fine_tune)
    for gate in [model.icm_gate, model.te_gate]:
        for parameter in gate.parameters():
            parameter.requires_grad = True


def mild_class_weights(rows, task, device):
    labels = np.asarray([int(float(row[task])) for row in rows if int(float(row[task])) >= 0])
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = np.clip(weights / weights.mean(), 0.5, 2.0)
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def evaluate_moe(model, loader, device):
    model.eval()
    stores = {task: {"true": [], "pred": [], "prob": []} for task in TASK_CLASSES}
    gate_values = {"icm_local": [], "te_local": []}
    records = []
    with torch.no_grad():
        for batch in loader:
            outputs, _, gates = model(batch, device)
            labels, valid = batch["labels"], batch["valid"]
            for name, values in gates.items():
                gate_values[name].extend(values.cpu().numpy().tolist())
            batch_records = [{"image": name} for name in batch["name"]]
            for task_index, task in enumerate(["icm", "te"]):
                probability = torch.softmax(outputs[task], dim=1).cpu().numpy()
                prediction = probability.argmax(axis=1)
                active = valid[:, task_index].numpy().astype(bool)
                truth = labels[:, task_index].numpy()
                stores[task]["true"].extend(truth[active].tolist())
                stores[task]["pred"].extend(prediction[active].tolist())
                stores[task]["prob"].extend(probability[active].tolist())
                for index in range(len(batch_records)):
                    if active[index]:
                        batch_records[index][f"{task}_true"] = int(truth[index])
                        batch_records[index][f"{task}_pred"] = int(prediction[index])
            records.extend(batch_records)
    metrics = {task.upper(): classification_metrics(store["true"], store["pred"], store["prob"]) for task, store in stores.items()}
    metrics["gate_summary"] = {name: {"mean": float(np.mean(values)), "std": float(np.std(values))} for name, values in gate_values.items()}
    return metrics, records


def train_moe(seed, train_rows, val_rows, test_rows, models, args, out_dir):
    seed_all(seed)
    model = ConstrainedMoE(*models).to(args.device)
    set_moe_stage(model, False)
    train_ds = AllExpertDataset(train_rows, args.image_dir, args.mask_dir, True, args.image_size, args.patch_size, args.patch_count)
    val_ds = AllExpertDataset(val_rows, args.image_dir, args.mask_dir, False, args.image_size, args.patch_size, args.patch_count)
    test_ds = AllExpertDataset(test_rows, args.image_dir, args.mask_dir, False, args.image_size, args.patch_size, args.patch_count)
    train_loader = DataLoader(train_ds, batch_size=args.moe_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=args.moe_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.moe_batch_size, shuffle=False, num_workers=args.num_workers)
    weights = {task: mild_class_weights(train_rows, task, args.device) for task in TASK_CLASSES}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gate_lr, weight_decay=args.weight_decay)
    best_score, best_state, history = -1.0, None, []
    for epoch in range(1, args.moe_epochs + 1):
        if epoch == args.gate_epochs + 1:
            set_moe_stage(model, True)
            gate_params = list(model.icm_gate.parameters()) + list(model.te_gate.parameters())
            gate_ids = {id(parameter) for parameter in gate_params}
            expert_params = [parameter for parameter in model.parameters() if id(parameter) not in gate_ids and parameter.requires_grad]
            optimizer = torch.optim.AdamW(
                [{"params": gate_params, "lr": args.gate_lr}, {"params": expert_params, "lr": args.finetune_lr}],
                weight_decay=args.weight_decay,
            )
        model.train()
        keep_batch_norm_eval(model)
        losses = []
        for batch in train_loader:
            outputs, auxiliary, _ = model(batch, args.device)
            labels = batch["labels"].to(args.device)
            valid = batch["valid"].to(args.device)
            loss = torch.zeros((), device=args.device)
            used = 0
            for task_index, task in enumerate(["icm", "te"]):
                active = valid[:, task_index] > 0
                if not active.any():
                    continue
                target = labels[:, task_index][active]
                loss = loss + F.cross_entropy(outputs[task][active], target, weight=weights[task])
                loss = loss + args.aux_weight * F.cross_entropy(auxiliary[f"global_{task}"][active], target, weight=weights[task])
                loss = loss + args.aux_weight * F.cross_entropy(auxiliary[f"local_{task}"][active], target, weight=weights[task])
                used += 1
            loss = loss / max(used, 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics, _ = evaluate_moe(model, val_loader, args.device)
        score = np.mean([0.5 * val_metrics[task]["balanced_accuracy"] + 0.5 * val_metrics[task]["macro_f1"] for task in ["ICM", "TE"]])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "score": float(score), "metrics": val_metrics})
        if score > best_score:
            best_score = float(score)
            best_state = clone_state(model)
    if best_state is None:
        raise RuntimeError("No MoE checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    test_metrics, records = evaluate_moe(model, test_loader, args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "seed": seed, "best_val_score": best_score}, out_dir / "best.pt")
    write_json(out_dir / "test_metrics.json", test_metrics)
    write_json(out_dir / "history.json", history)
    fieldnames = sorted({key for record in records for key in record})
    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return {"seed": seed, "mode": "constrained_moe", "best_val_score": best_score, **test_metrics}


def aggregate(runs):
    output = {}
    modes = sorted({run["mode"] for run in runs})
    for mode in modes:
        selected = [run for run in runs if run["mode"] == mode]
        output[mode] = {}
        tasks = sorted({run["task"].upper() for run in selected}) if "task" in selected[0] else ["ICM", "TE"]
        for task in tasks:
            task_runs = selected if "task" not in selected[0] else [run for run in selected if run["task"].upper() == task]
            output[mode][task] = {}
            for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"]:
                values = [run[task][metric] if task in run else run[metric] for run in task_runs]
                values = [value for value in values if value is not None]
                output[mode][task][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--expert-epochs", type=int, default=20)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--moe-epochs", type=int, default=12)
    parser.add_argument("--gate-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--moe-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patch-count", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    args = parser.parse_args()

    train_rows_all = read_rows(args.train_features)
    test_rows_all = read_rows(args.test_features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        common_train_rows, common_val_rows = train_test_split(train_rows_all, test_size=0.15, random_state=seed)
        write_json(
            args.out_dir / "splits" / f"common_seed_{seed}.json",
            {
                "seed": seed,
                "train_images": [row["image"] for row in common_train_rows],
                "val_images": [row["image"] for row in common_val_rows],
                "test_images": [row["image"] for row in test_rows_all],
            },
        )
        trained = {}
        for task in ["icm", "te"]:
            train_rows = valid_rows(common_train_rows, task)
            val_rows = valid_rows(common_val_rows, task)
            test_rows = valid_rows(test_rows_all, task)
            split_audit = {
                "seed": seed,
                "task": task,
                "train_images": [row["image"] for row in train_rows],
                "val_images": [row["image"] for row in val_rows],
                "test_images": [row["image"] for row in test_rows],
            }
            write_json(args.out_dir / "splits" / f"{task}_seed_{seed}.json", split_audit)
            modes = ["global", "icm_texture"] if task == "icm" else ["global", "te_boundary"]
            for mode in modes:
                print(f"TRAIN seed={seed} task={task} mode={mode}", flush=True)
                model, result = train_single(
                    task,
                    mode,
                    train_rows,
                    val_rows,
                    test_rows,
                    seed,
                    args,
                    args.out_dir / "experts" / task / mode / f"seed_{seed}",
                )
                trained[(task, mode)] = model
                runs.append(result)
                print(f"RESULT seed={seed} task={task} mode={mode} acc={result['accuracy']:.4f} ba={result['balanced_accuracy']:.4f} f1={result['macro_f1']:.4f}", flush=True)

        moe_train = [row for row in common_train_rows if int(float(row["icm"])) >= 0 or int(float(row["te"])) >= 0]
        moe_val = [row for row in common_val_rows if int(float(row["icm"])) >= 0 or int(float(row["te"])) >= 0]
        moe_test = [row for row in test_rows_all if int(float(row["icm"])) >= 0 or int(float(row["te"])) >= 0]
        models = [trained[("icm", "global")], trained[("icm", "icm_texture")], trained[("te", "global")], trained[("te", "te_boundary")]]
        print(f"TRAIN seed={seed} constrained_moe", flush=True)
        moe_result = train_moe(seed, moe_train, moe_val, moe_test, models, args, args.out_dir / "moe" / f"seed_{seed}")
        runs.append(moe_result)
        print(f"RESULT seed={seed} constrained_moe ICM_F1={moe_result['ICM']['macro_f1']:.4f} TE_F1={moe_result['TE']['macro_f1']:.4f}", flush=True)

    summary = {
        "task": "Independent ICM texture and TE spatial-boundary experts followed by constrained MoE",
        "protocol": "Kromp silver train/validation; gold test used once after validation checkpoint selection",
        "important_limit": "TE continuity is spatial continuity around a static blastocyst boundary, not temporal continuity.",
        "mask_source": str(args.mask_dir),
        "settings": vars(args),
        "aggregate": aggregate(runs),
        "per_seed": runs,
    }
    summary["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["settings"].items()}
    write_json(args.out_dir / "local_experts_constrained_moe_summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
