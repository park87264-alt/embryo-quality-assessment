from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFile, ImageFont


ImageFile.LOAD_TRUNCATED_IMAGES = True


RUN_RE = re.compile(r"RUN(\d+)", re.IGNORECASE)
TASKS = ["exp", "icm", "te"]
GRADE_MAP = {
    "exp": ["1", "2", "3", "4", "5"],
    "icm": ["A", "B", "C"],
    "te": ["A", "B", "C"],
}


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("gardner_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame_number(path: Path) -> int | None:
    match = RUN_RE.search(path.name)
    return int(match.group(1)) if match else None


def resolve_frame(directory: Path, target_frame: int) -> tuple[Path, int]:
    candidates = []
    for path in directory.glob("*.jpeg"):
        number = frame_number(path)
        if number is not None:
            candidates.append((abs(number - target_frame), number, path))
    if not candidates:
        raise FileNotFoundError(f"No RUN images in {directory}")
    _, actual_frame, path = min(candidates, key=lambda item: (item[0], item[1]))
    return path, actual_frame


def image_quality(path: Path) -> tuple[float, float, float]:
    """Measure structure in the central field while ignoring the bright dish rim."""
    image = Image.open(path).convert("L").resize((256, 256))
    array = np.asarray(image, dtype=np.float32) / 255.0
    center = array[64:192, 64:192]
    gradient_x = np.abs(np.diff(center, axis=1)).mean()
    gradient_y = np.abs(np.diff(center, axis=0)).mean()
    gradient = float((gradient_x + gradient_y) / 2.0)
    contrast = float(center.std())
    intensity_range = float(np.quantile(center, 0.95) - np.quantile(center, 0.05))
    return gradient, contrast, intensity_range


def select_samples(
    manifest_path: Path,
    frame_features_path: Path,
    n_samples: int,
    random_seed: int,
    phase: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    frames = pd.read_csv(frame_features_path)
    candidates = frames.loc[frames["phase"].astype(str) == phase].copy()
    candidates = candidates.sort_values(["embryo_id", "frame"]).groupby("embryo_id", as_index=False).tail(1)
    manifest_columns = ["embryo_id", "F0_dir", "ICM", "TE"]
    candidates = candidates.merge(manifest[manifest_columns], on="embryo_id", how="inner")

    # Keep the comparison blind to existing weak Nantes grades.
    candidates = candidates.loc[candidates["ICM"].isna() & candidates["TE"].isna()].copy()
    candidates = candidates.loc[candidates["F0_dir"].notna()].copy()
    image_paths, actual_frames = [], []
    quality_gradients, quality_contrasts, quality_ranges = [], [], []
    for row in candidates.itertuples(index=False):
        path, actual = resolve_frame(Path(str(row.F0_dir)), int(row.frame))
        image_paths.append(str(path))
        actual_frames.append(actual)
        gradient, contrast, intensity_range = image_quality(path)
        quality_gradients.append(gradient)
        quality_contrasts.append(contrast)
        quality_ranges.append(intensity_range)
    candidates["image_path"] = image_paths
    candidates["actual_frame"] = actual_frames
    candidates["quality_gradient"] = quality_gradients
    candidates["quality_contrast"] = quality_contrasts
    candidates["quality_range"] = quality_ranges

    # Fixed thresholds reject blank/out-of-focus F0 fields without consulting predictions.
    candidates = candidates.loc[
        (candidates["quality_gradient"] >= 0.008)
        & (candidates["quality_contrast"] >= 0.04)
        & (candidates["quality_range"] >= 0.12)
    ].copy()
    if len(candidates) < n_samples:
        raise RuntimeError(f"Only {len(candidates)} gradable {phase} embryos, need {n_samples}")
    sampled = (
        candidates.sample(n=n_samples, random_state=random_seed)
        .sort_values("embryo_id")
        .reset_index(drop=True)
    )
    sampled.insert(0, "sample_id", [f"N{i:02d}" for i in range(1, len(sampled) + 1)])
    return sampled


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def probability_columns(task: str, probabilities: np.ndarray) -> dict[str, float]:
    return {
        f"{task}_prob_{label}": float(probabilities[index])
        for index, label in enumerate(GRADE_MAP[task])
    }


def infer(samples: pd.DataFrame, module, checkpoints: list[Path], device: torch.device, image_size: int):
    transform = module.image_transform(False, image_size)
    models = []
    for checkpoint in checkpoints:
        model = module.TaskAttentionGardner(pretrained=False, dropout=0.25, use_attention=False).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        model.eval()
        models.append(model)

    records = []
    with torch.no_grad():
        for row in samples.itertuples(index=False):
            image = Image.open(row.image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).unsqueeze(0).to(device)
            seed_probabilities = {task: [] for task in TASKS}
            for model in models:
                outputs, _ = model(tensor)
                for task in TASKS:
                    seed_probabilities[task].append(
                        torch.softmax(outputs[task][0], dim=0).cpu().numpy()
                    )

            record = {
                "sample_id": row.sample_id,
                "embryo_id": str(row.embryo_id),
                "phase": str(row.phase),
                "requested_frame": int(row.frame),
                "actual_frame": int(row.actual_frame),
                "source_image": str(row.image_path),
                "quality_gradient": float(row.quality_gradient),
                "quality_contrast": float(row.quality_contrast),
                "quality_range": float(row.quality_range),
            }
            grades = {}
            for task in TASKS:
                stacked = np.stack(seed_probabilities[task], axis=0)
                mean_prob = stacked.mean(axis=0)
                predicted_index = int(mean_prob.argmax())
                sorted_prob = np.sort(mean_prob)[::-1]
                grade = GRADE_MAP[task][predicted_index]
                per_seed_index = stacked.argmax(axis=1)
                record[f"{task}_prediction"] = grade
                record[f"{task}_confidence"] = float(mean_prob[predicted_index])
                record[f"{task}_margin"] = float(sorted_prob[0] - sorted_prob[1])
                record[f"{task}_normalized_entropy"] = normalized_entropy(mean_prob)
                record[f"{task}_seed_agreement"] = int((per_seed_index == predicted_index).sum())
                record[f"{task}_per_seed"] = "|".join(GRADE_MAP[task][int(index)] for index in per_seed_index)
                record.update(probability_columns(task, mean_prob))
                grades[task] = grade
            record["gardner_prediction"] = f"{grades['exp']}{grades['icm']}{grades['te']}"
            records.append(record)
    return pd.DataFrame(records)


def fit_font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_contact_sheet(predictions: pd.DataFrame, output_dir: Path, show_predictions: bool):
    thumb_size = 360
    caption_height = 64 if show_predictions else 48
    columns = 2
    rows = math.ceil(len(predictions) / columns)
    canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = fit_font(19)
    small = fit_font(15)
    for index, row in predictions.iterrows():
        col, grid_row = index % columns, index // columns
        x, y = col * thumb_size, grid_row * (thumb_size + caption_height)
        image = Image.open(row["local_image"]).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        tile = Image.new("RGB", (thumb_size, thumb_size), "black")
        tile.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        canvas.paste(tile, (x, y))
        draw.text((x + 8, y + thumb_size + 5), f"{row['sample_id']}  {row['embryo_id']}  F0 RUN{row['actual_frame']}", fill="black", font=font)
        if show_predictions:
            draw.text(
                (x + 8, y + thumb_size + 31),
                f"Prediction: {row['gardner_prediction']}  conf E/I/T: "
                f"{row['exp_confidence']:.2f}/{row['icm_confidence']:.2f}/{row['te_confidence']:.2f}",
                fill="#333333",
                font=small,
            )
    suffix = "predictions" if show_predictions else "blind"
    canvas.save(output_dir / f"contact_sheet_{suffix}.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--phase", default="tEB")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    module = load_training_module(args.training_script)
    samples = select_samples(
        args.manifest, args.frame_features, args.n_samples, args.random_seed, args.phase
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictions = infer(samples, module, args.checkpoints, device, args.image_size)

    local_images = []
    for row in predictions.itertuples(index=False):
        source = Path(row.source_image)
        destination = image_dir / f"{row.sample_id}_{row.embryo_id}_RUN{row.actual_frame}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        local_images.append(str(destination))
    predictions["local_image"] = local_images
    predictions.to_csv(args.out_dir / "gardner_predictions_10.csv", index=False)
    (args.out_dir / "gardner_predictions_10.json").write_text(
        json.dumps(predictions.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    annotation = predictions[
        ["sample_id", "embryo_id", "phase", "actual_frame", "local_image"]
    ].copy()
    annotation["manual_expansion"] = ""
    annotation["manual_icm"] = ""
    annotation["manual_te"] = ""
    annotation["manual_gardner"] = ""
    annotation["notes"] = ""
    annotation.to_csv(args.out_dir / "manual_annotation_template.csv", index=False)
    build_contact_sheet(predictions, args.out_dir, show_predictions=False)
    build_contact_sheet(predictions, args.out_dir, show_predictions=True)

    metadata = {
        "sampling": f"Reproducible random sample of {args.n_samples} distinct Nantes embryos from a prediction-blind image-quality-filtered pool; latest uniformly sampled {args.phase} original F0 frame; existing Nantes ICM/TE labels excluded.",
        "random_seed": args.random_seed,
        "model": "Three-seed probability ensemble of Nantes-pretrained global ResNet18 fine-tuned on Kromp Gardner labels.",
        "checkpoints": [str(path) for path in args.checkpoints],
        "label_mapping": {
            "Expansion": "model classes 0-4 are reported as Gardner 1-5",
            "ICM": "model classes 0/1/2 are reported as A/B/C",
            "TE": "model classes 0/1/2 are reported as A/B/C",
        },
        "warning": "Cross-dataset exploratory inference only. Kromp test accuracy is not Nantes accuracy; Nantes accuracy requires independent manual labels for these exact images.",
    }
    (args.out_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(predictions[["sample_id", "embryo_id", "actual_frame", "gardner_prediction", "exp_confidence", "icm_confidence", "te_confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
