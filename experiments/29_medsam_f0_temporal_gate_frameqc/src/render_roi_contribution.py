"""Render a spatial ROI contribution map from actual masks and task gate weights.

The result is a region-weight visualization, not pixel-level Grad-CAM or evidence
of agreement with an expert. Supply an independent expert ROI mask to quantify
localization instead of judging the colors alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


REGIONS = ("global", "icm", "te", "zp")


def compute_density(masks: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if set(weights) != set(REGIONS):
        raise ValueError(f"Weights must contain exactly {REGIONS}")
    values = np.asarray([weights[region] for region in REGIONS], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-4):
        raise ValueError("ROI weights must be nonnegative, finite, and sum to one")
    shape = masks["global"].shape
    density = np.zeros(shape, dtype=np.float64)
    for region in REGIONS:
        mask = np.asarray(masks[region], dtype=bool)
        if mask.shape != shape:
            raise ValueError("All ROI masks must have the same shape")
        area = mask.sum()
        if area == 0 and weights[region] > 0:
            raise ValueError(f"Nonzero weight on empty {region} mask")
        if area:
            density += weights[region] * mask / area
    return density


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("L").resize(size, Image.Resampling.NEAREST)) > 127


def render(image_path: Path, masks: dict[str, Path], weights_path: Path, output: Path, expert_roi: Path | None) -> dict:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    mask_arrays = {"global": np.ones((image.height, image.width), dtype=bool)}
    mask_arrays.update({name: load_mask(path, image.size) for name, path in masks.items()})
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    density = compute_density(mask_arrays, weights)
    scaled = density / density.max() if density.max() else density
    # A fixed blue-to-red mapping is applied after measuring contributions.
    blue = np.asarray([27, 72, 154], dtype=np.float64)
    red = np.asarray([211, 42, 44], dtype=np.float64)
    heat = (blue[None, None] * (1 - scaled[..., None]) + red[None, None] * scaled[..., None]).astype(np.uint8)
    heat_image = Image.fromarray(heat, "RGB")
    overlay = Image.blend(image, heat_image, 0.55)
    canvas = Image.new("RGB", (image.width * 3, image.height), "white")
    for index, panel in enumerate((image, heat_image, overlay)):
        canvas.paste(panel, (image.width * index, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    result = {"kind": "ROI region contribution density, not pixel saliency", "weights": weights}
    if expert_roi is not None:
        expert = load_mask(expert_roi, image.size)
        result["expert_roi_area_fraction"] = float(expert.mean())
        result["contribution_mass_in_expert_roi"] = float(density[expert].sum() / density.sum()) if density.sum() else 0.0
        result["localization_lift_over_area"] = (
            result["contribution_mass_in_expert_roi"] / result["expert_roi_area_fraction"]
            if expert.any() else None
        )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--icm-mask", type=Path, required=True)
    parser.add_argument("--te-mask", type=Path, required=True)
    parser.add_argument("--zp-mask", type=Path, required=True)
    parser.add_argument("--task-weights", type=Path, required=True)
    parser.add_argument("--expert-roi", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.image, {"icm": args.icm_mask, "te": args.te_mask, "zp": args.zp_mask},
                    args.task_weights, args.output, args.expert_roi)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
