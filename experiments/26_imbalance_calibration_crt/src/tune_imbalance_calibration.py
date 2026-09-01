from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


METHODS = ["baseline_ce", "tuned_balanced_softmax", "tuned_crt"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_loader(fair, x, gate, labels, masks, indices, batch_size, shuffle, seed):
    kwargs: dict[str, Any] = {}
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(
        fair.CurveDataset(x, gate, labels, masks, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        **kwargs,
    )


def train_counts(imbalance, labels, masks, indices, n_classes):
    return imbalance.class_counts(labels, masks, indices, n_classes)


def balanced_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    counts: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    adjustment = float(tau) * counts.clamp_min(1.0).log().unsqueeze(0)
    return F.cross_entropy(logits + adjustment, targets)


def train_sequence_model(
    fair,
    x,
    gate,
    labels,
    masks,
    indices,
    common_state,
    device,
    seed,
    epochs,
    batch_size,
    lr,
    weight_decay,
    hidden,
    dropout,
    tau,
):
    fair.seed_all(seed)
    model = fair.CurvePhasePretrainer(71, hidden, dropout).to(device)
    model.load_state_dict(common_state, strict=True)
    loader = make_loader(fair, x, gate, labels, masks, indices, batch_size, True, seed)
    active_y = labels[indices][masks[indices] > 0]
    counts_np = np.bincount(active_y, minlength=len(fair.PHASES)).astype(np.float32)
    counts = torch.from_numpy(counts_np).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_gate, batch_y, batch_mask in loader:
            logits = model(batch_x.to(device), batch_gate.to(device))
            target = batch_y.to(device)
            active = batch_mask.to(device) > 0
            loss = balanced_softmax_loss(logits[active], target[active], counts, tau)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return model, history


def extract_frame_features(model, loader, device):
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for x, gate, y, mask in loader:
            z = model.encoder.encode_sequence(x.to(device)).cpu()
            z = torch.cat([z, gate.unsqueeze(-1)], dim=-1)
            active = mask > 0
            features.append(z[active])
            targets.append(y[active])
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def sampling_weights(targets: torch.Tensor, rho: float, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=n_classes).float()
    class_weight = torch.zeros_like(counts)
    present = counts > 0
    class_weight[present] = counts[present].pow(-float(rho))
    return class_weight[targets]


def train_classifier_head(
    head,
    features: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    rho: float,
    device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
):
    weights = sampling_weights(targets, rho, n_classes)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        losses = []
        for batch_features, batch_targets in loader:
            logits = head(batch_features.to(device))
            loss = F.cross_entropy(logits, batch_targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return head, history


def fresh_head(fair, common_state, hidden, dropout, device):
    template = fair.CurvePhasePretrainer(71, hidden, dropout)
    template.load_state_dict(common_state, strict=True)
    return copy.deepcopy(template.head).to(device)


def choose_candidate(candidates, baseline_accuracy: float, tolerance: float, key: str):
    eligible = [
        item for item in candidates
        if item["metrics"]["accuracy"] >= baseline_accuracy - tolerance
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            item["metrics"]["macro_f1_all_16"],
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["accuracy"],
            -float(item[key]),
        ),
    )


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1_all_16",
        "macro_f1_supported",
        "tail_recall",
        "tail_f1",
        "head_recall",
    ]
    for method in METHODS:
        selected = [run for run in runs if run["method"] == method]
        item = {}
        for metric in metrics:
            values = np.asarray(
                [run[metric] for run in selected if run.get(metric) is not None], dtype=np.float64
            )
            item[metric] = {
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "n": int(len(values)),
            }
        cm = np.asarray([run["confusion_matrix"] for run in selected], dtype=np.int64).sum(axis=0)
        item["summed_confusion_matrix"] = cm.tolist()
        output[method] = item
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-selected Balanced Softmax strength and two-stage classifier retraining"
    )
    parser.add_argument("--fair-script", type=Path, required=True)
    parser.add_argument("--imbalance-script", type=Path, required=True)
    parser.add_argument("--frame-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--tau-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--rho-values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--accuracy-tolerance", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--retrain-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--frames-per-embryo", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fair = load_module(args.fair_script, "fair_event_alignment")
    imbalance = load_module(args.imbalance_script, "class_imbalance_ablation")
    base, labels, masks, embryo_ids, feature_names = fair.load_nantes_sequences(
        args.frame_features, args.frames_per_embryo
    )
    all_idx = np.arange(len(base))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_runs = []
    selections = []

    for seed in args.seeds:
        outer_train, test_idx = train_test_split(all_idx, test_size=0.10, random_state=seed)
        fit_idx, val_idx = train_test_split(
            outer_train, test_size=1.0 / 9.0, random_state=seed + 1000
        )
        fair.seed_all(seed)
        template = fair.CurvePhasePretrainer(71, args.hidden, args.dropout)
        common_state = clone_state(template)
        common_hash = state_hash(common_state)

        # Inner gates and curves are fitted without using the outer test embryos.
        inner_gate, _, inner_gate_audit = fair.make_oof_soft_gates(
            base, labels, masks, embryo_ids, fit_idx, val_idx, seed
        )
        inner_x, inner_scalar = fair.variant_inputs(
            base, labels, masks, inner_gate, "soft_pred_gate"
        )
        fit_counts = train_counts(imbalance, labels, masks, fit_idx, len(fair.PHASES))
        val_loader = make_loader(
            fair, inner_x, inner_scalar, labels, masks, val_idx, args.batch_size, False, seed
        )

        tau_candidates = []
        ce_inner_model = None
        for tau in args.tau_values:
            model, history = train_sequence_model(
                fair, inner_x, inner_scalar, labels, masks, fit_idx, common_state, device,
                seed, args.epochs, args.batch_size, args.lr, args.weight_decay,
                args.hidden, args.dropout, tau,
            )
            metrics, _ = imbalance.evaluate(
                model, val_loader, device, list(fair.PHASES), fit_counts
            )
            tau_candidates.append({"tau": tau, "metrics": metrics})
            if float(tau) == 0.0:
                ce_inner_model = model
            pd.DataFrame(history).to_csv(
                args.out_dir / f"seed_{seed}_tau_{tau:g}_history.csv", index=False
            )
        if ce_inner_model is None:
            raise ValueError("tau-values must contain 0.0 as the validation CE reference")
        ce_val = next(item for item in tau_candidates if float(item["tau"]) == 0.0)["metrics"]
        chosen_tau = choose_candidate(
            tau_candidates, ce_val["accuracy"], args.accuracy_tolerance, "tau"
        )
        if chosen_tau is None:
            chosen_tau = next(item for item in tau_candidates if float(item["tau"]) == 0.0)

        # cRT: freeze the CE-trained encoder, reinitialize the classifier, then vary balance strength.
        fit_loader = make_loader(
            fair, inner_x, inner_scalar, labels, masks, fit_idx, args.batch_size, False, seed
        )
        fit_features, fit_targets = extract_frame_features(ce_inner_model, fit_loader, device)
        crt_candidates = [{"rho": -1.0, "name": "no_retrain", "metrics": ce_val}]
        for rho in args.rho_values:
            head = fresh_head(fair, common_state, args.hidden, args.dropout, device)
            head, _ = train_classifier_head(
                head, fit_features, fit_targets, len(fair.PHASES), rho, device, seed,
                args.retrain_epochs, args.head_batch_size, args.head_lr, args.weight_decay,
            )
            candidate_model = copy.deepcopy(ce_inner_model)
            candidate_model.head.load_state_dict(clone_state(head), strict=True)
            metrics, _ = imbalance.evaluate(
                candidate_model, val_loader, device, list(fair.PHASES), fit_counts
            )
            crt_candidates.append({"rho": rho, "name": "retrain", "metrics": metrics})
        chosen_crt = choose_candidate(
            crt_candidates, ce_val["accuracy"], args.accuracy_tolerance, "rho"
        )
        if chosen_crt is None:
            chosen_crt = crt_candidates[0]

        selection = {
            "seed": seed,
            "common_initialization_hash": common_hash,
            "fit_embryos": int(len(fit_idx)),
            "validation_embryos": int(len(val_idx)),
            "outer_train_embryos": int(len(outer_train)),
            "test_embryos": int(len(test_idx)),
            "accuracy_tolerance": args.accuracy_tolerance,
            "ce_validation_accuracy": ce_val["accuracy"],
            "tau_candidates": tau_candidates,
            "selected_tau": chosen_tau["tau"],
            "crt_candidates": crt_candidates,
            "selected_crt": {"name": chosen_crt["name"], "rho": chosen_crt["rho"]},
            "inner_gate_audit": inner_gate_audit,
        }
        selections.append(selection)
        write_json(args.out_dir / "selections" / f"seed_{seed}.json", selection)

        # Refit the selected methods on all 90% training embryos, then evaluate outer test once.
        outer_gate, outer_gate_state, outer_gate_audit = fair.make_oof_soft_gates(
            base, labels, masks, embryo_ids, outer_train, test_idx, seed
        )
        outer_x, outer_scalar = fair.variant_inputs(
            base, labels, masks, outer_gate, "soft_pred_gate"
        )
        outer_counts = train_counts(
            imbalance, labels, masks, outer_train, len(fair.PHASES)
        )
        test_loader = make_loader(
            fair, outer_x, outer_scalar, labels, masks, test_idx, args.batch_size, False, seed
        )

        baseline_model, baseline_history = train_sequence_model(
            fair, outer_x, outer_scalar, labels, masks, outer_train, common_state, device,
            seed, args.epochs, args.batch_size, args.lr, args.weight_decay,
            args.hidden, args.dropout, 0.0,
        )
        baseline_metrics, baseline_predictions = imbalance.evaluate(
            baseline_model, test_loader, device, list(fair.PHASES), outer_counts
        )

        if float(chosen_tau["tau"]) == 0.0:
            tuned_model = copy.deepcopy(baseline_model)
            tuned_history = baseline_history
        else:
            tuned_model, tuned_history = train_sequence_model(
                fair, outer_x, outer_scalar, labels, masks, outer_train, common_state, device,
                seed, args.epochs, args.batch_size, args.lr, args.weight_decay,
                args.hidden, args.dropout, float(chosen_tau["tau"]),
            )
        tuned_metrics, tuned_predictions = imbalance.evaluate(
            tuned_model, test_loader, device, list(fair.PHASES), outer_counts
        )

        if chosen_crt["name"] == "no_retrain":
            crt_model = copy.deepcopy(baseline_model)
            crt_history = []
        else:
            outer_train_loader = make_loader(
                fair, outer_x, outer_scalar, labels, masks, outer_train,
                args.batch_size, False, seed,
            )
            outer_features, outer_targets = extract_frame_features(
                baseline_model, outer_train_loader, device
            )
            crt_head = fresh_head(fair, common_state, args.hidden, args.dropout, device)
            crt_head, crt_history = train_classifier_head(
                crt_head, outer_features, outer_targets, len(fair.PHASES),
                float(chosen_crt["rho"]), device, seed, args.retrain_epochs,
                args.head_batch_size, args.head_lr, args.weight_decay,
            )
            crt_model = copy.deepcopy(baseline_model)
            crt_model.head.load_state_dict(clone_state(crt_head), strict=True)
        crt_metrics, crt_predictions = imbalance.evaluate(
            crt_model, test_loader, device, list(fair.PHASES), outer_counts
        )

        method_outputs = [
            ("baseline_ce", baseline_model, baseline_history, baseline_metrics, baseline_predictions),
            ("tuned_balanced_softmax", tuned_model, tuned_history, tuned_metrics, tuned_predictions),
            ("tuned_crt", crt_model, crt_history, crt_metrics, crt_predictions),
        ]
        for method, model, history, metrics, predictions in method_outputs:
            run_dir = args.out_dir / "runs" / method / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            pd.DataFrame(predictions).to_csv(run_dir / "test_predictions.csv", index=False)
            write_json(run_dir / "test_metrics.json", metrics)
            torch.save(
                {
                    "model": clone_state(model),
                    "method": method,
                    "seed": seed,
                    "selected_tau": chosen_tau["tau"],
                    "selected_crt": selection["selected_crt"],
                    "common_initialization_hash": common_hash,
                    "feature_names": feature_names,
                    "outer_gate_state": outer_gate_state,
                    "outer_gate_audit": outer_gate_audit,
                },
                run_dir / "final.pt",
            )
            run = {
                "seed": seed,
                "method": method,
                "selected_tau": chosen_tau["tau"],
                "selected_crt": selection["selected_crt"],
                "common_initialization_hash": common_hash,
                **metrics,
            }
            final_runs.append(run)
            print(
                f"seed={seed} method={method} tau={chosen_tau['tau']} "
                f"crt={selection['selected_crt']} acc={metrics['accuracy']:.4f} "
                f"ba={metrics['balanced_accuracy']:.4f} f1={metrics['macro_f1_all_16']:.4f} "
                f"tail={metrics['tail_recall']}",
                flush=True,
            )

    summary = {
        "task": "Validation-selected imbalance calibration and classifier retraining",
        "protocol": "80% fit / 10% validation / 10% untouched test by embryo",
        "input_variant": "soft_pred_gate",
        "selection_constraint": (
            f"validation accuracy must be at least CE accuracy minus {args.accuracy_tolerance:.3f}; "
            "maximize validation macro-F1 among eligible candidates"
        ),
        "tau_values": args.tau_values,
        "rho_values": args.rho_values,
        "aggregate": aggregate(final_runs),
        "selections": selections,
        "per_seed": final_runs,
    }
    write_json(args.out_dir / "imbalance_calibration_crt_summary.json", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
