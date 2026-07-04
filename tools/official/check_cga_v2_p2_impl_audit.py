"""P2A implementation/protocol audit for CGA-v2 seed42 failure."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from loss import CGALossConfig
from utils.cga_targets import CGATargetConfig, build_cga_targets


GATE = "Gate-CGA-v2-P2A-seed42-implementation-audit"
DECISION_MISMATCH = "P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED"
DECISION_NO_MISMATCH = "P2A_NO_IMPL_MISMATCH_STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42"
DECISION_INCOMPLETE = "P2A_AUDIT_INCOMPLETE_STOP"

AUDIT_SCOPE = [
    "dataset_split_identity",
    "eval_split_identity",
    "ohem_anchor_identity",
    "cga_identity",
    "cga_loss_target_identity",
    "checkpoint_identity",
    "metric_identity",
]

HISTORICAL_OHEM_SANITY = {
    "full_mIoU": {"reference": 0.83435, "min": 0.80, "max": 0.87},
    "hcval_mIoU": {"reference": 0.60479, "min": 0.55, "max": 0.66},
}

METRIC_KEYS = ["mIoU", "nIoU", "Precision", "Recall", "F1", "Pd", "FA", "FA_ppm", "FP_components"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path_text: str | None, root: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _record_mismatch(mismatches: list[str], code: str) -> None:
    if code not in mismatches:
        mismatches.append(code)


def _count_predictions(summary: dict[str, Any], root: Path, incomplete: list[str], label: str) -> int | None:
    pred_dir = _resolve(str(summary.get("prediction_dir", "")), root)
    if pred_dir is None or not pred_dir.exists():
        incomplete.append(f"{label}_prediction_dir_missing")
        return None
    return len(list(pred_dir.glob("*.png")))


def _read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def _metric_summary_check(
    summary: dict[str, Any],
    *,
    label: str,
    expected_dataset: str,
    expected_seed: int,
    expected_threshold: float,
    expected_epoch: int,
    expected_model: str,
    expected_split: str,
    root: Path,
    expected_count: int | None,
    mismatches: list[str],
    incomplete: list[str],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "label": label,
        "expected_model": expected_model,
        "expected_split": expected_split,
        "required_metrics_present": True,
    }
    expected = {
        "dataset": expected_dataset,
        "seed": expected_seed,
        "threshold": expected_threshold,
        "checkpoint_epoch": expected_epoch,
        "model": expected_model,
        "split": expected_split,
    }
    for key, value in expected.items():
        actual = summary.get(key)
        checks[key] = actual
        if actual != value:
            _record_mismatch(mismatches, f"{label}_{key}_mismatch")
    for key in METRIC_KEYS:
        if _as_float(summary.get(key)) is None:
            checks["required_metrics_present"] = False
            incomplete.append(f"{label}_{key}_missing_or_nonfinite")
    pred_count = _count_predictions(summary, root, incomplete, label)
    checks["prediction_count"] = pred_count
    checks["expected_prediction_count"] = expected_count
    if expected_count is not None and pred_count is not None and pred_count != expected_count:
        _record_mismatch(mismatches, f"{label}_prediction_count_mismatch")
    ckpt = _resolve(str(summary.get("checkpoint", "")), root)
    checks["checkpoint"] = str(ckpt) if ckpt else None
    if ckpt is None or not ckpt.exists():
        incomplete.append(f"{label}_checkpoint_missing")
    else:
        checks["checkpoint_sha256"] = sha256_file(ckpt)
    return checks


def _dataset_identity(p1: dict[str, Any], p2: dict[str, Any], dataset_name: str, mismatches: list[str]) -> dict[str, Any]:
    dataset_root = Path(str(p1.get("dataset_root", ""))) if p1.get("dataset_root") else None
    list_dir = dataset_root / "img_idx" if dataset_root else None
    summary = {
        "DATASET_NAME": dataset_name,
        "DATASET_DIR": str(dataset_root.parent) if dataset_root else None,
        "dataset_root": str(dataset_root) if dataset_root else None,
        "dataset_spec_sha256": p1.get("dataset_registry_sha256"),
        "train_list_path": str(list_dir / f"train_{dataset_name}.txt") if list_dir else None,
        "test_list_path": str(list_dir / f"test_{dataset_name}.txt") if list_dir else None,
        "hcval_list_path": str(list_dir / f"hcval_{dataset_name}.txt") if list_dir else None,
        "train_list_sha256": p1.get("train_list_sha256"),
        "test_list_sha256": p1.get("test_list_sha256"),
        "hcval_list_sha256": p1.get("hcval_list_sha256"),
        "train_count": p1.get("train_count"),
        "test_count": p1.get("test_count"),
        "hcval_count": p1.get("hcval_count"),
        "train_test_overlap_count": p1.get("train_test_overlap_count"),
        "duplicate_item_count": p1.get("duplicate_item_count"),
        "p1_gate_pass": p1.get("gate_pass"),
        "p1_decision": p1.get("decision"),
    }
    for key in ("train_list_sha256", "test_list_sha256", "hcval_list_sha256", "dataset_registry_sha256"):
        if p2.get(key) != p1.get(key):
            _record_mismatch(mismatches, f"p2_{key}_differs_from_p1")
    if p1.get("dataset_name") != dataset_name:
        _record_mismatch(mismatches, "p1_dataset_name_mismatch")
    if p2.get("dataset_name") != dataset_name:
        _record_mismatch(mismatches, "p2_dataset_name_mismatch")
    if p1.get("gate_pass") is not True:
        _record_mismatch(mismatches, "p1_preflight_not_passed")
    return summary


def _eval_identity(
    p1: dict[str, Any],
    p2: dict[str, Any],
    *,
    dataset_name: str,
    seed: int,
    threshold: float,
    epoch: int,
    baseline: str,
    candidate: str,
    root: Path,
    mismatches: list[str],
    incomplete: list[str],
) -> dict[str, Any]:
    full = p2.get("full", {})
    hcval = p2.get("hcval", {})
    return {
        "full": {
            "list_hash": p1.get("test_list_sha256"),
            "evaluated_images": p1.get("test_count"),
            "baseline": _metric_summary_check(
                full.get("baseline", {}),
                label="full_baseline",
                expected_dataset=dataset_name,
                expected_seed=seed,
                expected_threshold=threshold,
                expected_epoch=epoch,
                expected_model=baseline,
                expected_split="test",
                root=root,
                expected_count=p1.get("test_count"),
                mismatches=mismatches,
                incomplete=incomplete,
            ),
            "candidate": _metric_summary_check(
                full.get("candidate", {}),
                label="full_candidate",
                expected_dataset=dataset_name,
                expected_seed=seed,
                expected_threshold=threshold,
                expected_epoch=epoch,
                expected_model=candidate,
                expected_split="test",
                root=root,
                expected_count=p1.get("test_count"),
                mismatches=mismatches,
                incomplete=incomplete,
            ),
        },
        "hcval": {
            "available": bool(hcval.get("available")),
            "list_hash": p1.get("hcval_list_sha256"),
            "evaluated_images": p1.get("hcval_count"),
            "baseline": _metric_summary_check(
                hcval.get("baseline", {}),
                label="hcval_baseline",
                expected_dataset=dataset_name,
                expected_seed=seed,
                expected_threshold=threshold,
                expected_epoch=epoch,
                expected_model=baseline,
                expected_split="hcval",
                root=root,
                expected_count=p1.get("hcval_count"),
                mismatches=mismatches,
                incomplete=incomplete,
            ),
            "candidate": _metric_summary_check(
                hcval.get("candidate", {}),
                label="hcval_candidate",
                expected_dataset=dataset_name,
                expected_seed=seed,
                expected_threshold=threshold,
                expected_epoch=epoch,
                expected_model=candidate,
                expected_split="hcval",
                root=root,
                expected_count=p1.get("hcval_count"),
                mismatches=mismatches,
                incomplete=incomplete,
            ),
        },
        "metric_threshold": threshold,
        "mask_policy": "gt mask is binary via gt > 0",
        "image_resize_policy": "test loader pads/crops to original size",
        "prediction_resize_policy": "crop final probability map to original size before metrics",
    }


def _historical_anchor_check(p2: dict[str, Any], mismatches: list[str]) -> dict[str, Any]:
    full_miou = _as_float(p2.get("full", {}).get("baseline", {}).get("mIoU"))
    hcval_miou = _as_float(p2.get("hcval", {}).get("baseline", {}).get("mIoU"))
    out = {
        "mode": "sanity_reference_not_exact_requirement",
        "historical_reference": HISTORICAL_OHEM_SANITY,
        "full_mIoU": full_miou,
        "hcval_mIoU": hcval_miou,
        "full_within_sanity_range": False,
        "hcval_within_sanity_range": False,
    }
    if full_miou is not None:
        spec = HISTORICAL_OHEM_SANITY["full_mIoU"]
        out["full_within_sanity_range"] = bool(spec["min"] <= full_miou <= spec["max"])
        if not out["full_within_sanity_range"]:
            _record_mismatch(mismatches, "ohem_full_miou_outside_historical_sanity")
    if hcval_miou is not None:
        spec = HISTORICAL_OHEM_SANITY["hcval_mIoU"]
        out["hcval_within_sanity_range"] = bool(spec["min"] <= hcval_miou <= spec["max"])
        if not out["hcval_within_sanity_range"]:
            _record_mismatch(mismatches, "ohem_hcval_miou_outside_historical_sanity")
    return out


def _source_contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(encoding="utf-8")


def _train_log_identity(root: Path, model_name: str, seed: int, dataset_name: str, expected_epoch: int) -> dict[str, Any]:
    path = root / "results" / "official" / model_name / f"seed{seed}" / dataset_name / "train_log.jsonl"
    row = _read_last_jsonl(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "last_epoch": None if row is None else row.get("epoch"),
        "last_row": row,
        "epoch_matches": bool(row and row.get("epoch") == expected_epoch),
    }


def _sample_target_audit(p1: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    dataset_root = Path(str(p1.get("dataset_root", ""))) if p1.get("dataset_root") else None
    if dataset_root is None:
        return {"available": False, "reason": "dataset_root_missing"}
    train_list = dataset_root / "img_idx" / f"train_{dataset_name}.txt"
    if not train_list.exists():
        return {"available": False, "reason": "train_list_missing"}
    for raw in train_list.read_text(encoding="utf-8").splitlines()[:50]:
        item = raw.strip()
        if not item:
            continue
        item = Path(item).stem
        mask_path = dataset_root / "masks" / f"{item}.png"
        if not mask_path.exists():
            continue
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        if int((mask_arr > 0).sum()) == 0:
            continue
        mask = torch.from_numpy((mask_arr > 0).astype("float32"))[None, None]
        targets = build_cga_targets(mask, CGATargetConfig())
        sums = {k: float(v.sum().item()) for k, v in targets.items()}
        checks = {
            "available": True,
            "sample_item": item,
            "mask_shape": list(mask.shape),
            "target_sums": sums,
            "target_maps_non_empty": all(sums[k] > 0 for k in (
                "cga_center_target",
                "cga_boundary_target",
                "cga_scale_target",
                "cga_peak_target",
            )),
            "target_maps_same_spatial_size": all(tuple(v.shape[-2:]) == tuple(mask.shape[-2:]) for v in targets.values()),
            "scale_target_normalized": bool(
                targets["cga_scale_target"].min().item() >= 0.0 and targets["cga_scale_target"].max().item() <= 1.0
            ),
            "boundary_target_not_full_image": bool(sums["cga_boundary_target"] < float(mask.numel())),
            "peak_target_sparse": bool(sums["cga_peak_target"] < 0.25 * float(mask.numel())),
            "center_target_finite": bool(torch.isfinite(targets["cga_center_target"]).all().item()),
        }
        return checks
    return {"available": False, "reason": "no_nonempty_train_mask_sample_found"}


def _implementation_identity(
    root: Path,
    p1: dict[str, Any],
    p2: dict[str, Any],
    *,
    dataset_name: str,
    seed: int,
    epoch: int,
    baseline: str,
    candidate: str,
    mismatches: list[str],
    incomplete: list[str],
) -> dict[str, Any]:
    train_py = root / "train.py"
    test_py = root / "test.py"
    loss_py = root / "loss.py"
    cga_py = root / "model" / "CGA_MSHNet.py"
    net_py = root / "net.py"
    baseline_log = _train_log_identity(root, baseline, seed, dataset_name, epoch)
    candidate_log = _train_log_identity(root, candidate, seed, dataset_name, epoch)
    if not baseline_log["exists"]:
        incomplete.append("ohem_train_log_missing")
    if not candidate_log["exists"]:
        incomplete.append("cga_train_log_missing")
    if baseline_log["exists"] and not baseline_log["epoch_matches"]:
        _record_mismatch(mismatches, "ohem_train_log_epoch_mismatch")
    if candidate_log["exists"] and not candidate_log["epoch_matches"]:
        _record_mismatch(mismatches, "cga_train_log_epoch_mismatch")
    cga_last = candidate_log.get("last_row") or {}
    required_cga_terms = ["cga_center", "cga_boundary", "cga_scale", "cga_peak", "cga_w"]
    cga_terms_present = all(k in cga_last for k in required_cga_terms)
    cga_terms_finite = all(_as_float(cga_last.get(k)) is not None for k in required_cga_terms)
    if candidate_log["exists"] and not cga_terms_present:
        _record_mismatch(mismatches, "cga_loss_terms_missing_from_train_log")
    if candidate_log["exists"] and not cga_terms_finite:
        _record_mismatch(mismatches, "cga_loss_terms_nonfinite_in_train_log")
    target_audit = _sample_target_audit(p1, dataset_name)
    if not target_audit.get("available"):
        incomplete.append("cga_target_sample_audit_unavailable")
    elif not all(
        target_audit.get(k) is True
        for k in [
            "target_maps_non_empty",
            "target_maps_same_spatial_size",
            "scale_target_normalized",
            "boundary_target_not_full_image",
            "peak_target_sparse",
            "center_target_finite",
        ]
    ):
        _record_mismatch(mismatches, "cga_target_generation_semantics_mismatch")
    loss_cfg = CGALossConfig()
    target_cfg = CGATargetConfig()
    return {
        "training_determinism_and_checkpoint_identity": {
            "seed": seed,
            "torch_seed_set_in_train_py": _source_contains(train_py, "torch.manual_seed"),
            "cuda_seed_set_in_train_py": _source_contains(train_py, "torch.cuda.manual_seed_all"),
            "cudnn_deterministic": False,
            "cudnn_benchmark": True,
            "checkpoint_epoch": epoch,
            "baseline_train_log": baseline_log,
            "candidate_train_log": candidate_log,
        },
        "ohem_anchor_identity": {
            "model_name": baseline,
            "loss_name": "MSHNetOHEMLoss",
            "auxiliary_heads_disabled": _source_contains(net_py, 'name in {"mshnet", "mshnetohem", "ohem"}'),
            "cga_target_generation_disabled": baseline_log.get("last_row") is not None
            and not any(k.startswith("cga_") for k in baseline_log["last_row"]),
            "checkpoint_path_under_official_seed_dataset": f"results/official/{baseline}/seed{seed}/{dataset_name}/",
            "historical_anchor_sanity": _historical_anchor_check(p2, mismatches),
        },
        "cga_identity": {
            "model_name": candidate,
            "base_segmentation_path_active": _source_contains(cga_py, '"base_logits": final_logit'),
            "auxiliary_heads_active_in_training": _source_contains(cga_py, "self.cga_aux_head"),
            "eval_output_uses_final_logit_only": _source_contains(test_py, "extract_final_logit(output)"),
            "eval_uses_sigmoid_threshold": _source_contains(test_py, "torch.sigmoid") and _source_contains(test_py, "threshold"),
            "no_aux_output_in_test_metric": _source_contains(test_py, "logit = extract_final_logit(output)"),
        },
        "cga_loss_and_target_audit": {
            "lambda_center": loss_cfg.lambda_center,
            "lambda_boundary": loss_cfg.lambda_boundary,
            "lambda_scale": loss_cfg.lambda_scale,
            "lambda_peak": loss_cfg.lambda_peak,
            "start_epoch": loss_cfg.start_epoch,
            "ramp_epochs": loss_cfg.ramp_epochs,
            "target_config": {
                "center_radius": target_cfg.center_radius,
                "boundary_radius": target_cfg.boundary_radius,
                "peak_radius": target_cfg.peak_radius,
                "max_scale_area": target_cfg.max_scale_area,
            },
            "loss_terms_present": cga_terms_present,
            "loss_terms_finite": cga_terms_finite,
            "target_generation_sample": target_audit,
            "build_cga_targets_called_by_loss": _source_contains(loss_py, "build_cga_targets"),
        },
    }


def p3_guard_status(p2_summary: dict[str, Any], p2a_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    if p2_summary.get("gate_pass") is True:
        return {"allowed": True, "reason": "P2_SEED42_GATE_PASS"}
    if not p2a_summary:
        return {"allowed": False, "reason": "P2_FAILED_AND_P2A_NOT_CLOSED"}
    decision = p2a_summary.get("decision")
    if decision == DECISION_MISMATCH:
        return {"allowed": False, "reason": "P2A_MISMATCH_FOUND_BUT_RERUN_P2_NOT_PASS"}
    if decision == DECISION_NO_MISMATCH:
        return {"allowed": False, "reason": "P2A_NO_IMPL_MISMATCH_STOP_FORBIDS_P3"}
    return {"allowed": False, "reason": "P2A_AUDIT_INCOMPLETE_OR_UNKNOWN"}


def run_audit(
    *,
    dataset_name: str,
    seed: int,
    p1_summary: Path,
    p2_summary: Path,
    root: Path,
    threshold: float = 0.5,
    checkpoint_epoch: int = 400,
    baseline: str = "MSHNetOHEM",
    candidate: str = "MSHNetCGA",
) -> dict[str, Any]:
    mismatches: list[str] = []
    incomplete: list[str] = []
    p1 = _load_json(p1_summary)
    p2 = _load_json(p2_summary)
    if p2.get("seed") != seed:
        _record_mismatch(mismatches, "p2_seed_mismatch")
    if p2.get("threshold") != threshold:
        _record_mismatch(mismatches, "p2_threshold_mismatch")
    if p2.get("epoch") != checkpoint_epoch:
        _record_mismatch(mismatches, "p2_epoch_mismatch")
    if p2.get("baseline") != baseline:
        _record_mismatch(mismatches, "p2_baseline_name_mismatch")
    if p2.get("candidate") != candidate:
        _record_mismatch(mismatches, "p2_candidate_name_mismatch")
    dataset_identity = _dataset_identity(p1, p2, dataset_name, mismatches)
    eval_identity = _eval_identity(
        p1,
        p2,
        dataset_name=dataset_name,
        seed=seed,
        threshold=threshold,
        epoch=checkpoint_epoch,
        baseline=baseline,
        candidate=candidate,
        root=root,
        mismatches=mismatches,
        incomplete=incomplete,
    )
    implementation_identity = _implementation_identity(
        root,
        p1,
        p2,
        dataset_name=dataset_name,
        seed=seed,
        epoch=checkpoint_epoch,
        baseline=baseline,
        candidate=candidate,
        mismatches=mismatches,
        incomplete=incomplete,
    )
    if incomplete:
        decision = DECISION_INCOMPLETE
    elif mismatches:
        decision = DECISION_MISMATCH
    else:
        decision = DECISION_NO_MISMATCH
    summary = {
        "gate": GATE,
        "dataset": dataset_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "threshold": threshold,
        "checkpoint_epoch": checkpoint_epoch,
        "candidate": candidate,
        "baseline": baseline,
        "audit_scope": AUDIT_SCOPE,
        "p1_summary": str(p1_summary),
        "p2_summary": str(p2_summary),
        "dataset_split_identity": dataset_identity,
        "eval_split_identity": eval_identity,
        **implementation_identity,
        "metric_identity": {
            "metrics": METRIC_KEYS,
            "foreground_threshold": threshold,
            "binary_prediction_policy": "pred_prob > threshold",
            "binary_gt_policy": "gt_mask > 0",
            "component_fp_metric": "8-connected component count after target matching",
        },
        "mismatches": mismatches,
        "incomplete_items": incomplete,
        "audit_complete": not incomplete,
        "gate_pass": not incomplete,
        "decision": decision,
    }
    if decision == DECISION_MISMATCH:
        summary["next_allowed_action"] = "fix_only_identified_mismatch_then_rerun_P1_and_one_seed42_P2_once"
    elif decision == DECISION_NO_MISMATCH:
        summary["next_allowed_action"] = "stop_new_repo_cga_v2_as_aaai_main_method_evidence"
    else:
        summary["next_allowed_action"] = "stop_until_audit_inputs_are_complete"
    summary["p3_guard_status"] = p3_guard_status(p2, {"decision": decision})
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--checkpoint_epoch", type=int, default=400)
    p.add_argument("--baseline", default="MSHNetOHEM")
    p.add_argument("--candidate", default="MSHNetCGA")
    p.add_argument("--p1_summary", required=True)
    p.add_argument("--p2_summary", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--root", default=".")
    args = p.parse_args()
    root = Path(args.root).resolve()
    summary = run_audit(
        dataset_name=args.dataset_name,
        seed=args.seed,
        threshold=args.threshold,
        checkpoint_epoch=args.checkpoint_epoch,
        baseline=args.baseline,
        candidate=args.candidate,
        p1_summary=Path(args.p1_summary),
        p2_summary=Path(args.p2_summary),
        root=root,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["decision"] == DECISION_INCOMPLETE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
