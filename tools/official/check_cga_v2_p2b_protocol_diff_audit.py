"""P2B protocol-diff audit and one-time rerun gate for CGA-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset import sha256_file
from loss import CGALossConfig


GATE = "Gate-CGA-v2-P2B-protocol-diff-audit"
DECISION_PATCHED = "P2B_IMPL_MISMATCH_PATCHED_RERUN_P1_P2_ONCE"
DECISION_STOP = "P2B_NO_ACTIONABLE_IMPL_MISMATCH_STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42"
P2_GATE = "Gate-CGA-v2-P2-seed42-reproduction"
P2A_GATE = "Gate-CGA-v2-P2A-seed42-implementation-audit"
P2A_MISMATCH = "P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED"

HISTORICAL_OHEM = {
    "full_mIoU": 0.8343498886629878,
    "hcval_mIoU": 0.6047904191616766,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _candidate_paths(dataset_name: str) -> list[Path]:
    rel = Path("datasets") / dataset_name / "img_idx" / f"train_{dataset_name}.txt"
    roots = [
        Path("/home/AAAI/OHCM-MSHNet-1"),
        Path("/home/ly/AAAI/OHCM-MSHNet-1"),
        Path("/home/AAAI/BasicIRSTD-main1"),
        Path("/home/ly/AAAI/BasicIRSTD-main1"),
        Path("/home/AAAI/Reemm"),
        Path("/home/ly/AAAI/Reemm"),
    ]
    return [root / rel for root in roots]


def _find_frozen_train(dataset_name: str, explicit: str = "") -> dict[str, Any]:
    paths = [Path(explicit)] if explicit else _candidate_paths(dataset_name)
    candidates = []
    for path in paths:
        if path.exists():
            candidates.append({"path": str(path), "count": _count_lines(path), "sha256": sha256_file(path)})
    selected = None
    for row in candidates:
        if row["count"] == 697:
            selected = row
            break
    if selected is None and candidates:
        selected = candidates[0]
    return {"selected": selected, "candidates": candidates}


def _load_frozen_plan(path_text: str = "") -> dict[str, Any]:
    paths = [
        Path(path_text) if path_text else None,
        Path("/home/AAAI/OHCM-MSHNet-main/docs/internal/cga_v2/cga_v2_frozen_plan.json"),
        Path("/home/ly/AAAI/OHCM-MSHNet-main/docs/internal/cga_v2/cga_v2_frozen_plan.json"),
    ]
    for path in [p for p in paths if p is not None]:
        if path.exists():
            return {"path": str(path), "content": _load(path)}
    return {"path": None, "content": {}}


def _summary_identity_errors(p2: dict[str, Any], dataset: str, seed: int, threshold: float, epoch: int) -> list[str]:
    errors = []
    expected_top = {
        "gate": P2_GATE,
        "dataset_name": dataset,
        "seed": seed,
        "threshold": threshold,
        "epoch": epoch,
    }
    for key, value in expected_top.items():
        if p2.get(key) != value:
            errors.append(f"p2_{key}_mismatch")
    checks = [
        ("full_baseline", p2.get("full", {}).get("baseline", {}), "MSHNetOHEM", "test"),
        ("full_candidate", p2.get("full", {}).get("candidate", {}), "MSHNetCGA", "test"),
        ("hcval_baseline", p2.get("hcval", {}).get("baseline", {}), "MSHNetOHEM", "hcval"),
        ("hcval_candidate", p2.get("hcval", {}).get("candidate", {}), "MSHNetCGA", "hcval"),
    ]
    for label, row, model, split in checks:
        expected = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "threshold": threshold,
            "checkpoint_epoch": epoch,
            "split": split,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(f"{label}_{key}_mismatch")
    return errors


def _preflight_errors(preflight: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("train_list_sha256", "test_list_sha256", "hcval_list_sha256"):
        if not preflight.get(key):
            errors.append(f"missing_preflight_{key}")
    for key in ("train_count", "test_count", "hcval_count"):
        if preflight.get(key) is None:
            errors.append(f"missing_preflight_{key}")
    if preflight.get("train_test_overlap_count", 0) != 0:
        errors.append("train_test_overlap")
    if preflight.get("train_hcval_overlap_count", 0) != 0:
        errors.append("train_hcval_overlap")
    return errors


def _actual_split_state(preflight: dict[str, Any], dataset: str) -> dict[str, Any]:
    root = Path(str(preflight.get("dataset_root", "")))
    list_dir = root / "img_idx"
    out = {"dataset_root": str(root)}
    for split in ("train", "test", "hcval"):
        path = list_dir / f"{split}_{dataset}.txt"
        out[split] = {
            "path": str(path),
            "exists": path.exists(),
            "count": _count_lines(path) if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "preflight_count": preflight.get(f"{split}_count"),
            "preflight_sha256": preflight.get(f"{split}_list_sha256"),
        }
    return out


def run_audit(
    *,
    p2_summary: Path,
    p2a_summary: Path,
    preflight_summary: Path,
    output: Path | None = None,
    dataset: str = "NUDT-SIRST",
    seed: int = 42,
    threshold: float = 0.5,
    epoch: int = 400,
    frozen_train_list: str = "",
    frozen_plan: str = "",
) -> dict[str, Any]:
    p2 = _load(p2_summary)
    p2a = _load(p2a_summary)
    preflight = _load(preflight_summary)
    mismatch_categories: list[str] = []
    allowed_fix_scope: list[str] = []
    blocking_errors: list[str] = []

    blocking_errors.extend(_summary_identity_errors(p2, dataset, seed, threshold, epoch))
    if p2a.get("gate") != P2A_GATE:
        blocking_errors.append("p2a_gate_mismatch")
    if p2a.get("decision") != P2A_MISMATCH:
        blocking_errors.append("p2a_decision_not_impl_mismatch")
    blocking_errors.extend(_preflight_errors(preflight))

    actual_splits = _actual_split_state(preflight, dataset)
    frozen_train = _find_frozen_train(dataset, frozen_train_list)
    selected_train = frozen_train["selected"]
    split_patch_applied = False
    if selected_train:
        p1_train_sha = preflight.get("train_list_sha256")
        actual_train_sha = actual_splits["train"]["sha256"]
        if p1_train_sha != selected_train["sha256"]:
            mismatch_categories.append("split_identity_mismatch")
            allowed_fix_scope.append("restore_frozen_split_lists_only")
            split_patch_applied = actual_train_sha == selected_train["sha256"]
    frozen = _load_frozen_plan(frozen_plan)
    frozen_loss = frozen.get("content", {}).get("frozen_loss", {})
    p2a_loss = p2a.get("cga_loss_and_target_audit", {})
    current_loss = CGALossConfig()
    loss_patch_applied = False
    if frozen_loss:
        p2a_start = p2a_loss.get("start_epoch")
        p2a_ramp = p2a_loss.get("ramp_epochs")
        frozen_start = frozen_loss.get("start_epoch")
        frozen_ramp = frozen_loss.get("ramp_epochs")
        if p2a_start != frozen_start or p2a_ramp != frozen_ramp:
            mismatch_categories.append("loss_config_mismatch")
            allowed_fix_scope.append("restore_frozen_cga_loss_schedule_only")
            loss_patch_applied = current_loss.start_epoch == frozen_start and current_loss.ramp_epochs == frozen_ramp

    actionable_mismatch_found = bool(mismatch_categories)
    patch_status = {
        "split_identity_mismatch_patched": split_patch_applied if "split_identity_mismatch" in mismatch_categories else None,
        "loss_config_mismatch_patched": loss_patch_applied if "loss_config_mismatch" in mismatch_categories else None,
    }
    required_patches = [v for v in patch_status.values() if v is not None]
    patches_applied = bool(required_patches) and all(required_patches)
    rerun_allowed = actionable_mismatch_found and patches_applied and not blocking_errors
    decision = DECISION_PATCHED if rerun_allowed else DECISION_STOP
    full_ohem = p2.get("full", {}).get("baseline", {}).get("mIoU")
    hcval_ohem = p2.get("hcval", {}).get("baseline", {}).get("mIoU")
    summary = {
        "gate": GATE,
        "dataset": dataset,
        "seed": seed,
        "audit_complete": not blocking_errors,
        "p2_decision": p2.get("decision"),
        "p2a_decision": p2a.get("decision"),
        "mismatch_categories": sorted(set(mismatch_categories)),
        "actionable_mismatch_found": actionable_mismatch_found,
        "allowed_fix_scope": sorted(set(allowed_fix_scope)),
        "rerun_p1_p2_once_allowed": rerun_allowed,
        "decision": decision,
        "blocking_errors": blocking_errors,
        "patch_status": patch_status,
        "p2_summary": str(p2_summary),
        "p2a_summary": str(p2a_summary),
        "preflight_summary": str(preflight_summary),
        "dataset_split_identity": {
            "preflight": {
                "train_count": preflight.get("train_count"),
                "test_count": preflight.get("test_count"),
                "hcval_count": preflight.get("hcval_count"),
                "train_list_sha256": preflight.get("train_list_sha256"),
                "test_list_sha256": preflight.get("test_list_sha256"),
                "hcval_list_sha256": preflight.get("hcval_list_sha256"),
            },
            "actual": actual_splits,
            "frozen_train_reference": frozen_train,
        },
        "loss_config_identity": {
            "frozen_plan": frozen,
            "p2a_recorded_loss_config": p2a_loss,
            "current_code_loss_config": {
                "lambda_center": current_loss.lambda_center,
                "lambda_boundary": current_loss.lambda_boundary,
                "lambda_scale": current_loss.lambda_scale,
                "lambda_peak": current_loss.lambda_peak,
                "start_epoch": current_loss.start_epoch,
                "ramp_epochs": current_loss.ramp_epochs,
            },
        },
        "summary_identity_errors": blocking_errors,
        "historical_sanity_reference_only": {
            "historical_ohem": HISTORICAL_OHEM,
            "new_repo_ohem": {
                "full_mIoU": full_ohem,
                "hcval_mIoU": hcval_ohem,
            },
            "delta": {
                "full_mIoU": None if full_ohem is None else full_ohem - HISTORICAL_OHEM["full_mIoU"],
                "hcval_mIoU": None if hcval_ohem is None else hcval_ohem - HISTORICAL_OHEM["hcval_mIoU"],
            },
            "used_as_new_repo_evidence": False,
        },
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--p2_summary", required=True)
    p.add_argument("--p2a_summary", required=True)
    p.add_argument("--preflight_summary", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="NUDT-SIRST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--epoch", type=int, default=400)
    p.add_argument("--frozen_train_list", default="")
    p.add_argument("--frozen_plan", default="")
    args = p.parse_args()
    summary = run_audit(
        p2_summary=Path(args.p2_summary),
        p2a_summary=Path(args.p2a_summary),
        preflight_summary=Path(args.preflight_summary),
        output=Path(args.output),
        dataset=args.dataset,
        seed=args.seed,
        threshold=args.threshold,
        epoch=args.epoch,
        frozen_train_list=args.frozen_train_list,
        frozen_plan=args.frozen_plan,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["audit_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
