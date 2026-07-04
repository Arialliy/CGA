from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset import sha256_file
from loss import CGALossConfig
from tools.official.check_cga_v2_p2b_protocol_diff_audit import (
    DECISION_PATCHED,
    DECISION_STOP,
    run_audit,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _metric(model: str, split: str, *, dataset: str = "NUDT-SIRST", seed: int = 42) -> dict:
    return {
        "checkpoint_epoch": 400,
        "dataset": dataset,
        "epoch": 400,
        "mIoU": 0.8,
        "model": model,
        "seed": seed,
        "split": split,
        "threshold": 0.5,
    }


def _make_case(
    tmp_path: Path,
    *,
    actionable: bool = True,
    p2_gate: str = "Gate-CGA-v2-P2-seed42-reproduction",
    dataset: str = "NUDT-SIRST",
    seed: int = 42,
    missing_hash: bool = False,
    p2a_loss_start: int | None = 1,
) -> dict[str, Path]:
    ds_root = tmp_path / "datasets" / "NUDT-SIRST"
    list_dir = ds_root / "img_idx"
    current_train = _write(list_dir / "train_NUDT-SIRST.txt", "a\nb\nc\n")
    test_list = _write(list_dir / "test_NUDT-SIRST.txt", "d\n")
    hcval_list = _write(list_dir / "hcval_NUDT-SIRST.txt", "d\n")
    frozen_train = _write(tmp_path / "frozen" / "train_NUDT-SIRST.txt", "a\nb\nc\n" if actionable else "a\nb\nc\n")
    current = CGALossConfig()
    preflight_train_sha = "old-train-sha" if actionable else sha256_file(current_train)
    preflight = {
        "dataset_name": "NUDT-SIRST",
        "dataset_root": str(ds_root),
        "gate_pass": True,
        "train_count": 2 if actionable else 3,
        "test_count": 1,
        "hcval_count": 1,
        "train_list_sha256": None if missing_hash else preflight_train_sha,
        "test_list_sha256": sha256_file(test_list),
        "hcval_list_sha256": sha256_file(hcval_list),
        "train_test_overlap_count": 0,
        "train_hcval_overlap_count": 0,
    }
    p2 = {
        "gate": p2_gate,
        "dataset_name": dataset,
        "seed": seed,
        "threshold": 0.5,
        "epoch": 400,
        "decision": "P2_FAIL_IMPL_AUDIT_ALLOWED",
        "full": {
            "baseline": _metric("MSHNetOHEM", "test", dataset=dataset, seed=seed) | {"mIoU": 0.924},
            "candidate": _metric("MSHNetCGA", "test", dataset=dataset, seed=seed),
        },
        "hcval": {
            "baseline": _metric("MSHNetOHEM", "hcval", dataset=dataset, seed=seed) | {"mIoU": 0.856},
            "candidate": _metric("MSHNetCGA", "hcval", dataset=dataset, seed=seed),
        },
    }
    p2a = {
        "gate": "Gate-CGA-v2-P2A-seed42-implementation-audit",
        "decision": "P2A_IMPL_MISMATCH_FOUND_RERUN_P1_P2_ALLOWED",
        "cga_loss_and_target_audit": {
            "start_epoch": current.start_epoch if p2a_loss_start is None else p2a_loss_start,
            "ramp_epochs": current.ramp_epochs,
        },
    }
    frozen_plan = {
        "frozen_loss": {
            "lambda_center": current.lambda_center,
            "lambda_boundary": current.lambda_boundary,
            "lambda_scale": current.lambda_scale,
            "lambda_peak": current.lambda_peak,
            "start_epoch": current.start_epoch,
            "ramp_epochs": current.ramp_epochs,
        }
    }
    paths = {
        "p2": tmp_path / "p2.json",
        "p2a": tmp_path / "p2a.json",
        "preflight": tmp_path / "preflight.json",
        "frozen_train": frozen_train,
        "frozen_plan": tmp_path / "frozen_plan.json",
    }
    paths["p2"].write_text(json.dumps(p2), encoding="utf-8")
    paths["p2a"].write_text(json.dumps(p2a), encoding="utf-8")
    paths["preflight"].write_text(json.dumps(preflight), encoding="utf-8")
    paths["frozen_plan"].write_text(json.dumps(frozen_plan), encoding="utf-8")
    return paths


def _audit(paths: dict[str, Path]) -> dict:
    return run_audit(
        p2_summary=paths["p2"],
        p2a_summary=paths["p2a"],
        preflight_summary=paths["preflight"],
        frozen_train_list=str(paths["frozen_train"]),
        frozen_plan=str(paths["frozen_plan"]),
    )


def test_rejects_missing_p2_summary(tmp_path: Path) -> None:
    paths = _make_case(tmp_path)
    paths["p2"].unlink()
    with pytest.raises(FileNotFoundError):
        _audit(paths)


def test_rejects_wrong_p2_gate_name(tmp_path: Path) -> None:
    paths = _make_case(tmp_path, p2_gate="Wrong-Gate")
    summary = _audit(paths)
    assert summary["audit_complete"] is False
    assert "p2_gate_mismatch" in summary["blocking_errors"]
    assert summary["rerun_p1_p2_once_allowed"] is False


def test_rejects_wrong_dataset(tmp_path: Path) -> None:
    paths = _make_case(tmp_path, dataset="OTHER")
    summary = _audit(paths)
    assert summary["audit_complete"] is False
    assert "p2_dataset_name_mismatch" in summary["blocking_errors"]


def test_rejects_wrong_seed(tmp_path: Path) -> None:
    paths = _make_case(tmp_path, seed=43)
    summary = _audit(paths)
    assert summary["audit_complete"] is False
    assert "p2_seed_mismatch" in summary["blocking_errors"]


def test_rejects_missing_preflight_list_hashes(tmp_path: Path) -> None:
    paths = _make_case(tmp_path, missing_hash=True)
    summary = _audit(paths)
    assert summary["audit_complete"] is False
    assert "missing_preflight_train_list_sha256" in summary["blocking_errors"]


def test_detects_ohem_anchor_sanity_mismatch_without_using_as_evidence(tmp_path: Path) -> None:
    paths = _make_case(tmp_path)
    summary = _audit(paths)
    sanity = summary["historical_sanity_reference_only"]
    assert sanity["used_as_new_repo_evidence"] is False
    assert sanity["delta"]["full_mIoU"] > 0.08
    assert sanity["delta"]["hcval_mIoU"] > 0.20


def test_allows_rerun_only_when_actionable_mismatch_exists(tmp_path: Path) -> None:
    paths = _make_case(tmp_path, actionable=True, p2a_loss_start=1)
    summary = _audit(paths)
    assert summary["decision"] == DECISION_PATCHED
    assert summary["rerun_p1_p2_once_allowed"] is True
    assert "split_identity_mismatch" in summary["mismatch_categories"]
    assert "loss_config_mismatch" in summary["mismatch_categories"]


def test_stops_when_no_actionable_mismatch_exists(tmp_path: Path) -> None:
    paths = _make_case(tmp_path, actionable=False, p2a_loss_start=None)
    summary = _audit(paths)
    assert summary["decision"] == DECISION_STOP
    assert summary["actionable_mismatch_found"] is False
    assert summary["rerun_p1_p2_once_allowed"] is False
