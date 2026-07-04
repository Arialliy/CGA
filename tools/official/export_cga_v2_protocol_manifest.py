"""Export a run-local CGA-v2 protocol manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from dataset import dataset_registry_sha256, get_dataset_entry, sha256_file, split_path
from loss import CGALossConfig


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _split_hashes(dataset_dir: Path, dataset_name: str, registry: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "test", "hcval"):
        try:
            path = split_path(dataset_dir, dataset_name, split, registry)
        except KeyError:
            out[f"{split}_list_path"] = None
            out[f"{split}_list_sha256"] = None
            continue
        out[f"{split}_list_path"] = str(path)
        out[f"{split}_list_sha256"] = sha256_file(path) if path.exists() else None
    return out


def build_manifest(
    *,
    root: Path,
    dataset_dir: Path,
    dataset_name: str,
    seed: int,
    model: str,
    epoch: int,
    threshold: float,
    output_dir: Path,
    preflight_summary: Path | None,
    registry: Path,
    command_args: dict[str, Any],
) -> dict[str, Any]:
    registry_data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    entry = get_dataset_entry(dataset_name, registry)
    loss_cfg = CGALossConfig()
    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "model": model,
        "epoch": epoch,
        "threshold": threshold,
        "dataset_dir": str(dataset_dir),
        "dataset_spec_sha256": dataset_registry_sha256(registry),
        "dataset_registry": str(registry),
        "model_registry_entry": entry,
        "registry_dataset_names": sorted(registry_data.get("datasets", {})),
        "loss_name": "MSHNetCGALoss" if "cga" in model.lower() else "MSHNetOHEMLoss",
        "cga_loss_config": {
            "lambda_center": loss_cfg.lambda_center,
            "lambda_boundary": loss_cfg.lambda_boundary,
            "lambda_scale": loss_cfg.lambda_scale,
            "lambda_peak": loss_cfg.lambda_peak,
            "start_epoch": loss_cfg.start_epoch,
            "ramp_epochs": loss_cfg.ramp_epochs,
            "ohem_ratio": loss_cfg.ohem_ratio,
        },
        "eval_script": "test.py",
        "metric_version": "IRSTDMetrics:v1:global_mIoU_threshold_gt",
        "command_line_arguments": command_args,
        "git_commit": _git_commit(root),
    }
    manifest.update(_split_hashes(dataset_dir, dataset_name, registry))
    if preflight_summary and preflight_summary.exists():
        manifest["preflight_summary"] = str(preflight_summary)
        manifest["preflight"] = json.loads(preflight_summary.read_text(encoding="utf-8"))
    manifest["run_dir"] = str(output_dir / model / f"seed{seed}" / dataset_name)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", required=True)
    p.add_argument("--epoch", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ohem_ratio", type=float, default=0.01)
    p.add_argument("--output_dir", default="results/official")
    p.add_argument("--preflight_summary", default="")
    p.add_argument("--registry", default="configs/datasets.yaml")
    p.add_argument("--output", default="")
    args = p.parse_args()
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    preflight = Path(args.preflight_summary) if args.preflight_summary else None
    manifest = build_manifest(
        root=root,
        dataset_dir=Path(args.dataset_dir),
        dataset_name=args.dataset_name,
        seed=args.seed,
        model=args.model,
        epoch=args.epoch,
        threshold=args.threshold,
        output_dir=output_dir,
        preflight_summary=preflight,
        registry=Path(args.registry),
        command_args=vars(args),
    )
    out = Path(args.output) if args.output else output_dir / args.model / f"seed{args.seed}" / args.dataset_name / "protocol_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
