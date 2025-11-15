"""Scheduled maintenance helper for refreshing models and cleaning caches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = ROOT_DIR / "data_cache"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "model_metrics_latest.json"
SUMMARY_PATH_DEFAULT = REPORTS_DIR / "maintenance_summary.json"
HISTORY_PATH = REPORTS_DIR / "maintenance_log.jsonl"


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def clear_cache(dry_run: bool = False, patterns: List[str] | None = None) -> Dict[str, object]:
    if patterns is None:
        patterns = ["**/*.parquet", "**/*.json"]
    removed_files: List[str] = []
    bytes_freed = 0

    if not DATA_CACHE_DIR.exists():
        return {"removed": 0, "bytes_freed": 0, "preview": []}

    for pattern in patterns:
        for path in DATA_CACHE_DIR.glob(pattern):
            if not path.is_file():
                continue
            removed_files.append(str(path.relative_to(ROOT_DIR)))
            try:
                bytes_freed += path.stat().st_size
            except OSError:
                pass
            if not dry_run:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue

    return {
        "removed": len(removed_files),
        "bytes_freed": bytes_freed,
        "preview": removed_files[:5],
        "dry_run": dry_run,
    }


def run_training(dry_run: bool = False, extra_args: List[str] | None = None) -> Dict[str, object]:
    if dry_run:
        return {"status": "skipped (dry-run)", "duration_sec": 0.0}

    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "train_all.py")]
    if extra_args:
        cmd.extend(extra_args)
    started = time.time()
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    duration = time.time() - started
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "duration_sec": round(duration, 2),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.strip().splitlines()[-5:],
        "stderr_tail": result.stderr.strip().splitlines()[-5:],
    }


def read_latest_metrics() -> Dict[str, object]:
    if not METRICS_PATH.exists():
        return {}
    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    runs = payload.get("runs") or []
    summary = {
        "generated_at": payload.get("generated_at"),
        "runs": len(runs),
        "labels": [run.get("label") for run in runs],
    }
    accuracy = []
    macro_f1 = []
    for run in runs:
        metrics = run.get("classification_metrics") or {}
        if "accuracy" in metrics:
            accuracy.append(metrics["accuracy"])
        if "macro_f1" in metrics:
            macro_f1.append(metrics["macro_f1"])
    if accuracy:
        summary["accuracy_mean"] = float(sum(accuracy) / len(accuracy))
    if macro_f1:
        summary["macro_f1_mean"] = float(sum(macro_f1) / len(macro_f1))
    return summary


def write_summary(summary: Dict[str, object], path: Path, dry_run: bool = False) -> None:
    if dry_run:
        return
    _ensure_reports_dir()
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as history:
        history.write(json.dumps(summary, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh models and clean cache.")
    parser.add_argument("--skip-train", action="store_true", help="Skip running scripts/train_all.py.")
    parser.add_argument("--skip-cache", action="store_true", help="Skip clearing data_cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without making changes.")
    parser.add_argument(
        "--summary-path",
        default=str(SUMMARY_PATH_DEFAULT),
        help="Where to write the summary JSON (default: reports/maintenance_summary.json).",
    )
    parser.add_argument(
        "--train-extra-args",
        default="",
        help="Additional arguments passed to scripts/train_all.py (quoted string).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: Dict[str, object] = {
        "run_started_at": datetime.utcnow().isoformat(),
        "dry_run": args.dry_run,
    }

    if not args.skip_cache:
        summary["cache"] = clear_cache(dry_run=args.dry_run)
    else:
        summary["cache"] = {"status": "skipped"}

    if not args.skip_train:
        extra_args = args.train_extra_args.split() if args.train_extra_args else None
        summary["training"] = run_training(dry_run=args.dry_run, extra_args=extra_args)
    else:
        summary["training"] = {"status": "skipped"}

    metrics = read_latest_metrics()
    if metrics:
        summary["metrics_snapshot"] = metrics

    summary["run_finished_at"] = datetime.utcnow().isoformat()
    summary_path = Path(args.summary_path).expanduser()
    write_summary(summary, summary_path, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

