#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

META_PREFIX = "__"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_name_tag(model_name: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name.strip())
    tag = tag.strip("-._")
    return tag or "model"


def _task_result_file(mode_dir: Path, task: str, model_name: str) -> Path:
    model_tag = _model_name_tag(model_name)
    file_path = mode_dir / f"{task}__{model_tag}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Result file not found: {file_path}")
    return file_path


def _find_task_result_file(
    mode_dir: Path,
    task: str,
    model_name: str,
    explicit_result_file: Optional[str],
) -> Path:
    if explicit_result_file:
        p = Path(explicit_result_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--result-file does not exist: {p}")
        return p

    model_tag = _model_name_tag(model_name)
    pat = re.compile(rf"^{re.escape(task)}__{re.escape(model_tag)}(?:\.json|-.+\.json)$")
    rough_candidates = [
        p
        for p in mode_dir.glob(f"{task}__*.json")
        if pat.match(p.name) and not p.name.startswith(META_PREFIX)
    ]
    if not rough_candidates:
        raise FileNotFoundError(
            f"No result file found: mode={mode_dir.name} task={task} model={model_name} "
            f"(expected: {task}__{model_tag}.json or {task}__{model_tag}-*.json)"
        )

    candidates: List[Path] = []
    for p in rough_candidates:
        try:
            payload = _read_json(p)
            if isinstance(payload, dict) and str(payload.get("model_name", "")).strip() == model_name:
                candidates.append(p)
        except Exception:  
            continue

    if not candidates:
        rough_candidates = sorted(rough_candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        return rough_candidates[0]

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[`*_#]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?:;\"'()[]{}")
    return s


def _extract_final_text(pred: str) -> str:
    text = (pred or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else text

    m = re.search(r"(?:final\s+)?answer\s*:\s*(.+)$", candidate, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        m2 = re.search(r"(?:final\s+)?answer\s*:\s*(.+)$", text, flags=re.IGNORECASE)
        if m2:
            candidate = m2.group(1).strip()

    return candidate.strip()


def _extract_choice_letter(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"image\s*([A-D])\b",
        r"\b([A-D])\b",
        r"\(([A-D])\)",
        r"option\s*([A-D])\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _extract_yes_no(text: str) -> str | None:
    if not text:
        return None
    t = _normalize_text(text)
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return None


def _extract_ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"(?<!\d)(\d+)(?!\d)", text)]


def _parse_answer_as_set(ans: str) -> set[int] | None:
    t = _normalize_text(ans)
    if t == "none":
        return set()
    if re.fullmatch(r"\d+(\s*,\s*\d+)+", t):
        return {int(x.strip()) for x in t.split(",") if x.strip()}
    if re.fullmatch(r"\d+\s+and\s+\d+", t):
        nums = re.findall(r"\d+", t)
        return {int(nums[0]), int(nums[1])}
    return None


def _is_correct_with_eval_rule(gt_answer: str, pred_text: str) -> bool:
    gt = gt_answer.strip()
    pred = _extract_final_text(pred_text)

    # 1) A/B/C/D
    if re.fullmatch(r"[A-D]", gt, flags=re.IGNORECASE):
        letter = _extract_choice_letter(pred) or _extract_choice_letter(pred_text)
        return bool(letter and letter.upper() == gt.upper())

    # 2) Yes/No
    if gt.lower() in {"yes", "no"}:
        yn = _extract_yes_no(pred) or _extract_yes_no(pred_text)
        return bool(yn and yn == gt.lower())

    # 3) numeric set / pair / comma list / none
    gt_set = _parse_answer_as_set(gt)
    if gt_set is not None:
        pred_norm = _normalize_text(pred)
        if gt_set == set() and "none" in pred_norm:
            return True
        pred_nums = _extract_ints(pred)
        if not pred_nums:
            pred_nums = _extract_ints(pred_text)
        pred_set = set(pred_nums)
        return pred_set == gt_set

    # 4) single integer
    if re.fullmatch(r"\d+", gt):
        gt_num = int(gt)
        pred_nums = _extract_ints(pred)
        if not pred_nums:
            pred_nums = _extract_ints(pred_text)
        return bool(pred_nums and pred_nums[0] == gt_num)

    # 5) fallback normalized match
    gt_n = _normalize_text(gt)
    pred_n = _normalize_text(pred)
    if gt_n == pred_n:
        return True
    pred_full_n = _normalize_text(pred_text)
    return gt_n == pred_full_n


def _result_pred_text(item: Dict[str, Any]) -> str:
    pred = item.get("answer_pred")
    if isinstance(pred, str) and pred.strip():
        return pred
    model_output = item.get("model_output")
    if isinstance(model_output, dict):
        txt = model_output.get("text")
        if isinstance(txt, str):
            return txt
    return ""


def _calc_task_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", [])
    total = int(data.get("total", len(results) if isinstance(results, list) else 0) or 0)
    correct = int(data.get("correct", 0) or 0)
    errors = int(data.get("errors", 0) or 0)

    # If summary fields are missing or inconsistent, rebuild from sample-level flags.
    if isinstance(results, list):
        recomputed_total = len(results)
        recomputed_correct = 0
        for x in results:
            gt = str(x.get("answer_gt", ""))
            pred_text = _result_pred_text(x)
            if _is_correct_with_eval_rule(gt, pred_text):
                recomputed_correct += 1
        recomputed_errors = sum(1 for x in results if x.get("error"))
        if total != recomputed_total:
            total = recomputed_total
        if correct != recomputed_correct:
            correct = recomputed_correct
        if errors != recomputed_errors:
            errors = recomputed_errors

    acc = (correct / total) if total > 0 else 0.0
    return {
        "task": str(data.get("mode_task", "")) or "unknown_task",
        "model_name": str(data.get("model_name", "")),
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": acc,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compute single-task single-scene single-model accuracy and append to CSV")
    parser.add_argument("--mode", required=True, choices=["basic_own", "indoor", "region_basic", "region_complex"])
    parser.add_argument("--task", required=True, type=str, help="Task name, e.g. spatial_relation")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--result-file", type=str, default="", help="Optional: explicitly specify a result JSON path")
    parser.add_argument("--results-root", type=str, default=str(repo_root / "benchmark" / "results"))
    parser.add_argument(
        "--total-csv",
        type=str,
        default="",
        help="Total CSV path; if empty, auto-write to results_root/metrics_total-<model_variant>.csv",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional: additionally output current-task metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    mode_dir = results_root / args.mode

    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode result directory does not exist: {mode_dir}")

    task_file = _find_task_result_file(mode_dir, args.task, args.model_name, args.result_file)
    data = _read_json(task_file)
    if not isinstance(data, dict):
        raise ValueError(f"Result file is not a dict JSON: {task_file}")
    m = _calc_task_metrics(data)
    total = m["total"]
    correct = m["correct"]
    errors = m["errors"]
    overall_acc = m["accuracy"]

    file_stem = task_file.stem
    model_variant = file_stem.split("__", 1)[1] if "__" in file_stem else _model_name_tag(args.model_name)

    row = {
        "model_name": args.model_name,
        "model_variant": model_variant,
        "mode": args.mode,
        "task_name": args.task,
        "total": total,
        "correct": correct,
        "errors": errors,
        "acc": f"{overall_acc:.6f}",
        "result_file": str(task_file),
    }

    if args.total_csv.strip():
        total_csv = Path(args.total_csv).expanduser().resolve()
    else:
        total_csv = (results_root / f"metrics_total-{model_variant}.csv").resolve()
    total_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not total_csv.exists()
    with total_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "model_variant",
                "mode",
                "task_name",
                "total",
                "correct",
                "errors",
                "acc",
                "result_file",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    if args.output_json.strip():
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "task": args.task,
            "model_name": args.model_name,
            "model_variant": model_variant,
            "result_file": str(task_file),
            "metrics": {
                "total": total,
                "correct": correct,
                "errors": errors,
                "accuracy": overall_acc,
            },
            "appended_csv": str(total_csv),
        }
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[DONE] metrics_json: {output_json}")

    print("[DONE] metrics computed")
    print(
        f"[DONE] mode={args.mode} task={args.task} model={args.model_name} "
        f"total={total} correct={correct} errors={errors} acc={overall_acc:.6f}"
    )
    print(f"[DONE] result_file: {task_file}")
    print(f"[DONE] appended csv: {total_csv}")


if __name__ == "__main__":
    main()
