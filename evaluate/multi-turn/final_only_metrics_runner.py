#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def _find_region_dirs(workspace_root: Path) -> List[Path]:
    if not workspace_root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(workspace_root.iterdir()):
        if p.is_dir() and (p / "labels.json").is_file():
            out.append(p)
    return out


def _find_final_scene_path(region_dir: Path, step: str) -> str:
    cands: List[Path] = []
    if step:
        cands.extend(
            [
                region_dir / f"final_scene_steps{step}.blend",
                region_dir / f"final_scene_steps{step}.glb",
            ]
        )
    cands.extend([region_dir / "final_scene.blend", region_dir / "final_scene.glb"])
    for p in cands:
        if p.is_file():
            return str(p)
    return ""


def _infer_global_error_scene(region_info: Dict) -> str:
    src = (region_info or {}).get("source", {}) or {}
    direct = [
        src.get("error_scene"),
        src.get("error_blend"),
        src.get("global_error_scene"),
        src.get("global_error_blend"),
        src.get("source_scene_path"),  
    ]
    for p in direct:
        p = str(p or "").strip()
        if p and os.path.isfile(p):
            return p

    img_hints = list(src.get("error_images") or []) + list(src.get("initial_images") or [])
    for ip in img_hints:
        p = str(ip or "").replace("\\", "/")
        tags = list(re.finditer(r"([^/]+_steps\d+)_regions", p))
        if not tags:
            continue
        tag = tags[-1].group(1)
        d = os.path.dirname(p)
        visited = set()
        for _ in range(12):
            if (not d) or (d in visited):
                break
            visited.add(d)
            cands = [
                os.path.join(d, f"{tag}_global_error_scene.blend"),
                os.path.join(d, f"{tag}_global_error_scene.glb"),
            ]
            for c in cands:
                if os.path.isfile(c):
                    return c
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    return ""


def _safe_slug(x: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(x or "").strip())
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unknown"


def _normalize_action_list(v) -> List[str]:
    if isinstance(v, str):
        t = v.strip()
        return [t] if t else []
    if isinstance(v, list):
        out = []
        for x in v:
            t = str(x or "").strip()
            if t:
                out.append(t)
        return out
    return []


def _build_synthetic_actions(region_dirs: List[Path]) -> Tuple[Dict[str, str], List[Dict], List[str]]:
    object_actions: Dict[str, List[str]] = {}
    object_first_region: Dict[str, str] = {}
    warnings: List[str] = []

    for rd in region_dirs:
        labels_path = rd / "labels.json"
        actions_path = rd / "final_actions.json"
        fallback_actions_path = rd / "final_render_actions.json"
        if not labels_path.is_file():
            warnings.append(f"{rd.name}: missing labels.json")
            continue
        if not actions_path.is_file() and not fallback_actions_path.is_file():
            warnings.append(f"{rd.name}: missing final_actions/final_render_actions")
            continue

        labels = json.load(labels_path.open("r", encoding="utf-8")) or {}
        if actions_path.is_file():
            payload = json.load(actions_path.open("r", encoding="utf-8")) or {}
            consolidated = list(payload.get("consolidated_actions") or [])
        else:
            payload = json.load(fallback_actions_path.open("r", encoding="utf-8")) or {}
            consolidated = list(payload.get("consolidated_actions") or [])

        for item in consolidated:
            local_id = str(item.get("id", "")).strip()
            if not local_id:
                continue
            target_obj_id = str(labels.get(local_id, "")).strip()
            if not target_obj_id:
                warnings.append(f"{rd.name}: local label {local_id} has no object mapping")
                continue

            actions = _normalize_action_list(item.get("action", []))
            if not actions:
                continue

            if target_obj_id not in object_actions:
                object_actions[target_obj_id] = []
                object_first_region[target_obj_id] = rd.name
            else:
                if object_first_region.get(target_obj_id) != rd.name:
                    warnings.append(
                        f"duplicate object in regions: {target_obj_id} first={object_first_region.get(target_obj_id)} now={rd.name}"
                    )
            object_actions[target_obj_id].extend(actions)

    if not object_actions:
        return {}, [], warnings

    sorted_obj_ids = sorted(object_actions.keys())
    synthetic_labels: Dict[str, str] = {}
    synthetic_actions: List[Dict] = []
    for idx, obj_id in enumerate(sorted_obj_ids, start=1):
        sid = str(idx)
        synthetic_labels[sid] = obj_id
        synthetic_actions.append({"id": idx, "action": object_actions[obj_id]})
    return synthetic_labels, synthetic_actions, warnings


def _render_shared_after_once(
    group_dir: Path,
    source_scene_path: str,
    blender_bin: str,
    blender_script: str,
    project_root: str,
    unit_scale: float,
    region_mode: str,
    gltf_fallback: str,
    step: str,
) -> Tuple[str, str]:
    labels_path = group_dir / "labels.json"
    actions_path = group_dir / "final_render_actions.json"
    if not labels_path.is_file() or not actions_path.is_file():
        return "", "missing_synthetic_labels_or_actions"
    if not source_scene_path or (not os.path.isfile(source_scene_path)):
        return "", "global_error_scene_missing"

    out_dir = group_dir
    out_blend = group_dir / (f"final_scene_steps{step}.blend" if step else "final_scene.blend")
    env = os.environ.copy()
    env.update(
        {
            "BLENDER_REGION_DIR": str(group_dir),
            "BLENDER_OUTPUT_DIR": str(out_dir),
            "BLENDER_CUMULATIVE_PATH": str(actions_path),
            "BLENDER_INPUT_BLEND": source_scene_path if source_scene_path.lower().endswith(".blend") else "",
            "GLTF_PATH": source_scene_path if (not source_scene_path.lower().endswith(".blend")) else str(gltf_fallback),
            "BLENDER_OUTPUT_TOP": "top_final_shared_metrics_tmp.png",
            "BLENDER_OUTPUT_ISO": "isometric_final_shared_metrics_tmp.png",
            "BLENDER_OUTPUT_BLEND": str(out_blend),
            "BLENDER_UNIT_SCALE": str(float(unit_scale)),
            "BLENDER_SCENE_TYPE": "region",
            "BLENDER_REGION_MODE": region_mode,
            "BLENDER_ACTION_LOG_PATH": str(group_dir / "final_applied_actions_log_shared_metrics.json"),
        }
    )
    cmd = [blender_bin, "--background", "--python", blender_script]
    res = subprocess.run(cmd, env=env, cwd=project_root, capture_output=True, text=True)
    if res.returncode != 0 or (not out_blend.is_file()):
        msg = (res.stderr or res.stdout or "")[-1200:]
        return "", f"render_failed: {msg}"
    return str(out_blend), ""


def _run_metrics_single(
    metrics_script: str,
    step: str,
    blend_path: str,
    labels_json: str,
) -> Dict:
    with tempfile.TemporaryDirectory(prefix="metrics_after_only_") as td:
        out_path = Path(td) / "metrics_one.json"
        cmd = [
            "python",
            metrics_script,
            "--mode",
            "region",
            "--blend-path",
            blend_path,
            "--labels-json",
            labels_json,
            "--steps",
            str(step),
            "--output",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"metrics_failed({proc.returncode}): {(proc.stderr or '')[-1200:]}")
        payload = json.load(out_path.open("r", encoding="utf-8"))
        return payload


def _run_metrics_single_indoor(
    metrics_script: str,
    step: str,
    blend_path: str,
    scene_dir: str,
) -> Dict:
    with tempfile.TemporaryDirectory(prefix="metrics_after_only_indoor_") as td:
        out_path = Path(td) / "metrics_one.json"
        cmd = [
            "python",
            metrics_script,
            "--mode",
            "indoor",
            "--blend-path",
            blend_path,
            "--scene-dir",
            scene_dir,
            "--steps",
            str(step),
            "--output",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"metrics_failed({proc.returncode}): {(proc.stderr or '')[-1200:]}")
        payload = json.load(out_path.open("r", encoding="utf-8"))
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-only metrics runner with optional shared global replay.")
    parser.add_argument("--scene-type", choices=["region", "indoor"], default="region")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--step", default="")
    parser.add_argument("--metrics-script", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--use-global-after", action="store_true")
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--blender-script", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--unit-scale", type=float, default=20.0)
    parser.add_argument("--region-mode", default="complex")
    parser.add_argument("--metrics-workers", type=int, default=0)
    parser.add_argument("--gltf-fallback", default="")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    step = str(args.step or "").strip()
    metrics_script = str(args.metrics_script)

    scene_type = str(args.scene_type or "region").strip().lower()
    region_dirs = _find_region_dirs(workspace_root)
    region_entries: List[Dict] = []
    missing_final_scene_regions: List[str] = []

    for rd in region_dirs:
        final_path = _find_final_scene_path(rd, step)
        if not final_path:
            missing_final_scene_regions.append(rd.name)
            continue
        labels_path = str(rd / "labels.json")
        entry = {
            "region_dir": str(rd),
            "region_name": rd.name,
            "final_path": final_path,
            "labels_json": labels_path,
            "eval_blend_path": final_path,
            "eval_context": "local_final_scene",
            "eval_note": "",
            "group_key": "",
        }
        if scene_type == "region" and args.use_global_after:
            rp = rd / "region_info.json"
            region_info = {}
            if rp.is_file():
                try:
                    region_info = json.load(rp.open("r", encoding="utf-8"))
                except Exception:
                    region_info = {}
            global_src = _infer_global_error_scene(region_info)
            if global_src:
                entry["group_key"] = os.path.abspath(global_src)
            else:
                entry["eval_note"] = "global_error_scene_not_found"
        region_entries.append(entry)

    shared_groups_summary: List[Dict] = []
    if scene_type == "region" and args.use_global_after:
        groups: Dict[str, List[Dict]] = {}
        for e in region_entries:
            gk = str(e.get("group_key") or "")
            if gk:
                groups.setdefault(gk, []).append(e)

        shared_root = workspace_root / "_shared_final_scenes"
        shared_root.mkdir(parents=True, exist_ok=True)

        for idx, (source_scene, group_entries) in enumerate(sorted(groups.items()), start=1):
            region_paths = [Path(x["region_dir"]) for x in group_entries]
            synthetic_labels, synthetic_actions, merge_warnings = _build_synthetic_actions(region_paths)
            group_slug = _safe_slug(Path(source_scene).stem)
            group_dir = shared_root / f"shared_scene_{group_slug}_{idx}"
            group_dir.mkdir(parents=True, exist_ok=True)

            group_info = {
                "group_index": idx,
                "source_scene_path": source_scene,
                "region_count": len(group_entries),
                "region_names": [x["region_name"] for x in group_entries],
                "shared_blend_path": "",
                "status": "pending",
                "warnings": list(merge_warnings),
            }

            if not synthetic_labels or not synthetic_actions:
                group_info["status"] = "skipped_empty_actions"
                for e in group_entries:
                    if not e["eval_note"]:
                        e["eval_note"] = "shared_group_empty_actions_fallback_local"
                shared_groups_summary.append(group_info)
                continue

            with (group_dir / "labels.json").open("w", encoding="utf-8") as f:
                json.dump(synthetic_labels, f, ensure_ascii=False, indent=2)
            with (group_dir / "final_render_actions.json").open("w", encoding="utf-8") as f:
                json.dump({"consolidated_actions": synthetic_actions}, f, ensure_ascii=False, indent=2)

            shared_blend, render_note = _render_shared_after_once(
                group_dir=group_dir,
                source_scene_path=source_scene,
                blender_bin=args.blender_bin,
                blender_script=args.blender_script,
                project_root=args.project_root,
                unit_scale=float(args.unit_scale),
                region_mode=str(args.region_mode),
                gltf_fallback=str(args.gltf_fallback),
                step=step,
            )
            if shared_blend:
                group_info["status"] = "ok"
                group_info["shared_blend_path"] = shared_blend
                for e in group_entries:
                    e["eval_blend_path"] = shared_blend
                    e["eval_context"] = "shared_global_error_scene_replay_once"
            else:
                group_info["status"] = "render_failed_fallback_local"
                group_info["warnings"].append(render_note)
                for e in group_entries:
                    if not e["eval_note"]:
                        e["eval_note"] = render_note
            shared_groups_summary.append(group_info)

    # Run metrics per region (can be parallel)
    eval_tasks = list(region_entries)
    workers = int(args.metrics_workers or 0)
    if workers <= 0:
        workers = min(8, max(1, len(eval_tasks)))
    else:
        workers = max(1, min(workers, max(1, len(eval_tasks))))

    results: List[Dict] = []
    if scene_type == "indoor":
        count_keys = ("Overlap", "WallConflict", "Orientation")
    else:
        count_keys = ("Overlap", "RoadConflict", "Orientation")
    sum_counts = {k: 0 for k in count_keys}
    failures: List[Dict] = []

    def _worker(e: Dict) -> Dict:
        if scene_type == "indoor":
            payload = _run_metrics_single_indoor(
                metrics_script=metrics_script,
                step=step,
                blend_path=str(e["eval_blend_path"]),
                scene_dir=str(e["region_dir"]),
            )
        else:
            payload = _run_metrics_single(
                metrics_script=metrics_script,
                step=step,
                blend_path=str(e["eval_blend_path"]),
                labels_json=str(e["labels_json"]),
            )
        return {"entry": e, "payload": payload}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, e) for e in eval_tasks]
        for fut in concurrent.futures.as_completed(futs):
            try:
                row = fut.result()
            except Exception as exc:
                failures.append({"error": str(exc)})
                continue
            e = row["entry"]
            one = row["payload"]
            if scene_type == "indoor":
                scenes = list(one.get("scenes") or [])
                if not scenes:
                    raise RuntimeError(f"indoor output missing scenes: {e['region_dir']}")
                single = ((scenes[0] or {}).get("after") or {})
            else:
                single = one.get("single_blend", {}) or {}
            counts = single.get("error_type_counts", {}) or {}
            details = single.get("details", {}) or {}
            involved = {}
            for t in count_keys:
                td = details.get(t, {}) or {}
                ids = td.get("involved_building_label_ids", []) or []
                involved[t] = [int(x) for x in ids if str(x).isdigit()]
                sum_counts[t] += int(counts.get(t, 0) or 0)

            region_dir = Path(e["region_dir"])
            region_out = region_dir / (f"metrics_after_only_steps{step}.json" if step else "metrics_after_only.json")
            region_compact = {
                "mode": "indoor_after_only_single" if scene_type == "indoor" else "region_after_only_single",
                "steps": step,
                "region_dir": str(region_dir),
                "blend_path": str(e["final_path"]),
                "eval_blend_path": str(e["eval_blend_path"]),
                "eval_context": str(e["eval_context"]),
                "eval_note": str(e.get("eval_note", "") or ""),
                "labels_json": str(e["labels_json"]),
                "after": {
                    "error_type_counts": {k: int(counts.get(k, 0) or 0) for k in count_keys},
                    "total_errors": int(single.get("total_errors", 0) or 0),
                    "object_stats": single.get("object_stats", {}),
                    "involved_building_label_ids_by_type": involved,
                    "details": details,
                },
            }
            with region_out.open("w", encoding="utf-8") as f:
                json.dump(region_compact, f, ensure_ascii=False, indent=2)

            results.append(
                {
                    "region_dir": str(region_dir),
                    "blend_path": str(e["final_path"]),
                    "eval_blend_path": str(e["eval_blend_path"]),
                    "eval_context": str(e["eval_context"]),
                    "eval_note": str(e.get("eval_note", "") or ""),
                    "region_metrics_path": str(region_out),
                    "error_type_counts": {k: int(counts.get(k, 0) or 0) for k in count_keys},
                    "total_errors": int(single.get("total_errors", 0) or 0),
                }
            )

    results = sorted(results, key=lambda x: x["region_dir"])

    out = {
        "workspace_root": str(workspace_root),
        "steps": step,
        "mode": "indoor_final_only" if scene_type == "indoor" else "region_final_only",
        "summary": {
            "regions_total": len(region_dirs),
            "regions_with_final_scene": len(eval_tasks),
            "regions_evaluated": len(results),
            "regions_missing_final_scene": len(missing_final_scene_regions),
            "regions_metrics_failed": len(failures),
            "metrics_after_use_global_error_scene": bool(args.use_global_after) if scene_type == "region" else False,
            "shared_replay_groups_total": len(shared_groups_summary),
            "shared_replay_groups_ok": sum(1 for g in shared_groups_summary if g.get("status") == "ok"),
            "after_error_type_counts": sum_counts,
            "after_total_errors": int(sum(sum_counts.values())),
            "rule": "Final-only metrics: only detect errors on final_scene blend, no before comparison.",
        },
        "missing_final_scene_regions": missing_final_scene_regions,
        "metrics_failures": failures,
        "shared_replay_groups": shared_groups_summary,
        "regions": results,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] final-only metrics written: {out_path}")


if __name__ == "__main__":
    main()
