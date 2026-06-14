#!/usr/bin/env python3
"""
Indoor multi-step error QA constructor.
"""

import argparse
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "SpatialAct"))
BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", str(PROJECT_ROOT / "blender-3.2.2-linux-x64/blender")))

try:
    import bpy
    from mathutils import Vector
except ModuleNotFoundError:
    script_path = str(Path(__file__).resolve())
    cmd = [str(BLENDER_BIN), "--background", "--python", script_path, "--"] + sys.argv[1:]
    res = subprocess.run(cmd, env=os.environ.copy(), check=False)
    raise SystemExit(res.returncode)


BASE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "single_step" / "indoor_error_mode_construct.py"
_base_spec = importlib.util.spec_from_file_location("indoor_error_mode_base", BASE_SCRIPT_PATH)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT_PATH}")
base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(base)

INTERNSCENES_ROOT = Path(os.environ.get("INTERNSCENES_ROOT", "InternScenes"))
DEFAULT_GLB_DIR = INTERNSCENES_ROOT / "scenes/glb_files_wall_complex-10-15_clean_keep"
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_LAYOUT_ROOT = base.DEFAULT_LAYOUT_ROOT
DEFAULT_WHITELIST_JSON = base.DEFAULT_WHITELIST_JSON
DEFAULT_OUTPUT_BASE_DIR = PROJECT_ROOT / "benchmark/data/multi_step_error/indoor_scenes_complex-10-15_regions"


MULTI_MAX_LABELED_OBJECTS = 8

ONE_STEP_RETRY_PER_CASE = 8
FIXED_ISOMETRIC_MODE = getattr(base, "FIXED_ISOMETRIC_MODE", "isometric_north_ur")
OUTPUT_SUFFIX_BAOXIAN = ""


def _sync_overlay_style_with_base():
    """
    Keep 1m / N / arrow and label marker style exactly aligned with
    indoor_error_mode_construct.py.
    """
    base.LABEL_RADIUS = getattr(base, "LABEL_RADIUS", 13)
    base.LABEL_FONT_SIZE = getattr(base, "LABEL_FONT_SIZE", 17)
    base.BASE_LONG_EDGE = getattr(base, "BASE_LONG_EDGE", 1280)
    base.MIN_SHORT_EDGE = getattr(base, "MIN_SHORT_EDGE", 720)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Construct indoor multi-step error QA")
    parser.add_argument("--glb-dir", type=Path, default=DEFAULT_GLB_DIR)
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--whitelist-json", type=Path, default=DEFAULT_WHITELIST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means all")
    parser.add_argument("--region", type=str, default="", help="keyword filter")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--worker-index", type=int, default=-1, help="0-based worker index; -1 means parent mode")
    parser.add_argument("--steps", type=int, default=3, help="Number of error steps to inject per scene")
    
    parser.add_argument(
        "--wall-conflict-depth-ratio-threshold",
        type=float,
        default=float(base.WALL_CONFLICT_DEPTH_RATIO_THRESHOLD),
        help="Min penetration ratio (object-vs-wall thickness) to count as wall_conflict",
    )
    parser.add_argument(
        "--wall-conflict-visible-min-overflow-m",
        type=float,
        default=float(base.WALL_CONFLICT_VISIBLE_MIN_OVERFLOW_M),
        help="Min out-of-room overflow distance (meters) for visibly strong wall_conflict",
    )
    parser.add_argument(
        "--wall-conflict-visible-min-overflow-ratio",
        type=float,
        default=float(base.WALL_CONFLICT_VISIBLE_MIN_OVERFLOW_RATIO),
        help="Min out-of-room overflow ratio for visibly strong wall_conflict",
    )
    parser.add_argument(
        "--wall-conflict-visible-min-penetration-ratio",
        type=float,
        default=float(base.WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO),
        help="Min penetration ratio for visibly strong wall_conflict",
    )
    parser.add_argument(
        "--wall-conflict-inside-ratio-default",
        type=float,
        default=float(base.REGION_BOUNDARY_RATIO_DEFAULT),
        help="Inside-region ratio threshold for normal categories (lower -> less sensitive out_of_region)",
    )
    parser.add_argument(
        "--wall-conflict-inside-ratio-relaxed",
        type=float,
        default=float(base.REGION_BOUNDARY_RATIO_RELAXED),
        help="Inside-region ratio threshold for relaxed categories (lower -> less sensitive out_of_region)",
    )
    parser.add_argument(
        "--multi-step-attempt-multiplier",
        type=int,
        default=12,
        help="Per-step sampling budget multiplier for multi-step construction (default 12). Equivalent to old max_steps*12 cap.",
    )
    parser.add_argument(
        "--multi-step-variants-per-scene",
        type=int,
        default=1,
        help="How many independent multi-step QA variants to build per scene.",
    )
    parser.add_argument("--save-scene-glb", dest="save_scene_glb", action="store_true", default=True, help="Export final error-state GLB per scene (enabled by default)")
    parser.add_argument("--no-save-scene-glb", dest="save_scene_glb", action="store_false", help="Disable per-scene final error-state GLB export")
    return parser.parse_args(argv)


def _output_dir_with_steps(output_dir: Path, steps: int) -> Path:
    """
    Put steps tag in output dir name as:
    .../indoor_scenes_simple-5-10_{steps}_regions
    """
    p = output_dir.resolve()
    name = p.name
    steps_i = int(steps)

    def _with_baoxian(name_text: str) -> str:
        s = str(name_text)
        if s.endswith(OUTPUT_SUFFIX_BAOXIAN):
            return s
        return f"{s}{OUTPUT_SUFFIX_BAOXIAN}"

    already_canonical = re.match(
        r"^(indoor_scenes_(?:simple-5-10|complex-10-15))_(\d+)_regions(?:-baoxian)?$",
        name,
    )
    if already_canonical:
        existing_steps = int(already_canonical.group(2))
        if existing_steps == steps_i:
            return p.with_name(_with_baoxian(name))

    
    m = re.match(r"^(indoor_scenes_(?:simple-5-10|complex-10-15))(?:_\d+)?(?:_regions)?$", name)
    if m:
        return p.with_name(_with_baoxian(f"{m.group(1)}_{steps_i}_regions"))

    if name.endswith("_regions"):
        base_name = name[: -len("_regions")]
        base_name = re.sub(r"_\d+$", "", base_name)
        return p.with_name(_with_baoxian(f"{base_name}_{steps_i}_regions"))

  
    return p.with_name(_with_baoxian(f"{name}_{steps_i}_regions"))


def _write_json_atomic(path: Path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _snapshot_scene_state(id_to_entry: dict[str, dict]) -> dict[str, object]:
    snaps = {}
    for e in id_to_entry.values():
        for obj in e.get("group", []):
            if obj and obj.name not in snaps:
                snaps[obj.name] = obj.matrix_world.copy()
    return snaps


def _restore_scene_state(state: dict[str, object]):
    for name, mat in state.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.matrix_world = mat
    bpy.context.view_layer.update()


def _collect_forced_labels_from_steps(steps: list[dict]) -> list[int]:
    forced = set()
    for st in steps:
        try:
            forced.add(int(st["inject_action"]["id"]))
        except Exception:
            pass
        issue0 = (st.get("issue_meta", {}).get("issues") or [{}])[0]
        for x in issue0.get("object_labels", []):
            sx = str(x).strip()
            if sx.isdigit():
                forced.add(int(sx))
    return sorted(forced)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _label_category(label: int, label_to_oid: dict[int, str], id_to_entry: dict[str, dict]) -> str:
    oid = label_to_oid.get(int(label))
    if oid is None:
        return ""
    return str(id_to_entry.get(oid, {}).get("category", "")).lower()


def _is_orientation_target_candidate(
    label: int,
    meta: dict,
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    wall_bounds: dict | None,
) -> bool:
    obb = meta.get("obb", {}) if isinstance(meta, dict) else {}
    hx = _safe_float(obb.get("hx", meta.get("half_w", 0.0)))
    hy = _safe_float(obb.get("hy", meta.get("half_d", 0.0)))
    if min(hx, hy) < 1e-4:
        return False
    cat = _label_category(label, label_to_oid, id_to_entry)
    if base._round_like_category(cat):
        return False
    oid = label_to_oid.get(int(label))
    group = id_to_entry.get(oid, {}).get("group", []) if oid is not None else []
    if not group:
        return False
    if not base._is_orientation_candidate_shape_group(group, cat):
        return False
    ratio = max(hx, hy) / max(min(hx, hy), 1e-6)
    if ratio > _safe_float(base.ORIENTATION_MAX_ASPECT_RATIO):
        return False
    hz = _safe_float(obb.get("hz", meta.get("height", hx)))
    volume = hx * hy * hz
    if volume < _safe_float(base.ORIENTATION_MIN_VOLUME_THRESHOLD):
        return False
    if wall_bounds is not None:
        _side, nearest_wall_dist = _nearest_wall_side_from_meta(meta, wall_bounds)
        if nearest_wall_dist > _safe_float(base.ORIENTATION_WALL_NEAR_DIST_M):
            return False
    return True


def _wall_side_distances_from_meta(meta: dict, wall_bounds: dict | None) -> dict[str, float] | None:
    if wall_bounds is None:
        return None
    obb = meta.get("obb", {}) if isinstance(meta, dict) else {}
    center = obb.get("center", (None, None))
    try:
        cx = _safe_float(center[0])
        cy = _safe_float(center[1])
    except Exception:
        return None
    return {
        "W": _safe_float(cx - wall_bounds.get("min_x")),
        "E": _safe_float(wall_bounds.get("max_x") - cx),
        "S": _safe_float(cy - wall_bounds.get("min_y")),
        "N": _safe_float(wall_bounds.get("max_y") - cy),
    }


def _nearest_wall_side_from_meta(meta: dict, wall_bounds: dict | None) -> tuple[str, float]:
    ds = _wall_side_distances_from_meta(meta, wall_bounds)
    if not ds:
        return "W", 1e9
    side = min(ds.keys(), key=lambda k: ds[k])
    return side, float(ds[side])


def _normalize_axis_vec(v) -> tuple[float, float]:
    try:
        x = float(v[0])
        y = float(v[1])
    except Exception:
        return (1.0, 0.0)
    n = math.hypot(x, y)
    if n <= 1e-9:
        return (1.0, 0.0)
    return (x / n, y / n)


def _angle_between_axes_deg(a, b) -> float:
    ax, ay = _normalize_axis_vec(a)
    bx, by = _normalize_axis_vec(b)
    dot = max(-1.0, min(1.0, abs(ax * bx + ay * by)))
    return math.degrees(math.acos(dot))


def _axis_error_to_wall_side_deg(obb: dict, wall_side: str) -> tuple[float, str]:
    ax = _normalize_axis_vec(obb.get("ax", (1.0, 0.0)))
    ay = _normalize_axis_vec(obb.get("ay", (0.0, 1.0)))
    target = (0.0, 1.0) if wall_side in ("W", "E") else (1.0, 0.0)
    err_ax = _angle_between_axes_deg(ax, target)
    err_ay = _angle_between_axes_deg(ay, target)
    if err_ax <= err_ay:
        return err_ax, "ax"
    return err_ay, "ay"


def _orientation_nearest_side_labels(
    candidate_labels: list[int],
    metas_by_label: dict[int, dict],
    wall_bounds: dict | None,
) -> set[int]:
    if wall_bounds is None:
        return set()
    labels: list[int] = []
    for lb in candidate_labels:
        lb_i = int(lb)
        m = metas_by_label.get(lb_i)
        if m is None:
            continue
        ds = _wall_side_distances_from_meta(m, wall_bounds)
        if not ds:
            continue
        labels.append(lb_i)
    if not labels:
        return set()

    return set(labels)


def _scene_nonwhitelist_collision_pairs(
    labels: list[int],
    metas_by_label: dict[int, dict],
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    whitelist_pairs: set[tuple[str, str]],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    bvh_cache: dict[int, object] = {}

    def _get_bvh(lb: int):
        lb = int(lb)
        if lb in bvh_cache:
            return bvh_cache[lb]
        oid = label_to_oid.get(lb)
        if oid is None:
            bvh_cache[lb] = None
            return None
        group = id_to_entry.get(oid, {}).get("group")
        if not group:
            bvh_cache[lb] = None
            return None
        bvh_cache[lb] = base._build_group_bvh(group)
        return bvh_cache[lb]

    for i in range(len(labels)):
        a = int(labels[i])
        ma = metas_by_label.get(a)
        if ma is None:
            continue
        for j in range(i + 1, len(labels)):
            b = int(labels[j])
            mb = metas_by_label.get(b)
            if mb is None:
                continue
            ca = _label_category(a, label_to_oid, id_to_entry)
            cb = _label_category(b, label_to_oid, id_to_entry)
            if base._is_pair_whitelisted(ca, cb, whitelist_pairs):
                continue
            bvh_a = _get_bvh(a)
            bvh_b = _get_bvh(b)
            if bvh_a is None or bvh_b is None:
                continue
            hits = base._bvh_overlap_hit_count(bvh_a, bvh_b)
            if hits < int(base.BVH_OVERLAP_MIN_HITS):
                continue
            pen_depth = base._obb_penetration_depth_xy(ma, mb)
            if pen_depth < _safe_float(base.OVERLAP_MIN_PENETRATION_DEPTH_M):
                continue
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def _label_has_wall_conflict(
    label: int,
    metas_by_label: dict[int, dict],
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    wall_bounds: dict | None,
    wall_components: list | None,
) -> bool:
    lb = int(label)
    m = metas_by_label.get(lb)
    if m is None:
        return False
    oid = label_to_oid.get(lb)
    cat = str(id_to_entry.get(oid, {}).get("category", "")).lower() if oid is not None else ""
    if any(k in cat for k in base.WALL_CONFLICT_EXCLUDE_TARGET_KEYWORDS):
        return False

    out_of_region = False
    if wall_bounds is not None:
        center = m.get("obb", {}).get("center", (0.0, 0.0))
        try:
            cx = _safe_float(center[0])
            cy = _safe_float(center[1])
        except Exception:
            return False
        inside_ok = base._inside_wall_region_by_category_bbox(
            cx,
            cy,
            _safe_float(m.get("half_w")),
            _safe_float(m.get("half_d")),
            cat,
            wall_bounds,
            ratio_default=base.REGION_BOUNDARY_RATIO_DEFAULT,
            ratio_relaxed=base.REGION_BOUNDARY_RATIO_RELAXED,
        )
        out_of_region = (not inside_ok)

    penetrate_wall = False
    if base.SHAPELY_AVAILABLE and wall_components:
        group_objs = id_to_entry.get(oid, {}).get("group") if oid is not None else None
        near = base._nearest_wall_overlap_ratio_for_object(m, wall_components, group_objs=group_objs)
        penetrate_wall = _safe_float(near.get("ratio", 0.0)) > _safe_float(base.WALL_CONFLICT_DEPTH_RATIO_THRESHOLD)
    return bool(out_of_region or penetrate_wall)


def _scene_wall_conflict_labels(
    labels: list[int],
    metas_by_label: dict[int, dict],
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    wall_bounds: dict | None,
    wall_components: list | None,
) -> set[int]:
    out: set[int] = set()
    for lb in labels:
        lb_i = int(lb)
        if _label_has_wall_conflict(lb_i, metas_by_label, label_to_oid, id_to_entry, wall_bounds, wall_components):
            out.add(lb_i)
    return out


def _label_has_orientation_conflict(
    label: int,
    cur_metas_by_label: dict[int, dict],
    ref_metas_by_label: dict[int, dict],
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    wall_bounds: dict | None,
    eligible_labels: set[int] | None = None,
) -> bool:
    lb = int(label)
    cur_m = cur_metas_by_label.get(lb)
    if cur_m is None:
        return False
    if not _is_orientation_target_candidate(lb, cur_m, label_to_oid, id_to_entry, wall_bounds):
        return False
    if eligible_labels is None:
        all_labels = [int(x) for x in cur_metas_by_label.keys()]
        cand = [
            x
            for x in all_labels
            if (
                cur_metas_by_label.get(x) is not None
                and _is_orientation_target_candidate(x, cur_metas_by_label[x], label_to_oid, id_to_entry, wall_bounds)
            )
        ]
        eligible_labels = _orientation_nearest_side_labels(cand, cur_metas_by_label, wall_bounds)
    if lb not in eligible_labels:
        return False
    nearest_side, _near_dist = _nearest_wall_side_from_meta(cur_m, wall_bounds)
    d, _edge_axis = _axis_error_to_wall_side_deg(cur_m.get("obb") or {}, nearest_side)
    return bool(_safe_float(d) >= _safe_float(base.ORIENTATION_MIN_ANGLE_DELTA_DEG))


def _scene_orientation_conflict_labels(
    labels: list[int],
    cur_metas_by_label: dict[int, dict],
    ref_metas_by_label: dict[int, dict],
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    wall_bounds: dict | None,
) -> set[int]:
    candidate_labels: list[int] = []
    for lb in labels:
        lb_i = int(lb)
        if cur_metas_by_label.get(lb_i) is None:
            continue
        if _is_orientation_target_candidate(lb_i, cur_metas_by_label[lb_i], label_to_oid, id_to_entry, wall_bounds):
            candidate_labels.append(lb_i)
    eligible = _orientation_nearest_side_labels(candidate_labels, cur_metas_by_label, wall_bounds)
    out: set[int] = set()
    for lb in labels:
        lb_i = int(lb)
        if _label_has_orientation_conflict(
            lb_i,
            cur_metas_by_label,
            ref_metas_by_label,
            label_to_oid,
            id_to_entry,
            wall_bounds,
            eligible_labels=eligible,
        ):
            out.add(lb_i)
    return out


def _label_has_nonwhitelist_collision(
    label: int,
    overlap_pairs: set[tuple[int, int]],
) -> bool:
    lb = int(label)
    return any(lb in p for p in overlap_pairs)


def _one_step_exclusive_issue_ok(
    *,
    labels: list[int],
    after_metas: dict[int, dict],
    ref_metas: dict[int, dict],
    focus_label: int,
    main_issue_type: str,
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    whitelist_pairs: set[tuple[str, str]],
    wall_bounds: dict | None,
    wall_components: list | None,
    baseline_overlap_pairs: set[tuple[int, int]],
    baseline_wall_labels: set[int],
    baseline_orientation_labels: set[int],
) -> bool:
    overlap_pairs = _scene_nonwhitelist_collision_pairs(labels, after_metas, label_to_oid, id_to_entry, whitelist_pairs)
    wall_labels = _scene_wall_conflict_labels(labels, after_metas, label_to_oid, id_to_entry, wall_bounds, wall_components)
    orientation_labels = _scene_orientation_conflict_labels(labels, after_metas, ref_metas, label_to_oid, id_to_entry, wall_bounds)

    new_overlap_pairs = overlap_pairs - baseline_overlap_pairs
    new_wall_labels = wall_labels - baseline_wall_labels
    new_orientation_labels = orientation_labels - baseline_orientation_labels

    baseline_overlap_labels: set[int] = set()
    for pa, pb in baseline_overlap_pairs:
        baseline_overlap_labels.add(int(pa))
        baseline_overlap_labels.add(int(pb))

    lb = int(focus_label)

    if main_issue_type == base.ISSUE_ANGLE:
        if lb in baseline_orientation_labels:
            return False
        if not _label_has_orientation_conflict(lb, after_metas, ref_metas, label_to_oid, id_to_entry, wall_bounds):
            return False
        if _label_has_nonwhitelist_collision(lb, overlap_pairs):
            return False
        if lb in wall_labels:
            return False
        new_other_orientation = {x for x in new_orientation_labels if int(x) != lb}
        return (len(new_overlap_pairs) == 0) and (len(new_wall_labels) == 0) and (len(new_other_orientation) == 0)

    if main_issue_type == base.ISSUE_OVERLAP:
        if lb in baseline_overlap_labels:
            return False
        if not _label_has_nonwhitelist_collision(lb, overlap_pairs):
            return False
        if lb in wall_labels:
            return False
        if _label_has_orientation_conflict(lb, after_metas, ref_metas, label_to_oid, id_to_entry, wall_bounds):
            return False
        new_overlap_on_lb = {p for p in new_overlap_pairs if lb in p}
        if len(new_overlap_on_lb) == 0:
            return False
        if any(lb not in p for p in new_overlap_pairs):
            return False
        return (len(new_wall_labels) == 0) and (len(new_orientation_labels) == 0)

    if main_issue_type == base.ISSUE_WALL:
        if lb in baseline_wall_labels:
            return False
        if lb not in wall_labels:
            return False
        if _label_has_nonwhitelist_collision(lb, overlap_pairs):
            return False
        if _label_has_orientation_conflict(lb, after_metas, ref_metas, label_to_oid, id_to_entry, wall_bounds):
            return False
        new_other_wall = {x for x in new_wall_labels if int(x) != lb}
        return (len(new_overlap_pairs) == 0) and (len(new_orientation_labels) == 0) and (len(new_other_wall) == 0)

    return False


def _create_multi_step_anomalies(
    glb_name: str,
    labels: list[int],
    operable_labels: set[int],
    label_to_oid: dict[int, str],
    id_to_entry: dict[str, dict],
    whitelist_pairs: set[tuple[str, str]],
    wall_bounds: dict | None,
    wall_components: list | None,
    cam_top,
    render_ctx: dict,
    pivot_by_label: dict[int, Vector],
    max_steps: int,
    attempt_multiplier: int = 12,
) -> list[dict]:
    steps: list[dict] = []
    last_issue_main: str | None = None
    one_step_mode = int(max_steps) == 1

    case_ids = list(range(len(base.CASE_CYCLE)))
    attempt_multiplier = max(1, int(attempt_multiplier))
    max_attempts = max(max_steps * attempt_multiplier, 24)
    attempts = 0

    def _current_pivots() -> dict[int, Vector]:
        piv: dict[int, Vector] = {}
        for lb in labels:
            oid = label_to_oid.get(int(lb))
            if oid is None:
                continue
            group = id_to_entry.get(oid, {}).get("group", [])
            if not group:
                continue
            piv[int(lb)] = base._group_centroid_world(group)
        return piv

    def _extract_issue_labels(issue_meta: dict) -> set[int]:
        issue0 = (issue_meta.get("issues") or [{}])[0]
        issue_labels: set[int] = set()
        for x in issue0.get("object_labels", []):
            sx = str(x).strip()
            if sx.isdigit():
                issue_labels.add(int(sx))
        return issue_labels

    def _step_issue_still_present(
        step_rec: dict,
        after_overlap_pairs: list[tuple[int, int]],
        after_wall_labels: set[int],
        after_orientation_labels: set[int],
    ) -> bool:
        s_main = base._issue_main_type((step_rec.get("issue_meta") or {}).get("issues", [{}])[0])
        s_labels = set(step_rec.get("issue_labels", set()))
        if not s_labels:
            return True
        if s_main == base.ISSUE_OVERLAP:
            return any(s_labels.issubset({int(a), int(b)}) for a, b in after_overlap_pairs)
        if s_main == base.ISSUE_WALL:
            return len(s_labels.intersection(after_wall_labels)) > 0
        if s_main == base.ISSUE_ANGLE:
            return len(s_labels.intersection(after_orientation_labels)) > 0
        return True

    while len(steps) < max_steps and attempts < max_attempts:
        attempts += 1

        pool = case_ids[:]
        random.shuffle(pool)
        selected = None
        for case_id in pool:
            per_case_trials = int(ONE_STEP_RETRY_PER_CASE) if one_step_mode else 1
            excluded_action_texts_case: set[str] = set()
            for _ in range(max(1, per_case_trials)):
                cur_metas = base._rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
                cur_pivots = _current_pivots()
                anomaly = base._make_synthetic_anomaly(
                    glb_name=glb_name,
                    labels=labels,
                    metas_by_label=cur_metas,
                    label_to_oid=label_to_oid,
                    id_to_entry=id_to_entry,
                    sample_idx=int(case_id),
                    whitelist_pairs=whitelist_pairs,
                    wall_bounds=wall_bounds,
                    wall_components=wall_components,
                    cam_top=cam_top,
                    render_ctx=render_ctx,
                    pivot_by_label=cur_pivots,
                    excluded_inject_action_texts=excluded_action_texts_case,
                )
                if anomaly is None:
                    break

                issue_meta, inject_action = anomaly
                try:
                    inject_action_text = base._action_text(inject_action)
                except Exception:
                    inject_action_text = ""
                try:
                    target_label = int(inject_action.get("id"))
                except Exception:
                    continue
                if target_label not in operable_labels:
                    if inject_action_text:
                        excluded_action_texts_case.add(inject_action_text)
                    continue

                issue0 = (issue_meta.get("issues") or [{}])[0]
                issue_labels: set[int] = set()
                for x in issue0.get("object_labels", []):
                    sx = str(x).strip()
                    if sx.isdigit():
                        issue_labels.add(int(sx))
                if any(lb not in operable_labels for lb in issue_labels):
                    # Do not allow hidden/unlabeled objects to participate in the primary step issue.
                    if inject_action_text:
                        excluded_action_texts_case.add(inject_action_text)
                    continue

                g_tmp, o_tmp = base._apply_action(
                    inject_action, label_to_oid, id_to_entry, pivot_by_label=cur_pivots
                )
                try:
                    cur_after = base._rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
                    after_overlap = _scene_nonwhitelist_collision_pairs(
                        labels=labels,
                        metas_by_label=cur_after,
                        label_to_oid=label_to_oid,
                        id_to_entry=id_to_entry,
                        whitelist_pairs=whitelist_pairs,
                    )
                    after_wall = _scene_wall_conflict_labels(
                        labels=labels,
                        metas_by_label=cur_after,
                        label_to_oid=label_to_oid,
                        id_to_entry=id_to_entry,
                        wall_bounds=wall_bounds,
                        wall_components=wall_components,
                    )
                    after_orientation = _scene_orientation_conflict_labels(
                        labels=labels,
                        cur_metas_by_label=cur_after,
                        ref_metas_by_label=cur_metas,
                        label_to_oid=label_to_oid,
                        id_to_entry=id_to_entry,
                        wall_bounds=wall_bounds,
                    )
                finally:
                    base._restore_group(g_tmp, o_tmp)

                if steps and (not all(
                    _step_issue_still_present(
                        st,
                        after_overlap,
                        set(after_wall),
                        set(after_orientation),
                    )
                    for st in steps
                )):
                    if inject_action_text:
                        excluded_action_texts_case.add(inject_action_text)
                    continue

                selected = (case_id, issue_meta, inject_action)
                break
            if selected is not None:
                break

        if selected is None:
            continue

        case_id, issue_meta, inject_action = selected

        base._apply_action(inject_action, label_to_oid, id_to_entry, pivot_by_label=_current_pivots())

        reverse_action = base._reverse_action(inject_action)
        case_issue, case_action = base._target_case_by_id(case_id)
        steps.append(
            {
                "step_id": len(steps) + 1,
                "case_id": int(case_id + 1),
                "case_tag": f"step_{len(steps)+1}_{case_issue}_{case_action}",
                "issue_meta": issue_meta,
                "issue_labels": sorted(_extract_issue_labels(issue_meta)),
                "inject_action": inject_action,
                "reverse_action": reverse_action,
                "reverse_text": base._action_text(reverse_action),
            }
        )
        issue0 = (issue_meta.get("issues") or [{}])[0]
        last_issue_main = base._issue_main_type(issue0)

    return steps


def _build_multi_step_qa_text(mapped_steps: list[dict], label_category_map: dict[int, str]) -> tuple[str, str]:
    legend = base._label_legend_line(label_category_map)
    question = (
        "You are viewing two images of the same indoor scene with labeled objects.\n"
        "The scene contains multiple geometric anomalies introduced in multiple steps.\n"
        "Analyze the errors and provide a step-by-step repair plan.\n"
        "Allowed operation format:\n"
        "1. Move object <ID> <Direction> by <Distance>m\n"
        "2. Rotate object <ID> <clockwise/counter-clockwise> by <Angle>°\n"
        "3. Scale <up/down> object <ID> by <Percentage>%\n"
        "Top-view image: <image>\n"
        "Isometric-view image: <image>"
    )
    if legend:
        question += "\n" + legend

    answer_lines = []
    for i, st in enumerate(reversed(mapped_steps), start=1):
        answer_lines.append(f"{i}. {st['reverse_text']}")
    answer = "Fix plan:\n" + "\n".join(answer_lines)
    return question, answer


def process_scene_multi(
    glb_name: str,
    info: dict,
    glb_dir: Path,
    output_dir: Path,
    layout_root: Path,
    whitelist_pairs: set[tuple[str, str]],
    sample_idx: int,
    num_steps: int,
    save_scene_glb: bool = True,
    attempt_multiplier: int = 12,
    sample_variant: int = 1,
) -> dict | None:
    layout_info = info.get("layout_info", [])
    scene_name = info.get("scene_name", glb_name)
    layout_objects = base.convert_layout_to_objects(layout_info)
    if len(layout_objects) < base.MIN_OBJECTS:
        print(f"[DEBUG] skip {glb_name}: layout_objects={len(layout_objects)} < MIN_OBJECTS={base.MIN_OBJECTS}")
        return None

    glb_path = glb_dir / glb_name
    if not glb_path.exists():
        print(f"[DEBUG] skip {glb_name}: glb file not exists")
        return None

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))

    groups = base.collect_instance_groups(max_instance_idx=len(layout_objects))
    logical = []
    for obj in layout_objects:
        g = groups.get(obj["instance_index"])
        if g:
            logical.append(
                {
                    "id": obj["id"],
                    "instance_index": obj["instance_index"],
                    "category": str(obj.get("category", "unknown")),
                    "group": g,
                }
            )
    if len(logical) < base.MIN_OBJECTS:
        print(f"[DEBUG] skip {glb_name}: logical={len(logical)} < MIN_OBJECTS={base.MIN_OBJECTS}")
        return None

    base.set_render_and_world()
    bounds = base._scene_bounds_from_groups(groups)
    scene_center = Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"]))
    scene_radius = max(bounds["width"], bounds["depth"], 1.0)
    base.setup_lighting(scene_center, scene_radius)

    wall_path = base.build_wall_path(glb_name, layout_root)
    wall_bounds = base._import_wall_bounds(wall_path)
    wall_components = base._import_wall_components(wall_path)
    if base.USE_WALL_PROXY_RENDER:
        proxies = base._import_wall_proxy(wall_path)
        if proxies:
            print(f"[WALL] imported proxy walls: {len(proxies)} from {wall_path}")
            base._hide_original_wall_structures(wall_bounds)
    base._cap_wall_height(wall_bounds, top_margin=base.WALL_TOP_MARGIN)

    render_ctx = base.dc_utils.compute_render_frame(base._all_mesh_objects())
    cam_top, cam_top_data = base.create_preview_camera("TopCam")
    cam_iso, cam_iso_data = base.create_preview_camera("IsoCam")
    north_world_dir = base.dc_utils.canonical_north_world(render_ctx)
    base._set_move_basis_from_north(north_world_dir)

    dyn_w_top, dyn_h_top = base.dc_utils.dynamic_resolution_for_mode(render_ctx, "top", base.BASE_LONG_EDGE, base.MIN_SHORT_EDGE)
    bpy.context.scene.render.resolution_x = dyn_w_top
    bpy.context.scene.render.resolution_y = dyn_h_top
    base.setup_camera_for_mode(cam_top, cam_top_data, render_ctx, "top")
    bpy.context.view_layer.update()

    id_to_entry = {e["id"]: e for e in logical}
    valid_ids = [e["id"] for e in logical]
    rw_top = int(bpy.context.scene.render.resolution_x)
    rh_top = int(bpy.context.scene.render.resolution_y)
    valid_ids = [oid for oid in valid_ids if base._top_group_visibility_ok(id_to_entry[oid]["group"], cam_top, rw_top, rh_top)]
    if len(valid_ids) < base.MIN_OBJECTS:
        print(f"[DEBUG] skip {glb_name}: valid_ids(visibility)={len(valid_ids)} < MIN_OBJECTS={base.MIN_OBJECTS}")
        return None

    all_ids = [e["id"] for e in logical]
    all_label_map = base.assign_independent_labels([{"id": oid} for oid in all_ids])
    all_label_to_oid = {lb: oid for oid, lb in all_label_map.items()}
    all_labels = sorted(all_label_to_oid.keys())
    all_pivot_by_label = {}
    for lb in all_labels:
        oid = all_label_to_oid.get(lb)
        if oid is not None:
            all_pivot_by_label[lb] = base._group_centroid_world(id_to_entry[oid]["group"])

    visible_label_set = {int(all_label_map[oid]) for oid in valid_ids}
    visible_metas_by_label = {}
    for oid in valid_ids:
        e = id_to_entry[oid]
        lb = int(all_label_map[oid])
        visible_metas_by_label[lb] = base._build_meta_from_group(lb, oid, e["instance_index"], e["group"])
    visible_labels = sorted(visible_metas_by_label.keys())

    clean_state = _snapshot_scene_state(id_to_entry)

    steps_full = _create_multi_step_anomalies(
        glb_name=glb_name,
        labels=visible_labels,
        operable_labels=visible_label_set,
        label_to_oid=all_label_to_oid,
        id_to_entry=id_to_entry,
        whitelist_pairs=whitelist_pairs,
        wall_bounds=wall_bounds,
        wall_components=wall_components,
        cam_top=cam_top,
        render_ctx=render_ctx,
        pivot_by_label=all_pivot_by_label,
        max_steps=max(1, int(num_steps)),
        attempt_multiplier=attempt_multiplier,
    )

    req_steps = max(1, int(num_steps))
    if len(steps_full) < req_steps:
        _restore_scene_state(clean_state)
        print(f"[DEBUG] skip {glb_name}: cannot build enough multi-step anomalies ({len(steps_full)}/{req_steps})")
        return None

    forced_labels = _collect_forced_labels_from_steps(steps_full)
    max_labels = max(MULTI_MAX_LABELED_OBJECTS, len(forced_labels))
    selected_full_labels = base._select_operable_labels_with_forced(
        visible_labels,
        visible_metas_by_label,
        forced_labels,
        max_count=max_labels,
    )
    selected_oids = [oid for oid in valid_ids if int(all_label_map[oid]) in set(selected_full_labels)]

    label_map = base.assign_independent_labels([{"id": oid} for oid in selected_oids])
    label_to_oid = {lb: oid for oid, lb in label_map.items()}
    oid_to_label = {oid: lb for oid, lb in label_map.items()}
    label_category_map = {lb: str(id_to_entry[oid].get("category", "unknown")) for oid, lb in label_map.items()}

    mapped_steps = []
    for st in steps_full:
        issue_new, action_new = base._remap_issue_and_action_labels(
            st["issue_meta"],
            st["inject_action"],
            all_label_to_oid,
            oid_to_label,
        )
        if issue_new is None or action_new is None:
            _restore_scene_state(clean_state)
            print(f"[DEBUG] skip {glb_name}: remap failed for step {st.get('step_id')}")
            return None
        reverse_new = base._reverse_action(action_new)
        mapped_steps.append(
            {
                "step_id": int(st["step_id"]),
                "case_id": int(st["case_id"]),
                "case_tag": str(st["case_tag"]),
                "issue_meta": issue_new,
                "inject_action": action_new,
                "reverse_action": reverse_new,
                "reverse_text": base._action_text(reverse_new),
            }
        )

    stem = Path(glb_name).stem
    sample_variant = max(1, int(sample_variant))
    if sample_variant > 1:
        sample_dir = output_dir / stem / f"multi_step_case_{sample_variant}"
    else:
        sample_dir = output_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    top_error = sample_dir / "top_multi_error.png"
    iso_error = sample_dir / "isometric_multi_error.png"
    top_init = sample_dir / "top.png"
    iso_init = sample_dir / "isometric.png"

    raw_top_error = sample_dir / ".top_multi_error_raw.png"
    raw_iso_error = sample_dir / ".iso_multi_error_raw.png"
    raw_top_init = sample_dir / ".top_raw.png"
    raw_iso_init = sample_dir / ".iso_raw.png"
    final_glb_path = sample_dir / "error_scene.glb"
    final_glb_abs = ""

    try:
        top_error_abs = base._capture_labeled(
            cam_top,
            render_ctx,
            "top",
            id_to_entry,
            label_map,
            raw_top_error,
            top_error,
            wall_bounds=wall_bounds,
            wall_alpha=base.TOP_WALL_ALPHA,
            north_world_dir=north_world_dir,
        )
        iso_error_abs = base._capture_labeled(
            cam_iso,
            render_ctx,
            FIXED_ISOMETRIC_MODE,
            id_to_entry,
            label_map,
            raw_iso_error,
            iso_error,
            wall_bounds=wall_bounds,
            wall_alpha=base.ISO_WALL_ALPHA,
            north_world_dir=north_world_dir,
        )

        if save_scene_glb:
            bpy.ops.export_scene.gltf(
                filepath=str(final_glb_path),
                export_format='GLB',
                export_apply=True,
            )
            final_glb_abs = str(final_glb_path.resolve())

        _restore_scene_state(clean_state)

        top_init_abs = base._capture_labeled(
            cam_top,
            render_ctx,
            "top",
            id_to_entry,
            label_map,
            raw_top_init,
            top_init,
            wall_bounds=wall_bounds,
            wall_alpha=base.TOP_WALL_ALPHA,
            north_world_dir=north_world_dir,
        )
        iso_init_abs = base._capture_labeled(
            cam_iso,
            render_ctx,
            FIXED_ISOMETRIC_MODE,
            id_to_entry,
            label_map,
            raw_iso_init,
            iso_init,
            wall_bounds=wall_bounds,
            wall_alpha=base.ISO_WALL_ALPHA,
            north_world_dir=north_world_dir,
        )
    except Exception as e:
        _restore_scene_state(clean_state)
        print(f"[DEBUG] skip {glb_name}: render failed {e}")
        return None
    finally:
        for p in [raw_top_error, raw_iso_error, raw_top_init, raw_iso_init]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    question, answer = _build_multi_step_qa_text(mapped_steps, label_category_map)

    label_mapping = [
        {"scene_object_id": oid, "label_id": label_map[oid], "instance_index": id_to_entry[oid]["instance_index"]}
        for oid in sorted(selected_oids, key=lambda x: label_map[x])
    ]

    _restore_scene_state(clean_state)

    return {
        "case_id": int(sample_variant),
        "case_tag": f"multi_step_case_{sample_variant}",
        "glb_name": glb_name,
        "scene_name": scene_name,
        "object_count": len(valid_ids),
        "labeled_object_count": len(selected_oids),
        "label_mapping": label_mapping,
        "steps": len(mapped_steps),
        "anomalies": mapped_steps,
        "images": {
            "top": top_error_abs,
            "isometric": iso_error_abs,
            "top_init": top_init_abs,
            "isometric_init": iso_init_abs,
            "error_scene_glb": final_glb_abs,
        },
        "qa": {
            "multi_step": {
                "task_type": "multi_step_error_modify",
                "question": question,
                "answer": answer,
                "images": [top_error_abs, iso_error_abs],
                "initial_images": [top_init_abs, iso_init_abs],
            }
        },
    }


def _merge_worker_outputs(output_dir: Path, workers: int, steps: int) -> Path | None:
    parts = [output_dir / f"metadata_indoor_steps{steps}.worker_{i}_of_{workers}.json" for i in range(workers)]
    if not all(p.exists() for p in parts):
        return None

    merged: list[dict] = []
    for p in parts:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged.extend(data)

    out_path = output_dir / f"metadata_indoor_steps{steps}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    for p in parts:
        try:
            p.unlink()
        except Exception:
            pass

    return out_path


def _cleanup_empty_scene_dir(output_dir: Path, glb_name: str):
    scene_dir = output_dir / Path(glb_name).stem
    if not scene_dir.exists() or not scene_dir.is_dir():
        return
    try:
        has_files = any(p.is_file() for p in scene_dir.rglob("*"))
        if not has_files:
            import shutil
            shutil.rmtree(scene_dir, ignore_errors=True)
    except Exception:
        pass


def main():
    args = parse_args()
    random.seed(args.seed)
    _sync_overlay_style_with_base()
    args.multi_step_attempt_multiplier = max(1, int(args.multi_step_attempt_multiplier))
    args.multi_step_variants_per_scene = max(1, int(args.multi_step_variants_per_scene))

  
    base.WALL_CONFLICT_DEPTH_RATIO_THRESHOLD = float(args.wall_conflict_depth_ratio_threshold)
    base.WALL_CONFLICT_VISIBLE_MIN_OVERFLOW_M = float(args.wall_conflict_visible_min_overflow_m)
    base.WALL_CONFLICT_VISIBLE_MIN_OVERFLOW_RATIO = float(args.wall_conflict_visible_min_overflow_ratio)
    base.WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO = float(args.wall_conflict_visible_min_penetration_ratio)
    base.REGION_BOUNDARY_RATIO_DEFAULT = float(args.wall_conflict_inside_ratio_default)
    base.REGION_BOUNDARY_RATIO_RELAXED = float(args.wall_conflict_inside_ratio_relaxed)

    glb_dir = args.glb_dir.resolve()
    mapping_json = args.mapping_json.resolve()
    layout_root = args.layout_root.resolve()
    whitelist_json = args.whitelist_json.resolve()
    output_dir = _output_dir_with_steps(args.output_dir, args.steps)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not glb_dir.exists():
        raise FileNotFoundError(f"GLB dir not found: {glb_dir}")
    if not mapping_json.exists():
        raise FileNotFoundError(f"Mapping json not found: {mapping_json}")

    with open(mapping_json, "r", encoding="utf-8") as f:
        scene_mapping = json.load(f)
    whitelist_pairs = base.load_whitelist_pairs(whitelist_json)

    glb_files = sorted([p.name for p in glb_dir.glob("*.glb") if p.is_file()])
    matched_pairs = []
    for k in glb_files:
        map_key = None
        if k in scene_mapping:
            map_key = k
        elif k.endswith("_clean.glb"):
            base_k = k[:-10] + ".glb"
            if base_k in scene_mapping:
                map_key = base_k
        if map_key is not None:
            matched_pairs.append((k, map_key))

    scene_keys = [k for k, _ in matched_pairs]
    map_keys = {k: mk for k, mk in matched_pairs}
    if args.region:
        kw = args.region.strip().lower()
        scene_keys = [
            k for k in scene_keys
            if kw in k.lower() or kw in str(scene_mapping.get(map_keys[k], {}).get("scene_name", "")).lower()
        ]
    if args.max_scenes > 0:
        scene_keys = scene_keys[: args.max_scenes]

    indexed_scene_keys = list(enumerate(scene_keys))

    if args.workers > 1 and args.worker_index < 0:
        script_path = str(Path(__file__).resolve())
        base_args = []
        if args.max_scenes > 0:
            base_args += ["--max-scenes", str(args.max_scenes)]
        if args.region:
            base_args += ["--region", args.region]
        base_args += [
            "--glb-dir", str(glb_dir),
            "--mapping-json", str(mapping_json),
            "--layout-root", str(layout_root),
            "--whitelist-json", str(whitelist_json),
            "--output-dir", str(args.output_dir),
            "--seed", str(args.seed),
            "--workers", str(args.workers),
            "--steps", str(args.steps),
            "--wall-conflict-depth-ratio-threshold", str(args.wall_conflict_depth_ratio_threshold),
            "--wall-conflict-visible-min-overflow-m", str(args.wall_conflict_visible_min_overflow_m),
            "--wall-conflict-visible-min-overflow-ratio", str(args.wall_conflict_visible_min_overflow_ratio),
            "--wall-conflict-visible-min-penetration-ratio", str(args.wall_conflict_visible_min_penetration_ratio),
            "--wall-conflict-inside-ratio-default", str(args.wall_conflict_inside_ratio_default),
            "--wall-conflict-inside-ratio-relaxed", str(args.wall_conflict_inside_ratio_relaxed),
            "--multi-step-attempt-multiplier", str(args.multi_step_attempt_multiplier),
            "--multi-step-variants-per-scene", str(args.multi_step_variants_per_scene),
        ]
        if not args.save_scene_glb:
            base_args += ["--no-save-scene-glb"]

        print("=" * 80)
        print(f"Launching {args.workers} parallel workers (steps={args.steps})...")
        procs = []
        for wi in range(args.workers):
            cmd = [
                str(BLENDER_BIN),
                "--background",
                "--python",
                script_path,
                "--",
                *base_args,
                "--worker-index",
                str(wi),
            ]
            procs.append((wi, subprocess.Popen(cmd, env=os.environ.copy())))

        bad = []
        for wi, p in procs:
            rc = p.wait()
            if rc != 0:
                bad.append((wi, rc))
        if bad:
            raise RuntimeError(f"Some workers failed: {bad}")

        merged = _merge_worker_outputs(output_dir, args.workers, args.steps)
        if merged is None:
            raise RuntimeError("Failed to merge worker outputs")

        print("=" * 80)
        print(f"Parallel done. Output: {merged}")
        return

    if args.workers > 1 and args.worker_index >= 0:
        indexed_scene_keys = [x for x in indexed_scene_keys if (x[0] % args.workers) == args.worker_index]
        print(f"[Worker {args.worker_index}/{args.workers}] Processing {len(indexed_scene_keys)} scenes")

    print("=" * 80)
    print(f"Indoor multi-step error QA construct (steps={args.steps})")
    print(f"Total glb files: {len(glb_files)}")
    print(f"Matched in mapping: {len(indexed_scene_keys)}")
    if args.region:
        print(f"Region filter: {args.region}")

    if args.workers > 1 and args.worker_index >= 0:
        out_json = output_dir / f"metadata_indoor_steps{args.steps}.worker_{args.worker_index}_of_{args.workers}.json"
    else:
        out_json = output_dir / f"metadata_indoor_steps{args.steps}.json"

    all_results = []
    for local_idx, (global_idx, glb_name) in enumerate(indexed_scene_keys, start=1):
        scene_results = []
        for variant_idx in range(1, int(args.multi_step_variants_per_scene) + 1):
            variant_seed = int(args.seed) + int(global_idx) * 10000 + int(variant_idx) * 31
            random.seed(variant_seed)
            try:
                res = process_scene_multi(
                    glb_name=glb_name,
                    info=scene_mapping[map_keys[glb_name]],
                    glb_dir=glb_dir,
                    output_dir=output_dir,
                    layout_root=layout_root,
                    whitelist_pairs=whitelist_pairs,
                    sample_idx=global_idx + variant_idx,
                    num_steps=args.steps,
                    save_scene_glb=bool(args.save_scene_glb),
                    attempt_multiplier=args.multi_step_attempt_multiplier,
                    sample_variant=variant_idx,
                )
            except Exception as e:
                print(f"[{local_idx}/{len(indexed_scene_keys)}] variant {variant_idx}: skip {glb_name}: exception {e}")
                continue
            if not res:
                print(
                    f"[{local_idx}/{len(indexed_scene_keys)}] variant {variant_idx}/"
                    f"{int(args.multi_step_variants_per_scene)} skip {glb_name}: invalid variant"
                )
                continue
            all_results.append(res)
            print(
                f"[{local_idx}/{len(indexed_scene_keys)}] done {glb_name} variant={variant_idx}/"
                f"{int(args.multi_step_variants_per_scene)}: steps={res.get('steps', 0)}, "
                f"labeled={res.get('labeled_object_count', 0)}"
            )
            scene_results.append(res)
        if not scene_results:
            _cleanup_empty_scene_dir(output_dir, glb_name)
            if int(args.multi_step_variants_per_scene) > 1:
                print(f"[{local_idx}/{len(indexed_scene_keys)}] skip {glb_name}: no valid variants")
            else:
                print(f"[{local_idx}/{len(indexed_scene_keys)}] skip {glb_name}: invalid scene")
        _write_json_atomic(out_json, all_results)
        print(
            f"[CHECKPOINT] {local_idx}/{len(indexed_scene_keys)} {glb_name} -> "
            f"{out_json.name} (valid_samples={len(all_results)})"
        )

    _write_json_atomic(out_json, all_results)

    print("=" * 80)
    total_attempts = len(indexed_scene_keys) * int(args.multi_step_variants_per_scene)
    print(
        f"Done. valid_samples={len(all_results)}/{total_attempts} "
        f"(steps={args.steps}, variants_per_scene={int(args.multi_step_variants_per_scene)})"
    )
    print(f"Output: {out_json}")


if __name__ == "__main__":
    main()
