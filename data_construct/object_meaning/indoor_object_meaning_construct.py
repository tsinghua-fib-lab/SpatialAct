#!/usr/bin/env python3
"""
Indoor object-meaning QA constructor.

Build 4 QA categories for indoor 3D scenes:
1) Counting
2) Object identification
3) Relative position
4) Distance / nearest-farthest

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import shutil
import subprocess
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


DEFAULT_GLB_DIR = INTERNSCENES_ROOT / "scenes/glb_files_wall_complex-10-15_clean_keep"
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark/data/object_meaning/indoor_scenes_complex-10-15"
DEFAULT_CATEGORY_DIST_JSON = INTERNSCENES_ROOT / "data_process/category_distribution.json"

BLENDER_BIN = PROJECT_ROOT / "blender-3.2.2-linux-x64/blender"
RENDER_SCRIPT = PROJECT_ROOT / "benchmark/data_construct/utils.py"
SHARED_UTILS_PATH = PROJECT_ROOT / "benchmark/data_construct/utils.py"

_utils_spec = importlib.util.spec_from_file_location("dc_render_utils_task0_meaning", SHARED_UTILS_PATH)
if _utils_spec is None or _utils_spec.loader is None:
    raise RuntimeError(f"Cannot load shared utils from {SHARED_UTILS_PATH}")
dc_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(dc_utils)

# Priority categories are chosen from high-frequency + visually clear classes
# in InternScenes statistics (category_distribution.json) and current clean subset.
DEFAULT_COUNT_PREF_CATEGORIES = ["cabinet", "chair", "table", "shelf", "couch", "bed", "window", "desk", "wardrobe", "toilet", "bathtub", "sink"]
DEFAULT_IDENTIFY_PAIR_PREF_CATEGORIES = ["cabinet", "chair", "table", "shelf", "couch", "bed", "desk", "wardrobe", "refrigerator", "toilet"]
DEFAULT_RELATIVE_REF_PREF_CATEGORIES = ["table", "cabinet", "couch", "bed", "shelf", "desk", "wardrobe", "refrigerator", "bathtub", "toilet", "sink"]
DEFAULT_DISTANCE_REF_PREF_CATEGORIES = ["table", "cabinet", "couch", "bed", "shelf", "desk", "wardrobe", "refrigerator", "bathtub", "toilet", "sink"]

COUNT_PREF_CATEGORIES = list(DEFAULT_COUNT_PREF_CATEGORIES)
IDENTIFY_PAIR_PREF_CATEGORIES = list(DEFAULT_IDENTIFY_PAIR_PREF_CATEGORIES)
RELATIVE_REF_PREF_CATEGORIES = list(DEFAULT_RELATIVE_REF_PREF_CATEGORIES)
DISTANCE_REF_PREF_CATEGORIES = list(DEFAULT_DISTANCE_REF_PREF_CATEGORIES)
FIXED_ISOMETRIC_MODE = "isometric_north_ur"

QA_DISTRACTOR_EXTRA_POOL = [
    "chair",
    "table",
    "cabinet",
    "bed",
    "couch",
    "sink",
    "window",
    "wardrobe",
    "desk",
    "shelf",
    "mirror",
    "plant",
]

# Exclude small/object-level categories from QA targets/references/options.
QA_EXCLUDE_PHRASES = {
    "toilet paper",
    "paper towel",
    "tissue box",
    "soap dispenser",
    "soap dish",
    "tooth brush",
    "toothbrush",
    "tooth paste",
    "toothpaste",
    "toilet brush",
}
QA_EXCLUDE_TOKENS = {
    "cup",
    "mug",
    "bottle",
    "plate",
    "bowl",
    "spoon",
    "fork",
    "knife",
    "bag",
    "backpack",
    "shoe",
    "slipper",
    "sock",
    "clothes",
    "jacket",
    "hat",
    "book",
    "phone",
    "remote",
    "keyboard",
    "mouse",
    "wallet",
    "key",
    "toy",
    "towel",
    "tissue",
    "paper",
}

PREF_GENERAL_AVOID_PHRASES = {
    "unknown",
    "object",
    "decoration",
    "picture",
    "poster",
    "painting",
    "frame",
    "socket",
    "switch",
    "light",
    "lamp",
    "clock",
    "curtain",
    "blinds",
    "radiator",
}
PREF_IDENTIFY_AVOID_PHRASES = {
    "door",
    "window frame",
    "doorframe",
    "window",
}
PREF_SPATIAL_REF_AVOID_PHRASES = {
    "door",
    "doorframe",
    "window frame",
    "window",
    "mirror",
}

PREF_MIN_GLOBAL_FREQ = 300
PREF_IDENTIFY_MIN_MEDIAN_VOL = 0.15
PREF_SPATIAL_MIN_MEDIAN_VOL = 0.18

CARDINAL_OPTIONS_EN = ["north", "south", "east", "west"]
EIGHT_DIRECTIONS_EN = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
LABEL_RADIUS = 13
LABEL_FONT_SIZE = 16
TOP_WALL_ALPHA = 1.0
ISO_WALL_ALPHA = 0.55
VIS_MIN_AREA_PX = 120.0
TASK_TYPES_USE_LABELED_IMAGES = {
    "object_identification_name",
    "object_identification_mcq",
    "relative_position_direction_mcq",
    "distance_nearest",
    "distance_farthest",
}
TASK_TYPES_IMAGES_EQ_REF_IMAGES = {
    "object_identification_name",
    "object_identification_mcq",
    "relative_position_side",
    "relative_position_direction_mcq",
    "distance_nearest",
    "distance_farthest",
}
LEFT_STRIP_UI_SCALE_MULT = 1.3
LEFT_STRIP_UI_SCALE_MULT_TOP = 1.55
RELATIVE_DIRECTION_MIN_MARGIN = 0.12
RELATIVE_DIRECTION_MIN_AXIS_WORLD = 0.12
RELATIVE_DIRECTION_MIN_AXIS_PX_FALLBACK = 6.0
DISTANCE_EXTREME_MIN_MARGIN = 0.15
DISTANCE_EXTREME_MAX_REF_TRIES = 24
CAMERA_TOP_DIST_SCALE = 2.9
CAMERA_ISO_DIST_SCALE = 3.2
CAMERA_FIT_MARGIN = 1.10
CAMERA_SAFETY_SCALE = 1.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct indoor object-meaning QA")
    parser.add_argument("--glb-dir", type=Path, default=DEFAULT_GLB_DIR)
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--category-dist-json",
        type=Path,
        default=DEFAULT_CATEGORY_DIST_JSON,
        help="Category distribution JSON for deriving preference categories",
    )
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes")
    parser.add_argument("--scene-keyword", type=str, default="", help="Filter scene key by substring")
    parser.add_argument("--choices", type=int, default=4, help="MCQ option count")
    parser.add_argument("--min-objects", type=int, default=4)
    parser.add_argument("--keep-box-json", action="store_true", help="Keep intermediate boxes JSON files")
    parser.add_argument("--force-rerender", action="store_true")
    return parser.parse_args()


def _safe_int(x) -> tuple[int, str]:
    s = str(x)
    try:
        return 0, int(s)
    except Exception:
        return 1, s


def _norm_category(cat: str) -> str:
    s = str(cat or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def _category_matches(cat: str, keyword: str) -> bool:
    c = _norm_category(cat)
    k = _norm_category(keyword)
    if not c or not k:
        return False
    if c == k:
        return True
    c_tokens = set(c.split())
    k_tokens = set(k.split())
    if c_tokens and k_tokens and (c_tokens & k_tokens):
        return True
    return k in c or c in k


def _category_equal(cat_a: str, cat_b: str) -> bool:
    return _norm_category(cat_a) == _norm_category(cat_b)


def _contains_any_phrase(cat: str, phrases: set[str]) -> bool:
    c = _norm_category(cat)
    if not c:
        return False
    for p in phrases:
        if _norm_category(p) in c:
            return True
    return False


def _is_excluded_category_for_qa(cat: str) -> bool:
    c = _norm_category(cat)
    if not c or c == "unknown":
        return True

    for phrase in QA_EXCLUDE_PHRASES:
        if phrase in c:
            return True

    tokens = set(c.split())
    if not tokens:
        return True
    if tokens & QA_EXCLUDE_TOKENS:
        return True
    singularized = {t[:-1] for t in tokens if t.endswith("s") and len(t) > 3}
    if singularized & QA_EXCLUDE_TOKENS:
        return True
    return False


def _filter_objects_for_qa(objects: list[dict]) -> list[dict]:
    return [o for o in objects if not _is_excluded_category_for_qa(str(o.get("category", "")))]


def _is_good_spatial_ref_category(cat: str) -> bool:
    blocked = PREF_GENERAL_AVOID_PHRASES | PREF_SPATIAL_REF_AVOID_PHRASES | {"curtain", "window", "door"}
    return not _contains_any_phrase(cat, blocked)


def _load_category_frequency(dist_json: Path) -> dict[str, int]:
    if not dist_json.exists():
        return {}
    try:
        with open(dist_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    dist = data.get("distribution", [])
    if not isinstance(dist, list):
        return {}
    out: dict[str, int] = {}
    for item in dist:
        if not isinstance(item, dict):
            continue
        c = _norm_category(item.get("category", ""))
        if not c:
            continue
        try:
            out[c] = int(item.get("count", 0))
        except Exception:
            continue
    return out


def _category_local_counts_and_volumes(mapping: dict) -> tuple[dict[str, int], dict[str, float]]:
    local_counts: dict[str, int] = {}
    vols: dict[str, list[float]] = {}
    for payload in mapping.values():
        if not isinstance(payload, dict):
            continue
        layout_info = payload.get("layout_info", [])
        if not isinstance(layout_info, list):
            continue
        for item in layout_info:
            if not isinstance(item, dict):
                continue
            cat = _norm_category(item.get("category", ""))
            if not cat:
                continue
            local_counts[cat] = local_counts.get(cat, 0) + 1
            bbox = item.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) < 6:
                continue
            try:
                sx, sy, sz = abs(float(bbox[3])), abs(float(bbox[4])), abs(float(bbox[5]))
            except Exception:
                continue
            v = sx * sy * sz
            if v <= 1e-8:
                continue
            vols.setdefault(cat, []).append(v)

    med_vol = {c: float(statistics.median(vs)) for c, vs in vols.items() if vs}
    return local_counts, med_vol


def _derive_pref_categories_from_stats(mapping: dict, dist_json: Path) -> dict[str, list[str]]:
    global_freq = _load_category_frequency(dist_json)
    local_counts, med_vol = _category_local_counts_and_volumes(mapping)
    observed = sorted(local_counts.keys())

    def _base_ok(cat: str) -> bool:
        if _is_excluded_category_for_qa(cat):
            return False
        if _contains_any_phrase(cat, PREF_GENERAL_AVOID_PHRASES):
            return False
        gf = global_freq.get(cat, 0)
        lc = local_counts.get(cat, 0)
        return gf >= PREF_MIN_GLOBAL_FREQ or lc >= 2

    candidates = [c for c in observed if _base_ok(c)]

    def _score(cat: str) -> float:
        gf = float(global_freq.get(cat, 0))
        lc = float(local_counts.get(cat, 0))
        mv = float(med_vol.get(cat, 0.0))
        return math.log1p(gf) + 0.75 * math.log1p(lc) + 0.95 * math.log1p(max(mv, 1e-8) * 1000.0)

    ranked = sorted(candidates, key=_score, reverse=True)

    def _pick(limit: int, predicate, fallback: list[str]) -> list[str]:
        out: list[str] = []
        for c in ranked:
            if predicate(c) and c not in out:
                out.append(c)
            if len(out) >= limit:
                break
        for c in fallback:
            cn = _norm_category(c)
            if cn in observed and cn not in out and predicate(cn):
                out.append(cn)
            if len(out) >= limit:
                break
        return out

    count_pref = _pick(
        12,
        lambda c: (not _contains_any_phrase(c, PREF_IDENTIFY_AVOID_PHRASES)) and med_vol.get(c, 0.0) >= 0.12,
        DEFAULT_COUNT_PREF_CATEGORIES,
    )
    identify_pref = _pick(
        10,
        lambda c: (not _contains_any_phrase(c, PREF_IDENTIFY_AVOID_PHRASES))
        and med_vol.get(c, 0.0) >= PREF_IDENTIFY_MIN_MEDIAN_VOL,
        DEFAULT_IDENTIFY_PAIR_PREF_CATEGORIES,
    )
    relative_pref = _pick(
        10,
        lambda c: _is_good_spatial_ref_category(c)
        and med_vol.get(c, 0.0) >= PREF_SPATIAL_MIN_MEDIAN_VOL
        and local_counts.get(c, 0) >= 2,
        DEFAULT_RELATIVE_REF_PREF_CATEGORIES,
    )
    distance_pref = _pick(
        10,
        lambda c: _is_good_spatial_ref_category(c)
        and med_vol.get(c, 0.0) >= PREF_SPATIAL_MIN_MEDIAN_VOL
        and local_counts.get(c, 0) >= 2,
        DEFAULT_DISTANCE_REF_PREF_CATEGORIES,
    )

    return {
        "count": count_pref or list(DEFAULT_COUNT_PREF_CATEGORIES),
        "identify": identify_pref or list(DEFAULT_IDENTIFY_PAIR_PREF_CATEGORIES),
        "relative_ref": relative_pref or list(DEFAULT_RELATIVE_REF_PREF_CATEGORIES),
        "distance_ref": distance_pref or list(DEFAULT_DISTANCE_REF_PREF_CATEGORIES),
    }


def _choice_labels(n: int) -> list[str]:
    base = ["A", "B", "C", "D", "E", "F"]
    return base[:n]


def _build_mcq_text(options: list[str], answer_idx: int) -> tuple[str, str]:
    labels = _choice_labels(len(options))
    lines = [f"{labels[i]}. {options[i]}" for i in range(len(options))]
    return "\n".join(lines), labels[answer_idx]


def _xy_dist(a: dict, b: dict) -> float:
    dx = a["center_xy"][0] - b["center_xy"][0]
    dy = a["center_xy"][1] - b["center_xy"][1]
    return math.hypot(dx, dy)


def _cardinal_from_ref_to_obj(obj: dict, ref: dict) -> tuple[str, float]:
    dx = obj["center_xy"][0] - ref["center_xy"][0]
    dy = obj["center_xy"][1] - ref["center_xy"][1]
    ax = abs(dx)
    ay = abs(dy)
    if ax >= ay:
        direction = "east" if dx >= 0 else "west"
        margin = (ax - ay) / max(ax, 1e-6)
    else:
        direction = "north" if dy >= 0 else "south"
        margin = (ay - ax) / max(ay, 1e-6)
    return direction, margin


def _cardinal_membership_from_ref_to_obj(
    obj: dict, ref: dict, min_axis: float = RELATIVE_DIRECTION_MIN_AXIS_WORLD
) -> tuple[set[str], dict[str, float]]:
    dx = float(obj["center_xy"][0] - ref["center_xy"][0])
    dy = float(obj["center_xy"][1] - ref["center_xy"][1])
    scale = max(abs(dx), abs(dy), 1e-6)
    dirs: set[str] = set()
    scores: dict[str, float] = {}
    if dx >= min_axis:
        dirs.add("east")
        scores["east"] = abs(dx) / scale
    if dx <= -min_axis:
        dirs.add("west")
        scores["west"] = abs(dx) / scale
    if dy >= min_axis:
        dirs.add("north")
        scores["north"] = abs(dy) / scale
    if dy <= -min_axis:
        dirs.add("south")
        scores["south"] = abs(dy) / scale
    return dirs, scores


def _direction8_from_ref_to_obj(obj: dict, ref: dict) -> tuple[str, float]:
    dx = float(obj["center_xy"][0] - ref["center_xy"][0])
    dy = float(obj["center_xy"][1] - ref["center_xy"][1])
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return "north", 0.0

    angle = math.degrees(math.atan2(dy, dx))
    bins = [
        ("east", 0.0),
        ("northeast", 45.0),
        ("north", 90.0),
        ("northwest", 135.0),
        ("west", 180.0),
        ("southwest", -135.0),
        ("south", -90.0),
        ("southeast", -45.0),
    ]

    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    best_dir = "north"
    best_diff = 180.0
    for dname, dangle in bins:
        diff = _ang_diff(angle, dangle)
        if diff < best_diff:
            best_diff = diff
            best_dir = dname
    margin = max(0.0, min(1.0, (22.5 - best_diff) / 22.5))
    return best_dir, margin


def _normalize2(vx: float, vy: float) -> tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-8:
        return 0.0, 1.0
    return vx / n, vy / n


def _parse_north_world_xy(payload: dict) -> tuple[float, float] | None:
    nw = payload.get("north_world_xy", None) if isinstance(payload, dict) else None
    if not (isinstance(nw, list) and len(nw) == 2):
        return None
    try:
        nx, ny = float(nw[0]), float(nw[1])
    except Exception:
        return None
    n = math.hypot(nx, ny)
    if n < 1e-8:
        return None
    return (nx / n, ny / n)


def _canonical_north_world_xy_from_objects(objects: list[dict]) -> tuple[float, float]:
    if not objects:
        return (0.0, 1.0)
    xs = [float(o["center_xy"][0]) for o in objects if isinstance(o.get("center_xy"), list) and len(o["center_xy"]) >= 2]
    ys = [float(o["center_xy"][1]) for o in objects if isinstance(o.get("center_xy"), list) and len(o["center_xy"]) >= 2]
    if not xs or not ys:
        return (0.0, 1.0)
    ext_x = max(xs) - min(xs)
    ext_y = max(ys) - min(ys)
    # Match shared renderer canonical_north_world(): if Y-span dominates, north is -X; else +Y.
    if ext_y > ext_x:
        return (-1.0, 0.0)
    return (0.0, 1.0)


def _get_world_basis(payload: dict, objects_by_id: dict[str, dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    parsed = _parse_north_world_xy(payload)
    if parsed is None:
        nx, ny = 0.0, 1.0
    else:
        nx, ny = parsed
    north = _normalize2(nx, ny)
    east = _normalize2(north[1], -north[0])
    return north, east


def _direction8_from_world_basis(
    target_obj: dict,
    ref_obj: dict,
    north_vec: tuple[float, float],
    east_vec: tuple[float, float],
) -> tuple[str, float]:
    dx = float(target_obj["center_xy"][0] - ref_obj["center_xy"][0])
    dy = float(target_obj["center_xy"][1] - ref_obj["center_xy"][1])
    x = dx * east_vec[0] + dy * east_vec[1]
    y = dx * north_vec[0] + dy * north_vec[1]
    if abs(x) < 1e-8 and abs(y) < 1e-8:
        return "north", 0.0

    angle = math.degrees(math.atan2(y, x))
    bins = [
        ("east", 0.0),
        ("northeast", 45.0),
        ("north", 90.0),
        ("northwest", 135.0),
        ("west", 180.0),
        ("southwest", -135.0),
        ("south", -90.0),
        ("southeast", -45.0),
    ]

    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    best_dir = "north"
    best_diff = 180.0
    for dname, dangle in bins:
        diff = _ang_diff(angle, dangle)
        if diff < best_diff:
            best_diff = diff
            best_dir = dname
    margin = max(0.0, min(1.0, (22.5 - best_diff) / 22.5))
    return best_dir, margin


def _cardinal_membership_from_world_basis(
    target_obj: dict,
    ref_obj: dict,
    north_vec: tuple[float, float],
    east_vec: tuple[float, float],
    min_axis: float = RELATIVE_DIRECTION_MIN_AXIS_WORLD,
) -> tuple[set[str], dict[str, float]]:
    dx = float(target_obj["center_xy"][0] - ref_obj["center_xy"][0])
    dy = float(target_obj["center_xy"][1] - ref_obj["center_xy"][1])
    x = dx * east_vec[0] + dy * east_vec[1]
    y = dx * north_vec[0] + dy * north_vec[1]
    scale = max(abs(x), abs(y), min_axis, 1e-6)
    dirs: set[str] = set()
    scores: dict[str, float] = {}
    if x >= min_axis:
        dirs.add("east")
        scores["east"] = abs(x) / scale
    if x <= -min_axis:
        dirs.add("west")
        scores["west"] = abs(x) / scale
    if y >= min_axis:
        dirs.add("north")
        scores["north"] = abs(y) / scale
    if y <= -min_axis:
        dirs.add("south")
        scores["south"] = abs(y) / scale
    return dirs, scores


def _direction8_from_top_payload(target_obj: dict, ref_obj: dict, top_payload: dict) -> tuple[str, float] | None:
    """
    Compute 8-direction using top-view screen coordinates so that direction
    semantics are guaranteed to align with rendered N-up indicator.
    """
    ct = _center_from_payload(top_payload, int(target_obj["instance_index"]))
    cr = _center_from_payload(top_payload, int(ref_obj["instance_index"]))
    if ct is None or cr is None:
        return None

    # Screen: +x right, +y down. Convert to math axis with +y as north/up.
    dx = float(ct[0] - cr[0])
    dy = float(cr[1] - ct[1])
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return "north", 0.0

    angle = math.degrees(math.atan2(dy, dx))
    bins = [
        ("east", 0.0),
        ("northeast", 45.0),
        ("north", 90.0),
        ("northwest", 135.0),
        ("west", 180.0),
        ("southwest", -135.0),
        ("south", -90.0),
        ("southeast", -45.0),
    ]

    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    best_dir = "north"
    best_diff = 180.0
    for dname, dangle in bins:
        diff = _ang_diff(angle, dangle)
        if diff < best_diff:
            best_diff = diff
            best_dir = dname
    margin = max(0.0, min(1.0, (22.5 - best_diff) / 22.5))
    return best_dir, margin


def _cardinal_from_top_payload(target_obj: dict, ref_obj: dict, top_payload: dict) -> tuple[str, float] | None:
    """
    Compute 4-direction using top-view screen coordinates so that direction
    semantics are guaranteed to align with rendered N-up indicator.
    """
    ct = _center_from_payload(top_payload, int(target_obj["instance_index"]))
    cr = _center_from_payload(top_payload, int(ref_obj["instance_index"]))
    if ct is None or cr is None:
        return None

    # Screen: +x right, +y down. Convert to math axis with +y as north/up.
    dx = float(ct[0] - cr[0])
    dy = float(cr[1] - ct[1])
    ax = abs(dx)
    ay = abs(dy)
    if ax >= ay:
        direction = "east" if dx >= 0 else "west"
        margin = (ax - ay) / max(ax, 1e-6)
    else:
        direction = "north" if dy >= 0 else "south"
        margin = (ay - ax) / max(ay, 1e-6)
    return direction, margin


def _relative_direction_min_axis_px(top_payload: dict) -> float:
    try:
        ppm = float(top_payload.get("pixels_per_meter", 0.0))
    except Exception:
        ppm = 0.0
    if ppm > 1e-6:
        return max(4.0, ppm * RELATIVE_DIRECTION_MIN_AXIS_WORLD)
    return RELATIVE_DIRECTION_MIN_AXIS_PX_FALLBACK


def _cardinal_membership_from_top_payload(
    target_obj: dict, ref_obj: dict, top_payload: dict
) -> tuple[set[str], dict[str, float]] | None:
    ct = _center_from_payload(top_payload, int(target_obj["instance_index"]))
    cr = _center_from_payload(top_payload, int(ref_obj["instance_index"]))
    if ct is None or cr is None:
        return None

    # Screen: +x right, +y down. Convert to math axis with +y as north/up.
    dx = float(ct[0] - cr[0])
    dy = float(cr[1] - ct[1])
    min_axis = _relative_direction_min_axis_px(top_payload)
    scale = max(abs(dx), abs(dy), min_axis, 1e-6)
    dirs: set[str] = set()
    scores: dict[str, float] = {}
    if dx >= min_axis:
        dirs.add("east")
        scores["east"] = abs(dx) / scale
    if dx <= -min_axis:
        dirs.add("west")
        scores["west"] = abs(dx) / scale
    if dy >= min_axis:
        dirs.add("north")
        scores["north"] = abs(dy) / scale
    if dy <= -min_axis:
        dirs.add("south")
        scores["south"] = abs(dy) / scale
    return dirs, scores


def _direction_option_pool(answer_dir: str) -> list[str]:
    if answer_dir not in EIGHT_DIRECTIONS_EN:
        return CARDINAL_OPTIONS_EN[:]
    i = EIGHT_DIRECTIONS_EN.index(answer_dir)
    left = EIGHT_DIRECTIONS_EN[(i - 1) % len(EIGHT_DIRECTIONS_EN)]
    right = EIGHT_DIRECTIONS_EN[(i + 1) % len(EIGHT_DIRECTIONS_EN)]
    return [d for d in EIGHT_DIRECTIONS_EN if d not in {answer_dir, left, right}]


def _convert_layout_to_objects(layout_info: list[dict]) -> list[dict]:
    raw = []
    for idx, item in enumerate(layout_info):
        if not isinstance(item, dict):
            continue
        if "id" not in item:
            continue
        bbox = item.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 2:
            continue
        try:
            cx = float(bbox[0])
            cy = float(bbox[1])
        except Exception:
            continue
        cat_raw = str(item.get("category", "unknown"))
        raw.append(
            {
                "id": str(item["id"]),
                "instance_index": int(idx),
                "category": _norm_category(cat_raw) or "unknown",
                "category_raw": cat_raw,
                "center_xy": [cx, cy],
            }
        )

    return sorted(raw, key=lambda x: _safe_int(x["id"]))


def _choose_category(objects: list[dict], pref: list[str]) -> str:
    cats = sorted({o["category"] for o in objects})
    if not cats:
        return "object"
    for p in pref:
        matched = [c for c in cats if _category_matches(c, p)]
        if matched:
            return matched[0]
    return random.choice(cats)


def _objects_in_category(objects: list[dict], category: str, exact: bool = True) -> list[dict]:
    if exact:
        return [o for o in objects if _category_equal(o["category"], category)]
    return [o for o in objects if _category_matches(o["category"], category)]


def _category_counts(objects: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in objects:
        c = str(o.get("category", "")).strip()
        if not c:
            continue
        counts[c] = counts.get(c, 0) + 1
    return counts


def _unique_category_objects(objects: list[dict]) -> list[dict]:
    counts = _category_counts(objects)
    return [o for o in objects if counts.get(o["category"], 0) == 1]


def _semantic_confusable(cat_a: str, cat_b: str) -> bool:
    a = _norm_category(cat_a)
    b = _norm_category(cat_b)
    if not a or not b or a == b:
        return False
    if a in b or b in a:
        return True
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return False
    return ta.issubset(tb) or tb.issubset(ta)


def _unique_semantic_objects(objects: list[dict]) -> list[dict]:
    unique_exact = _unique_category_objects(objects)
    all_cats = [o["category"] for o in objects]
    out = []
    for o in unique_exact:
        c = o["category"]
        if any((_semantic_confusable(c, oc) for oc in all_cats if oc != c)):
            continue
        out.append(o)
    return out


def _category_confusable_in_scene(category: str, all_categories: list[str]) -> bool:
    c = _norm_category(category)
    if not c:
        return True
    return any(_semantic_confusable(c, oc) for oc in all_categories if _norm_category(oc) != c)


def _objects_with_unambiguous_semantic(objects: list[dict]) -> list[dict]:
    all_cats = [o["category"] for o in objects]
    return [o for o in objects if not _category_confusable_in_scene(o["category"], all_cats)]


def _choose_unique_object(objects: list[dict], pref: list[str]) -> dict | None:
    unique_objs = _unique_category_objects(objects)
    if not unique_objs:
        return None
    for p in pref:
        for o in unique_objs:
            if _category_matches(o["category"], p):
                return o
    return random.choice(unique_objs)


def _sample_distractors(objects: list[dict], target_obj: dict, k: int, forbid_ids: set[str] | None = None) -> list[dict]:
    forbid_ids = forbid_ids or set()
    pool = [o for o in objects if o["id"] != target_obj["id"] and o["id"] not in forbid_ids]
    if len(pool) < k - 1:
        return []
    return random.sample(pool, k - 1)


def _distance_ref_candidates(unique_objs: list[dict]) -> list[dict]:
    """
    Prioritize preferred reference categories, then fall back to the rest.
    """
    ordered: list[dict] = []
    used_ids: set[str] = set()
    for pref_cat in DISTANCE_REF_PREF_CATEGORIES:
        matched = [o for o in unique_objs if _category_matches(o["category"], pref_cat) and o["id"] not in used_ids]
        random.shuffle(matched)
        for o in matched:
            used_ids.add(o["id"])
            ordered.append(o)
    leftovers = [o for o in unique_objs if o["id"] not in used_ids]
    random.shuffle(leftovers)
    ordered.extend(leftovers)
    return ordered


def _distance_extreme_margin(sorted_cands: list[tuple[float, dict]], mode: str) -> float:
    """
    Compute normalized gap between best and second-best extreme candidates.
    """
    if len(sorted_cands) < 2:
        return 0.0

    best_dist = float(sorted_cands[0][0])
    second_dist = float(sorted_cands[1][0])
    if mode == "nearest":
        # sorted ascending: best_dist <= second_dist
        return (second_dist - best_dist) / max(second_dist, 1e-6)
    # sorted descending: best_dist >= second_dist
    return (best_dist - second_dist) / max(best_dist, 1e-6)


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_local_label_map(label_object_ids: list[str]) -> dict[str, int]:
    # Stable per-QA numbering: independent and sorted by original object id.
    ids = sorted(_dedup_keep_order(label_object_ids), key=_safe_int)
    return {oid: i + 1 for i, oid in enumerate(ids)}


def _label_dict_for_metadata(label_map: dict[str, int], objects_by_id: dict[str, dict]) -> dict[str, dict]:
    out = {}
    by_label = sorted([(lb, oid) for oid, lb in label_map.items()], key=lambda x: x[0])
    for lb, oid in by_label:
        obj = objects_by_id.get(oid)
        if obj is None:
            continue
        out[str(lb)] = {
            "object_id": oid,
            "category": obj["category"],
            "instance_index": obj["instance_index"],
        }
    return out


def _qa_counting(objects: list[dict]) -> dict | None:
    counts = _category_counts(objects)
    all_cats = list(counts.keys())
    repeated_cats = {c for c, n in counts.items() if n >= 2 and (not _category_confusable_in_scene(c, all_cats))}
    candidate_pool = [o for o in objects if o["category"] in repeated_cats] if repeated_cats else list(objects)
    if not repeated_cats:
        candidate_pool = _objects_with_unambiguous_semantic(objects)
    if not candidate_pool:
        candidate_pool = list(objects)
    if not candidate_pool:
        return None

    target_cat = _choose_category(candidate_pool, COUNT_PREF_CATEGORIES)
    targets = sorted(_objects_in_category(objects, target_cat, exact=True), key=lambda x: _safe_int(x["id"]))
    if not targets:
        return None
    label_ids = [o["id"] for o in targets]
    return {
        "task_type": "counting",
        "question": f"How many {target_cat} objects are in the room? Answer with only one number.",
        "answer": str(len(targets)),
        "answer_text": str(len(targets)),
        "target_category": target_cat,
        "label_object_ids": label_ids,
    }


def _qa_identify_object1(objects: list[dict], choices: int) -> dict | None:
    if not objects:
        return None
    choices = max(2, int(choices))
    target = random.choice(objects)
    target_cat = target["category"]
    scene_cats = sorted({o["category"] for o in objects if o["category"] != target_cat})
    if len(scene_cats) < choices - 1:
        for c in QA_DISTRACTOR_EXTRA_POOL:
            if c != target_cat and c not in scene_cats:
                scene_cats.append(c)
            if len(scene_cats) >= choices - 1:
                break
    if len(scene_cats) < choices - 1:
        return None
    distractors = random.sample(scene_cats, choices - 1)
    options = [target_cat] + distractors
    random.shuffle(options)
    answer_idx = options.index(target_cat)
    choice_text, answer_label = _build_mcq_text(options, answer_idx)
    label_ids = [target["id"]]
    return {
        "task_type": "object_identification_name",
        "question": "What is object 1?\n" + choice_text,
        "answer": answer_label,
        "answer_text": target_cat,
        "options": options,
        "target_object_id": target["id"],
        "label_object_ids": label_ids,
    }


def _qa_identify_mcq(objects: list[dict], choices: int) -> dict | None:
    if len(objects) < choices:
        return None

    unique_objs = _unique_semantic_objects(objects)
    if not unique_objs:
        return None

    pair_cat = _choose_category(unique_objs, IDENTIFY_PAIR_PREF_CATEGORIES)
    candidates = _objects_in_category(unique_objs, pair_cat, exact=True)
    if len(candidates) != 1:
        return None
    target = random.choice(candidates)
    distractor_pool = [
        o
        for o in objects
        if o["id"] != target["id"] and (not _semantic_confusable(o["category"], target["category"]))
    ]
    if len(distractor_pool) < choices - 1:
        return None
    distractors = random.sample(distractor_pool, choices - 1)

    option_objs = [target] + distractors
    random.shuffle(option_objs)

    option_ids = [o["id"] for o in option_objs]
    label_map = _build_local_label_map(option_ids)
    rendered_options = [str(label_map[oid]) for oid in option_ids]
    answer_idx = option_ids.index(target["id"])
    answer_number = rendered_options[answer_idx]

    if _category_matches(pair_cat, "shoe") or _category_matches(pair_cat, "slipper"):
        prompt = "Which numbered object is a pair of slippers?"
    else:
        prompt = f"Which numbered object is a {pair_cat}?"

    return {
        "task_type": "object_identification_mcq",
        "question": prompt + " Answer with only one number.",
        "answer": answer_number,
        "answer_text": answer_number,
        "target_category": pair_cat,
        "target_object_id": target["id"],
        "label_object_ids": option_ids,
    }


def _qa_relative_side(objects: list[dict]) -> dict | None:
    unique_objs = _unique_semantic_objects(objects)
    if len(unique_objs) < 2:
        return None
    pairs = []
    for a in unique_objs:
        for b in unique_objs:
            if a["id"] == b["id"]:
                continue
            direction, margin = _direction8_from_ref_to_obj(a, b)
            dist = _xy_dist(a, b)
            pairs.append((margin, dist, a, b, direction))
    if not pairs:
        return None
    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    margin, _, target, ref, direction = pairs[0]
    if float(margin) < 0.2:
        return None
    cat_counts = _category_counts(objects)
    if cat_counts.get(target["category"], 0) != 1 or cat_counts.get(ref["category"], 0) != 1:
        return None

    option_pool = _direction_option_pool(direction)
    if len(option_pool) < 3:
        return None
    options = [direction] + random.sample(option_pool, 3)
    random.shuffle(options)
    answer_idx = options.index(direction)
    choice_text, answer_label = _build_mcq_text(options, answer_idx)

    return {
        "task_type": "relative_position_side",
        "question": f"Which side of the {ref['category']} is the {target['category']} on?\n{choice_text}",
        "answer": answer_label,
        "answer_text": direction,
        "target_object_id": target["id"],
        "reference_object_id": ref["id"],
        "target_semantic": target["category"],
        "reference_semantic": ref["category"],
        "direction_confidence_margin": round(float(margin), 4),
        "label_object_ids": [],
    }


def _rewrite_relative_side_with_top_payload(item: dict, objects_by_id: dict[str, dict], top_payload: dict) -> None:
    if item.get("task_type") != "relative_position_side":
        return
    tid = str(item.get("target_object_id", ""))
    rid = str(item.get("reference_object_id", ""))
    target = objects_by_id.get(tid)
    ref = objects_by_id.get(rid)
    if target is None or ref is None:
        return
    world_north, world_east = _get_world_basis(top_payload, objects_by_id)
    direction, margin = _direction8_from_world_basis(target, ref, world_north, world_east)
    option_pool = _direction_option_pool(direction)
    if len(option_pool) < 3:
        return
    options = [direction] + random.sample(option_pool, 3)
    random.shuffle(options)
    answer_idx = options.index(direction)
    choice_text, answer_label = _build_mcq_text(options, answer_idx)
    item["question"] = f"Which side of the {ref['category']} is the {target['category']} on?\n{choice_text}"
    item["answer"] = answer_label
    item["answer_text"] = direction
    item["direction_confidence_margin"] = round(float(margin), 4)


def _rewrite_relative_direction_with_top_payload(
    item: dict,
    objects_by_id: dict[str, dict],
    top_payload: dict,
    label_object_ids: list[str],
    category_counts: dict[str, int],
) -> bool:
    if item.get("task_type") != "relative_position_direction_mcq":
        return True
    tid = str(item.get("target_object_id", ""))
    rid = str(item.get("reference_object_id", ""))
    target = objects_by_id.get(tid)
    ref = objects_by_id.get(rid)
    if target is None or ref is None:
        return False
    ref_cat_count = int(category_counts.get(ref["category"], 0))
    # Reference semantic must be unique in the scene.
    if ref_cat_count != 1:
        return False
    world_north, world_east = _get_world_basis(top_payload, objects_by_id)
    target_dirs, target_scores = _cardinal_membership_from_world_basis(target, ref, world_north, world_east)
    if not target_dirs:
        return False

    label_ids = [str(x) for x in label_object_ids if str(x)]
    option_ids = [oid for oid in label_ids if oid != rid]
    if tid not in option_ids:
        return False
    label_map = _build_local_label_map(option_ids)
    if tid not in label_map:
        return False

    # Multi-side rule: northeast counts as both north and east.
    # Pick a direction for which the target is the unique option object on that side.
    sorted_dirs = sorted(target_dirs, key=lambda x: float(target_scores.get(x, 0.0)), reverse=True)
    direction = None
    margin = 0.0
    for cand_dir in sorted_dirs:
        same_dir_ids: list[str] = []
        for oid in option_ids:
            obj = objects_by_id.get(oid)
            if obj is None:
                continue
            dirs_o, _scores_o = _cardinal_membership_from_world_basis(obj, ref, world_north, world_east)
            if cand_dir in dirs_o:
                same_dir_ids.append(oid)
        if len(same_dir_ids) == 1 and same_dir_ids[0] == tid:
            direction = cand_dir
            margin = float(target_scores.get(cand_dir, 0.0))
            break
    if direction is None:
        return False

    ref_phrase = f"the {ref['category']}"
    answer_number = str(label_map[tid])

    item["question"] = (
        "An object can be on multiple sides at once (e.g., northwest counts as both north and west). "
        f"Which numbered object is {direction} of {ref_phrase}? Answer with only one number."
    )
    item["answer"] = answer_number
    item["answer_text"] = answer_number
    item["direction"] = direction
    item["direction_confidence_margin"] = round(float(margin), 4)
    item.pop("options", None)
    return True


def _qa_relative_direction_mcq(objects: list[dict], choices: int) -> dict | None:
    if len(objects) < choices + 1:
        return None

    safe_objects = _objects_with_unambiguous_semantic(objects)
    if len(safe_objects) >= choices + 1:
        objects = safe_objects

    category_counts = _category_counts(objects)
    preferred_refs: list[dict] = []
    for cat in RELATIVE_REF_PREF_CATEGORIES:
        for obj in objects:
            if (
                _category_matches(obj["category"], cat)
                and _is_good_spatial_ref_category(obj["category"])
                and int(category_counts.get(obj["category"], 0)) == 1
            ):
                preferred_refs.append(obj)
    preferred_ids = {o["id"] for o in preferred_refs}
    fallback_refs = [
        o
        for o in objects
        if (
            o["id"] not in preferred_ids
            and _is_good_spatial_ref_category(o["category"])
            and int(category_counts.get(o["category"], 0)) == 1
        )
    ]
    candidate_refs = preferred_refs + fallback_refs
    if not candidate_refs:
        return None

    by_direction: dict[str, list[tuple[float, dict, dict, float]]] = {d: [] for d in CARDINAL_OPTIONS_EN}
    for ref in candidate_refs:
        best_by_dir: dict[str, tuple[float, float, dict]] = {}
        for o in objects:
            if o["id"] == ref["id"]:
                continue
            dirs, scores = _cardinal_membership_from_ref_to_obj(o, ref)
            if not dirs:
                continue
            dx = float(o["center_xy"][0] - ref["center_xy"][0])
            dy = float(o["center_xy"][1] - ref["center_xy"][1])
            for direction in dirs:
                margin = float(scores.get(direction, 0.0))
                axis_dist = abs(dx) if direction in {"east", "west"} else abs(dy)
                old = best_by_dir.get(direction)
                if old is None or (margin, axis_dist) > (old[0], old[1]):
                    best_by_dir[direction] = (margin, axis_dist, o)
        for direction, (margin, axis_dist, target) in best_by_dir.items():
            strength = float(margin) * 2.0 + float(axis_dist)
            by_direction[direction].append((strength, ref, target, float(margin)))

    available_dirs = [d for d, vals in by_direction.items() if vals]
    if not available_dirs:
        return None

    direction = random.choice(available_dirs)
    by_direction[direction].sort(key=lambda x: x[0], reverse=True)
    top = by_direction[direction][: min(5, len(by_direction[direction]))]
    _, ref, target, margin = random.choice(top)
    distractor_pool = [
        o
        for o in objects
        if o["id"] not in {target["id"], ref["id"]}
        and (not _semantic_confusable(o["category"], target["category"]))
        and (not _semantic_confusable(o["category"], ref["category"]))
        and (direction not in _cardinal_membership_from_ref_to_obj(o, ref)[0])
    ]
    if len(distractor_pool) < choices - 1:
        return None
    distractors = random.sample(distractor_pool, choices - 1)

    option_objs = [target] + distractors
    random.shuffle(option_objs)
    option_ids = [o["id"] for o in option_objs]

    label_ids = option_ids
    label_map = _build_local_label_map(label_ids)

    rendered_options = [str(label_map[oid]) for oid in option_ids]
    answer_idx = option_ids.index(target["id"])
    ref_phrase = f"the {ref['category']}"
    answer_number = rendered_options[answer_idx]

    return {
        "task_type": "relative_position_direction_mcq",
        "question": (
            "An object can be on multiple sides at once (e.g., northwest counts as both north and west). "
            f"Which numbered object is {direction} of {ref_phrase}? Answer with only one number."
        ),
        "answer": answer_number,
        "answer_text": answer_number,
        "direction": direction,
        "target_object_id": target["id"],
        "reference_object_id": ref["id"],
        "reference_category": ref["category"],
        "direction_confidence_margin": round(float(margin), 4),
        "label_object_ids": label_ids,
    }


def _qa_nearest(objects: list[dict], choices: int) -> dict | None:
    if len(objects) < choices + 1:
        return None

    unique_objs = _unique_semantic_objects(objects)
    if len(unique_objs) < choices + 1:
        return None

    ref_candidates = _distance_ref_candidates(unique_objs)
    if not ref_candidates:
        return None

    tried = 0
    for ref in ref_candidates:
        if tried >= DISTANCE_EXTREME_MAX_REF_TRIES:
            break
        tried += 1

        cands = [(_xy_dist(o, ref), o) for o in unique_objs if o["id"] != ref["id"]]
        if len(cands) < choices:
            continue
        cands.sort(key=lambda x: x[0])

        extreme_margin = _distance_extreme_margin(cands, mode="nearest")
        if extreme_margin < DISTANCE_EXTREME_MIN_MARGIN:
            continue

        target = cands[0][1]
        distractor_pool = [o for _, o in cands[1:] if o["id"] != target["id"]]
        if len(distractor_pool) < choices - 1:
            continue
        distractors = random.sample(distractor_pool, choices - 1)

        option_objs = [target] + distractors
        random.shuffle(option_objs)
        option_ids = [o["id"] for o in option_objs]

        label_ids = option_ids
        label_map = _build_local_label_map(label_ids)
        rendered_options = [str(label_map[oid]) for oid in option_ids]
        answer_idx = option_ids.index(target["id"])
        answer_number = rendered_options[answer_idx]

        return {
            "task_type": "distance_nearest",
            "question": f"Which numbered object is closest to the {ref['category']}? Answer with only one number.",
            "answer": answer_number,
            "answer_text": answer_number,
            "target_object_id": target["id"],
            "reference_object_id": ref["id"],
            "reference_category": ref["category"],
            "reference_semantic": ref["category"],
            "distance_extreme_margin": round(float(extreme_margin), 4),
            "label_object_ids": label_ids,
        }

    return None


def _qa_farthest(objects: list[dict], choices: int) -> dict | None:
    if len(objects) < choices + 1:
        return None

    unique_objs = _unique_semantic_objects(objects)
    if len(unique_objs) < choices + 1:
        return None

    ref_candidates = _distance_ref_candidates(unique_objs)
    if not ref_candidates:
        return None

    tried = 0
    for ref in ref_candidates:
        if tried >= DISTANCE_EXTREME_MAX_REF_TRIES:
            break
        tried += 1

        cands = [(_xy_dist(o, ref), o) for o in unique_objs if o["id"] != ref["id"]]
        if len(cands) < choices:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        extreme_margin = _distance_extreme_margin(cands, mode="farthest")
        if extreme_margin < DISTANCE_EXTREME_MIN_MARGIN:
            continue

        target = cands[0][1]
        distractor_pool = [o for _, o in cands[1:] if o["id"] != target["id"]]
        if len(distractor_pool) < choices - 1:
            continue
        distractors = random.sample(distractor_pool, choices - 1)

        option_objs = [target] + distractors
        random.shuffle(option_objs)
        option_ids = [o["id"] for o in option_objs]

        label_ids = option_ids
        label_map = _build_local_label_map(label_ids)
        rendered_options = [str(label_map[oid]) for oid in option_ids]
        answer_idx = option_ids.index(target["id"])
        answer_number = rendered_options[answer_idx]

        return {
            "task_type": "distance_farthest",
            "question": f"Which numbered object is farthest from the {ref['category']}? Answer with only one number.",
            "answer": answer_number,
            "answer_text": answer_number,
            "target_object_id": target["id"],
            "reference_object_id": ref["id"],
            "reference_category": ref["category"],
            "reference_semantic": ref["category"],
            "distance_extreme_margin": round(float(extreme_margin), 4),
            "label_object_ids": label_ids,
        }

    return None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolved_render_output_path(out_image: Path, view_type: str) -> Path:
    # Shared renderer now respects explicit OUT_PATH for single CAMERA_MODE.
    # Keep requested filename stable (e.g. raw_views/isometric.png).
    _ = view_type
    suffix = out_image.suffix if out_image.suffix else ".png"
    return out_image.with_suffix(suffix)


def _render_canonical_view(
    glb_path: Path,
    out_image: Path,
    view_type: str,
    force_rerender: bool = False,
    wall_alpha: float = 1.0,
    keep_box_json: bool = False,
    north_world_xy: tuple[float, float] | None = None,
) -> tuple[Path, dict]:
    out_image.parent.mkdir(parents=True, exist_ok=True)
    actual_out = _resolved_render_output_path(out_image, view_type)
    box_json = out_image.with_name(f".{out_image.stem}.{view_type}.boxes.json")

    if (not force_rerender) and actual_out.exists() and box_json.exists():
        return actual_out, _load_json(box_json)

    if not BLENDER_BIN.exists():
        raise FileNotFoundError(f"Blender not found: {BLENDER_BIN}")
    if not RENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Render script not found: {RENDER_SCRIPT}")

    env = os.environ.copy()
    env["GLB_PATH"] = str(glb_path)
    env["OUT_PATH"] = str(out_image)
    env["CAMERA_MODE"] = view_type
    env["BOX_JSON_PATH"] = str(box_json)
    env["SKIP_RENDER"] = "0"
    # Keep canonical north policy consistent with task1_spatial_relation.
    env["NORTH_POLICY"] = "top_up"
    env["NORTH_SCREEN_MODE"] = "camera_basis"
    env["SETUP_CAMERA_NORTH_BIND"] = "0"
    env["TOP_DISTANCE_SCALE"] = str(float(CAMERA_TOP_DIST_SCALE))
    env["ISOMETRIC_DISTANCE_SCALE"] = str(float(CAMERA_ISO_DIST_SCALE))
    env["CAMERA_FIT_MARGIN"] = str(float(CAMERA_FIT_MARGIN))
    env["CAMERA_FIT_SAFETY"] = str(float(CAMERA_SAFETY_SCALE))
    env["WALL_ALPHA"] = str(float(wall_alpha))
    env["TOP_WALL_ALPHA"] = str(float(TOP_WALL_ALPHA))
    env["ISO_WALL_ALPHA"] = str(float(ISO_WALL_ALPHA))

    cmd = [str(BLENDER_BIN), "--background", "--python", str(RENDER_SCRIPT)]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    payload = _load_json(box_json)
    if (not keep_box_json) and box_json.exists():
        try:
            box_json.unlink()
        except Exception:
            pass
    return actual_out, payload


def _load_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _center_from_payload(payload: dict, instance_index: int) -> tuple[int, int] | None:
    k = str(instance_index)
    centers = payload.get("centers_by_instance_index", {}) if isinstance(payload, dict) else {}
    boxes = payload.get("boxes_by_instance_index", {}) if isinstance(payload, dict) else {}

    c = centers.get(k)
    if isinstance(c, list) and len(c) == 2:
        try:
            return int(c[0]), int(c[1])
        except Exception:
            pass

    b = boxes.get(k)
    if isinstance(b, list) and len(b) == 4:
        try:
            x1, y1, x2, y2 = [int(v) for v in b]
            return (x1 + x2) // 2, (y1 + y2) // 2
        except Exception:
            pass
    return None


def _annotate_numbers(
    base_image: Path,
    out_image: Path,
    payload: dict,
    objects_by_id: dict[str, dict],
    label_map: dict[str, int],
) -> str:
    out_image.parent.mkdir(parents=True, exist_ok=True)

    if not base_image.exists():
        return str(base_image.resolve())

    if not PIL_AVAILABLE:
        shutil.copy2(base_image, out_image)
        return str(out_image.resolve())

    img = Image.open(base_image).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _load_font(LABEL_FONT_SIZE)

    for oid, label in label_map.items():
        obj = objects_by_id.get(oid)
        if obj is None:
            continue
        center = _center_from_payload(payload, int(obj["instance_index"]))
        if center is None:
            continue
        cx, cy = center
        cx = max(16, min(img.width - 16, cx))
        cy = max(16, min(img.height - 16, cy))
        r = LABEL_RADIUS
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 60), outline=(255, 255, 255), width=2)
        txt = str(label)
        try:
            draw.text((cx, cy), txt, fill=(255, 255, 255), font=font, anchor="mm")
        except TypeError:
            bb = draw.textbbox((0, 0), txt, font=font)
            draw.text((cx - (bb[2] - bb[0]) // 2, cy - (bb[3] - bb[1]) // 2), txt, fill=(255, 255, 255), font=font)

    nv = payload.get("north_screen_vector", [0.0, -1.0]) if isinstance(payload, dict) else [0.0, -1.0]
    try:
        north_vec = (float(nv[0]), float(nv[1]))
    except Exception:
        north_vec = (0.0, -1.0)

    px_per_unit = 70.0
    try:
        ppu = float(payload.get("pixels_per_meter", 70.0)) if isinstance(payload, dict) else 70.0
        if ppu > 1e-6:
            px_per_unit = ppu
    except Exception:
        pass

    stem_lower = base_image.stem.lower()
    scale_mult = LEFT_STRIP_UI_SCALE_MULT_TOP if stem_lower == "top" else LEFT_STRIP_UI_SCALE_MULT
    img = _compose_with_left_strip_scaled(
        img,
        north_vec,
        px_per_unit,
        ui_scale_mult=scale_mult,
    )
    img.save(out_image)
    return str(out_image.resolve())


def _compose_with_left_strip_scaled(
    img,
    north_vec: tuple[float, float],
    px_per_unit: float,
    ui_scale_mult: float = 1.0,
    model_pad: int = 6,
    gap: int = 6,
):
    """
    Task-local variant of left-strip compose with an adjustable UI scale multiplier,
    so N arrow and 1m marker can be enlarged without affecting other tasks.
    """
    if not PIL_AVAILABLE:
        return img

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        bbox = (0, 0, w, h)
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - model_pad)
    y1 = max(0, y1 - model_pad)
    x2 = min(w, x2 + model_pad)
    y2 = min(h, y2 + model_pad)
    cropped = img.crop((x1, y1, x2, y2))
    cw, ch = cropped.size

    ui_scale_base = max(0.58, min(1.0, ch / 460.0))
    ui_scale = ui_scale_base * max(0.8, float(ui_scale_mult))
    strip_w = int(round(190 * ui_scale))
    bar_px = int(round(max(10.0, (px_per_unit if px_per_unit > 0 else 70.0))))
    strip_w = max(strip_w, int(round(52 * ui_scale)) + bar_px)
    strip_h = int(round((14 + 70 + 8 + 48 + 14) * ui_scale))
    real_gap = max(4, int(round(gap * ui_scale)))
    out_w = strip_w + real_gap + cw
    out_h = max(ch, strip_h)
    canvas = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    cx = strip_w + real_gap
    cy = (out_h - ch) // 2
    canvas.paste(cropped, (cx, cy), cropped)

    draw = ImageDraw.Draw(canvas)
    strip_y = (out_h - strip_h) // 2
    dc_utils.draw_standard_north_scale(
        draw,
        out_w,
        out_h,
        north_vec,
        px_per_unit,
        origin_x=0,
        origin_y=strip_y,
        ui_scale=ui_scale,
        base_img=canvas,
    )
    return canvas


def _object_bbox_area(payload: dict, instance_index: int) -> float:
    boxes = payload.get("boxes_by_instance_index", {}) if isinstance(payload, dict) else {}
    b = boxes.get(str(instance_index))
    if not isinstance(b, list) or len(b) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except Exception:
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _qa_focus_object_ids(qa: dict) -> list[str]:
    ids = []
    for k in ["target_object_id", "reference_object_id"]:
        v = qa.get(k)
        if isinstance(v, str) and v:
            ids.append(v)
    return _dedup_keep_order(ids)


def _pick_best_iso_mode(
    qa: dict,
    objects_by_id: dict[str, dict],
    iso_payloads: dict[str, dict],
) -> tuple[str | None, bool]:
    focus_ids = _qa_focus_object_ids(qa)
    if not iso_payloads:
        return None, False
    if not focus_ids:
        first_mode = sorted(iso_payloads.keys())[0]
        return first_mode, True

    scored = []
    for mode, payload in iso_payloads.items():
        total = 0.0
        miss = 0
        for oid in focus_ids:
            obj = objects_by_id.get(oid)
            if obj is None:
                miss += 1
                continue
            area = _object_bbox_area(payload, int(obj["instance_index"]))
            if area < VIS_MIN_AREA_PX:
                miss += 1
            total += area
        scored.append((miss, total, mode))

    scored.sort(key=lambda x: (x[0], -x[1], x[2]))
    best_miss, _, best_mode = scored[0]
    return best_mode, (best_miss == 0)


def _all_labels_renderable(
    label_map: dict[str, int],
    payload: dict,
    objects_by_id: dict[str, dict],
) -> bool:
    for oid in label_map.keys():
        obj = objects_by_id.get(oid)
        if obj is None:
            return False
        center = _center_from_payload(payload, int(obj["instance_index"]))
        if center is None:
            return False
    return True


def _build_scene_qa_objects(objects: list[dict], choices: int) -> list[dict]:
    qa_items = []

    for builder in [
        lambda: _qa_counting(objects),
        lambda: _qa_identify_object1(objects, choices),
        lambda: _qa_identify_mcq(objects, choices),
        lambda: _qa_relative_side(objects),
        lambda: _qa_relative_direction_mcq(objects, choices),
        lambda: _qa_nearest(objects, choices),
        lambda: _qa_farthest(objects, choices),
    ]:
        q = builder()
        if q is not None:
            qa_items.append(q)

    return qa_items


def build_scene_sample(
    scene_name: str,
    scene_payload: dict,
    glb_dir: Path,
    sample_dir: Path,
    choices: int,
    min_objects: int,
    keep_box_json: bool,
    force_rerender: bool,
) -> dict | None:
    layout_info = scene_payload.get("layout_info", [])
    if not isinstance(layout_info, list):
        return None

    objects = _convert_layout_to_objects(layout_info)
    if len(objects) < min_objects:
        return None

    qa_objects = _filter_objects_for_qa(objects)
    if len(qa_objects) < min_objects:
        return None

    objects_by_id = {o["id"]: o for o in objects}
    qa_items = _build_scene_qa_objects(qa_objects, choices)

    glb_path = glb_dir / scene_name
    if not glb_path.exists():
        return None

    raw_dir = sample_dir / "raw_views"
    qa_img_dir = sample_dir / "qa_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    qa_img_dir.mkdir(parents=True, exist_ok=True)

    top_raw = raw_dir / "top.png"
    iso_modes = [FIXED_ISOMETRIC_MODE]
    iso_req_paths = {
        FIXED_ISOMETRIC_MODE: raw_dir / "isometric.png",
    }

    try:
        top_raw, top_payload = _render_canonical_view(
            glb_path,
            top_raw,
            "top",
            force_rerender=force_rerender,
            wall_alpha=TOP_WALL_ALPHA,
            keep_box_json=keep_box_json,
        )
        iso_raws: dict[str, Path] = {}
        iso_payloads: dict[str, dict] = {}
        for mode in iso_modes:
            rp = iso_req_paths[mode]
            img_p, payload = _render_canonical_view(
                glb_path,
                rp,
                mode,
                force_rerender=force_rerender,
                wall_alpha=ISO_WALL_ALPHA,
                keep_box_json=keep_box_json,
            )
            if payload:
                iso_raws[mode] = img_p
                iso_payloads[mode] = payload
    except Exception as e:
        print(f"[WARN] render failed for {scene_name}: {e}")
        return None

    if not top_payload or not iso_payloads:
        print(f"[WARN] payload missing for {scene_name}")
        return None

    qa_out = []
    category_counts = _category_counts(objects)

    for qi, qa in enumerate(qa_items, start=1):
        label_ids = qa.get("label_object_ids", [])
        if not isinstance(label_ids, list):
            label_ids = []
        label_ids = [str(x) for x in label_ids]

        item = dict(qa)
        item.pop("label_object_ids", None)
        _rewrite_relative_side_with_top_payload(item, objects_by_id, top_payload)
        if not _rewrite_relative_direction_with_top_payload(
            item, objects_by_id, top_payload, label_ids, category_counts
        ):
            continue

        best_iso_mode, vis_ok = _pick_best_iso_mode(item, objects_by_id, iso_payloads)
        if best_iso_mode is None:
            continue
        if _qa_focus_object_ids(item) and (not vis_ok):
            continue

        best_iso_raw = iso_raws[best_iso_mode]
        best_iso_payload = iso_payloads[best_iso_mode]
        raw_images = [str(top_raw.resolve()), str(best_iso_raw.resolve())]
        item["images"] = list(raw_images)
        if item.get("task_type") in TASK_TYPES_USE_LABELED_IMAGES:
            item["raw_images"] = list(raw_images)
        if label_ids:
            label_map = _build_local_label_map(label_ids)
            # Ensure all labels can be rendered on top view.
            # This avoids broken QA visuals like numbering starting from 2
            # because label 1 is missing in the image payload.
            if not _all_labels_renderable(label_map, top_payload, objects_by_id):
                continue
            top_ref_out = qa_img_dir / f"qa{qi:02d}_top_ref.png"
            iso_ref_out = qa_img_dir / f"qa{qi:02d}_isometric_ref.png"
            top_ref = _annotate_numbers(top_raw, top_ref_out, top_payload, objects_by_id, label_map)
            iso_ref = _annotate_numbers(best_iso_raw, iso_ref_out, best_iso_payload, objects_by_id, label_map)
            item["ref_images"] = [top_ref, iso_ref]
            item["qa_label_map"] = _label_dict_for_metadata(label_map, objects_by_id)
            if item.get("task_type") in TASK_TYPES_USE_LABELED_IMAGES:
                item["images"] = [top_ref, iso_ref]
        else:
            # For relative-position-side QA, use images with north indicator (no labels).
            if item.get("task_type") == "relative_position_side":
                top_ref_out = qa_img_dir / f"qa{qi:02d}_top_ref.png"
                iso_ref_out = qa_img_dir / f"qa{qi:02d}_isometric_ref.png"
                top_ref = _annotate_numbers(top_raw, top_ref_out, top_payload, objects_by_id, {})
                iso_ref = _annotate_numbers(best_iso_raw, iso_ref_out, best_iso_payload, objects_by_id, {})
                item["images"] = [top_ref, iso_ref]
                item["ref_images"] = [top_ref, iso_ref]
            else:
                item["ref_images"] = []
            item["qa_label_map"] = {}
        if item.get("task_type") in TASK_TYPES_IMAGES_EQ_REF_IMAGES:
            item.pop("ref_images", None)
        item["selected_isometric_mode"] = best_iso_mode
        qa_out.append(item)

    keep_iso_paths = set()
    if FIXED_ISOMETRIC_MODE in iso_raws:
        keep_iso_paths.add(str(iso_raws[FIXED_ISOMETRIC_MODE].resolve()))
    for it in qa_out:
        imgs = it.get("images", [])
        if isinstance(imgs, list) and len(imgs) >= 2:
            keep_iso_paths.add(str(Path(imgs[1]).resolve()))
    for mode, p in iso_raws.items():
        ap = str(p.resolve())
        if ap in keep_iso_paths:
            continue
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    return {
        "scene_name": scene_name,
        "scene_stem": Path(scene_name).stem,
        "scene_dir": str(sample_dir.resolve()),
        "glb_path": str(glb_path.resolve()),
        "render_images": {
            "top": str(top_raw.resolve()),
            "isometric": str(iso_raws[FIXED_ISOMETRIC_MODE].resolve()) if FIXED_ISOMETRIC_MODE in iso_raws else "",
        },
        "object_count": len(objects),
        "objects": [
            {
                "object_id": o["id"],
                "instance_index": o["instance_index"],
                "category": o["category"],
                "center_xy": o["center_xy"],
            }
            for o in objects
        ],
        "qa": qa_out,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if not args.mapping_json.exists():
        raise FileNotFoundError(f"mapping json not found: {args.mapping_json}")

    with open(args.mapping_json, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f"expected dict mapping, got {type(mapping).__name__}")

    pref = _derive_pref_categories_from_stats(mapping, args.category_dist_json)
    global COUNT_PREF_CATEGORIES, IDENTIFY_PAIR_PREF_CATEGORIES, RELATIVE_REF_PREF_CATEGORIES, DISTANCE_REF_PREF_CATEGORIES
    COUNT_PREF_CATEGORIES = list(pref["count"])
    IDENTIFY_PAIR_PREF_CATEGORIES = list(pref["identify"])
    RELATIVE_REF_PREF_CATEGORIES = list(pref["relative_ref"])
    DISTANCE_REF_PREF_CATEGORIES = list(pref["distance_ref"])
    print(f"[INFO] count pref categories: {COUNT_PREF_CATEGORIES}")
    print(f"[INFO] identify pref categories: {IDENTIFY_PAIR_PREF_CATEGORIES}")
    print(f"[INFO] relative ref pref categories: {RELATIVE_REF_PREF_CATEGORIES}")
    print(f"[INFO] distance ref pref categories: {DISTANCE_REF_PREF_CATEGORIES}")

    scene_keys = sorted(mapping.keys())
    if args.scene_keyword.strip():
        kw = args.scene_keyword.strip().lower()
        scene_keys = [k for k in scene_keys if kw in k.lower()]
    if args.max_scenes > 0:
        scene_keys = scene_keys[: args.max_scenes]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for idx, scene_name in enumerate(scene_keys, start=1):
        sample_dir = output_dir / Path(scene_name).stem
        sample = build_scene_sample(
            scene_name=scene_name,
            scene_payload=mapping.get(scene_name, {}),
            glb_dir=args.glb_dir,
            sample_dir=sample_dir,
            choices=max(2, int(args.choices)),
            min_objects=max(2, int(args.min_objects)),
            keep_box_json=bool(args.keep_box_json),
            force_rerender=bool(args.force_rerender),
        )
        if sample is None:
            print(f"[{idx}/{len(scene_keys)}] skip {scene_name}")
            continue
        samples.append(sample)
        print(f"[{idx}/{len(scene_keys)}] done {scene_name}: qa={len(sample['qa'])}")

    out_json = output_dir / "metadata_indoor.json"
    metadata = {
        "task": "indoor_object_meaning",
        "source_mapping_json": str(args.mapping_json),
        "glb_dir": str(args.glb_dir),
        "seed": args.seed,
        "scene_count": len(samples),
        "samples": samples,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_json} (scene_count={len(samples)})")


if __name__ == "__main__":
    main()
