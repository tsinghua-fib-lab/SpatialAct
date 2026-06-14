#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import random
import re
import argparse
import shutil
import subprocess
import time
import concurrent.futures
from itertools import combinations

random.seed(10480)

# --------------------------------------------------------------------------------------
# Paths / Args
# --------------------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.expanduser('~/SpatialAct'))
EMBEDDED_BLENDER_BIN = os.path.join(PROJECT_ROOT, "blender-3.2.2-linux-x64", "blender")

try:
    import bpy
except ModuleNotFoundError:
    if os.path.exists(EMBEDDED_BLENDER_BIN):
        cmd = [
            EMBEDDED_BLENDER_BIN,
            "--background",
            "--python",
            os.path.abspath(__file__),
            "--",
            *sys.argv[1:],
        ]
        print(f"[Launcher] bpy not found, relaunching with embedded Blender: {EMBEDDED_BLENDER_BIN}")
        raise SystemExit(subprocess.call(cmd))
    raise ModuleNotFoundError(
        "No module named 'bpy'. Embedded blender not found at "
        f"{EMBEDDED_BLENDER_BIN}. Please verify blender path."
    )

from mathutils import Vector, Matrix
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "blender_scripts"))

from src.building_labels import (
    BuildingLabeler,
    is_building as check_is_building,
    is_road,
    apply_road_material,
    create_road_material,
)

parser = argparse.ArgumentParser(description=" QA Blender")
parser.add_argument("--region", type=str, default=None, help="Region name, e.g., region_a, region_b")
parser.add_argument("--mode", type=str, default="basic")
parser.add_argument("--min_region", type=int, default=0)
parser.add_argument("--max_regions", type=int, default=0)
parser.add_argument("--workers", type=int, default=1)


argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = argv[1:]
args, _ = parser.parse_known_args(argv)

def _normalize_region_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _infer_mode(mode_arg: str, region_name: str) -> str:
    if mode_arg:
        m = mode_arg.strip().lower()
        if m in {"basic", "complex"}:
            return m
    if region_name.endswith("_complex"):
        return "complex"
    return "basic"


def _canonical_region_name(region_name: str, mode: str) -> str:
    name = _normalize_region_name(region_name)
    if mode == "complex":
        return name if name.endswith("_complex") else f"{name}_complex"
    if mode == "basic" and name.endswith("_complex"):
        return name[:-8]
    return name


RAW_REGION_NAME = args.region if args.region else os.environ.get("REGION_NAME", "region")
INPUT_MODE = _infer_mode(args.mode if args.mode else os.environ.get("MODE", ""), _normalize_region_name(RAW_REGION_NAME))
REGION_NAME = _canonical_region_name(RAW_REGION_NAME, INPUT_MODE)
MIN_REGION = args.min_region
MAX_REGIONS = args.max_regions
WORKERS = max(1, int(args.workers))

def _first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0] if candidates else ""


def resolve_input_paths(mode: str, region_name: str):
    default_region_dir = os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{region_name}_kmeans")

    if mode == "complex":
        complex_result_dir = os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{region_name}")
        base_region_name = region_name[:-8] if region_name.endswith("_complex") else region_name
        base_region_dir = os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{base_region_name}_kmeans")
        combined_region_partition_dir = os.path.join(PROJECT_ROOT, "shared_outputs/combined/region_partition")
        region_dir = os.environ.get("REGION_DIR", base_region_dir)

        default_blend_path = _first_existing_path([
            os.path.join(complex_result_dir, "osm_reference_clean.blend"),
            os.path.join(complex_result_dir, "osm_reference.blend"),
            os.path.join(PROJECT_ROOT, f"osm_scene_0228/{base_region_name}_osm_scene_0228/osm_reference_clean.blend"),
            os.path.join(PROJECT_ROOT, f"osm_scene_0228/{base_region_name}_osm_scene_0228/osm_reference.blend"),
        ])
        default_region_data_path = _first_existing_path([
            os.path.join(complex_result_dir, "region_data_clean.json"),
            os.path.join(region_dir, "region_data_clean.json"),
        ])
        default_building_region_map_path = _first_existing_path([
            os.path.join(combined_region_partition_dir, "building_region_map.json"),
            os.path.join(region_dir, "building_region_map.json"),
        ])
    else:
        region_dir = os.environ.get("REGION_DIR", default_region_dir)
        default_blend_path = _first_existing_path([
            os.path.join(PROJECT_ROOT, f"osm_scene_0228/{region_name}_osm_scene_0228/osm_reference_clean.blend"),
            os.path.join(PROJECT_ROOT, f"osm_scene_0228/{region_name}_osm_scene_0228/osm_reference.blend"),
        ])
        default_region_data_path = _first_existing_path([
            os.path.join(region_dir, "region_data_clean.json"),
            os.path.join(region_dir, "region_data.json"),
        ])
        default_building_region_map_path = os.path.join(region_dir, "building_region_map.json")

    return {
        "region_dir": region_dir,
        "blend_path": os.environ.get("BLEND_PATH", default_blend_path),
        "region_data_path": os.environ.get("REGION_DATA_PATH", default_region_data_path),
        "building_region_map_path": os.environ.get("BUILDING_REGION_MAP_PATH", default_building_region_map_path),
    }


_input_paths = resolve_input_paths(INPUT_MODE, REGION_NAME)
REGION_DIR = _input_paths["region_dir"]  # legacy alias: metadata directory
BLEND_PATH = _input_paths["blend_path"]
REGION_DATA_PATH = _input_paths["region_data_path"]
BUILDING_REGION_MAP_PATH = _input_paths["building_region_map_path"]
KNOWN_BASIC_REGIONS = {"region_a", "region_b", "region_c"}

if os.path.exists(BLEND_PATH):
    print(f"[INFO] Using blend file: {BLEND_PATH}")
else:
    print(f"[WARN] Blend file does not exist (will fail later unless overridden): {BLEND_PATH}")

if INPUT_MODE == "basic" and REGION_NAME not in KNOWN_BASIC_REGIONS:
    print(f"[WARN] Region '{REGION_NAME}' is not in known basic set: {sorted(KNOWN_BASIC_REGIONS)}")

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "benchmark/data/error_mode_clean"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{REGION_NAME}_regions")


def validate_required_inputs():
    missing = []
    for key, path in {
        "BLEND_PATH": BLEND_PATH,
        "REGION_DATA_PATH": REGION_DATA_PATH,
        "BUILDING_REGION_MAP_PATH": BUILDING_REGION_MAP_PATH,
    }.items():
        if not path or not os.path.exists(path):
            missing.append(f"{key} -> {path}")

    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n  - " + "\n  - ".join(missing)
        )

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
RESOLUTION = int(os.environ.get("RESOLUTION", "0"))  # <=0 keep scene/default resolution

# Anomaly types
ISSUE_OVERLAP = "overlap"
ISSUE_ANGLE = "orientation"
ISSUE_ROAD = "road_conflict"
ISSUE_SCALE = "scale"
TARGET_QA_ANOMALY_TYPES = [
    "overlap_by_move",
    "overlap_by_rotate",
    "overlap_by_scale",
    "road_conflict_by_move",
    "road_conflict_by_rotate",
    "road_conflict_by_scale",
    "orientation",
]


MOVE_DIRS = ["North", "South", "East", "West", "NE", "NW", "SE", "SW"]
MOVE_DISTS_M = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
ANGLE_ROTATE_CHOICES_DEG = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100, 110, 120, 135, 150]
ORIENTATION_ROTATE_CHOICES_DEG = [30, 35, 40, 45, 50, 55, 60, 120, 125, 130, 135, 140, 145, 150]
SCALE_PERCENT_CHOICES = [10, 15, 20, 25, 30, 35, 40, 50]

# Unit conversion
UNIT_SCALE = 20.0  # 1 Unit in question = 20.0 units in scene
ROAD_CONFLICT_AREA_THRESHOLD = float(os.environ.get("ROAD_CONFLICT_AREA_THRESHOLD", "0.1"))
# For branches that must guarantee "no road conflict introduced", use any-overlap strictness.
STRICT_NO_ROAD_CONFLICT_AREA_THRESHOLD = float(
    os.environ.get("STRICT_NO_ROAD_CONFLICT_AREA_THRESHOLD", "0.0")
)
ORIENTATION_MAX_ROAD_DIST_M = float(
    os.environ.get("ORIENTATION_MAX_ROAD_DIST_M", "8.0" if INPUT_MODE == "complex" else "2.0")
)
ORIENTATION_NEAREST_ROAD_RADIUS = float(os.environ.get("ORIENTATION_NEAREST_ROAD_RADIUS", "500.0"))
ORIENTATION_ROAD_EDGE_SAMPLE_MAX = int(os.environ.get("ORIENTATION_ROAD_EDGE_SAMPLE_MAX", "1200"))
ORIENTATION_MAIN_ROAD_LOCAL_RADIUS = float(os.environ.get("ORIENTATION_MAIN_ROAD_LOCAL_RADIUS", "80.0"))
FRAME_CONSTRAINT_MARGIN = float(os.environ.get("FRAME_CONSTRAINT_MARGIN", "0.0"))

# Scale marker config
SCALE_MARK_LENGTH_M = 20.0  # 20 units = 1 unit display
SCALE_BAR_THICKNESS = 2
SCALE_TICK_THICKNESS = 1.0
SCALE_TICK_LEN_Y = 3.5
SCALE_MARK_MARGIN = 1.2
SCALE_MARK_Z_EPS = 0.5
SCALE_LABEL_SIZE = 10.0  # Much larger for visibility
SCALE_LABEL_Z_OFF = 2.0
SCALE_PAD_FOR_PLACEMENT = 0.25

# Label sizing
LABEL_SIZE_RATIO = 0.07
LABEL_Z_PAD_RATIO = 0.03
LABEL_COLOR = (0.02, 0.02, 0.02, 1.0)
WHITE_FILM_ALPHA = 0.92

# Camera / framing (match task4 style, with larger margin ~2x)
TOP_FIT_MARGIN_RATIO = float(os.environ.get("TOP_FIT_MARGIN_RATIO", "0.40"))
ISO_FIT_MARGIN_RATIO = float(os.environ.get("ISO_FIT_MARGIN_RATIO", "0.40"))
MIN_FIT_ORTHO_SCALE = float(os.environ.get("MIN_FIT_ORTHO_SCALE", "2.2"))
TOP_NORTH_ARROW_SIZE = int(os.environ.get("TOP_NORTH_ARROW_SIZE", "96"))
SHOW_ROADS_IN_REGION_RENDER = os.environ.get("SHOW_ROADS_IN_REGION_RENDER", "1").lower() in ("1", "true", "yes")
REFRAME_PAD_RATIO = float(os.environ.get("REFRAME_PAD_RATIO", "0.45"))
REFRAME_CONTENT_FILL_RATIO = float(os.environ.get("REFRAME_CONTENT_FILL_RATIO", "1.00"))
FINAL_TRIM_PAD_PX = int(os.environ.get("FINAL_TRIM_PAD_PX", "180"))

# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def world_bounds_from_obj(obj: bpy.types.Object) -> dict:
    """World-space bounds from actual mesh vertices (more accurate than bound_box)."""
    mw = obj.matrix_world

    # Revert to vertices to MATCH BuildingLabeler implementation exactly
    # BuildingLabeler.get_building_bounds uses vertices.
    if obj.type == "MESH" and obj.data and hasattr(obj.data, "vertices") and obj.data.vertices:
        verts = [mw @ Vector(v.co) for v in obj.data.vertices]
    elif hasattr(obj, "bound_box") and obj.bound_box:
        verts = [mw @ Vector(co) for co in obj.bound_box]
    else:
        return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0, "center_x": 0, "center_y": 0, "half_w": 0, "half_d": 0}

    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    return {
        "min_x": min_x, "max_x": max_x,
        "min_y": min_y, "max_y": max_y,
        "min_z": min_z, "max_z": max_z,
        "center_x": (min_x + max_x) / 2,
        "center_y": (min_y + max_y) / 2,
        "center_z": (min_z + max_z) / 2,
        "half_w": (max_x - min_x) / 2,
        "half_d": (max_y - min_y) / 2,
    }


def world_geometry_centroid_from_obj(obj: bpy.types.Object) -> Vector:
    """World-space geometry centroid from mesh vertices (fallback to AABB center)."""
    if obj and obj.type == "MESH" and obj.data and hasattr(obj.data, "vertices") and obj.data.vertices:
        mw = obj.matrix_world
        n = len(obj.data.vertices)
        sx = sy = sz = 0.0
        for v in obj.data.vertices:
            wv = mw @ Vector(v.co)
            sx += wv.x
            sy += wv.y
            sz += wv.z
        inv_n = 1.0 / float(n)
        return Vector((sx * inv_n, sy * inv_n, sz * inv_n))

    b = world_bounds_from_obj(obj)
    return Vector((b["center_x"], b["center_y"], b["center_z"]))


def _is_obj_inside_frame_bounds(obj: bpy.types.Object, frame_bounds: dict | None, margin: float = 0.0) -> bool:
    """
    Keep transformed object inside the region frame used for rendering.
    We use XY bounds because top/iso framing is region-based in XY.
    """
    if obj is None or not frame_bounds:
        return True
    try:
        b = world_bounds_from_obj(obj)
        m = float(margin)
        return (
            b["min_x"] >= float(frame_bounds["min_x"]) - m and
            b["max_x"] <= float(frame_bounds["max_x"]) + m and
            b["min_y"] >= float(frame_bounds["min_y"]) - m and
            b["max_y"] <= float(frame_bounds["max_y"]) + m
        )
    except Exception:
        return True


def compute_scene_bounds(objs: list[bpy.types.Object]) -> dict:
    """Compute scene bounds from a list of objects"""
    bnds = [world_bounds_from_obj(o) for o in objs if o and o.type == "MESH"]
    if not bnds:
        return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0,
                "center_x": 0, "center_y": 0, "center_z": 0, "width": 0, "depth": 0, "height": 0}
    min_x = min(b["min_x"] for b in bnds)
    max_x = max(b["max_x"] for b in bnds)
    min_y = min(b["min_y"] for b in bnds)
    max_y = max(b["max_y"] for b in bnds)
    min_z = min(b["min_z"] for b in bnds)
    max_z = max(b["max_z"] for b in bnds)
    return {
        "min_x": float(min_x), "max_x": float(max_x),
        "min_y": float(min_y), "max_y": float(max_y),
        "min_z": float(min_z), "max_z": float(max_z),
        "center_x": float((min_x + max_x) / 2),
        "center_y": float((min_y + max_y) / 2),
        "center_z": float((min_z + max_z) / 2),
        "width": float(max_x - min_x),
        "depth": float(max_y - min_y),
        "height": float(max_z - min_z),
    }


def does_location_move_geometry(obj: bpy.types.Object, delta_x: float = UNIT_SCALE) -> bool:
    """Test if changing obj.location changes its world bounds center."""
    b0 = world_bounds_from_obj(obj)
    old_loc = obj.location.copy()
    obj.location.x += float(delta_x)
    bpy.context.view_layer.update()
    b1 = world_bounds_from_obj(obj)
    obj.location = old_loc
    bpy.context.view_layer.update()
    dx = abs(b1["center_x"] - b0["center_x"])
    return dx > (abs(delta_x) * 0.6)


def bake_mesh_to_local_and_reset_transform(obj: bpy.types.Object) -> None:
    """
    Bake current world transform into mesh vertex positions, then reset matrix_world to identity.
    After this, obj.location will reliably move the geometry.
    """
    if obj.type != "MESH" or obj.data is None:
        return

    mw = obj.matrix_world.copy()
    mesh = obj.data

    for v in mesh.vertices:
        v.co = mw @ v.co
    mesh.update()

    obj.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()


def ensure_buildings_movable(building_objs: list[bpy.types.Object]) -> None:
    """
    If location does not move geometry, bake all buildings once.
    Also set origin to geometry center for proper rotation.
    """
    if not building_objs:
        return

    
    for obj in building_objs:
        if obj and obj.type == 'MESH' and obj.data:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

    bpy.context.view_layer.update()

    test = building_objs[0]
    if does_location_move_geometry(test):
        print("[INFO] location affects geometry -> no bake needed")
        return

    print("[WARN] location does NOT affect geometry -> baking buildings to local space...")
    for o in building_objs:
        bake_mesh_to_local_and_reset_transform(o)

    bpy.context.view_layer.update()
    if not does_location_move_geometry(building_objs[0]):
        print("[ERROR] bake did not make buildings movable; move/swap after render may still be wrong.")


def _iter_object_world_vertices(obj):
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    if not mesh.vertices:
        return
    used = set()
    for poly in mesh.polygons:
        for vid in poly.vertices:
            used.add(int(vid))
    if used:
        for vid in used:
            yield obj.matrix_world @ mesh.vertices[vid].co
    else:
        for v in mesh.vertices:
            yield obj.matrix_world @ v.co


def _render_resolution_px() -> tuple[int, int]:
    scene = bpy.context.scene
    rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    return max(1, rx), max(1, ry)


def _camera_half_extents(cam: bpy.types.Object) -> tuple[float, float]:
    frame = cam.data.view_frame(scene=bpy.context.scene)
    hx = max(abs(v.x) for v in frame)
    hy = max(abs(v.y) for v in frame)
    return float(hx), float(hy)


def _unit_bar_length_px(cam: bpy.types.Object, unit_world: float = 1.0) -> float:
    if cam is None or cam.type != "CAMERA" or cam.data.type != "ORTHO":
        return 0.0
    hx, _ = _camera_half_extents(cam)
    rx, _ = _render_resolution_px()
    if hx <= 1e-8:
        return 0.0
    px_per_world_x = float(rx) / (2.0 * hx)
    return float(unit_world) * px_per_world_x


def fit_ortho_camera_to_objects(
    cam: bpy.types.Object,
    objects: list[bpy.types.Object],
    margin_ratio: float = 0.08,
    min_ortho_scale: float = 2.2,
) -> None:
    if cam is None or cam.type != "CAMERA" or cam.data.type != "ORTHO":
        return
    world_points = []
    for obj in objects:
        for p in _iter_object_world_vertices(obj):
            world_points.append(p)
    if not world_points:
        return

    wx = [float(p.x) for p in world_points]
    wy = [float(p.y) for p in world_points]
    cam.location.x = 0.5 * (min(wx) + max(wx))
    cam.location.y = 0.5 * (min(wy) + max(wy))
    bpy.context.view_layer.update()

    fit_half_x = 0.5 * (max(wx) - min(wx))
    fit_half_y = 0.5 * (max(wy) - min(wy))
    rx, ry = _render_resolution_px()
    aspect = float(rx) / float(max(1, ry))
    needed_scale = max(2.0 * fit_half_x, 2.0 * fit_half_y * aspect)
    cam.data.ortho_scale = max(float(min_ortho_scale), needed_scale * (1.0 + float(margin_ratio)))


def inflate_ortho_scale(cam: bpy.types.Object, margin_ratio: float) -> None:
    if cam is None or cam.type != "CAMERA" or cam.data.type != "ORTHO":
        return
    cam.data.ortho_scale = float(cam.data.ortho_scale) * (1.0 + float(max(0.0, margin_ratio)))


def create_white_film_material(name: str = "Mask_WhiteFilm") -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if WHITE_FILM_ALPHA < 0.995 else "OPAQUE"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs["Strength"].default_value = 1.0
    if WHITE_FILM_ALPHA < 0.995:
        tr = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = WHITE_FILM_ALPHA
        links.new(tr.outputs["BSDF"], mix.inputs[1])
        links.new(em.outputs["Emission"], mix.inputs[2])
        links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def apply_white_film_to_buildings(building_ids: list[str]) -> None:
    mat = create_white_film_material()
    for bid in building_ids:
        obj = bpy.data.objects.get(bid)
        if not obj or obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.visible_shadow = False
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.shadow = False


def set_road_visibility(visible: bool) -> None:
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not is_road(obj.name):
            continue
        obj.hide_set(not visible)
        obj.hide_render = (not visible)


def create_black_label_material(name: str = "Label_Black_Mat") -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = LABEL_COLOR
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Specular"].default_value = 0.28
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


# --------------------------------------------------------------------------------------
# PCA OBB footprint computation
# --------------------------------------------------------------------------------------
def compute_pca_obb_xy(world_xy_pts: list[tuple[float, float]]) -> dict | None:
    """
    Given 2D points in world XY, compute an oriented bounding box using PCA.
    Return:
      angle (radians), ax, ay, hx, hy, center (cx,cy)
    """
    if not world_xy_pts:
        return None
    n = len(world_xy_pts)
    mx = sum(p[0] for p in world_xy_pts) / n
    my = sum(p[1] for p in world_xy_pts) / n
    xx = sum((p[0] - mx) ** 2 for p in world_xy_pts)
    yy = sum((p[1] - my) ** 2 for p in world_xy_pts)
    xy = sum((p[0] - mx) * (p[1] - my) for p in world_xy_pts)

    if abs(xy) < 1e-9:
        angle = 0.0 if xx >= yy else (math.pi / 2.0)
    else:
        angle = 0.5 * math.atan2(2.0 * xy, xx - yy)

    ca = math.cos(angle)
    sa = math.sin(angle)
    ax = (ca, sa)
    ay = (-sa, ca)

    rot = []
    for (x, y) in world_xy_pts:
        dx = x - mx
        dy = y - my
        rx = ca * dx + sa * dy
        ry = -sa * dx + ca * dy
        rot.append((rx, ry))

    rxs = [p[0] for p in rot]
    rys = [p[1] for p in rot]
    min_rx, max_rx = min(rxs), max(rxs)
    min_ry, max_ry = min(rys), max(rys)

    hx = 0.5 * (max_rx - min_rx)
    hy = 0.5 * (max_ry - min_ry)
    crx = 0.5 * (min_rx + max_rx)
    cry = 0.5 * (min_ry + max_ry)
    cx = mx + ca * crx - sa * cry
    cy = my + sa * crx + ca * cry

    return {
        "angle": float(angle),
        "ax": (float(ax[0]), float(ax[1])),
        "ay": (float(ay[0]), float(ay[1])),
        "hx": float(hx),
        "hy": float(hy),
        "center": (float(cx), float(cy)),
    }


def build_meta_from_obj(obj: bpy.types.Object, label_id: int) -> dict:
    """
    Build collision/render meta from the current object:
    - pos: obj.location xy ()
    - half_w/half_d: world AABB half extents
    - obb: PCA OBB on footprint sampled from vertices projected to XY
    """
    b = world_bounds_from_obj(obj)

    pos_x = float(obj.location.x)
    pos_y = float(obj.location.y)

    mw = obj.matrix_world
    verts = obj.data.vertices
    pts = []
    step = max(1, len(verts) // 500)
    for i in range(0, len(verts), step):
        w = mw @ verts[i].co
        pts.append((float(w.x), float(w.y)))

    obb = compute_pca_obb_xy(pts)
    if obb is None:
        obb = {
            "angle": 0.0,
            "ax": (1.0, 0.0),
            "ay": (0.0, 1.0),
            "hx": b["half_w"],
            "hy": b["half_d"],
            "center": (pos_x, pos_y),
        }

    return {
        "label_id": int(label_id),
        "original_id": obj.name,
        "pos": (pos_x, pos_y),
        "half_w": float(b["half_w"]),
        "half_d": float(b["half_d"]),
        "obb": obb,
    }


def get_building_obj(region_building_ids: list, id_to_obj: dict) -> dict:
    """Get building objects by ID mapping"""
    return {bid: obj for bid, obj in id_to_obj.items() if bid in region_building_ids}


def setup_render(resolution: int | tuple[int, int] = 1080):
    """Setup render settings."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 32
    if isinstance(resolution, tuple):
        rx, ry = int(resolution[0]), int(resolution[1])
    else:
        rx = int(resolution)
        ry = int(resolution)
    scene.render.resolution_x = max(1, rx)
    scene.render.resolution_y = max(1, ry)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True


def force_disable_all_shadows_and_world():
    """Disable all shadows and world shading"""
    scene = bpy.context.scene
    scene.cycles.use_shadows = False
    scene.cycles.use_progressive = False
    scene.cycles.use_shadow_highlight = False
    scene.cycles.blur_shadow = 0

    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            if hasattr(obj.data, "use_shadow"):
                obj.data.use_shadow = False
            if hasattr(obj.data, "cast_shadow"):
                obj.data.cast_shadow = False

    for obj in bpy.data.objects:
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.cast_shadow = False
            obj.cycles_visibility.receive_shadow = False


def clear_labels_only():
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Label_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for curve in list(bpy.data.curves):
        if curve.name.startswith("Label_"):
            bpy.data.curves.remove(curve)



def clear_overlays_only():
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Scale_") or obj.name.startswith("Overlay_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for curve in list(bpy.data.curves):
        if curve.name.startswith("Scale_") or curve.name.startswith("Overlay_"):
            bpy.data.curves.remove(curve)


def create_label_material(name: str, strength: float = 5.0):
    """Create emission material for labels"""
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = strength
    emission.inputs["Color"].default_value = (1, 1, 1, 1)

    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def rect_overlaps_any_building(rect_min_x, rect_max_x, rect_min_y, rect_max_y, objs, pad: float) -> bool:
    """Check if rectangle overlaps with any building"""
    for o in objs:
        b = world_bounds_from_obj(o)
        if not (
            (rect_max_x + pad) < (b["min_x"] - pad) or
            (rect_min_x - pad) > (b["max_x"] + pad) or
            (rect_max_y + pad) < (b["min_y"] - pad) or
            (rect_min_y - pad) > (b["max_y"] + pad)
        ):
            return True
    return False


def pick_scale_marker_position(bounds: dict, building_objs: list) -> tuple:
    """Pick a position for scale marker
    """
    w = SCALE_MARK_LENGTH_M + 1.2
    h = SCALE_TICK_LEN_Y + 1.0

    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])

    candidates = [
        (min_x + SCALE_MARK_MARGIN, max_y - SCALE_MARK_MARGIN - h),  # top-left (preferred - inside scene)
        (min_x + SCALE_MARK_MARGIN, min_y + SCALE_MARK_MARGIN),  # bottom-left
        (max_x - SCALE_MARK_MARGIN - w, max_y - SCALE_MARK_MARGIN - h),  # top-right
        (max_x - SCALE_MARK_MARGIN - w, min_y + SCALE_MARK_MARGIN),  # bottom-right
        (min_x + SCALE_MARK_MARGIN, (min_y + max_y) * 0.5 - h * 0.5),
        (max_x - SCALE_MARK_MARGIN - w, (min_y + max_y) * 0.5 - h * 0.5),
    ]

    for (x0, y0) in candidates:
        if not rect_overlaps_any_building(
            x0, x0 + w, y0, y0 + h, building_objs, pad=0.25
        ):
            return float(x0), float(y0)

    return float(max_x - SCALE_MARK_MARGIN - w), float(min_y + SCALE_MARK_MARGIN)


def is_isometric_camera(cam: bpy.types.Object) -> bool:
    """Check if camera is isometric based on name"""
    # Top-down camera has "Top" in name, isometric has "Iso" in name
    return "Iso" in cam.name


def add_scale_marker(bounds: dict, building_objs: list, cam: bpy.types.Object, length_m: float = 1.0) -> None:
    """
    Add scale marker to scene
    Marker in XY plane, visible in top view:
    - bar along X
    - two ticks extend along Y
    - placed at z = max_z + eps
    - "1 Unit" text label
    """
    x0, y0 = pick_scale_marker_position(bounds, building_objs)
    z = float(bounds["max_z"]) + SCALE_MARK_Z_EPS
    bar_y = y0 + SCALE_TICK_LEN_Y * 0.5

    # Use white material for scale marker (visible on dark background)
    mat = create_label_material("Scale_Mark_Mat", strength=10.0)

    created = []

    # Bar - create explicitly with correct dimensions from start
    mesh = bpy.data.meshes.new("Scale_Bar_Mesh")
    bar = bpy.data.objects.new("Scale_Bar", mesh)
    bpy.context.collection.objects.link(bar)

    bar_verts = [
        (x0, bar_y - SCALE_BAR_THICKNESS/2, z - SCALE_BAR_THICKNESS/2),
        (x0 + length_m, bar_y - SCALE_BAR_THICKNESS/2, z - SCALE_BAR_THICKNESS/2),
        (x0 + length_m, bar_y + SCALE_BAR_THICKNESS/2, z - SCALE_BAR_THICKNESS/2),
        (x0, bar_y + SCALE_BAR_THICKNESS/2, z - SCALE_BAR_THICKNESS/2),
        (x0, bar_y - SCALE_BAR_THICKNESS/2, z + SCALE_BAR_THICKNESS/2),
        (x0 + length_m, bar_y - SCALE_BAR_THICKNESS/2, z + SCALE_BAR_THICKNESS/2),
        (x0 + length_m, bar_y + SCALE_BAR_THICKNESS/2, z + SCALE_BAR_THICKNESS/2),
        (x0, bar_y + SCALE_BAR_THICKNESS/2, z + SCALE_BAR_THICKNESS/2),
    ]
    bar_faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    mesh.from_pydata(bar_verts, [], bar_faces)
    mesh.update()
    bar.data.materials.append(mat)
    created.append(bar)

    # Left Tick
    tick_mesh = bpy.data.meshes.new("Scale_Tick_L_Mesh")
    tick_l = bpy.data.objects.new("Scale_Tick_L", tick_mesh)
    bpy.context.collection.objects.link(tick_l)
    tick_l_verts = [
        (x0 - SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x0 + SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x0 + SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x0 - SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x0 - SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
        (x0 + SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
        (x0 + SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
        (x0 - SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
    ]
    tick_faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    tick_mesh.from_pydata(tick_l_verts, [], tick_faces)
    tick_mesh.update()
    tick_l.data.materials.append(mat)
    created.append(tick_l)

    # Right Tick
    tick_r_mesh = bpy.data.meshes.new("Scale_Tick_R_Mesh")
    tick_r = bpy.data.objects.new("Scale_Tick_R", tick_r_mesh)
    bpy.context.collection.objects.link(tick_r)
    x_r = x0 + length_m
    tick_r_verts = [
        (x_r - SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x_r + SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x_r + SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x_r - SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z - SCALE_TICK_THICKNESS/2),
        (x_r - SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
        (x_r + SCALE_TICK_THICKNESS/2, bar_y - SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
        (x_r + SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
        (x_r - SCALE_TICK_THICKNESS/2, bar_y + SCALE_TICK_LEN_Y/2, z + SCALE_TICK_THICKNESS/2),
    ]
    tick_r_faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
    tick_r_mesh.from_pydata(tick_r_verts, [], tick_r_faces)
    tick_r_mesh.update()
    tick_r.data.materials.append(mat)
    created.append(tick_r)

    # Disable shadows & ensure render visible
    for o in created:
        o.hide_render = False
        o.visible_shadow = False
        if hasattr(o, "cycles_visibility"):
            o.cycles_visibility.shadow = False
            o.cycles_visibility.diffuse = False
            o.cycles_visibility.glossy = False
            o.cycles_visibility.ambient_occlusion = False

    # Label - always show "1 Unit" as the reference scale
    curve = bpy.data.curves.new("Scale_Label_curve", type="FONT")
    curve.body = "1 Unit"
    curve.size = float(SCALE_LABEL_SIZE)
    curve.align_x = "LEFT"
    curve.align_y = "CENTER"
    curve.extrude = 0.2  # Thicker for bold look
    curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new("Scale_Label", curve)
    bpy.context.collection.objects.link(text_obj)
    text_obj.location = (float(x0 + length_m + 4.0), float(bar_y), float(z + SCALE_LABEL_Z_OFF))
    text_obj.data.materials.clear()
    text_obj.data.materials.append(create_label_material("Scale_Label_Mat", strength=5.0))

    # Billboard constraint for the label - use DAMPED_TRACK for isometric view
    if "Iso" in cam.name:
        dt = text_obj.constraints.new(type="DAMPED_TRACK")
        dt.target = cam
        dt.track_axis = "TRACK_NEGATIVE_Z"
    else:
        bb = text_obj.constraints.new(type="LOCKED_TRACK")
        bb.target = cam
        bb.track_axis = "TRACK_Z"
        bb.lock_axis = "LOCK_Y"

        lr = text_obj.constraints.new(type="LIMIT_ROTATION")
        lr.owner_space = "LOCAL"
        lr.use_limit_z = True
        lr.min_z = 0.0
        lr.max_z = 0.0

    text_obj.hide_render = False
    text_obj.visible_shadow = False
    if hasattr(text_obj, "cycles_visibility"):
        text_obj.cycles_visibility.shadow = False
        text_obj.cycles_visibility.diffuse = False
        text_obj.cycles_visibility.glossy = False
        text_obj.cycles_visibility.ambient_occlusion = False

    bpy.context.view_layer.update()


def add_label(obj: bpy.types.Object, label_text: str, cam: bpy.types.Object, font_size: float, z_top: float):
    """Add dynamic QA label for building."""
    b = world_bounds_from_obj(obj)
    cx = b["center_x"]
    cy = b["center_y"]

    curve = bpy.data.curves.new(f"Label_{label_text}_curve", type="FONT")
    curve.body = str(label_text)
    curve.size = float(font_size)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    # Match BuildingLabeler: no extrude? BuildingLabeler doesn't set extrude.
    # curve.extrude = max(0.02, float(font_size) * 0.02)
    curve.fill_mode = "BOTH"

    t = bpy.data.objects.new(f"Label_{label_text}", curve)
    t["qa_dynamic_label"] = True
    bpy.context.collection.objects.link(t)
    
    # Match BuildingLabeler z-offset logic more closely?
    # BuildingLabeler: label_z = bounds['max_z'] + font_size * 0.3
    # region_error_mode: z_top calculated in rebuild_labels... which is max_z + ...
    # We will use the z_top passed in, assuming it's reasonably calculated.
    t.location = (float(cx), float(cy), float(z_top))

    t.rotation_euler = (0, 0, 0)
    
    t.data.materials.clear()
    t.data.materials.append(create_black_label_material(f"Label_{label_text}_mat"))

    t.hide_render = False
    
    # BuildingLabeler logic for shadow visibility
    t.visible_shadow = False
    if hasattr(t, "cycles_visibility"):
        t.cycles_visibility.shadow = False
        t.cycles_visibility.diffuse = False
        t.cycles_visibility.glossy = False
        t.cycles_visibility.ambient_occlusion = False
        t.cycles_visibility.cast_shadow = False
        t.cycles_visibility.receive_shadow = False

    bpy.context.view_layer.update()



def render_view(
    cam: bpy.types.Object,
    output_path: str,
    add_north: bool = True,
    north_world_dir: Vector = Vector((0.0, 1.0, 0.0)),
    add_unit_bar: bool = False,
) -> None:
    """Render view and post-process with north panel and optional 1 unit bar."""
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    if add_north:
        reframe_image_focus_buildings(
            output_path,
            arrow_size=TOP_NORTH_ARROW_SIZE,
            reserve_for_north=True,
            building_bbox=None,
        )
        draw_north_arrow(
            output_path,
            cam=cam,
            north_world_dir=north_world_dir,
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=None,
            add_unit_bar=add_unit_bar,
        )
        trim_image_to_alpha_bbox(output_path, pad_px=FINAL_TRIM_PAD_PX)


def rebuild_labels_for_current_objects(
    cam: bpy.types.Object,
    id_to_obj: dict,
    bounds: dict,
    font_size: float,
) -> None:
    
    # Force update to ensure object matrices are correct before calculating bounds for labels
    bpy.context.view_layer.update()
    
    clear_labels_only()

    for label_id, obj in sorted(id_to_obj.items()):
        if obj and obj.type == "MESH":
            # Match preprocess_clean / BuildingLabeler logic:
            # Per-building height calculation
            b_bounds = world_bounds_from_obj(obj)
            z_top = b_bounds["max_z"] + font_size * 0.3

            add_label(
                obj=obj,
                label_text=str(label_id),
                cam=cam,
                font_size=font_size,
                z_top=z_top,
            )

    bpy.context.view_layer.update()


def _compute_north_screen_vec(cam, north_world_dir: Vector) -> Vector:
    m = cam.matrix_world.to_3x3()
    cam_right = (m @ Vector((1.0, 0.0, 0.0))).normalized()
    cam_up = (m @ Vector((0.0, 1.0, 0.0))).normalized()
    d = north_world_dir.normalized()
    screen_vec = Vector((d.dot(cam_right), -d.dot(cam_up)))
    if screen_vec.length < 1e-6:
        return Vector((0.0, -1.0))
    screen_vec.normalize()
    return screen_vec


def _draw_north_arrow_via_system_python(
    image_path: str,
    screen_vec: Vector,
    arrow_color=(255, 0, 0, 255),
    arrow_size=120,
    corner_idx: int | None = 0,
    unit_bar_px: float | None = None,
    unit_bar_text: str = "1 unit",
) -> bool:
    payload = {
        "image_path": image_path,
        "screen_vec": [float(screen_vec.x), float(screen_vec.y)],
        "arrow_color": [int(arrow_color[0]), int(arrow_color[1]), int(arrow_color[2]), int(arrow_color[3])],
        "arrow_size": int(arrow_size),
        "corner_idx": (int(corner_idx) if corner_idx is not None else None),
        "unit_bar_px": (float(unit_bar_px) if unit_bar_px is not None else None),
        "unit_bar_text": str(unit_bar_text),
        "unit_bar_color": [255, 255, 255, 255],
    }
    helper_script = r"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

def load_font(sz):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, sz)
            except Exception:
                pass
    return ImageFont.load_default()

def norm2(x, y):
    n = (x * x + y * y) ** 0.5
    if n < 1e-8:
        return 0.0, -1.0
    return x / n, y / n

payload = json.loads(sys.argv[1])
image_path = payload["image_path"]
if not os.path.exists(image_path):
    raise SystemExit(0)
vx, vy = norm2(float(payload["screen_vec"][0]), float(payload["screen_vec"][1]))
arrow_color = tuple(payload["arrow_color"])
arrow_size = int(payload["arrow_size"])
unit_bar_color = tuple(int(v) for v in payload.get("unit_bar_color", [255, 255, 255, 255]))

img = Image.open(image_path).convert("RGBA")
draw = ImageDraw.Draw(img)
w, h = img.size
alpha = img.getchannel("A")
pix = img.load()
margin = max(16, int(arrow_size * 0.20))
unit_bar_px = payload.get("unit_bar_px", None)
unit_bar_text = str(payload.get("unit_bar_text", "1 unit"))
bar_need_px = 0
if unit_bar_px is not None:
    try:
        u = float(unit_bar_px)
        if u < 0.0:
            u = 0.0
        bar_need_px = int(u + 0.999) + int(arrow_size * 0.56)
    except Exception:
        bar_need_px = 0
panel_w = max(int(arrow_size * 1.75), bar_need_px)
panel_h = int(arrow_size * 2.25)
cands = [
    (w - margin - panel_w, margin),
    (margin, margin),
    (w - margin - panel_w, h - margin - panel_h),
    (margin, h - margin - panel_h),
]
corner_idx = payload.get("corner_idx", 0)

def overlap_ratio(px, py):
    x1 = max(0, int(px)); y1 = max(0, int(py))
    x2 = min(w, x1 + panel_w); y2 = min(h, y1 + panel_h)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    patch = alpha.crop((x1, y1, x2, y2))
    hist = patch.histogram()
    non_transparent = (x2 - x1) * (y2 - y1) - hist[0]
    return non_transparent / float((x2 - x1) * (y2 - y1))

def is_building_pixel(r, g, b, a):
    if a <= 0:
        return False
    is_yellow_road = (r > 150 and g > 150 and b < 120 and abs(r - g) < 80)
    return (not is_yellow_road)

def panel_overlaps_buildings(px, py):
    x1 = max(0, int(px))
    y1 = max(0, int(py))
    x2 = min(w, x1 + panel_w)
    y2 = min(h, y1 + panel_h)
    if x2 <= x1 or y2 <= y1:
        return False
    for yy in range(y1, y2):
        for xx in range(x1, x2):
            r, g, b, a = pix[xx, yy]
            if is_building_pixel(r, g, b, a):
                return True
    return False

if isinstance(corner_idx, int) and 0 <= corner_idx < len(cands):
    bx, by = cands[corner_idx]
else:
    bx, by = min(cands, key=lambda xy: overlap_ratio(xy[0], xy[1]))

x1, y1 = int(bx), int(by)
x2, y2 = x1 + panel_w, y1 + panel_h

if panel_overlaps_buildings(x1, y1):
    old_w, old_h = w, h
    ext_w = panel_w + 2 * margin
    ext_top = max(0, panel_h + 2 * margin - old_h)
    new_w = old_w + ext_w
    new_h = old_h + ext_top
    new_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    new_img.paste(img, (0, ext_top), img)
    img = new_img
    draw = ImageDraw.Draw(img)
    pix = img.load()
    w, h = img.size
    x1 = old_w + max(0, (ext_w - panel_w) // 2)
    y1 = margin

x2, y2 = x1 + panel_w, y1 + panel_h
draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 220))
layout_shift = -0.10 * panel_h
cx = x1 + panel_w * 0.63
cy = y1 + panel_h * 0.34 + layout_shift
arrow_len = arrow_size * 0.52
head = arrow_size * 0.20
shaft_w = max(3, int(arrow_size * 0.05))
px, py = -vy, vx

tip = (cx + vx * arrow_len * 0.5, cy + vy * arrow_len * 0.5)
bottom = (cx - vx * arrow_len * 0.5, cy - vy * arrow_len * 0.5)
wing_base = (tip[0] - vx * head, tip[1] - vy * head)
left_wing = (wing_base[0] + px * head * 0.58, wing_base[1] + py * head * 0.58)
right_wing = (wing_base[0] - px * head * 0.58, wing_base[1] - py * head * 0.58)
draw.line([bottom, tip], fill=arrow_color, width=shaft_w)
draw.polygon([tip, left_wing, right_wing], fill=arrow_color)
font = load_font(max(16, int(arrow_size * 0.34)))
if font is not None:
    draw.text((x1 + panel_w * 0.20, y1 + panel_h * 0.50 + layout_shift), "N", fill=arrow_color, font=font)

if unit_bar_px is not None:
    try:
        bar_len = max(1.0, float(unit_bar_px))
        bar_y = y1 + panel_h * 0.84 + layout_shift
        bar_cx = x1 + panel_w * 0.50
        bar_left = bar_cx - bar_len * 0.5
        bar_right = bar_cx + bar_len * 0.5
        max_right = x1 + panel_w * 0.94
        if bar_right > max_right:
            shift = bar_right - max_right
            bar_left -= shift
            bar_right -= shift
        min_left = x1 + panel_w * 0.06
        if bar_left < min_left:
            bar_left = min_left
            bar_right = bar_left + bar_len
        bar_w = max(3, int(arrow_size * 0.045))
        tick_h = max(4, int(arrow_size * 0.085))
        draw.line([(bar_left, bar_y), (bar_right, bar_y)], fill=unit_bar_color, width=bar_w)
        draw.line([(bar_left, bar_y - tick_h), (bar_left, bar_y + tick_h)], fill=unit_bar_color, width=bar_w)
        draw.line([(bar_right, bar_y - tick_h), (bar_right, bar_y + tick_h)], fill=unit_bar_color, width=bar_w)
        lbl_font = load_font(max(16, int(arrow_size * 0.24)))
        if lbl_font is not None:
            bbox = draw.textbbox((0, 0), unit_bar_text, font=lbl_font)
            tw = bbox[2] - bbox[0]
            tx2 = (bar_left + bar_right - tw) * 0.5
            ty2 = bar_y + tick_h + max(3, int(arrow_size * 0.035))
            if ty2 + (bbox[3] - bbox[1]) > y1 + panel_h:
                ty2 = y1 + panel_h - (bbox[3] - bbox[1]) - 2
            if ty2 < y1 + 2:
                ty2 = y1 + 2
            draw.text((tx2, ty2), unit_bar_text, fill=unit_bar_color, font=lbl_font)
    except Exception:
        pass
img.save(image_path)
"""
    try:
        subprocess.run(
            ["python3", "-c", helper_script, json.dumps(payload, ensure_ascii=False)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except Exception as e:
        print(f"[WARN] draw north failed for {image_path}: {e}")
        return False


def draw_north_arrow(
    image_path: str,
    cam,
    north_world_dir=Vector((0.0, 1.0, 0.0)),
    arrow_size: int = TOP_NORTH_ARROW_SIZE,
    corner_idx: int | None = None,
    add_unit_bar: bool = False,
) -> str:
    if not os.path.exists(image_path):
        return image_path
    sv = _compute_north_screen_vec(cam, north_world_dir)
    unit_bar_px = _unit_bar_length_px(cam, unit_world=SCALE_MARK_LENGTH_M) if add_unit_bar else None
    _draw_north_arrow_via_system_python(
        image_path,
        sv,
        arrow_color=(255, 0, 0, 255),
        arrow_size=arrow_size,
        corner_idx=corner_idx,
        unit_bar_px=unit_bar_px,
        unit_bar_text="1 unit",
    )
    return image_path


def reframe_image_focus_buildings(
    image_path: str,
    pad_ratio: float | None = None,
    content_fill_ratio: float | None = None,
    arrow_size: int | None = None,
    reserve_for_north: bool = True,
    building_bbox: tuple[int, int, int, int] | None = None,
) -> bool:
    if not os.path.exists(image_path):
        return False
    if pad_ratio is None:
        pad_ratio = REFRAME_PAD_RATIO
    if content_fill_ratio is None:
        content_fill_ratio = REFRAME_CONTENT_FILL_RATIO
    payload = {
        "image_path": image_path,
        "pad_ratio": float(max(0.0, min(0.5, pad_ratio))),
        "content_fill_ratio": float(max(0.35, min(1.0, content_fill_ratio))),
        "arrow_size": (int(arrow_size) if arrow_size is not None else None),
        "reserve_for_north": bool(reserve_for_north),
        "building_bbox": (
            [int(building_bbox[0]), int(building_bbox[1]), int(building_bbox[2]), int(building_bbox[3])]
            if building_bbox is not None else None
        ),
    }
    helper_script = r"""
import json, os, sys
from PIL import Image

payload = json.loads(sys.argv[1])
image_path = payload["image_path"]
if not os.path.exists(image_path):
    raise SystemExit(0)

pad_ratio = float(payload.get("pad_ratio", 0.10))
bbox = payload.get("building_bbox", None)

img = Image.open(image_path).convert("RGBA")
w, h = img.size
pix = img.load()

if isinstance(bbox, list) and len(bbox) == 4:
    x1 = max(0, min(w - 1, int(bbox[0])))
    y1 = max(0, min(h - 1, int(bbox[1])))
    x2 = max(0, min(w - 1, int(bbox[2])))
    y2 = max(0, min(h - 1, int(bbox[3])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
else:
    xs = []
    ys = []
    for y in range(h):
        for x in range(w):
            r, g, b, a = pix[x, y]
            if a <= 0:
                continue
            if (r > 140 and g > 140 and b > 140) and (abs(r - g) < 45 and abs(r - b) < 45 and abs(g - b) < 45):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        for y in range(h):
            for x in range(w):
                if pix[x, y][3] > 0:
                    xs.append(x)
                    ys.append(y)
    if not xs or not ys:
        raise SystemExit(0)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
bw = max(1, x2 - x1 + 1)
bh = max(1, y2 - y1 + 1)
pad_x = int(round(bw * pad_ratio))
pad_y = int(round(bh * pad_ratio))

x1 = max(0, x1 - pad_x)
y1 = max(0, y1 - pad_y)
x2 = min(w - 1, x2 + pad_x)
y2 = min(h - 1, y2 + pad_y)

crop = img.crop((x1, y1, x2 + 1, y2 + 1))
cw, ch = crop.size
if cw <= 0 or ch <= 0:
    raise SystemExit(0)
crop.save(image_path)
"""
    try:
        subprocess.run(
            ["python3", "-c", helper_script, json.dumps(payload, ensure_ascii=False)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except Exception as e:
        print(f"  [reframe] Warning: reframing failed for {image_path}: {e}")
        return False


def trim_image_to_alpha_bbox(image_path: str, pad_px: int = 0) -> bool:
    if not os.path.exists(image_path):
        return False
    payload = {
        "image_path": image_path,
        "pad_px": int(max(0, pad_px)),
    }
    helper_script = r"""
import json, os, sys
from PIL import Image

payload = json.loads(sys.argv[1])
image_path = payload["image_path"]
pad = int(payload.get("pad_px", 0))
if not os.path.exists(image_path):
    raise SystemExit(0)

img = Image.open(image_path).convert("RGBA")
w, h = img.size
x1 = y1 = x2 = y2 = None
pix = img.load()
for y in range(h):
    for x in range(w):
        r, g, b, a = pix[x, y]
        if a <= 0:
            continue
        is_yellow_road = (r > 150 and g > 150 and b < 120 and abs(r - g) < 80)
        if is_yellow_road:
            continue
        if x1 is None:
            x1 = x2 = x
            y1 = y2 = y
        else:
            if x < x1: x1 = x
            if y < y1: y1 = y
            if x > x2: x2 = x
            if y > y2: y2 = y

if x1 is None:
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise SystemExit(0)
    x1, y1, x2, y2 = bbox
else:
    x2 += 1
    y2 += 1

x1 = max(0, x1 - pad)
y1 = max(0, y1 - pad)
x2 = min(w, x2 + pad)
y2 = min(h, y2 + pad)
if x2 <= x1 or y2 <= y1:
    raise SystemExit(0)
img.crop((x1, y1, x2, y2)).save(image_path)
"""
    try:
        subprocess.run(
            ["python3", "-c", helper_script, json.dumps(payload, ensure_ascii=False)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except Exception as e:
        print(f"  [trim] Warning: trim failed for {image_path}: {e}")
        return False


from mathutils.geometry import convex_hull_2d, intersect_line_line_2d


def _cross_2d(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_on_segment_2d(p, a, b, eps: float = 1e-6) -> bool:
    cross = abs(_cross_2d(a, b, p))
    if cross > eps:
        return False
    min_x, max_x = min(a[0], b[0]) - eps, max(a[0], b[0]) + eps
    min_y, max_y = min(a[1], b[1]) - eps, max(a[1], b[1]) + eps
    return min_x <= p[0] <= max_x and min_y <= p[1] <= max_y


def _segments_intersect_2d(a1, a2, b1, b2, eps: float = 1e-6) -> bool:
    inter = intersect_line_line_2d(a1, a2, b1, b2)
    if inter is not None:
        return True

    return (
        _point_on_segment_2d(a1, b1, b2, eps) or
        _point_on_segment_2d(a2, b1, b2, eps) or
        _point_on_segment_2d(b1, a1, a2, eps) or
        _point_on_segment_2d(b2, a1, a2, eps)
    )


def _point_in_polygon_2d(point, polygon, eps: float = 1e-6) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]

        if _point_on_segment_2d(point, a, b, eps):
            return True

        xi, yi = a
        xj, yj = b
        if abs(yj - yi) < eps:
            continue

        hit = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        if hit:
            inside = not inside

    return inside


def _polygon_edges(poly):
    for i in range(len(poly)):
        yield poly[i], poly[(i + 1) % len(poly)]


def _polygon_overlap_2d(poly_a, poly_b, eps: float = 1e-6) -> bool:
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False

    for a1, a2 in _polygon_edges(poly_a):
        for b1, b2 in _polygon_edges(poly_b):
            if _segments_intersect_2d(a1, a2, b1, b2, eps):
                return True

    if _point_in_polygon_2d(poly_a[0], poly_b, eps):
        return True
    if _point_in_polygon_2d(poly_b[0], poly_a, eps):
        return True

    return False


def _polygon_centroid_2d(poly):
    if not poly:
        return (0.0, 0.0)
    return (
        sum(p[0] for p in poly) / len(poly),
        sum(p[1] for p in poly) / len(poly),
    )


def _polygon_offset_from_centroid(poly, pad: float):
    if abs(pad) < 1e-9 or len(poly) < 3:
        return poly

    cx, cy = _polygon_centroid_2d(poly)
    out = []
    for x, y in poly:
        dx = x - cx
        dy = y - cy
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            out.append((x, y))
            continue
        scale = (norm + pad) / norm
        if scale <= 0:
            scale = 1e-6
        out.append((cx + dx * scale, cy + dy * scale))
    return out


def extract_footprint_polygon_xy(obj: bpy.types.Object, max_points: int = 4000):

    if obj is None or obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return []

    mw = obj.matrix_world
    verts = obj.data.vertices
    step = max(1, len(verts) // max_points)

    pts = []
    for i in range(0, len(verts), step):
        w = mw @ verts[i].co
        pts.append(Vector((float(w.x), float(w.y))))

    if len(pts) < 3:
        return []

    hull_idx = convex_hull_2d(pts)
    poly = [(float(pts[i].x), float(pts[i].y)) for i in hull_idx]

    if len(poly) >= 3 and _cross_2d(poly[0], poly[1], poly[2]) < 0:
        poly.reverse()

    return poly


def clean_get_region_bounds(building_objs, buffer=10.0):
    if not building_objs:
        return {'min_x': -1e6, 'max_x': 1e6, 'min_y': -1e6, 'max_y': 1e6}
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for obj in building_objs:
        if obj is None: continue
        # Use bound_box for speed
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_x = min(min_x, world_corner.x)
            min_y = min(min_y, world_corner.y)
            max_x = max(max_x, world_corner.x)
            max_y = max(max_y, world_corner.y)
            
    return {
        'min_x': min_x - buffer,
        'max_x': max_x + buffer,
        'min_y': min_y - buffer,
        'max_y': max_y + buffer
    }

def clean_is_face_in_bounds(face_verts, bounds, buffer=5.0, mode: str = "bbox_overlap"):
    if not face_verts:
        return False

    bmin_x = bounds['min_x'] - buffer
    bmax_x = bounds['max_x'] + buffer
    bmin_y = bounds['min_y'] - buffer
    bmax_y = bounds['max_y'] + buffer

    # Robust mode: include face if its XY bbox intersects query bounds.
    # This avoids missing long road faces whose centroids are outside the local query box.
    if mode == "bbox_overlap":
        fmin_x = min(v[0] for v in face_verts)
        fmax_x = max(v[0] for v in face_verts)
        fmin_y = min(v[1] for v in face_verts)
        fmax_y = max(v[1] for v in face_verts)
        return not (fmax_x < bmin_x or fmin_x > bmax_x or fmax_y < bmin_y or fmin_y > bmax_y)

    # Legacy mode: centroid-in-bounds.
    cx = sum(v[0] for v in face_verts) / len(face_verts)
    cy = sum(v[1] for v in face_verts) / len(face_verts)
    return (bmin_x <= cx <= bmax_x and bmin_y <= cy <= bmax_y)

def clean_extract_valid_road_faces(road_obj, region_bounds, buffer=5.0, filter_mode: str = "bbox_overlap"):
    valid_faces = []
    if road_obj.type != 'MESH': return []
    mesh = road_obj.data
    mw = road_obj.matrix_world
    world_verts = [mw @ v.co for v in mesh.vertices]
    for poly in mesh.polygons:
        face_vs = [world_verts[i] for i in poly.vertices]
        coords_2d = [(v.x, v.y) for v in face_vs]
        if clean_is_face_in_bounds(coords_2d, region_bounds, buffer, mode=filter_mode):
            valid_faces.append({
                'verts': face_vs,
                'indices': list(poly.vertices),
                'normal': poly.normal
            })
    return valid_faces

def clean_create_max_region_roads(region_id, building_ids):
    b_objs = []
    for bid in building_ids:
        if bid in bpy.data.objects:
            b_objs.append(bpy.data.objects[bid])
    
    bounds = clean_get_region_bounds(b_objs, buffer=10.0)
    
    road_objs = [obj for obj in bpy.data.objects if obj.type == "MESH" and ("road" in obj.name.lower() or "path" in obj.name.lower())]
    
    new_verts = []
    new_faces = []
    vert_cursor = 0
    
    original_hidden_states = {}
    
    for r_obj in road_objs:
        original_hidden_states[r_obj.name] = (r_obj.hide_render, r_obj.hide_viewport)
        r_obj.hide_render = True
        r_obj.hide_viewport = True
        
        valid_items = clean_extract_valid_road_faces(
            r_obj, bounds, buffer=5.0, filter_mode="bbox_overlap"
        )
        
        for item in valid_items:
            face_indices = []
            for v in item['verts']:
                new_verts.append((v.x, v.y, v.z))
                face_indices.append(vert_cursor)
                vert_cursor += 1
            new_faces.append(face_indices)
    
    if not new_faces:
        return None, original_hidden_states

    mesh = bpy.data.meshes.new(name=f"Region_{region_id}_Roads_Temp")
    mesh.from_pydata(new_verts, [], new_faces)
    mesh.update()
    
    obj = bpy.data.objects.new(f"Region_{region_id}_Roads_Temp_Obj", mesh)
    bpy.context.collection.objects.link(obj)
    

    if road_objs and road_objs[0].data.materials:
        obj.data.materials.append(road_objs[0].data.materials[0])
    else:

        mat = create_road_material()
        obj.data.materials.append(mat)
        
    return obj, original_hidden_states

def clean_cleanup_temp_roads(temp_obj, hidden_states):
    if temp_obj:
        try:
            mesh = temp_obj.data
            bpy.data.objects.remove(temp_obj, do_unlink=True)
            if mesh:
                try: bpy.data.meshes.remove(mesh)
                except: pass
        except: pass
            
    for obj_name, (hr, hv) in hidden_states.items():
        obj = bpy.data.objects.get(obj_name)
        if obj:
            try:
                obj.hide_render = hr
                obj.hide_viewport = hv
            except: pass


def _check_target_building_overlap(
    target_obj: bpy.types.Object,
    building_objs: list,
    pad: float = 0.0,
    id_to_obj: dict = None,
):
   
    if target_obj is None:
        return (False, None, None) if id_to_obj else False

    # Force update the evaluated depsgraph to ensure we get the latest transformed geometry
    # bpy.context.view_layer.update() # Assume caller has updated view layer


    poly_a = get_building_footprint(target_obj)
    
    if poly_a is None or poly_a.is_empty:

        pts = extract_footprint_polygon_xy(target_obj)
        if pts and len(pts) >= 3:
            try:
                poly_a = Polygon(pts)
            except Exception: pass
    
    if poly_a is None or poly_a.is_empty:
         return (False, None, None) if id_to_obj else False

    # Pre-calculate bounds for AABB check
    min_x, min_y, max_x, max_y = poly_a.bounds

    for obj_b in building_objs:
        if obj_b is None or obj_b == target_obj:
            continue
            
        # Optimization: AABB check first
        # world_bounds_from_obj uses object.bound_box * matrix_world (always up to date)
        b_b = world_bounds_from_obj(obj_b)
        if (max_x < b_b["min_x"] or min_x > b_b["max_x"] or
            max_y < b_b["min_y"] or min_y > b_b["max_y"]):
            continue

        poly_b = get_building_footprint(obj_b)
        
        if poly_b is None or poly_b.is_empty:
             pts_b = extract_footprint_polygon_xy(obj_b)
             if pts_b and len(pts_b) >= 3:
                 try:
                    poly_b = Polygon(pts_b)
                 except Exception: pass

        if poly_b is None or poly_b.is_empty:
             continue # Skip invalid building B

        if poly_a.intersects(poly_b):
             try:
                 intersection = poly_a.intersection(poly_b)
                 inter_area = intersection.area
             except Exception:
                 inter_area = 0.0
                 
             if inter_area > 1e-9:
                if id_to_obj:
                    label_a = None
                    label_b = None
                    for label, o in id_to_obj.items():
                        if o == target_obj: label_a = label
                        if o == obj_b: label_b = label
                    return True, label_a, label_b
                return True

    if id_to_obj:
        return False, None, None
    return False


def render_top_view_with_labels(
    cam: bpy.types.Object,
    output_path: str,
    bounds: dict,
    building_objs: list,
    id_to_obj: dict,
    with_scale_marker: bool,
    label_font_size: float,
    region_max_dim: float,
) -> None:
    clear_overlays_only()
    rebuild_labels_for_current_objects(cam, id_to_obj, bounds, label_font_size)
    bpy.context.view_layer.update()
    _ = with_scale_marker  # legacy arg; scale bar is now always in north panel
    render_view(cam, output_path, add_unit_bar=True)


# --------------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------------
def load_region_data():
    """Load region data"""
    with open(REGION_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_building_region_map():
    
    with open(BUILDING_REGION_MAP_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _compute_selected_region_tasks():

    region_data = load_region_data()
    all_regions_raw = region_data.get("regions", [])
    all_regions_raw = sorted(all_regions_raw, key=lambda r: int(r.get("region_id", 0)))

    start_idx = max(0, int(MIN_REGION))
    total_available = len(all_regions_raw)
    if start_idx > total_available:
        start_idx = total_available

    if MAX_REGIONS > 0:
        end_idx = min(start_idx + int(MAX_REGIONS), total_available)
    else:
        end_idx = total_available

    tasks = []
    for abs_idx in range(start_idx, end_idx):
        region = all_regions_raw[abs_idx]
        region_id = int(region.get("region_id", abs_idx))
        tasks.append({"abs_idx": abs_idx, "region_id": region_id})

    return tasks, total_available, start_idx, end_idx


def _run_parallel_worker_task(task: dict, worker_output_root: str, logs_dir: str):
    """ worker region"""
    abs_idx = int(task["abs_idx"])
    region_id = int(task["region_id"])
    log_path = os.path.join(logs_dir, f"region_{region_id}_idx_{abs_idx}.log")

    child_args = [
        "--mode", INPUT_MODE,
        "--region", REGION_NAME,
        "--min_region", str(abs_idx),
        "--max_regions", "1",
        "--workers", "1",
    ]

    if os.path.exists(EMBEDDED_BLENDER_BIN):
        cmd = [
            EMBEDDED_BLENDER_BIN,
            "--background",
            "--python",
            os.path.abspath(__file__),
            "--",
            *child_args,
        ]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), *child_args]

    env = os.environ.copy()
    env["OUTPUT_ROOT"] = worker_output_root
    env["REGION_WORKER_CHILD"] = "1"

    os.makedirs(logs_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    return {
        "task": task,
        "returncode": int(proc.returncode),
        "worker_output_root": worker_output_root,
        "log_path": log_path,
    }


def _merge_parallel_outputs(worker_results: list, tasks: list):
   
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged_qa = []
    failed_regions = []

    for res in worker_results:
        task = res["task"]
        region_id = int(task["region_id"])
        abs_idx = int(task["abs_idx"])
        worker_root = res["worker_output_root"]
        worker_out_dir = os.path.join(worker_root, f"{REGION_NAME}_regions")
        qa_path = os.path.join(worker_out_dir, f"qa_{REGION_NAME}.json")
        stats_path = os.path.join(worker_out_dir, f"stats_{REGION_NAME}.json")

        if res["returncode"] != 0:
            failed_regions.append({
                "region_id": region_id,
                "region_index": abs_idx,
                "reason": f"worker exit code {res['returncode']}",
                "log_path": res["log_path"],
            })
            continue

        if os.path.exists(qa_path):
            try:
                with open(qa_path, "r", encoding="utf-8") as f:
                    qa_data = json.load(f)
                if isinstance(qa_data, list):
                    merged_qa.extend(qa_data)
            except Exception as e:
                failed_regions.append({
                    "region_id": region_id,
                    "region_index": abs_idx,
                    "reason": f"failed to load worker qa json: {e}",
                    "log_path": res["log_path"],
                })
        else:
            failed_regions.append({
                "region_id": region_id,
                "region_index": abs_idx,
                "reason": "worker qa json missing",
                "log_path": res["log_path"],
            })

        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    worker_stats = json.load(f)
                for fr in worker_stats.get("failed_regions", []):
                    if isinstance(fr, dict):
                        fr_copy = dict(fr)
                        fr_copy.setdefault("region_index", abs_idx)
                        failed_regions.append(fr_copy)
            except Exception as e:
                failed_regions.append({
                    "region_id": region_id,
                    "region_index": abs_idx,
                    "reason": f"failed to load worker stats json: {e}",
                    "log_path": res["log_path"],
                })

        if os.path.exists(worker_out_dir):
            for name in os.listdir(worker_out_dir):
                if not name.startswith("region_"):
                    continue
                src = os.path.join(worker_out_dir, name)
                dst = os.path.join(OUTPUT_DIR, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)

    anomaly_order = {t: i for i, t in enumerate(TARGET_QA_ANOMALY_TYPES)}
    merged_qa.sort(
        key=lambda x: (
            int(x.get("region_id", -1)),
            anomaly_order.get(x.get("requested_anomaly_type", ""), 999),
            str(x.get("requested_anomaly_type", "")),
        )
    )

    success_count = len(merged_qa)
    synthetic_count = sum(1 for x in merged_qa if x.get("is_synthetic"))
    existing_count = success_count - synthetic_count
    total_regions = len(tasks)
    total_target_qas = total_regions * len(TARGET_QA_ANOMALY_TYPES)

    qa_output_path = os.path.join(OUTPUT_DIR, f"qa_{REGION_NAME}.json")
    with open(qa_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_qa, f, ensure_ascii=False, indent=2)

    merged_stats = {
        "region": REGION_NAME,
        "total_regions": total_regions,
        "total_target_qas": total_target_qas,
        "target_anomaly_types": TARGET_QA_ANOMALY_TYPES,
        "workers": WORKERS,
        "success_count": success_count,
        "existing_anomaly_count": existing_count,
        "synthetic_anomaly_count": synthetic_count,
        "failed_count": len(failed_regions),
        "failed_regions": failed_regions,
    }
    stats_output_path = os.path.join(OUTPUT_DIR, f"stats_{REGION_NAME}.json")
    with open(stats_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_stats, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total regions: {total_regions}")
    print(f"  GoalQA: {total_target_qas}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(failed_regions)}")
    print(f"  QA : {qa_output_path}")
    print(f"  Statistics: {stats_output_path}")


def run_parallel_workers():
    tasks, total_available, start_idx, end_idx = _compute_selected_region_tasks()
    print(
        f"[PARALLEL] workers={WORKERS}, selected index range=[{start_idx}, {end_idx}), "
        f"total_available={total_available}, tasks={len(tasks)}",
        flush=True,
    )

    if not tasks:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        qa_output_path = os.path.join(OUTPUT_DIR, f"qa_{REGION_NAME}.json")
        stats_output_path = os.path.join(OUTPUT_DIR, f"stats_{REGION_NAME}.json")
        with open(qa_output_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "region": REGION_NAME,
                    "total_regions": 0,
                    "total_target_qas": 0,
                    "target_anomaly_types": TARGET_QA_ANOMALY_TYPES,
                    "workers": WORKERS,
                    "success_count": 0,
                    "existing_anomaly_count": 0,
                    "synthetic_anomaly_count": 0,
                    "failed_count": 0,
                    "failed_regions": [],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("[PARALLEL] No regions selected, done.")
        return

    parallel_tmp_root = os.path.join(
        OUTPUT_ROOT,
        f".parallel_tmp_{REGION_NAME}_{int(time.time())}",
    )
    logs_dir = os.path.join(parallel_tmp_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    worker_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = []
        for task in tasks:
            worker_output_root = os.path.join(parallel_tmp_root, f"worker_idx_{task['abs_idx']}")
            os.makedirs(worker_output_root, exist_ok=True)
            submit_log_path = os.path.join(
                logs_dir,
                f"region_{int(task['region_id'])}_idx_{int(task['abs_idx'])}.log",
            )
            print(
                f"[PARALLEL] submit region_id={int(task['region_id'])}, idx={int(task['abs_idx'])}, "
                f"log={submit_log_path}",
                flush=True,
            )
            futures.append(
                executor.submit(
                    _run_parallel_worker_task,
                    task,
                    worker_output_root,
                    logs_dir,
                )
            )

        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            worker_results.append(res)
            rid = res["task"]["region_id"]
            idx = res["task"]["abs_idx"]
            print(
                f"[PARALLEL] region_id={rid}, idx={idx}, returncode={res['returncode']}, "
                f"log={res['log_path']}"
            )

    worker_results.sort(key=lambda x: int(x["task"]["abs_idx"]))
    _merge_parallel_outputs(worker_results, tasks)

    keep_tmp = str(os.environ.get("KEEP_PARALLEL_TMP", "0")).strip().lower() in {"1", "true", "yes"}
    if not keep_tmp:
        try:
            shutil.rmtree(parallel_tmp_root)
        except Exception:
            pass


def get_building_label(region_labels_path):
    
    labels_path = os.path.join(region_labels_path, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------------------
# QA Generation Functions
# --------------------------------------------------------------------------------------
def get_anomaly_type_mapping():
    """Anomaly types"""
    return {
        ISSUE_OVERLAP: "A",
        ISSUE_ANGLE: "C",
        ISSUE_ROAD: "B",
        ISSUE_SCALE: "D",
        "scale_small": "D",
        "scale_large": "D",
    }


def qa1_mcq_what_problem(issue_meta: dict, images: list[str]) -> dict:
    """QA1:  - """
    t = issue_meta["issues"][0]["type"] if issue_meta.get("issues") else None
    mapping = get_anomaly_type_mapping()

    choices = [
        "A. Buildings overlap each other",
        "B. A building conflicts the road network",
        "C. A building is rotated at an abnormal angle (not aligned with nearby roads or inconsistent with the orientation of adjacent buildings)",
    ]

    q = (
        "You are viewing a top-down view of a 3D scene containing:\n"
        "- Multiple buildings of varying heights, labeled with numbers (1, 2, 3, etc.)\n"
        "- A yellow road network\n"
        "Question: Examine the scene carefully and identify what problem exists. Choose one option.\n"
        "Top-view image: <image>"
    )

    return {
        "question": q + "\n" + "\n".join(choices),
        "answer": mapping.get(t, "A"),
        "task_type": "top_error_identify",
        "meta": issue_meta,
        "images": images
    }


def qa2_mcq_what_problem(issue_meta: dict, images: list[str]) -> dict:
    """QA2: top + isometric"""
    t = issue_meta["issues"][0]["type"] if issue_meta.get("issues") else None
    mapping = get_anomaly_type_mapping()

    choices = [
        "A. Buildings overlap each other",
        "B. A building conflicts the road network",
        "C. A building is rotated at an abnormal angle (not aligned with nearby roads or inconsistent with the orientation of adjacent buildings)",
    ]

    q = (
        "You are viewing two images of a 3D scene: a top-down view and an isometric view.\n"
        "The scene contains:\n"
        "- Multiple buildings of varying heights, labeled with numbers (1, 2, 3, etc.)\n"
        "- A yellow road network\n"
        "Question: Examine both views carefully and identify what problem exists. Choose one option.\n"
        "Top-view image: <image>\n"
        "Isometric-view image: <image>"
    )

    return {
        "question": q + "\n" + "\n".join(choices),
        "answer": mapping.get(t, "A"),
        "task_type": "top_isometric_error_identify",
        "meta": issue_meta,
        "images": images
    }


# Direction mapping
DIR_TO_VEC = {
    # 4 cardinal directions
    "North": (0.0, 1.0), "South": (0.0, -1.0), "East": (1.0, 0.0), "West": (-1.0, 0.0),
    "N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0), "W": (-1.0, 0.0),
    "north": (0.0, 1.0), "south": (0.0, -1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0),
    # 4 diagonal directions
    "NE": (1.0, 1.0), "NW": (-1.0, 1.0), "SE": (1.0, -1.0), "SW": (-1.0, -1.0),
    "northeast": (1.0, 1.0), "northwest": (-1.0, 1.0), "southeast": (1.0, -1.0), "southwest": (-1.0, -1.0),
    "Northeast": (1.0, 1.0), "Northwest": (-1.0, 1.0), "Southeast": (1.0, -1.0), "Southwest": (-1.0, -1.0),
}


DIR_CODE_TO_WORD = {
    "N": "north", "S": "south", "E": "east", "W": "west",
    "NE": "northeast", "NW": "northwest", "SE": "southeast", "SW": "southwest",
}

DIR_WORD_TO_CODE = {v: k for k, v in DIR_CODE_TO_WORD.items()}

DIR_ALIASES_TO_CODE = {
    "n": "N", "north": "N", "N": "N", "North": "N",
    "s": "S", "south": "S", "S": "S", "South": "S",
    "e": "E", "east": "E", "E": "E", "East": "E",
    "w": "W", "west": "W", "W": "W", "West": "W",
    "ne": "NE", "northeast": "NE", "NE": "NE", "Northeast": "NE",
    "nw": "NW", "northwest": "NW", "NW": "NW", "Northwest": "NW",
    "se": "SE", "southeast": "SE", "SE": "SE", "Southeast": "SE",
    "sw": "SW", "southwest": "SW", "SW": "SW", "Southwest": "SW",
}

OPPOSITE_DIR_CODE = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "NW": "SE", "SE": "NW", "SW": "NE",
}


def _normalize_dir_code(direction: str, default: str = "E") -> str:
    if not direction:
        return default
    key = str(direction).strip()
    return DIR_ALIASES_TO_CODE.get(key, DIR_ALIASES_TO_CODE.get(key.lower(), default))


def _dir_code_to_full_name(direction: str) -> str:
    code = _normalize_dir_code(direction)
    word = DIR_CODE_TO_WORD.get(code, "east")
    return word.capitalize()


def _get_opposite_dir_code(direction: str) -> str:
    code = _normalize_dir_code(direction)
    return OPPOSITE_DIR_CODE.get(code, "W")


def _get_dir_vector(direction: str) -> tuple[float, float]:
    code = _normalize_dir_code(direction)
    return DIR_TO_VEC.get(code, (1.0, 0.0))


DIR_FULL_NAME = {
    "N": "North", "S": "South", "E": "East", "W": "West",
    "NE": "Northeast", "NW": "Northwest", "SE": "Southeast", "SW": "Southwest",
    "north": "North", "south": "South", "east": "East", "west": "West",
    "northeast": "Northeast", "northwest": "Northwest", "southeast": "Southeast", "southwest": "Southwest",
}

def _get_full_dir_name(abbrev: str) -> str:
    """Get full direction name"""
    normalized = _normalize_dir_code(abbrev, default="")
    if normalized:
        return _dir_code_to_full_name(normalized)
    return DIR_FULL_NAME.get(abbrev, abbrev)


def _parse_choice_text(choice_text: str) -> dict | None:
    """Parse option text and return operation parameter dict"""
    choice_text = choice_text.strip()

    
    if choice_text.startswith("Move building"):
        
        m = re.match(
            r"^Move building\s+(\S+)\s+(\S+)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s*(?:units?|Unit|m)?$",
            choice_text,
            flags=re.IGNORECASE,
        )
        if m:
            return {"op": "move", "id": m.group(1), "dir": m.group(2), "dist": float(m.group(3))}
        return None

    
    if choice_text.startswith("Rotate building"):
        parts = choice_text.split()
        try:
            # parts: ["Rotate", "building", "X", "direction", "by", "Y°"]
            bid = parts[2]
            direction = parts[3]  # clockwise or counter-clockwise
            deg = int(parts[5].rstrip("°"))  # parts[4]"by"
            return {"op": "rotate", "id": bid, "dir": direction, "deg": deg}
        except (IndexError, ValueError):
            return None

    
    if choice_text.startswith("Scale"):
        parts = choice_text.split()
        try:
            # parts: ["Scale", "up/down", "building", "X", "by", "Y%"]
            scale_dir = parts[1]  # up or down
            bid = parts[3]        # building number
            percent = float(parts[5].rstrip("%"))  # parts[4]"by"
            return {"op": "scale", "id": bid, "dir": scale_dir, "percent": percent}
        except (IndexError, ValueError):
            return None

    return None


def _snapshot_obj_transform(obj: bpy.types.Object) -> dict:
    """Save object transform"""
    return {
        "matrix_world": obj.matrix_world.copy(),
    }


def _restore_obj_transform(obj: bpy.types.Object, snap: dict) -> None:
    """Restore object transform"""
    obj.matrix_world = snap["matrix_world"].copy()
    bpy.context.view_layer.update()


def _format_percent_value(value: float) -> str:
    v = round(float(value), 1)
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.1f}"


def _build_inverse_scale_action_text(bid: str, injected_scale_factor: float) -> str:
    """scale_factor"""
    sf = max(1e-6, float(injected_scale_factor))
    if sf > 1.0:
        
        down_pct = (1.0 - 1.0 / sf) * 100.0
        return f"Scale down building {bid} by {_format_percent_value(down_pct)}%"
    if sf < 1.0:
        
        up_pct = (1.0 / sf - 1.0) * 100.0
        return f"Scale up building {bid} by {_format_percent_value(up_pct)}%"
    return f"Scale up building {bid} by 0%"


NICE_SCALE_REVERSE_PCTS = [10, 15, 20, 25, 30, 35, 40, 50]


def _sample_scale_factor_from_nice_reverse(allow_up_injection: bool = True, allow_down_injection: bool = True) -> tuple[float, float, str]:
    """
    “” scale_factor
    : (scale_factor, reverse_pct, reverse_dir)
      - reverse_dir == "down"  Scale down ... by reverse_pct%
      - reverse_dir == "up"    Scale up ... by reverse_pct%
    """
    modes = []
    if allow_up_injection:
        modes.append("down")  # 
    if allow_down_injection:
        modes.append("up")    # 
    if not modes:
        modes = ["down"]

    reverse_dir = random.choice(modes)
    reverse_pct = float(random.choice(NICE_SCALE_REVERSE_PCTS))
    r = reverse_pct / 100.0

    if reverse_dir == "down":
        # 1/sf = 1-r => sf = 1/(1-r)
        scale_factor = 1.0 / max(1e-6, (1.0 - r))
    else:
        # 1/sf - 1 = r => sf = 1/(1+r)
        scale_factor = 1.0 / (1.0 + r)

    return float(scale_factor), reverse_pct, reverse_dir


def _build_candidate_fix_actions_from_inject(inject_action: dict, bid: str) -> list[str]:
    """ inject_action  correct choice Failed"""
    candidates: list[str] = []
    op = str(inject_action.get("op", "")).lower()

    if op == "move":
        move_dir = inject_action.get("dir", inject_action.get("dir_word", "E"))
        dist = inject_action.get("dist_m", 1.0)
        opposite_code = _get_opposite_dir_code(str(move_dir))
        try:
            dist_val = float(dist)
        except Exception:
            dist_val = 1.0
        dist_text = _format_percent_value(dist_val)
        candidates.append(f"Move building {bid} {_dir_code_to_full_name(opposite_code)} by {dist_text} units")

        move_dists = [dist_val] + list(MOVE_DISTS_M) + [0.5, 0.8, 1.2, 1.5, 2.0]
        for d in ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]:
            full_name = _dir_code_to_full_name(d)
            for m in move_dists:
                m_text = _format_percent_value(m)
                candidates.append(f"Move building {bid} {full_name} by {m_text} units")

    elif op == "rotate":
        deg = inject_action.get("deg", 0)
        try:
            deg_val = int(round(float(deg)))
        except Exception:
            deg_val = 0

        clockwise = bool(inject_action.get("clockwise", False))
        reverse_dir = "clockwise" if not clockwise else "counter-clockwise"
        if deg_val > 0:
            candidates.append(f"Rotate building {bid} {reverse_dir} by {deg_val}°")

        angle_pool = sorted(set([deg_val] + ORIENTATION_ROTATE_CHOICES_DEG + [5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 150]))
        for a in angle_pool:
            if a <= 0:
                continue
            candidates.append(f"Rotate building {bid} clockwise by {a}°")
            candidates.append(f"Rotate building {bid} counter-clockwise by {a}°")

    elif op == "scale":
        scale_factor = inject_action.get("scale_factor", None)
        if scale_factor is not None:
            try:
                candidates.append(_build_inverse_scale_action_text(bid, float(scale_factor)))
            except Exception:
                pass

        for pct in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75]:
            p_text = _format_percent_value(pct)
            candidates.append(f"Scale up building {bid} by {p_text}%")
            candidates.append(f"Scale down building {bid} by {p_text}%")

    else:
        for d in ["North", "South", "East", "West"]:
            for m in [0.5, 1.0, 1.5, 2.0]:
                m_text = _format_percent_value(m)
                candidates.append(f"Move building {bid} {d} by {m_text} units")
        for a in [15, 30, 45, 60, 90]:
            candidates.append(f"Rotate building {bid} clockwise by {a}°")
            candidates.append(f"Rotate building {bid} counter-clockwise by {a}°")
        for pct in [10, 20, 25, 30]:
            p_text = _format_percent_value(pct)
            candidates.append(f"Scale up building {bid} by {p_text}%")
            candidates.append(f"Scale down building {bid} by {p_text}%")

    
    return list(dict.fromkeys(candidates))


def _apply_exact_inverse_scale_if_possible(obj: bpy.types.Object, inject_action: dict) -> bool:
    """ inject_action  scale_factor1/scale_factor"""
    if obj is None:
        return False
    if str(inject_action.get("op", "")).lower() != "scale":
        return False

    sf = inject_action.get("scale_factor")
    if sf is None:
        return False

    try:
        sf = max(1e-6, float(sf))
    except Exception:
        return False

    inv_sf = 1.0 / sf
    inv_sf = max(0.05, float(inv_sf))

    bounds = world_bounds_from_obj(obj)
    center = Vector((bounds["center_x"], bounds["center_y"], bounds["min_z"]))
    current_matrix = obj.matrix_world.copy()
    scale_mat = Matrix.Diagonal((inv_sf, inv_sf, inv_sf, 1.0))
    obj.matrix_world = Matrix.Translation(center) @ scale_mat @ Matrix.Translation(-center) @ current_matrix
    bpy.context.view_layer.update()
    return True


def _apply_exact_inverse_rotate_if_possible(obj: bpy.types.Object, inject_action: dict) -> bool:
    """ inject_action  rotate"""
    if obj is None:
        return False
    if str(inject_action.get("op", "")).lower() != "rotate":
        return False

    deg = inject_action.get("deg")
    clockwise = inject_action.get("clockwise")
    if deg is None or clockwise is None:
        return False

    try:
        deg = float(deg)
        clockwise = bool(clockwise)
    except Exception:
        return False

    signed_deg = deg if clockwise else -deg
    angle_rad = math.radians(signed_deg)

    
    bounds = world_bounds_from_obj(obj)
    center = Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"]))
    current_matrix = obj.matrix_world.copy()
    rot_mat = Matrix.Rotation(angle_rad, 4, 'Z')
    obj.matrix_world = Matrix.Translation(center) @ rot_mat @ Matrix.Translation(-center) @ current_matrix
    bpy.context.view_layer.update()
    return True


def _apply_choice(obj: bpy.types.Object, parsed: dict) -> bool:
    op = parsed.get("op")
    try:
        if op == "move":
            direction = parsed.get("dir", "North")
            dist = parsed.get("dist", 1.0)
            
            dx, dy = _get_dir_vector(direction)

            
            is_diagonal = (dx != 0 and dy != 0)
            scale = 1.0 / math.sqrt(2) if is_diagonal else 1.0
 
            obj.location.x += float(dx * dist * UNIT_SCALE * scale)
            obj.location.y += float(dy * dist * UNIT_SCALE * scale)
            bpy.context.view_layer.update()
            return True
            
        elif op == "rotate":
            direction = parsed.get("dir", "clockwise")
            deg = parsed.get("deg", 45)
            
            # Clockwise is negative rotation around Z
            sgn = -1.0 if direction == "clockwise" else 1.0
            angle_rad = float(sgn * math.radians(deg))

            # Rotate around AABB center
            bounds = world_bounds_from_obj(obj)
            center = Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"]))

            current_matrix = obj.matrix_world.copy()
            rot_mat = Matrix.Rotation(angle_rad, 4, 'Z')
            
            obj.matrix_world = (
                Matrix.Translation(center) @ 
                rot_mat @ 
                Matrix.Translation(-center) @ 
                current_matrix
            )
            bpy.context.view_layer.update()
            return True
            
        elif op == "scale":
            scale_dir = parsed.get("dir", "up")
            percent = parsed.get("percent", 20)
            factor = 1.0 + (percent / 100.0) if scale_dir == "up" else 1.0 - (percent / 100.0)
            factor = max(0.05, float(factor))
            
            # Scale around building base center to keep object grounded
            bounds = world_bounds_from_obj(obj)
            center = Vector((bounds["center_x"], bounds["center_y"], bounds["min_z"]))
            
            # Create uniform scale matrix
            scale_mat = Matrix.Scale(factor, 4)
            
            current_matrix = obj.matrix_world.copy()
            
            obj.matrix_world = (
                Matrix.Translation(center) @ 
                scale_mat @ 
                Matrix.Translation(-center) @ 
                current_matrix
            )
            
            bpy.context.view_layer.update()
            return True
            
    except Exception as e:
        print(f"[WARN] Failed to apply choice: {e}")
        import traceback
        traceback.print_exc()
        return False
    return False


def _check_building_overlap(building_objs: list, exclude_obj: bpy.types.Object = None, pad: float = 0.0, id_to_obj: dict = None) -> bool:
    """buffer"""
    
    # Filter objects
    objs = [o for o in building_objs if o != exclude_obj]
    if len(objs) < 2:
         if id_to_obj: return False, None, None
         return False

    # Cache footprints to avoid re-computing for same object in multiple pairs
    footprints = {}
    
    def get_fp(obj):
        if obj in footprints: return footprints[obj]
        
        fp = None
        try:
            fp = get_building_footprint(obj)
        except: pass
        
        if fp is None or fp.is_empty:
             pts = extract_footprint_polygon_xy(obj)
             if len(pts) >= 3: fp = Polygon(pts)
             
        footprints[obj] = fp
        return fp

    for i in range(len(objs)):
        obj_a = objs[i]
        b_a = world_bounds_from_obj(obj_a)
        
        for j in range(i + 1, len(objs)):
            obj_b = objs[j]
            b_b = world_bounds_from_obj(obj_b)
            
            # Fast AABB Check
            if (b_a["max_x"] < b_b["min_x"] or
                b_a["min_x"] > b_b["max_x"] or
                b_a["max_y"] < b_b["min_y"] or
                b_a["min_y"] > b_b["max_y"]):
                continue
            
            # Accurate Check
            poly_a = get_fp(obj_a)
            poly_b = get_fp(obj_b)
            
            if not poly_a or not poly_b: continue
            
            if poly_a.intersects(poly_b):
                 try:
                     area = poly_a.intersection(poly_b).area
                 except: area = 0.0
                     
                 if area > 1e-9:
                    if id_to_obj:
                        label_a = next((k for k, v in id_to_obj.items() if v == obj_a), None)
                        label_b = next((k for k, v in id_to_obj.items() if v == obj_b), None)
                        return True, label_a, label_b
                    return True

    if id_to_obj:
        return False, None, None
    return False


def _find_nearest_building_direction(target_obj: bpy.types.Object, other_objs: list) -> dict:
    """GoalMove directions

    Args:
        target_obj: Goal
        other_objs: 

    Returns:
        :
        - dir:  ("North", "South", "East", "West",  "NE", "NW", "SE", "SW")
        - dir_word: 
        - dist_m: 
        - nearest_obj: 
        - nearest_label: 
        - dx, dy: xy
    """
    if not other_objs:
        return None

    
    target_bounds = world_bounds_from_obj(target_obj)
    target_center_x = (target_bounds["min_x"] + target_bounds["max_x"]) / 2
    target_center_y = (target_bounds["min_y"] + target_bounds["max_y"]) / 2

    
    nearest_obj = None
    min_dist = float('inf')
    nearest_bounds = None

    for obj in other_objs:
        if obj == target_obj:
            continue
        bounds = world_bounds_from_obj(obj)
        center_x = (bounds["min_x"] + bounds["max_x"]) / 2
        center_y = (bounds["min_y"] + bounds["max_y"]) / 2

        
        dist = math.sqrt((center_x - target_center_x)**2 + (center_y - target_center_y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_obj = obj
            nearest_bounds = bounds

    if not nearest_obj:
        return None

    
    nearest_center_x = (nearest_bounds["min_x"] + nearest_bounds["max_x"]) / 2
    nearest_center_y = (nearest_bounds["min_y"] + nearest_bounds["max_y"]) / 2

    dx = nearest_center_x - target_center_x
    dy = nearest_center_y - target_center_y

    
    
    target_width = target_bounds["max_x"] - target_bounds["min_x"]
    
    
    center_dist = math.sqrt(dx * dx + dy * dy)

    
    
    if random.random() < 0.5:
        offset_factor = random.uniform(0.1, 0.3)  # 
    else:
        offset_factor = random.uniform(-0.3, -0.1)  # 
    total_required_dist = center_dist * (1 + offset_factor)

    
    if total_required_dist < 0.5:
        total_required_dist = 0.5

    
    total_required_dist = total_required_dist / UNIT_SCALE

    
    total_required_dist = round(total_required_dist, 1)

    
    # theta: 0 = east, pi/2 = north, pi = west, -pi/2 = south
    theta = math.atan2(dy, dx)

    
    sector = math.pi / 4.0  # 45
    centers = [
        ("east", 0.0),
        ("northeast", math.pi / 4.0),
        ("north", math.pi / 2.0),
        ("northwest", 3.0 * math.pi / 4.0),
        ("west", math.pi),
        ("southwest", -3.0 * math.pi / 4.0),
        ("south", -math.pi / 2.0),
        ("southeast", -math.pi / 4.0),
    ]

    def _min_abs_angle_diff(a, b):
        diff = (a - b + math.pi) % (2 * math.pi) - math.pi
        if diff < -math.pi:
            diff += 2 * math.pi
        return abs(diff)

    best_dir = "east"
    best_diff = 1e9
    for d, c in centers:
        diff = _min_abs_angle_diff(theta, c)
        if diff < best_diff:
            best_diff = diff
            best_dir = d

    dir_word = best_dir  #  spatial_relation 

    
    dir_map = {
        "east": "E", "north": "N", "west": "W", "south": "S",
        "northeast": "NE", "northwest": "NW",
        "southeast": "SE", "southwest": "SW"
    }

    return {
        "dir": dir_map.get(dir_word, "E"),
        "dir_word": dir_word,
        "dist_m": total_required_dist,
        "nearest_obj": nearest_obj,
        "dx": dx,
        "dy": dy
    }


def _find_nearest_road_direction(target_obj: bpy.types.Object, road_objs: list) -> dict:
    """GoalMove directions

    Args:
        target_obj: Goal
        road_objs: 

    Returns:
        :
        - dir: 
        - dir_word: 
        - dist_m: 
        - nearest_obj: 
    """
    if not road_objs:
        return None

    
    target_bounds = world_bounds_from_obj(target_obj)
    target_center_x = (target_bounds["min_x"] + target_bounds["max_x"]) / 2
    target_center_y = (target_bounds["min_y"] + target_bounds["max_y"]) / 2

    
    nearest_road = None
    min_dist = float('inf')
    nearest_point_xy = None

    for road in road_objs:
        local_best_dist = float('inf')
        local_best_xy = None

        try:
            if road.type == "MESH" and road.data and len(road.data.vertices) > 0:
                verts = road.data.vertices
                
                step = max(1, len(verts) // 3000)
                for vid in range(0, len(verts), step):
                    v_w = road.matrix_world @ verts[vid].co
                    dx_v = v_w.x - target_center_x
                    dy_v = v_w.y - target_center_y
                    d = math.hypot(dx_v, dy_v)
                    if d < local_best_dist:
                        local_best_dist = d
                        local_best_xy = (v_w.x, v_w.y)
        except Exception:
            local_best_dist = float('inf')
            local_best_xy = None

        if local_best_xy is None:
            bounds = world_bounds_from_obj(road)
            center_x = (bounds["min_x"] + bounds["max_x"]) / 2
            center_y = (bounds["min_y"] + bounds["max_y"]) / 2
            local_best_xy = (center_x, center_y)
            local_best_dist = math.hypot(center_x - target_center_x, center_y - target_center_y)

        if local_best_dist < min_dist:
            min_dist = local_best_dist
            nearest_road = road
            nearest_point_xy = local_best_xy

    if not nearest_road:
        return None

    
    nearest_center_x, nearest_center_y = nearest_point_xy

    dx = nearest_center_x - target_center_x
    dy = nearest_center_y - target_center_y

    
    center_dist = math.hypot(dx, dy)
    
    nearest_dist_m = center_dist / UNIT_SCALE

    
    
    if random.random() < 0.5:
        offset_factor = random.uniform(0.1, 0.3)  # 
    else:
        offset_factor = random.uniform(-0.3, -0.1)  # 
    total_required_dist = center_dist * (1 + offset_factor)

    
    if total_required_dist < 0.5:
        total_required_dist = 0.5

    
    total_required_dist = total_required_dist / UNIT_SCALE

    total_required_dist = round(total_required_dist, 1)

    
    theta = math.atan2(dy, dx)

    sector = math.pi / 4.0
    centers = [
        ("east", 0.0),
        ("northeast", math.pi / 4.0),
        ("north", math.pi / 2.0),
        ("northwest", 3.0 * math.pi / 4.0),
        ("west", math.pi),
        ("southwest", -3.0 * math.pi / 4.0),
        ("south", -math.pi / 2.0),
        ("southeast", -math.pi / 4.0),
    ]

    def _min_abs_angle_diff(a, b):
        diff = (a - b + math.pi) % (2 * math.pi) - math.pi
        if diff < -math.pi:
            diff += 2 * math.pi
        return abs(diff)

    best_dir = "east"
    best_diff = 1e9
    for d, c in centers:
        diff = _min_abs_angle_diff(theta, c)
        if diff < best_diff:
            best_diff = diff
            best_dir = d

    dir_word = best_dir

    dir_map = {
        "east": "E", "north": "N", "west": "W", "south": "S",
        "northeast": "NE", "northwest": "NW",
        "southeast": "SE", "southwest": "SW"
    }

    return {
        "dir": dir_map.get(dir_word, "E"),
        "dir_word": dir_word,
        "dist_m": total_required_dist,
        "nearest_dist_m": round(nearest_dist_m, 3),
        "nearest_obj": nearest_road,
        "dx": dx,
        "dy": dy
    }


# --------------------------------------------------------------------------------------
# Collision Detection Helpers
# --------------------------------------------------------------------------------------

def _get_object_state_status(obj: bpy.types.Object, building_objs: list, road_objs: list,
                            misalign_threshold_deg: float = 20.0,
                            scale_threshold_ratio: float = 0.7,
                            original_scale: tuple = None,
                            original_rot_z: float = None) -> dict:
    """Get the current error status of the object: B-B, B-R, Misalign, Scale"""

    # Check orientation FIRST because it's fast
    # Use modulo 90 relative check if original_rot_z is provided
    # Updated to pass road_objs for context-aware check
    is_misaligned = _check_orientation_issue(obj, road_objs=road_objs, threshold_deg=misalign_threshold_deg, original_rot_z=original_rot_z)
    
    # Check B-R
    has_br = _check_road_conflict(road_objs, [], target_obj=obj)
    
    # Check B-B
    res = _check_target_building_overlap(obj, building_objs, pad=0.0)
    has_bb = res if isinstance(res, bool) else res[0]

    return {
        "has_bb": has_bb,
        "has_br": has_br,
        "is_misaligned": is_misaligned
    }

def get_building_footprint(obj, base_tolerance=0.5, normal_threshold=0.9):
    """ (, mesh)."""
    # depsgraph = bpy.context.evaluated_depsgraph_get() # Avoid creating new one every time if possible, or use one context
    # Usually better to get current context's depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    try:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
    except Exception:
        return None
        
    if mesh is None or len(mesh.vertices) < 3:
        return None

    world_matrix = obj.matrix_world
    
    
    transform = world_matrix.to_3x3()
    valid_polygons = []

    for poly in mesh.polygons:
        world_coords = [world_matrix @ mesh.vertices[idx].co for idx in poly.vertices]
        if len(world_coords) < 3:
            continue
        
        
        world_normal = transform @ poly.normal
        
        
        
        if abs(world_normal.z) < 0.7:
            continue

        coords_2d = [(v.x, v.y) for v in world_coords]
        face_poly = Polygon(coords_2d)
        if not face_poly.is_valid:
            face_poly = face_poly.buffer(0)
            
        if not face_poly.is_empty and face_poly.area > 0.01:
            valid_polygons.append(face_poly)

    obj_eval.to_mesh_clear()

    if not valid_polygons:
        return None

    
    try:
        footprint = unary_union(valid_polygons)
        if footprint.is_empty:
            return None
        return footprint
    except Exception as e:
        print(f"Error joining polygons for {obj.name}: {e}")
        return None


def get_road_faces_in_region(road_obj, region_bounds=None, buffer=5.0):
    """ (Calls clean_extract_valid_road_faces)"""
    if region_bounds is None:
        region_bounds = {'min_x': -1e6, 'max_x': 1e6, 'min_y': -1e6, 'max_y': 1e6}

    road_faces = []
    
    # Use clean_extract_valid_road_faces to ensure consistency with rendering
    # This guarantees that the faces used for collision detection are exactly 
    # the ones used for creating the region mesh (Visual = Collision)
    valid_data = clean_extract_valid_road_faces(
        road_obj, region_bounds, buffer, filter_mode="bbox_overlap"
    )  # Reuse rendering extraction with robust bounds filter
    
    for item in valid_data:
        face_verts = item['verts'] # List of Vectors
        coords_2d = [(v.x, v.y) for v in face_verts]
        
        if len(coords_2d) < 3: continue

        face_poly = Polygon(coords_2d)
        if not face_poly.is_valid: face_poly = face_poly.buffer(0)
        if face_poly.is_empty: continue
        
        # 'verts' is kept for potential debug, though mostly 'geom' is used
        road_faces.append({'geom': face_poly, 'verts': coords_2d})
            
    return road_faces


def get_bounds_from_geom(geom):
    """ shapely """
    min_x, min_y, max_x, max_y = geom.bounds
    return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}


def extract_overlap_coords(overlap_geom):
    
    if overlap_geom.is_empty:
        return []

    geom_type = overlap_geom.geom_type
    if geom_type == 'Polygon':
        return list(overlap_geom.exterior.coords)
    if geom_type == 'MultiPolygon':
        largest = max(overlap_geom.geoms, key=lambda g: g.area)
        return list(largest.exterior.coords)
    if geom_type == 'GeometryCollection':
        polys = [g for g in overlap_geom.geoms if g.geom_type == 'Polygon']
        if not polys:
            return []
        largest = max(polys, key=lambda g: g.area)
        return list(largest.exterior.coords)
    return []


def _check_road_conflict(
    road_objs: list,
    building_objs: list,
    target_obj: bpy.types.Object = None,
    threshold_area: float | None = None
) -> bool:
    """Check if target building (or list) overlaps with any road object."""
    if not road_objs:
        return False
    if threshold_area is None:
        threshold_area = ROAD_CONFLICT_AREA_THRESHOLD

    buildings_to_check = [target_obj] if target_obj else building_objs
    buildings_to_check = [b for b in buildings_to_check if b]
    
    if not buildings_to_check:
        return False

    # Get combined bounds of buildings to optimize road fetch
    min_x, min_y = 1e9, 1e9
    max_x, max_y = -1e9, -1e9
    
    for b in buildings_to_check:
        bb = world_bounds_from_obj(b)
        min_x = min(min_x, bb["min_x"])
        min_y = min(min_y, bb["min_y"])
        max_x = max(max_x, bb["max_x"])
        max_y = max(max_y, bb["max_y"])
        
    region_bounds = {'min_x': min_x - 30, 'max_x': max_x + 30, 'min_y': min_y - 30, 'max_y': max_y + 30}
    
    # Check against ALL road objects within bounds
    road_polys = []
    for r_obj in road_objs:
        faces = get_road_faces_in_region(r_obj, region_bounds=region_bounds, buffer=10.0)
        for f in faces:
            road_polys.append(f['geom'])
            
    if not road_polys:
        return False
        
    road_union = unary_union(road_polys)
    if road_union.is_empty:
        return False

    # Check intersection
    for b in buildings_to_check:
        # Prioritize accurate footprint
        poly_b = get_building_footprint(b)
        if poly_b is None or poly_b.is_empty:
             pts = extract_footprint_polygon_xy(b)
             if len(pts) >= 3:
                 try: poly_b = Polygon(pts)
                 except: pass
        
        if poly_b is None or poly_b.is_empty: continue

        if poly_b.intersects(road_union):
            try:
                area = poly_b.intersection(road_union).area
                if area > threshold_area:
                    return True
            except: pass
                
    return False


def _normalize_angle_180(deg: float) -> float:
    d = float(deg) % 180.0
    if d < 0.0:
        d += 180.0
    return d


def _angle_diff_180(a_deg: float, b_deg: float) -> float:
    a = _normalize_angle_180(a_deg)
    b = _normalize_angle_180(b_deg)
    d = abs(a - b)
    return min(d, 180.0 - d)


def _axis_error_deg(curr_deg: float, axis_deg: float) -> float:
    """Error to an axis system: min(parallel error, perpendicular error)."""
    return min(_angle_diff_180(curr_deg, axis_deg), _angle_diff_180(curr_deg, axis_deg + 90.0))


def _polygon_major_axis_deg(poly) -> float | None:
    if poly is None or poly.is_empty:
        return None
    try:
        mrr = poly.minimum_rotated_rectangle
    except Exception:
        return None
    if mrr is None or mrr.is_empty:
        return None
    try:
        coords = list(mrr.exterior.coords)
    except Exception:
        return None
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        if math.hypot(dx, dy) > 1e-6:
            return _normalize_angle_180(math.degrees(math.atan2(dy, dx)))
    return None


def _largest_polygon(poly):
    if poly is None or poly.is_empty:
        return None
    if getattr(poly, "geom_type", "") == "MultiPolygon":
        polys = [g for g in poly.geoms if g and (not g.is_empty)]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    return poly


def _extract_orientation_footprint(target_obj: bpy.types.Object):
    if target_obj is None:
        return None
    footprint = get_building_footprint(target_obj)
    if footprint is None or footprint.is_empty:
        pts = extract_footprint_polygon_xy(target_obj)
        if len(pts) >= 3:
            try:
                footprint = Polygon(pts)
            except Exception:
                footprint = None
    return _largest_polygon(footprint)


def _building_edge_axis_deg(target_obj: bpy.types.Object) -> float | None:
    """
    Building direction from a real footprint edge:
    - rectangle: usually picks one long edge
    - L-shape: picks dominant outer edge direction
    """
    footprint = _extract_orientation_footprint(target_obj)
    if footprint is None or footprint.is_empty:
        return None

    simple = footprint.simplify(0.15, preserve_topology=True)
    if simple is None or simple.is_empty or not hasattr(simple, "exterior"):
        simple = footprint

    coords = list(simple.exterior.coords) if hasattr(simple, "exterior") else []
    if len(coords) < 4:
        return _polygon_major_axis_deg(footprint)

    best_len = -1.0
    best_ang = None
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        seg_len = math.hypot(dx, dy)
        if seg_len < 0.5:
            continue
        ang = _normalize_angle_180(math.degrees(math.atan2(dy, dx)))
        if seg_len > best_len:
            best_len = seg_len
            best_ang = ang

    if best_ang is not None:
        return best_ang
    return _polygon_major_axis_deg(footprint)


_ROAD_EDGE_SAMPLE_CACHE = {}


def _get_sampled_road_edges(road_obj: bpy.types.Object):
    """
    Cache sampled road edges: [(mid_x, mid_y, angle_deg, length), ...]
    """
    if road_obj is None or road_obj.type != "MESH" or road_obj.data is None:
        return []
    try:
        key = (
            road_obj.name_full,
            len(road_obj.data.vertices),
            len(road_obj.data.edges),
        )
    except Exception:
        key = (road_obj.name, 0, 0)
    cached = _ROAD_EDGE_SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached

    samples = []
    try:
        mesh = road_obj.data
        if len(mesh.edges) <= 0:
            _ROAD_EDGE_SAMPLE_CACHE[key] = samples
            return samples

        step = max(1, len(mesh.edges) // max(1, ORIENTATION_ROAD_EDGE_SAMPLE_MAX))
        for i in range(0, len(mesh.edges), step):
            edge = mesh.edges[i]
            v1_w = road_obj.matrix_world @ mesh.vertices[edge.vertices[0]].co
            v2_w = road_obj.matrix_world @ mesh.vertices[edge.vertices[1]].co
            dx = float(v2_w.x - v1_w.x)
            dy = float(v2_w.y - v1_w.y)
            seg_len = math.hypot(dx, dy)
            if seg_len <= 0.5:
                continue
            ang = _normalize_angle_180(math.degrees(math.atan2(dy, dx)))
            mid_x = float((v1_w.x + v2_w.x) * 0.5)
            mid_y = float((v1_w.y + v2_w.y) * 0.5)
            samples.append((mid_x, mid_y, ang, seg_len))
    except Exception:
        samples = []

    _ROAD_EDGE_SAMPLE_CACHE[key] = samples
    return samples


def _nearest_road_axis_deg(target_obj: bpy.types.Object, road_objs: list, radius: float = None) -> float | None:
    """
    Find nearest main-road axis around object in XY plane.
    1) Locate nearest road edge sample.
    2) On that local main road area, use the longest nearby road edge as direction.
    """
    if target_obj is None or not road_objs:
        return None

    if radius is None:
        radius = ORIENTATION_NEAREST_ROAD_RADIUS

    center = target_obj.location
    center_x = float(center.x)
    center_y = float(center.y)
    best_dist = float("inf")
    nearest_road = None
    nearest_edge_ang = None

    for r_obj in road_objs:
        if r_obj is None or r_obj.type != "MESH":
            continue
        edge_samples = _get_sampled_road_edges(r_obj)
        if not edge_samples:
            continue
        for mid_x, mid_y, ang, _seg_len in edge_samples:
            d = math.hypot(mid_x - center_x, mid_y - center_y)
            if d > radius:
                continue
            if d < best_dist:
                best_dist = d
                nearest_road = r_obj
                nearest_edge_ang = ang

    if nearest_road is None:
        return None

    local_best_len = -1.0
    local_best_ang = None
    local_radius = max(15.0, float(ORIENTATION_MAIN_ROAD_LOCAL_RADIUS))
    for mid_x, mid_y, ang, seg_len in _get_sampled_road_edges(nearest_road):
        d = math.hypot(mid_x - center_x, mid_y - center_y)
        if d > local_radius:
            continue
        if seg_len > local_best_len:
            local_best_len = seg_len
            local_best_ang = ang

    if local_best_ang is None:
        local_best_ang = nearest_edge_ang

    if local_best_ang is None:
        return None
    return _normalize_angle_180(local_best_ang)


def _check_orientation_issue(target_obj: bpy.types.Object, road_objs: list = None, threshold_deg: float = 20.0, original_rot_z: float = None) -> bool:
    """
    Orientation anomaly rule:
    - Prefer nearest-road reference: object major direction should be parallel/perpendicular to nearest road axis.
    - If original_rot_z is provided, compare against original axis-parallel/perpendicular.
    """
    if target_obj is None:
        return False

    current_axis_deg = _building_edge_axis_deg(target_obj)
    if current_axis_deg is None:
        current_rot_z = target_obj.rotation_euler.z if target_obj.rotation_euler else 0.0
        current_axis_deg = _normalize_angle_180(math.degrees(current_rot_z))

    if original_rot_z is not None:
        current_rot_z = target_obj.rotation_euler.z if target_obj.rotation_euler else 0.0
        current_rot_deg = math.degrees(current_rot_z)
        ref_rot_deg = math.degrees(float(original_rot_z))
        delta_rot = current_rot_deg - ref_rot_deg
        ref_axis_deg = _normalize_angle_180(current_axis_deg - delta_rot)
        err = _axis_error_deg(current_axis_deg, ref_axis_deg)
        return err > float(threshold_deg)

    if not road_objs:
        return False

    nearest_axis_deg = _nearest_road_axis_deg(target_obj, road_objs, radius=ORIENTATION_NEAREST_ROAD_RADIUS)
    if nearest_axis_deg is None:
        return False

    err = _axis_error_deg(current_axis_deg, nearest_axis_deg)
    return err > float(threshold_deg)


def _check_nearby_road_orthogonality(target_obj: bpy.types.Object, road_objs: list, radius: float = 40.0) -> bool:
    """Check if roads within radius form ~90 deg intersections."""
    if not road_objs: return False
    
    center = target_obj.location
    nearby_vecs = []
    
    # Collect nearby road edge vectors
    for r_obj in road_objs:
        # Quick bounds check
        dist = (Vector((r_obj.location.x, r_obj.location.y, 0)) - Vector((center.x, center.y, 0))).length
        if dist > radius + max(r_obj.dimensions): continue
        
        try:
             # Just sample mesh edges for direction
             mesh = r_obj.data
             # Check random sample of edges to save time
             indices = range(0, len(mesh.edges), max(1, len(mesh.edges)//20))
             for i in indices:
                 edge = mesh.edges[i]
                 v1_w = r_obj.matrix_world @ mesh.vertices[edge.vertices[0]].co
                 v2_w = r_obj.matrix_world @ mesh.vertices[edge.vertices[1]].co
                 
                 mid = (v1_w + v2_w) / 2
                 if (mid - center).length < radius:
                     vec = v2_w - v1_w
                     vec.z = 0
                     if vec.length > 0.5:
                         nearby_vecs.append(vec.normalized())
        except: pass
        
        if len(nearby_vecs) > 30: break
    
    if len(nearby_vecs) < 2: return False
    
    # Check for ~90 deg pairs
    for i in range(len(nearby_vecs)):
        for j in range(i+1, len(nearby_vecs)):
             # angle in radians
             angle = nearby_vecs[i].angle(nearby_vecs[j])
             deg = math.degrees(angle)
             # Normalize angle to 0-90
             deg = deg % 180
             if deg > 90: deg = 180 - deg
             
             # If close to 90 deg (e.g. 75-105 range)
             if abs(deg - 90) < 15:
                 return True
                 
    return False


def _is_orientation_candidate_shape(target_obj: bpy.types.Object) -> bool:
    if target_obj is None:
        return False

    try:
        footprint = get_building_footprint(target_obj)
        if footprint is None or footprint.is_empty:
            pts = extract_footprint_polygon_xy(target_obj)
            if len(pts) >= 3:
                footprint = Polygon(pts)
    except Exception:
        return False

    if footprint is None or footprint.is_empty:
        return False

    try:
        if footprint.geom_type == "MultiPolygon":
            polys = [g for g in footprint.geoms if g and (not g.is_empty)]
            if not polys:
                return False
            footprint = max(polys, key=lambda g: g.area)
    except Exception:
        return False

    area = float(getattr(footprint, "area", 0.0) or 0.0)
    perimeter = float(getattr(footprint, "length", 0.0) or 0.0)
    if area <= 1e-4 or perimeter <= 1e-4:
        return False

    mrr = footprint.minimum_rotated_rectangle
    if mrr is None or mrr.is_empty:
        return False
    mrr_area = float(getattr(mrr, "area", 0.0) or 0.0)
    if mrr_area <= 1e-6:
        return False

    convex = footprint.convex_hull
    convex_area = float(getattr(convex, "area", 0.0) or 0.0)
    if convex_area <= 1e-6:
        return False

    rect_ratio = area / mrr_area
    convex_ratio = area / convex_area

    mrr_coords = list(mrr.exterior.coords)
    axis_angle = None
    for i in range(len(mrr_coords) - 1):
        x1, y1 = mrr_coords[i]
        x2, y2 = mrr_coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        if math.hypot(dx, dy) > 1e-6:
            axis_angle = math.degrees(math.atan2(dy, dx))
            break
    if axis_angle is None:
        return False

    def _angle_diff_mod_180(a: float, b: float) -> float:
        d = abs((a - b) % 180.0)
        if d > 90.0:
            d = 180.0 - d
        return d

    coords = list(footprint.exterior.coords)
    if len(coords) < 4:
        return False

    total_edge_len = 0.0
    orth_edge_len = 0.0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        seg_len = math.hypot(dx, dy)
        if seg_len < 0.25:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        d_main = _angle_diff_mod_180(ang, axis_angle)
        d_orth = _angle_diff_mod_180(ang, axis_angle + 90.0)
        total_edge_len += seg_len
        if min(d_main, d_orth) <= 18.0:
            orth_edge_len += seg_len

    if total_edge_len <= 1e-6:
        return False
    orth_ratio = orth_edge_len / total_edge_len
    if orth_ratio < 0.75:
        return False

    simple = footprint.simplify(0.15, preserve_topology=True)
    if simple is None or simple.is_empty or not hasattr(simple, "exterior"):
        simple = footprint
    s_coords = list(simple.exterior.coords) if hasattr(simple, "exterior") else []
    if len(s_coords) < 4:
        return False

    corner_count = 0
    right_corner_count = 0
    n = len(s_coords) - 1
    for i in range(n):
        p_prev = s_coords[(i - 1) % n]
        p_cur = s_coords[i]
        p_next = s_coords[(i + 1) % n]

        v1x, v1y = (p_prev[0] - p_cur[0]), (p_prev[1] - p_cur[1])
        v2x, v2y = (p_next[0] - p_cur[0]), (p_next[1] - p_cur[1])
        len1 = math.hypot(v1x, v1y)
        len2 = math.hypot(v2x, v2y)
        if len1 < 0.25 or len2 < 0.25:
            continue

        dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (len1 * len2)))
        ang = math.degrees(math.acos(dot))

        if abs(ang - 180.0) <= 20.0:
            continue

        corner_count += 1
        if abs(ang - 90.0) <= 25.0:
            right_corner_count += 1

    if corner_count == 0:
        return False
    right_ratio = right_corner_count / corner_count
    if right_ratio < 0.75:
        return False

    is_rectangle = (corner_count <= 5) and (rect_ratio >= 0.82) and (convex_ratio >= 0.95)
    is_l_shape = (6 <= corner_count <= 8) and (0.45 <= rect_ratio <= 0.82) and (0.55 <= convex_ratio <= 0.92)

    return is_rectangle or is_l_shape


def _check_scale_issue(target_obj: bpy.types.Object, original_scale: tuple = None, threshold_ratio: float = 0.7) -> bool:
    """

    Args:
        target_obj: Goal
        original_scale:  (x, y, z)
        threshold_ratio: /
    """
    if target_obj is None:
        return False

    current_scale = target_obj.scale

    if original_scale is None:
        
        avg_scale = (abs(current_scale.x) + abs(current_scale.y) + abs(current_scale.z)) / 3.0
        return avg_scale < threshold_ratio or avg_scale > (1.0 / threshold_ratio)

    
    scale_ratio_x = abs(current_scale.x) / max(abs(original_scale[0]), 0.001)
    scale_ratio_y = abs(current_scale.y) / max(abs(original_scale[1]), 0.001)
    scale_ratio_z = abs(current_scale.z) / max(abs(original_scale[2]), 0.001)

    avg_ratio = (scale_ratio_x + scale_ratio_y + scale_ratio_z) / 3.0

    return avg_ratio < threshold_ratio or avg_ratio > (1.0 / threshold_ratio)


def _choice_makes_scene_clean(choice_text: str, building_objs: list, target_building_id: str) -> bool:
    
    parsed = _parse_choice_text(choice_text)
    if parsed is None:
        return False

    
    target_obj = None
    for obj in building_objs:
        
        
        pass

    
    

    return False


def generate_distractors(issue_type, bid, correct_action_text="", building_objs=None, id_to_obj=None, num_distractors=3):
    """ - """
    distractors = []

    
    correct_op = None
    if correct_action_text:
        parsed = _parse_choice_text(correct_action_text)
        if parsed:
            correct_op = parsed.get("op")

    
    if correct_op != "move":
        
        for _ in range(num_distractors):
            d = random.choice(MOVE_DIRS)
            m = random.choice(MOVE_DISTS_M)
            distractors.append(f"Move building {bid} {_get_full_dir_name(d)} by {m} units")

    if correct_op != "rotate" and len(distractors) < num_distractors:
        
        for _ in range(num_distractors - len(distractors)):
            deg = random.choice(ORIENTATION_ROTATE_CHOICES_DEG)
            direction = "clockwise" if random.random() < 0.5 else "counter-clockwise"
            distractors.append(f"Rotate building {bid} {direction} by {deg}°")

    if correct_op != "scale" and len(distractors) < num_distractors:
        
        if random.random() < 0.5:
            distractors.append(f"Scale up building {bid} by 25%")
        else:
            distractors.append(f"Scale down building {bid} by 25%")

    
    distractors = list(dict.fromkeys(distractors))
    return distractors[:num_distractors]


def generate_distractors_with_filter(issue_type, bid, correct_action_text="", building_objs=None, id_to_obj=None, num_distractors=3):
    """
    Generate distractors (incorrect options).
    A Valid Distractor must:
      1. Fail to fix the main issue (still broken).
      OR
      2. Fix the main issue BUT introduce a NEW issue (B-B, B-R, or Misalignment).
      
    If an option fixes the main issue and introduces NO new issues, it is a CORRECT answer (invalid distractor).
    """
    if building_objs is None or id_to_obj is None:
        return generate_distractors(issue_type, bid, correct_action_text, num_distractors=num_distractors)

    # Get road objects for checking B-R conflicts
    road_objs = [obj for obj in bpy.data.objects if obj.type == "MESH" and ("road" in obj.name.lower() or "Road" in obj.name)]

    try:
        target_label = int(bid) if str(bid).isdigit() else 1
    except:
        target_label = 1
    target_obj = id_to_obj.get(target_label)

    if target_obj is None:
        return generate_distractors(issue_type, bid, correct_action_text, num_distractors=num_distractors)

    # Snapshot current ERROR state
    original_transform = _snapshot_obj_transform(target_obj)

    # Define the "Main Issue" key based on issue_type
    main_issue_key = None
    
    # Normalize issue type
    t = str(issue_type).lower()
    
    if t in ["building_collision", "overlap", "overlap_by_move", "overlap_by_rotate", "overlap_by_scale", "issue_overlap"]:
        main_issue_key = "has_bb"
    elif t in ["road_collision", "road_conflict", "issue_road", "road"]:
        main_issue_key = "has_br"
    elif t in ["misalignment", "orientation", "angle", "issue_angle"]:
        main_issue_key = "is_misaligned"
    
    # Identify directions to avoid (correct answer + neighbors)
    avoid_dirs = set()
    parsed_correct = _parse_choice_text(correct_action_text)
    if parsed_correct and parsed_correct.get("op") == "move":
        cd = parsed_correct.get("dir")
        # Normalize to full name
        cd_full = _get_full_dir_name(cd)
        avoid_dirs.add(cd_full)
        
        neighbors_map = {
            "North": ["Northwest", "Northeast"],
            "South": ["Southwest", "Southeast"],
            "East": ["Northeast", "Southeast"],
            "West": ["Northwest", "Southwest"],
            "Northeast": ["North", "East"],
            "Northwest": ["North", "West"],
            "Southeast": ["South", "East"],
            "Southwest": ["South", "West"]
        }
        for n in neighbors_map.get(cd_full, []):
            avoid_dirs.add(n)

    # Generate a pool of candidates
    candidates = []
    
    # 1. Move candidates
    for d in MOVE_DIRS:
        full_d = _get_full_dir_name(d)
        if full_d in avoid_dirs:
            continue
            
        for m in MOVE_DISTS_M:
            candidates.append(f"Move building {bid} {full_d} by {m} units")
            
    # 2. Rotate candidates (Construct large pool of angles)
    angles = [15, 30, 45, 90, 10, 20, 5, 25, 35, 60, 120]
    for deg in angles:
        candidates.append(f"Rotate building {bid} clockwise by {deg}°")
        candidates.append(f"Rotate building {bid} counter-clockwise by {deg}°")
        
    # 3. Scale candidates
    for pct in [10, 25, 50, 75]:
        candidates.append(f"Scale up building {bid} by {pct}%")
        candidates.append(f"Scale down building {bid} by {pct}%")

    # Shuffle
    random.shuffle(candidates)
    
    unique_distractors = set()
    valid_distractors = []
    
    try:
        for cand in candidates:
            if len(unique_distractors) >= num_distractors:
                break
                
            if cand == correct_action_text:
                continue
                
            # Parse
            parsed = _parse_choice_text(cand)
            if not parsed: 
                continue
                
            # Restore to Error State before applying
            _restore_obj_transform(target_obj, original_transform)
            
            # Apply candidate
            try:
                _apply_choice(target_obj, parsed)
                bpy.context.view_layer.update()
            except Exception as e:
                # If application fails, skip
                continue
            
            # Check Status
            status = _get_object_state_status(target_obj, building_objs, road_objs)
            
            # Logic:
            # Distractor is VALID if:
            #   (Main Issue Persists) 
            #   OR 
            #   (Main Issue Fixed BUT New Issue Exists)
            
            main_still_broken = False
            if main_issue_key and status.get(main_issue_key):
                main_still_broken = True
                
            is_valid = False
            if main_still_broken:
                # Valid: Failed to fix
                is_valid = True
            else:
                # Main Fixed. Check for new issues.
                has_any_issues = status['has_bb'] or status['has_br'] or status['is_misaligned']
                if has_any_issues:
                    # Valid: Fixed main, but broke something else
                    is_valid = True
                else:
                    # Invalid: accidental correct fix
                    is_valid = False

            if is_valid:
                if cand not in unique_distractors:
                    unique_distractors.add(cand)
                    valid_distractors.append(cand)

    finally:
        # Always restore finally (in case exception occurring during loop)
        _restore_obj_transform(target_obj, original_transform)
        bpy.context.view_layer.update()

    # Fill if not enough
    final_list = list(unique_distractors)
    tries = 0
    while len(final_list) < num_distractors and tries < 50:
        tries += 1
        # Generate random dummy that likely won't work or collide
        m_dist = random.uniform(0.1, 0.5)
        d_cand = f"Move building {bid} North by {m_dist:.2f} units"
        if d_cand not in unique_distractors and d_cand != correct_action_text:
             unique_distractors.add(d_cand)
             final_list.append(d_cand)
             
    return final_list[:num_distractors]



def qa3_fix(issue_meta, inject_action, region_labels, images, building_objs=None, id_to_obj=None, option_images=None, fixed_options_text=None):
    """QA3: """
    issue = issue_meta["issues"][0]
    t = issue["type"]

    building_labels = issue.get("building_labels", [])
    bid = building_labels[0] if building_labels else "1"

    if t == ISSUE_OVERLAP:
        if len(building_labels) >= 2:
            known = f"A problem is detected in the image: Building {building_labels[0]} overlaps with another building."
        else:
            known = f"A problem is detected in the image: Building {bid} overlaps with another building."

    elif t == ISSUE_ROAD or t == "road_conflict":
        known = f"A problem is detected in the image: Building {bid} overlaps with the road."

    elif t == ISSUE_SCALE or t == "scale_small" or t == "scale_large":
        known = f"A problem is detected in the image: Building {bid} has an abnormal scale."

    elif t == ISSUE_ANGLE:
        known = f"A problem is detected in the image: Building {bid} has an abnormal orientation."

    else:
        known = "A problem is detected in the image: Unknown."

    correct_action = inject_action.get("reverse_choice_text", "")

    if fixed_options_text:
        all_options = fixed_options_text
    else:
        
        distractors = generate_distractors_with_filter(
            t, bid, correct_action, building_objs, id_to_obj, num_distractors=3
        )

        all_options = [correct_action] + distractors
        all_options = list(dict.fromkeys(all_options))

        
        while len(all_options) < 4:
            
            if t == ISSUE_OVERLAP or t == ISSUE_ROAD:
                d = random.choice(MOVE_DIRS)
                m = random.choice(MOVE_DISTS_M)
                new_option = f"Move building {bid} {_get_full_dir_name(d)} by {m} units"
            elif t == ISSUE_ANGLE:
                deg = random.choice(ORIENTATION_ROTATE_CHOICES_DEG)
                direction = "clockwise" if random.random() < 0.5 else "counter-clockwise"
                new_option = f"Rotate building {bid} {direction} by {deg}°"
            else:
                
                pct = random.choice(SCALE_PERCENT_CHOICES)
                new_option = f"Scale up building {bid} by {pct}%" if random.random() < 0.5 else f"Scale down building {bid} by {pct}%"

            if new_option not in all_options:
                all_options.append(new_option)

    
    reference_images = []
    if fixed_options_text and option_images and len(option_images) == len(fixed_options_text):
        final_options = fixed_options_text
        final_images = images
        reference_images = option_images
    else:
        raise ValueError("[ERROR] For QA3 modification task, fixed_options_text and option_images MUST be provided and match in length!")

    letters = ["A", "B", "C", "D"]
    choices = [f"{letters[i]}. {final_options[i]}" for i in range(min(4, len(final_options)))]
    ans_letter = letters[final_options.index(correct_action)]

    q = (
        f"{known}\n"
        "Choose ONE action to fix it in ONE step. The fix should resolve the issue without introducing new problems.\n\n"
        "Reference information:\n"
        "- Buildings are labeled with numbers (1, 2, 3, etc.) at their center positions.\n"
        "- Directions in the image: North=up, South=down, East=right, West=left.\n"
        "- There is a 1 unit scale bar in the North panel for distance reference.\n"
        "- Possible issues to avoid introducing: (1) A building overlaps another building, (2) A building overlaps the yellow road network, (3) A building has an abnormal rotation angle (not aligned with roads).\n"
        "Top-view image: <image>"
    )

    return {
        "question": q + "\n\n" + "\n".join(choices),
        "answer": ans_letter,
        "task_type": "top_error_modify",
        "meta": {"issue_meta": issue_meta, "inject_action": inject_action},
        "images": final_images,
        "reference_images": reference_images,
    }


# --------------------------------------------------------------------------------------
# Synthetic Anomaly Generation
# --------------------------------------------------------------------------------------
def create_synthetic_anomaly(region_info, region_labels):
    
    building_ids = region_info.get("building_ids", [])
    if not building_ids:
        return None

    
    
    
    
    anomaly_types = [
        "overlap_by_move", "overlap_by_rotate", "overlap_by_scale",
        "road_conflict", "orientation"
    ]
    anomaly_type = random.choice(anomaly_types)

    building_id = random.choice(building_ids)

    bid = None
    for label_num, bid_original in region_labels.items():
        if bid_original == building_id:
            bid = label_num
            break
    if not bid:
        bid = "1"

    if anomaly_type == "overlap_by_move":
        d = random.choice(MOVE_DIRS)
        m = round(random.uniform(0.5, 6.0), 1)  # 1
        d_code = _normalize_dir_code(d, default="E")
        opposite_code = _get_opposite_dir_code(d_code)
        fix_action = f"Move building {bid} {_dir_code_to_full_name(opposite_code)} by {m} units"

        anomaly = {
            "type": "overlap",
            "severity": "high",
            "building_ids": [building_id],
            "description": f"Building {bid} overlaps with another building",
            "metric_value": 0.3,
            "synthetic": True,
            "inject_action": {
                "op": "move",
                "dir": d_code,
                "dir_word": DIR_CODE_TO_WORD.get(d_code, "east"),
                "dist_m": m,
                "reverse_choice_text": fix_action,
            }
        }

    elif anomaly_type == "overlap_by_rotate":
        candidates = []

        rotate_angles = list(range(5, 180, 5))
        random.shuffle(rotate_angles)

        for deg in rotate_angles:
            for direction in ["clockwise", "counter-clockwise"]:
                reverse_direction = "counter-clockwise" if direction == "clockwise" else "clockwise"
                fix_action = f"Rotate building {bid} {reverse_direction} by {deg}°"

                inject_action = {
                    "op": "rotate",
                    "deg": deg,
                    "clockwise": direction == "clockwise",
                    "reverse_choice_text": fix_action,
                }

                candidates.append({
                    "type": "overlap",
                    "severity": "high",
                    "building_ids": [building_id],
                    "description": f"Building {bid} overlaps with another building",
                    "metric_value": float(deg),
                    "synthetic": True,
                    "inject_action": inject_action,
                    "candidate_params": {"deg": deg, "direction": direction}
                })

        random.shuffle(candidates)
        return candidates

    elif anomaly_type == "overlap_by_scale":
        
        scale_factor, reverse_pct, reverse_dir = _sample_scale_factor_from_nice_reverse(
            allow_up_injection=True,
            allow_down_injection=True,
        )
        fix_action = _build_inverse_scale_action_text(bid, scale_factor)

        anomaly = {
            "type": "overlap",
            "severity": "high",
            "building_ids": [building_id],
            "description": f"Building {bid} overlaps with another building",
            "metric_value": 0.3,
            "synthetic": True,
            "inject_action": {
                "op": "scale",
                "scale_type": "" if scale_factor > 1.0 else "",
                "scale_factor": scale_factor,
                "reverse_choice_text": fix_action,
                "reverse_pct": reverse_pct,
                "reverse_dir": reverse_dir,
            }
        }

    elif anomaly_type == "orientation":
        deg = random.choice(ORIENTATION_ROTATE_CHOICES_DEG)
        direction = "clockwise" if random.random() < 0.5 else "counter-clockwise"
        reverse_direction = "counter-clockwise" if direction == "clockwise" else "clockwise"
        fix_action = f"Rotate building {bid} {reverse_direction} by {deg}°"

        anomaly = {
            "type": "orientation",
            "severity": "high",
            "building_ids": [building_id],
            "description": f"Building {bid} has an abnormal orientation (rotated {deg}°)",
            "metric_value": float(deg),
            "synthetic": True,
            "inject_action": {
                "op": "rotate",
                "deg": deg,
                "clockwise": direction == "clockwise",
                "reverse_choice_text": fix_action,
            }
        }

    elif anomaly_type in ["scale_small", "scale_large"]:
        scale_type = "" if anomaly_type == "scale_small" else ""
        if scale_type == "":
            scale_action = f"Scale up building {bid} by 20%"
            fix_action = f"Scale down building {bid} by 20%"
        else:
            scale_action = f"Scale down building {bid} by 20%"
            fix_action = f"Scale up building {bid} by 20%"

        anomaly = {
            "type": anomaly_type,
            "severity": "high",
            "building_ids": [building_id],
            "description": f"Building {bid} has an abnormal scale ({scale_type})",
            "metric_value": 0.3,
            "synthetic": True,
            "inject_action": {
                "op": "scale",
                "scale_type": scale_type,
                "reverse_choice_text": fix_action,
            }
        }

    return anomaly


def create_synthetic_anomaly_by_type(region_info, building_ids, anomaly_type: str,
                                     target_obj=None, other_objs=None, id_to_obj=None,
                                     road_objs_override=None, frame_bounds: dict | None = None):
    """ - 

    Task 1: B-B Collision (overlap_by_move, overlap_by_rotate, overlap_by_scale)
            Constraint: has_bb=True, has_br=False, is_misaligned=False
    Task 2: B-R Conflict (road_conflict)
            Constraint: has_br=True, has_bb=False, is_misaligned=False
    Task 3: Misalignment (orientation)
            Constraint: is_misaligned=True, has_bb=False, has_br=False
    """
    if not building_ids:
        building_ids = region_info.get("building_ids", [])
        if not building_ids:
            print(f"[DEBUG] create_synthetic_anomaly_by_type: building_ids is empty!")
            return None

    # Determine candidates to try (list of building_ids)
    candidate_bids = []
    
    # If target_obj is forced, try only that one
    if target_obj is not None and id_to_obj is not None:
        found_target_id = None
        for label, obj in id_to_obj.items():
            if obj == target_obj:
                label_num = int(label)
                if 1 <= label_num <= len(building_ids):
                    found_target_id = building_ids[label_num - 1]
                break
        
        if found_target_id:
            candidate_bids = [found_target_id]
        else:

            candidate_bids = list(building_ids)
            random.shuffle(candidate_bids)
    else:
        # Try all buildings in random order
        candidate_bids = list(building_ids)
        random.shuffle(candidate_bids)

    # Environment objects for checking
    if road_objs_override is not None:
        all_road_objs = [obj for obj in road_objs_override if obj and obj.type == "MESH"]
    else:
        all_road_objs = [obj for obj in bpy.data.objects if obj.type == "MESH" and ("road" in obj.name.lower() or "path" in obj.name.lower())]
    
    all_building_objs_list = []
    if id_to_obj:
        all_building_objs_list = list(id_to_obj.values())
    elif other_objs:
         all_building_objs_list = other_objs + ([target_obj] if target_obj else [])


    # Helper: Check Strict Constraints
    def check_constraints_for_type(obj, a_type):
        status = _get_object_state_status(obj, all_building_objs_list, all_road_objs)
        # B-B types
        if a_type in ["overlap_by_move", "overlap_by_rotate", "overlap_by_scale"]:
            # Need B-B. Relaxed: Ignore B-R conflict to ensure generation in dense maps.
            # Originally: res = status["has_bb"] and not status["has_br"]
            res = status["has_bb"]
            if not res:
                # print(f"[DEBUG_CONSTRAINT] {a_type} check failed for {obj.name}: has_bb={status['has_bb']}, has_br={status['has_br']}")
                pass
            return res
        # B-R types
        elif a_type in ["road_conflict", "road_conflict_by_move", "road_conflict_by_rotate", "road_conflict_by_scale"]:
             # Need B-R, NO B-B, NO Misalign
             res = status["has_br"] and not status["has_bb"] and not status["is_misaligned"]
             if not res:
                # print(f"[DEBUG_CONSTRAINT] {a_type} check failed for {obj.name}: has_bb={status['has_bb']}, has_br={status['has_br']}, is_misaligned={status['is_misaligned']}")
                pass
             return res
        # Misalign
        elif a_type == "orientation":
             # Need Misalign, NO B-B, NO B-R
             res = status["is_misaligned"] and not status["has_bb"] and not status["has_br"]
             if not res:
                print(f"[DEBUG_CONSTRAINT] {a_type} check failed for {obj.name}: has_bb={status['has_bb']}, has_br={status['has_br']}, is_misaligned={status['is_misaligned']}")
             return res
        return False

    
    
    orientation_shape_candidate_ids = set()
    if anomaly_type == "orientation":
        for bid in candidate_bids:
            obj_tmp = bpy.data.objects.get(bid)
            if obj_tmp and _is_orientation_candidate_shape(obj_tmp):
                orientation_shape_candidate_ids.add(bid)

    # Loop over candidates until success
    for building_id in candidate_bids:
        # Resolve target object
        target_obj = bpy.data.objects.get(building_id)
        
        # Determine bid (label string)
        bid = "1"
        try:
            full_idx = building_ids.index(building_id)
            bid = str(full_idx + 1)
        except ValueError: pass
        
        if not target_obj:
             if id_to_obj: # Fallback lookup
                 try: target_obj = id_to_obj.get(int(bid))
                 except: pass
        
        if not target_obj:
            continue # Skip invalid building

        # Define other_objs for this target (excluding self)
        current_other_objs = [o for o in all_building_objs_list if o != target_obj]

        # Helper to dry run a transformation
        def try_modification_and_check(obj, transform_func, a_type):
            snap = _snapshot_obj_transform(obj)
            baseline_br = False
            baseline_bb = False
            if a_type == "orientation":
                baseline_br = _check_road_conflict(
                    all_road_objs,
                    [],
                    target_obj=obj,
                    threshold_area=STRICT_NO_ROAD_CONFLICT_AREA_THRESHOLD,
                )
                baseline_bb = _check_target_building_overlap(obj, all_building_objs_list, pad=0.0)
                if isinstance(baseline_bb, tuple):
                    baseline_bb = baseline_bb[0]
            try:
                transform_func(obj)
                bpy.context.view_layer.update()
                if not _is_obj_inside_frame_bounds(obj, frame_bounds, margin=FRAME_CONSTRAINT_MARGIN):
                    print("[DEBUG] FAIL: Target object moved out of frame bounds")
                    return False
                
                # Use snapshot for original rotation reference
                if snap.get("rotation") is not None:
                    original_rot_z = snap.get("rotation").z
                elif snap.get("matrix_world") is not None:
                    original_rot_z = snap.get("matrix_world").to_euler('XYZ').z
                else:
                    original_rot_z = obj.rotation_euler.z
                
                # Orientation should be judged by road context (absolute misalignment),
                # not only by relative rotation delta from original snapshot.
                if a_type == "orientation":
                    status = _get_object_state_status(obj, all_building_objs_list, all_road_objs, original_rot_z=None)
                else:
                    status = _get_object_state_status(obj, all_building_objs_list, all_road_objs, original_rot_z=original_rot_z)
                
                # DEBUG PRINT
                if a_type == "orientation":
                    print(f"[DEBUG] Orientation check: has_br={status['has_br']}, has_bb={status['has_bb']}, is_misaligned={status['is_misaligned']}")
                
                # B-B types
                if a_type in ["overlap_by_move", "overlap_by_rotate", "overlap_by_scale"]:
                    res = status["has_bb"]
                    # Strict check: Must NOT have any B-R overlap.
                    if _check_road_conflict(
                        all_road_objs,
                        [],
                        target_obj=obj,
                        threshold_area=STRICT_NO_ROAD_CONFLICT_AREA_THRESHOLD,
                    ):
                        return False
                    if status["is_misaligned"]:
                        return False
                    return res

                # B-R types
                elif a_type in ["road_conflict", "road_conflict_by_move", "road_conflict_by_rotate", "road_conflict_by_scale"]:
                     if not status["has_br"]: return False
                     
                     # Strict check: Must NOT have B-B (Building Overlap)
                     # Using 0.0 pad to be strict
                     strict_bb = _check_target_building_overlap(obj, all_building_objs_list, pad=0.0) 
                     if isinstance(strict_bb, tuple): strict_bb = strict_bb[0]
                     if strict_bb: return False
                     if status["is_misaligned"]:
                         return False
                     
                     return True

                # Misalign
                elif a_type == "orientation":
                     # Must be close to roads so the orientation issue is visually meaningful
                     nearest_road_info = _find_nearest_road_direction(obj, all_road_objs) if all_road_objs else None
                     nearest_dist = nearest_road_info.get("nearest_dist_m", nearest_road_info.get("dist_m", 1e9)) if nearest_road_info else 1e9
                     if nearest_dist > ORIENTATION_MAX_ROAD_DIST_M:
                         print(
                             f"[DEBUG] FAIL: Too far from road for orientation "
                             f"(dist={nearest_dist:.2f}m > {ORIENTATION_MAX_ROAD_DIST_M:.2f}m)"
                         )
                         return False

                     
                     new_br = _check_road_conflict(
                         all_road_objs,
                         [],
                         target_obj=obj,
                         threshold_area=STRICT_NO_ROAD_CONFLICT_AREA_THRESHOLD,
                     )
                     if (not baseline_br) and new_br:
                         print("[DEBUG] FAIL: New road conflict introduced (strict any-overlap)")
                         return False
                     
                     
                     new_bb = _check_target_building_overlap(obj, all_building_objs_list, pad=0.0)
                     if isinstance(new_bb, tuple):
                         new_bb = new_bb[0]
                     if (not baseline_bb) and new_bb:
                         print(f"[DEBUG] FAIL: New building conflict introduced (strict)")
                         return False
                     
                     res = status["is_misaligned"]
                     if not res:
                         print(f"[DEBUG] FAIL: Not misaligned")
                     return res
                return False

            finally:
                _restore_obj_transform(obj, snap)
                bpy.context.view_layer.update()


        # --- Task 1: B-B Collision ---
        if anomaly_type == "overlap_by_move":
            # Smart direction if possible, else random
            # For move, we want to hit another building
            use_smart = len(current_other_objs) > 0
            candidates_to_try = []
            
            if use_smart:
                d_info = _find_nearest_building_direction(target_obj, current_other_objs)
                if d_info:
                     candidates_to_try.append((d_info["dir_word"], d_info["dist_m"]))
            
         
            for _ in range(5):
                 candidates_to_try.append((random.choice(MOVE_DIRS), round(random.uniform(0.5, 6.0), 1)))

            for d, m in candidates_to_try:
                 # Define Apply
                 def apply_move(o):
                     v = _get_dir_vector(d)
                     dx, dy = v
                     scale = 1.0
                     if dx!=0 and dy!=0: scale = 1.0 / math.sqrt(2)
                     o.location.x += dx * m * UNIT_SCALE * scale
                     o.location.y += dy * m * UNIT_SCALE * scale
                
                 if target_obj and try_modification_and_check(target_obj, apply_move, anomaly_type):
                   
                     d_code = _normalize_dir_code(d, default="E")
                     opp = _get_opposite_dir_code(d_code)
                     fix_action = f"Move building {bid} {_dir_code_to_full_name(opp)} by {m} units"

                     return {
                        "type": "overlap",
                        "severity": "high",
                        "building_ids": [building_id],
                        "description": f"Building {bid} overlaps with another building",
                        "metric_value": 0.3,
                        "synthetic": True,
                        "inject_action": {
                            "op": "move", 
                            "dir": d_code, 
                            "dir_word": DIR_CODE_TO_WORD.get(d_code, "east"), 
                            "dist_m": m,
                            "reverse_choice_text": fix_action
                        }
                     }

        elif anomaly_type == "overlap_by_rotate":
            candidates = []
    
            rotate_angles = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 110, 120, 135, 150]
            
            for deg in rotate_angles:
                for direction in ["clockwise", "counter-clockwise"]:
                    if target_obj:
                        sgn = -1 if direction == "clockwise" else 1
                        rad = math.radians(deg) * sgn
                        def apply_rot(o):
                            # Ensure consistency with apply_inject_action
                            temp_action = {"op": "rotate", "deg": deg, "clockwise": (direction == "clockwise")}
                            apply_inject_action(o, temp_action, UNIT_SCALE)
                        
                        if not try_modification_and_check(target_obj, apply_rot, anomaly_type):
                            continue
                    
                    # If valid or no target_obj to check
                    rev_dir = "counter-clockwise" if direction == "clockwise" else "clockwise"
                    fix_action = f"Rotate building {bid} {rev_dir} by {deg}°"
                    
                    candidates.append({
                        "type": "overlap", 
                        "severity": "high", 
                        "building_ids": [building_id],
                        "description": f"Building {bid} overlaps with another building",
                        "metric_value": float(deg),
                        "synthetic": True,
                        "inject_action": {
                            "op": "rotate", "deg": deg, "clockwise": (direction=="clockwise"),
                            "reverse_choice_text": fix_action
                        },
                        "candidate_params": {"deg": deg, "direction": direction}
                    })
            
            if candidates:
                random.shuffle(candidates)
                return candidates

        elif anomaly_type == "overlap_by_scale":
            candidates = []
            
            reverse_pcts = NICE_SCALE_REVERSE_PCTS

            for reverse_pct in reverse_pcts:
                scale_factor = 1.0 / max(1e-6, (1.0 - reverse_pct / 100.0))
                if target_obj:
                    def apply_scale(o):
                        temp_action = {
                            "op": "scale",
                            "scale_type": "",
                            "scale_factor": scale_factor
                        }
                        apply_inject_action(o, temp_action, UNIT_SCALE)

                    if not try_modification_and_check(target_obj, apply_scale, anomaly_type):
                        continue

                fix_action = _build_inverse_scale_action_text(bid, scale_factor)
                inject_pct = (scale_factor - 1.0) * 100.0
                candidates.append({
                    "type": "overlap", "severity": "high", "building_ids": [building_id],
                    "description": f"Building {bid} overlaps with another building",
                    "metric_value": 0.3, "synthetic": True,
                    "inject_action": {
                        "op": "scale", "scale_type": "", "scale_factor": scale_factor,
                        "reverse_choice_text": fix_action
                    },
                    "candidate_params": {"inject_pct": round(inject_pct, 1), "reverse_pct": reverse_pct}
                })
            
            if candidates:
                random.shuffle(candidates)
                return candidates

        # --- Task 2: B-R Collision ---
        elif anomaly_type in ["road_conflict", "road_conflict_by_move"]:
             # Similar to move, but targeting road
             # Use all_road_objs for smart check
             use_smart = len(all_road_objs) > 0
             candidates_to_try = []
             
             if use_smart:
                 d_info = _find_nearest_road_direction(target_obj, all_road_objs)
                 if d_info:
                      candidates_to_try.append((d_info["dir_word"], d_info["dist_m"]))
             
             for _ in range(5):
                 candidates_to_try.append((random.choice(MOVE_DIRS), round(random.uniform(0.5, 6.0), 1)))
                 
             for d, m in candidates_to_try:
                 def apply_move(o):
                     v = _get_dir_vector(d)
                     dx, dy = v
                     scale = 1.0
                     if dx!=0 and dy!=0: scale = 1.0 / math.sqrt(2)
                     o.location.x += dx * m * UNIT_SCALE * scale
                     o.location.y += dy * m * UNIT_SCALE * scale
                
                 if target_obj and try_modification_and_check(target_obj, apply_move, anomaly_type):
                     # Valid B-R strictly
                     d_code = _normalize_dir_code(d, default="E")
                     opp = _get_opposite_dir_code(d_code)
                     fix_action = f"Move building {bid} {_dir_code_to_full_name(opp)} by {m} units"
                     
                     return {
                        "type": "road_conflict", "severity": "high", "building_ids": [building_id],
                        "description": f"Building {bid} overlaps with the road",
                        "metric_value": 0.3, "synthetic": True,
                        "inject_action": {
                            "op": "move", "dir": d_code, "dir_word": DIR_CODE_TO_WORD.get(d_code, "east"), "dist_m": m,
                            "reverse_choice_text": fix_action
                        }
                     }

        elif anomaly_type == "road_conflict_by_rotate":
            candidates = []
            # Similarly, we can try other angles.
            rotate_angles = [30, 45, 60, 75, 90]
            
            for deg in rotate_angles:
                for direction in ["clockwise", "counter-clockwise"]:
                    if target_obj:
                        sgn = -1 if direction == "clockwise" else 1
                        rad = math.radians(deg) * sgn
                        def apply_rot(o):
                            # Ensure consistency with apply_inject_action (rotate around center)
                            temp_action = {"op": "rotate", "deg": deg, "clockwise": (direction == "clockwise")}
                            apply_inject_action(o, temp_action, UNIT_SCALE)
                        
                        if not try_modification_and_check(target_obj, apply_rot, anomaly_type):
                            continue
                    
                    rev_dir = "counter-clockwise" if direction == "clockwise" else "clockwise"
                    fix_action = f"Rotate building {bid} {rev_dir} by {deg}°"
                    
                    candidates.append({
                        "type": "road_conflict", "severity": "high", "building_ids": [building_id],
                        "description": f"Building {bid} overlaps with the road (rotated {deg}°)",
                        "metric_value": float(deg), "synthetic": True,
                        "inject_action": {
                            "op": "rotate", "deg": deg, "clockwise": (direction=="clockwise"),
                            "reverse_choice_text": fix_action
                        },
                        "candidate_params": {"deg": deg, "direction": direction}
                    })
            
            if candidates:
                random.shuffle(candidates)
                return candidates

        elif anomaly_type == "road_conflict_by_scale":
            candidates = []
            reverse_pcts = [20, 25, 30, 35, 40]

            for reverse_pct in reverse_pcts:
                scale_factor = 1.0 / max(1e-6, (1.0 - reverse_pct / 100.0))
                if target_obj:
                    def apply_scale(o):
                        temp_action = {
                            "op": "scale",
                            "scale_type": "",
                            "scale_factor": scale_factor
                        }
                        apply_inject_action(o, temp_action, UNIT_SCALE)

                    if not try_modification_and_check(target_obj, apply_scale, anomaly_type):
                        continue

                fix_action = _build_inverse_scale_action_text(bid, scale_factor)
                inject_pct = (scale_factor - 1.0) * 100.0
                candidates.append({
                    "type": "road_conflict", "severity": "high", "building_ids": [building_id],
                    "description": f"Building {bid} overlaps with the road (scaled up {_format_percent_value(inject_pct)}%)",
                    "metric_value": 0.3, "synthetic": True,
                    "inject_action": {
                        "op": "scale", "scale_type": "", "scale_factor": scale_factor,
                        "reverse_choice_text": fix_action
                    },
                    "candidate_params": {"inject_pct": round(inject_pct, 1), "reverse_pct": reverse_pct}
                })
            
            if candidates:
                random.shuffle(candidates)
                return candidates

        # --- Task 3: Misalignment ---
        elif anomaly_type == "orientation":
            if orientation_shape_candidate_ids and (building_id not in orientation_shape_candidate_ids):
                continue

            candidates = []
    
            rotate_angles = [30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85]
            
            for deg in rotate_angles:
                for direction in ["clockwise", "counter-clockwise"]:
                    if target_obj:
                        sgn = -1 if direction == "clockwise" else 1
                        rad = math.radians(deg) * sgn
                        
                        def apply_rot(o):
                            # Ensure consistency with apply_inject_action (rotate around center)
                            temp_action = {"op": "rotate", "deg": deg, "clockwise": (direction == "clockwise")}
                            apply_inject_action(o, temp_action, UNIT_SCALE)
                        
                        if not try_modification_and_check(target_obj, apply_rot, anomaly_type):
                            continue
                    
                    rev_dir = "counter-clockwise" if direction == "clockwise" else "clockwise"
                    fix_action = f"Rotate building {bid} {rev_dir} by {deg}°"
                    
                    candidates.append({
                        "type": "orientation", "severity": "high", "building_ids": [building_id],
                        "description": f"Building {bid} has an abnormal orientation (rotated {deg}°)",
                        "metric_value": float(deg), "synthetic": True,
                        "inject_action": {
                            "op": "rotate", "deg": deg, "clockwise": (direction=="clockwise"),
                            "reverse_choice_text": fix_action
                        },
                        "candidate_params": {"deg": deg, "direction": direction}
                    })
            
            if candidates:
                random.shuffle(candidates)
                return candidates

    return None


# --------------------------------------------------------------------------------------
# Blender-specific Operations
# --------------------------------------------------------------------------------------
def apply_inject_action(obj: bpy.types.Object, inject_action: dict, unit_scale: float = 20.0):
    """Apply inject action (move/rotate/scale) to building object"""
    op = inject_action.get("op", "")

    if op == "move":
        dir_word = inject_action.get("dir_word", inject_action.get("dir", "E"))
        dist_m = inject_action.get("dist_m", 1.0)

        dx, dy = _get_dir_vector(dir_word)

        
        is_diagonal = (dx != 0 and dy != 0)
        scale = 1.0 / math.sqrt(2) if is_diagonal else 1.0
        # Apply movement (in scene units)
        obj.location.x += dx * dist_m * unit_scale * scale
        obj.location.y += dy * dist_m * unit_scale * scale

    elif op == "rotate":
        deg = inject_action.get("deg", 45)
        clockwise = inject_action.get("clockwise", True)

        signed_deg = -deg if clockwise else deg
        angle_rad = math.radians(signed_deg)

        print(f"[DEBUG] Rotate: obj={obj.name}, deg={signed_deg}")

        bounds = world_bounds_from_obj(obj)
        center = Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"]))

        current_matrix = obj.matrix_world.copy()
        
        # Rotation
        rot_mat = Matrix.Rotation(angle_rad, 4, 'Z')

        new_matrix = (
            Matrix.Translation(center)
            @ rot_mat
            @ Matrix.Translation(-center)
            @ current_matrix
        )

        obj.matrix_world = new_matrix
        bpy.context.view_layer.update()


    elif op == "scale":
        scale_type = inject_action.get("scale_type", "")
        
        scale_factor = inject_action.get("scale_factor")
        if scale_factor is None:
            scale_factor = 0.8 if scale_type == "" else 1.25
        scale_factor = max(0.05, float(scale_factor))

        print(f"[DEBUG] Scale: obj={obj.name}, factor={scale_factor}")

        
        
        bounds = world_bounds_from_obj(obj)
        aabb_center = Vector((bounds["center_x"], bounds["center_y"], bounds["min_z"]))

        print(f"[DEBUG] AABB center: {aabb_center}")
        print(f"[DEBUG] Object location before: {obj.matrix_world.to_translation()}")

        
        current_matrix = obj.matrix_world.copy()

        
        scale_mat = Matrix.Diagonal((scale_factor, scale_factor, scale_factor, 1.0))

        
        
        T_center = Matrix.Translation(aabb_center)
        T_center_inv = Matrix.Translation(-aabb_center)

        
        new_matrix = T_center @ scale_mat @ T_center_inv @ current_matrix

        
        obj.matrix_world = new_matrix

        bpy.context.view_layer.update()

        print(f"[DEBUG] Object location after: {obj.matrix_world.to_translation()}")

def restore_object(obj: bpy.types.Object, original_state):
    """Restore object to original transform"""
    
    obj.matrix_world = original_state
    
    bpy.context.view_layer.update()


# --------------------------------------------------------------------------------------
# Main Processing
# --------------------------------------------------------------------------------------
def generate_qa_for_region(
    region_id: int,
    region_info: dict,
    region_data: dict,
    labeler: BuildingLabeler,
    output_dir: str,
    sample_idx: int = 0,
    anomaly_type_override: str = None,
):
    """ QA """

    
    building_ids = region_info.get("building_ids", [])
    if not building_ids:
        print(f"  [WARN] Region {region_id} has no building_ids")
        return None

    # Build id_to_obj mapping (label number -> Blender object)
    
    objs = []
    id_to_obj = {}
    for i, bid in enumerate(building_ids):
        o = bpy.data.objects.get(bid)
        if o and o.type == "MESH":
            objs.append(o)
            id_to_obj[i + 1] = o

    if len(objs) < 3:
        print(f"  [WARN] Region {region_id} has only {len(objs)} valid building objects")
        # Still continue but will likely fail later

    # Ensure these objects are movable by location (bake if needed)
    ensure_buildings_movable(objs)

    
    metas = []
    for i in range(len(objs)):
        metas.append(build_meta_from_obj(objs[i], label_id=i + 1))

    
    
    anomaly_types = TARGET_QA_ANOMALY_TYPES
    if anomaly_type_override:
        if anomaly_type_override not in anomaly_types:
            return {"failed": True, "reason": f"Unsupported anomaly_type: {anomaly_type_override}"}
        anomaly_type = anomaly_type_override
    else:
        anomaly_type = anomaly_types[region_id % len(anomaly_types)]


    
    region_building_ids = region_info.get("building_ids", [])

    
    
    original_state_before_anomaly = {}
    print(f"[DEBUG] Saving original state for region {region_id}:")
    for bid in region_building_ids:
        obj = bpy.data.objects.get(bid)
        if obj:
            
            original_state_before_anomaly[bid] = obj.matrix_world.copy()
            world_loc = obj.matrix_world.to_translation()
            print(f"  Building ({obj.name}): world_loc=({world_loc.x:.2f},{world_loc.y:.2f},{world_loc.z:.2f})")

    # Create clean region roads (temp) before anomaly generation/validation
    temp_road, hidden_states = clean_create_max_region_roads(region_id, region_building_ids)
    road_objs_for_conflict = [temp_road] if temp_road else []
    frame_bounds_for_constraints = labeler.calculate_region_bounds(region_building_ids)

    
    
    selected_anomaly = create_synthetic_anomaly_by_type(
        region_info, building_ids, anomaly_type,
        id_to_obj=id_to_obj,
        road_objs_override=road_objs_for_conflict,
        frame_bounds=frame_bounds_for_constraints,
    )
    
                
    use_existing = False

    
    if isinstance(selected_anomaly, list):
        
        selected_anomaly = selected_anomaly[0] if selected_anomaly else None

    if not selected_anomaly:
        print(f"  [FAIL] Region {region_id}: Failed to create anomaly")
        clean_cleanup_temp_roads(temp_road, hidden_states)
        return None

    print(f"  [SYNTHETIC] Region {region_id} created synthetic anomaly: {selected_anomaly['type']}")

    
    bid_to_obj = {}
    anomaly_buildings = selected_anomaly.get("building_ids", [])

    
    for bid in anomaly_buildings:
        for i, bid_original in enumerate(building_ids):
            if bid_original == bid:
                label_num = i + 1
                obj = id_to_obj.get(label_num)
                if obj:
                    bid_to_obj[bid] = (obj, obj.matrix_world.copy())
                break

    
    inject_action = selected_anomaly.get("inject_action", {})

    
    output_region_dir = os.path.join(output_dir, f"region_{region_id}", anomaly_type)
    os.makedirs(output_region_dir, exist_ok=True)

    
    region_building_ids = region_info.get("building_ids", [])
    
    labeler.set_region_visibility(region_building_ids)
    set_road_visibility(SHOW_ROADS_IN_REGION_RENDER)
    
    # Ensure temp road is visible
    if temp_road:
        temp_road.hide_render = False
        temp_road.hide_viewport = False

    labeler.clear_mask_materials()
    labeler.clear_all_labels()
    clear_labels_only()
    if INPUT_MODE == "complex":
        
        print("[INFO] Complex mode: keep original building materials/colors (no white film).")
    else:
        
        apply_white_film_to_buildings(region_building_ids)
    
    # Apply road material to ensure visual consistency (Yellow Roads)
    # The geometry is already consistent via clean_extract_valid_road_faces
    if SHOW_ROADS_IN_REGION_RENDER:
        apply_road_material()

    
    obj_to_id = {obj.name: label for label, obj in id_to_obj.items() if obj}
    region_objs = [obj for bid in region_building_ids if bid in bpy.data.objects]
    
    region_objs = [bpy.data.objects.get(bid) for bid in region_building_ids if bpy.data.objects.get(bid)]
    region_objs = [obj for obj in region_objs if obj]

    print(f"[DEBUG] Region {region_id}: region_building_ids={len(region_building_ids)}, region_objs={len(region_objs)}")

    if not region_objs:
        print(f"  [WARN] No building objects found for region {region_id}")
        clean_cleanup_temp_roads(temp_road, hidden_states)
        return None

    
    # bounds = compute_scene_bounds(region_objs)
    # Use labeler.calculate_region_bounds to stay 100% consistent with BuildingLabeler
    bounds = labeler.calculate_region_bounds(region_building_ids)
    
    if not bounds:
        print(f"  [WARN] Failed to calculate bounds for region {region_id}")
        clean_cleanup_temp_roads(temp_road, hidden_states)
        return None
    clean_bounds = dict(bounds)


    region_max_dim = max(bounds["width"], bounds["depth"])
    label_font_size = region_max_dim * LABEL_SIZE_RATIO

    # Create cameras for this region
    cam_top = labeler.setup_camera_top_down(bounds, f"Camera_Top_{region_id}")
    cam_iso = labeler.setup_camera_isometric(bounds, f"Camera_Iso_{region_id}")
    fit_ortho_camera_to_objects(
        cam_top,
        region_objs,
        margin_ratio=TOP_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    inflate_ortho_scale(cam_iso, ISO_FIT_MARGIN_RATIO)

    
    
    top_path = os.path.join(output_region_dir, "top.png")
    render_top_view_with_labels(
        cam_top, top_path, bounds, region_objs,
        id_to_obj, True, label_font_size, region_max_dim
    )

    
    iso_path = os.path.join(output_region_dir, "isometric.png")
    render_top_view_with_labels(
        cam_iso, iso_path, bounds, region_objs,
        id_to_obj, False, label_font_size, region_max_dim
    )

    
    if cam_top and cam_top.name in bpy.data.objects:
        bpy.data.objects.remove(cam_top)
    if cam_iso and cam_iso.name in bpy.data.objects:
        bpy.data.objects.remove(cam_iso)

    
    
    for bid, (obj, original_matrix) in bid_to_obj.items():
        restore_object(obj, original_matrix)

    
    
    anomaly_bid = selected_anomaly.get("building_ids", [])[0] if selected_anomaly.get("building_ids") else None

    
    road_objs = [temp_road] if temp_road else []

    
    has_road = len(road_objs) > 0
    if "road_conflict" in anomaly_type and not has_road:
        fail_reason = f"Region {region_id} has no road objects, cannot create {anomaly_type} anomaly"
        print(f"  [WARN] {fail_reason}")
        clean_cleanup_temp_roads(temp_road, hidden_states)
        return {"failed": True, "reason": fail_reason}

    
    original_state = original_state_before_anomaly

    
    
    anomaly_buildings_for_labels = selected_anomaly.get("building_ids", [])
    temp_building_labels = []
    for bid_original in anomaly_buildings_for_labels:
        for i, bid_map in enumerate(building_ids):
            if bid_map == bid_original:
                temp_building_labels.append(str(i + 1))
                break
        else:
            temp_building_labels.append(bid_original)
    target_label_str = temp_building_labels[0] if temp_building_labels else "1"

    shuffled_building_ids = building_ids.copy()
    random.shuffle(shuffled_building_ids)

    selected_anomaly = None

    for retry_idx, target_building_id in enumerate(shuffled_building_ids, start=1):
        
        
        if anomaly_type in ["overlap_by_move", "overlap_by_rotate", "overlap_by_scale", "orientation"] or "road_conflict" in anomaly_type:

            target_label_num = building_ids.index(target_building_id) + 1
            target_obj = id_to_obj.get(target_label_num)

            
            if anomaly_type in ["overlap_by_move", "overlap_by_rotate", "overlap_by_scale"]:
                other_objs = [obj for label, obj in id_to_obj.items() if label != target_label_num]
            elif "road_conflict" in anomaly_type:
                other_objs = road_objs
            else:
                other_objs = road_objs

            
            anomaly_candidates = create_synthetic_anomaly_by_type(
                region_info, building_ids, anomaly_type,
                target_obj=target_obj, other_objs=other_objs, id_to_obj=id_to_obj,
                road_objs_override=road_objs, frame_bounds=clean_bounds
            )
        else:
            
            anomaly_candidates = create_synthetic_anomaly_by_type(
                region_info, building_ids, anomaly_type,
                road_objs_override=road_objs, frame_bounds=clean_bounds
            )

        
        if isinstance(anomaly_candidates, list):
            
            candidates = anomaly_candidates
        elif anomaly_candidates:
            
            candidates = [anomaly_candidates]
        else:
            print(f"  [DEBUG] Region {region_id}: Failed to create anomaly for building {target_building_id}, skipping to next...")
            continue

        
        selected_anomaly = None
        for candidate in candidates:
            
            inject_action = candidate.get("inject_action", {})
            anomaly_bid = candidate.get("building_ids", [])[0] if candidate.get("building_ids") else None
            target_label_str = str(building_ids.index(anomaly_bid) + 1) if anomaly_bid in building_ids else "1"

            
            bid_to_obj = {}
            anomaly_buildings = candidate.get("building_ids", [])
            for bid in anomaly_buildings:
                for i, bid_original in enumerate(building_ids):
                    if bid_original == bid:
                        label_num = i + 1
                        obj = id_to_obj.get(label_num)
                        if obj:
                            
                            bid_to_obj[bid] = (obj, original_state.get(bid, obj.matrix_world.copy()))
                        break

            
            
            for label_num, obj in id_to_obj.items():
                if obj and obj.name in original_state:
                    restore_object(obj, original_state[obj.name])

            
            if anomaly_bid:
                
                target_obj = bpy.data.objects.get(anomaly_bid)
                target_label = None  #  label
                if target_obj:
                    apply_inject_action(target_obj, inject_action, UNIT_SCALE)

            bpy.context.view_layer.update()

            
            target_obj = bpy.data.objects.get(anomaly_bid)

            error_occurred = False
            overlapped_label = None  # 
            if anomaly_type in ["overlap_by_move", "overlap_by_rotate", "overlap_by_scale"]:
                
                # result = _check_building_overlap(region_objs, exclude_obj=None, pad=0.02, id_to_obj=id_to_obj)
                result = _check_target_building_overlap(
                    target_obj=target_obj,
                    building_objs=region_objs,
                    pad=0.0,
                    id_to_obj=id_to_obj,
                )

                if isinstance(result, tuple):
                    error_occurred, overlapped_label_a, overlapped_label_b = result
                    print(f"[DEBUG] Overlap check result: {error_occurred}, labels: {overlapped_label_a}, {overlapped_label_b}")
                    
                    target_label = int(target_label_str) if target_label_str.isdigit() else 1
                    if overlapped_label_a == target_label:
                        overlapped_label = overlapped_label_b
                    else:
                        overlapped_label = overlapped_label_a
                else:
                    print(f"[DEBUG] Overlap check result: {error_occurred}")
                status_now = _get_object_state_status(target_obj, region_objs, road_objs, original_rot_z=None)
                error_occurred = bool(error_occurred and (not status_now["has_br"]) and (not status_now["is_misaligned"]))
            elif "road_conflict" in anomaly_type:
                
                error_occurred = _check_road_conflict(road_objs, region_objs, target_obj)
                status_now = _get_object_state_status(target_obj, region_objs, road_objs, original_rot_z=None)
                strict_bb_now = _check_target_building_overlap(target_obj, region_objs, pad=0.0)
                if isinstance(strict_bb_now, tuple):
                    strict_bb_now = strict_bb_now[0]
                error_occurred = bool(error_occurred and (not strict_bb_now) and (not status_now["is_misaligned"]))
            elif anomaly_type == "orientation":
                
    
                error_occurred = _check_orientation_issue(target_obj, road_objs=road_objs, threshold_deg=15.0)
                strict_br_now = _check_road_conflict(
                    road_objs,
                    [],
                    target_obj=target_obj,
                    threshold_area=STRICT_NO_ROAD_CONFLICT_AREA_THRESHOLD,
                )
                strict_bb_now = _check_target_building_overlap(target_obj, region_objs, pad=0.0)
                if isinstance(strict_bb_now, tuple):
                    strict_bb_now = strict_bb_now[0]
                error_occurred = bool(error_occurred and (not strict_br_now) and (not strict_bb_now))
            elif anomaly_type in ["scale_small", "scale_large"]:
                
                
                original_scale = None
                for bid, (obj, original_matrix) in bid_to_obj.items():
                    if bid == anomaly_bid:
                        
                        
                        original_scale = tuple(obj.scale)
                        break
                
                error_occurred = _check_scale_issue(target_obj, original_scale, threshold_ratio=0.65)

            if error_occurred and not _is_obj_inside_frame_bounds(
                target_obj,
                clean_bounds,
                margin=FRAME_CONSTRAINT_MARGIN,
            ):
                print("[DEBUG] FAIL: Target object is out of frame bounds after inject action")
                error_occurred = False

            
            candidate_params = candidate.get("candidate_params", {})
            print(f"[DEBUG] Retry {retry_idx + 1}: building={target_label_str}, type={anomaly_type}, params={candidate_params}, error_occurred={error_occurred}")

            if error_occurred:
                
                selected_anomaly = candidate
                
                if anomaly_type in ["overlap_by_move", "overlap_by_rotate", "overlap_by_scale"] and overlapped_label:
                    selected_anomaly["overlapped_building_label"] = str(overlapped_label)
                    
                    moving_label = str(target_label_str)
                    selected_anomaly["description"] = f"Building {moving_label} overlaps with Building {overlapped_label}"
                print(f"[DEBUG] SUCCESS: Created {anomaly_type} on building {target_label_str} with params {candidate_params}")
                break
            else:
                
                print(f"[DEBUG] Candidate failed with params {candidate_params}, trying next candidate...")
                continue
        else:
            
            print(f"[DEBUG] All candidates failed for building {target_label_str}, trying next building...")
            continue

        
        if selected_anomaly:
            break

    else:
        
        fail_reason = f"Failed to create {anomaly_type} after trying all buildings and all parameters"
        print(f"[WARN] Region {region_id}: {fail_reason}, skipping...")
        
        for label_num, obj in id_to_obj.items():
            if obj and obj.name in original_state:
                restore_object(obj, original_state[obj.name])
        # Return failure reason
        clean_cleanup_temp_roads(temp_road, hidden_states)
        return {"failed": True, "reason": fail_reason}

    
    inject_action = selected_anomaly.get("inject_action", {})

    
    print(f"[DEBUG] Final selected anomaly: type={selected_anomaly.get('type')}, building_ids={selected_anomaly.get('building_ids')}, inject_action={inject_action}")

    
    anomaly_bid = selected_anomaly.get("building_ids", [])[0] if selected_anomaly.get("building_ids") else None

    
    for bid in region_building_ids:
        obj = bpy.data.objects.get(bid)
        if obj and bid in original_state:
            restore_object(obj, original_state[bid])

    
    if anomaly_bid:
        target_obj = bpy.data.objects.get(anomaly_bid)
        print(f"[DEBUG] Target building: obj_name={target_obj.name if target_obj else 'None'}, inject_action={inject_action}")

        if target_obj:
            
            original_world_loc = target_obj.matrix_world.to_translation()
            original_loc_x = original_world_loc.x
            original_loc_y = original_world_loc.y
            original_loc_z = original_world_loc.z

            loc_before = (original_loc_x, original_loc_y, original_loc_z)
            rot_before = target_obj.rotation_euler.z
            print(f"[DEBUG] Before apply: building ({target_obj.name}) world_loc={loc_before}, rot_z={rot_before}")
            apply_inject_action(target_obj, inject_action, UNIT_SCALE)

            
            after_world_loc = target_obj.matrix_world.to_translation()
            loc_after = (after_world_loc.x, after_world_loc.y, after_world_loc.z)
            rot_after = target_obj.rotation_euler.z
            print(f"[DEBUG] After apply: building {target_label} ({target_obj.name}) world_loc={loc_after}, rot_z={rot_after}")
            print(f"[DEBUG] World location changed: {loc_before != loc_after}, Rotation changed: {rot_before != rot_after}")

    
    target_label = None
    if anomaly_bid:
        for i, bid_original in enumerate(building_ids):
            if bid_original == anomaly_bid:
                target_label = i + 1
                break
    if target_label:
        obj = id_to_obj.get(target_label)
        if obj:
            print(f"[DEBUG] Final state of building {target_label}: location={obj.location}, rotation={obj.rotation_euler}, scale={obj.scale}")

    
    print(f"[DEBUG] All buildings final state:")
    for label_num, obj in sorted(id_to_obj.items()):
        if obj and obj.name in region_building_ids:
            print(f"  Building {label_num} ({obj.name}): loc=({obj.location.x:.2f},{obj.location.y:.2f}), rot_z={obj.rotation_euler.z:.4f}")

    
    if selected_anomaly.get("type") == "overlap":
        # result = _check_building_overlap(region_objs, exclude_obj=None, pad=0.02, id_to_obj=id_to_obj)
        result = _check_target_building_overlap(
                target_obj=target_obj,
                building_objs=region_objs,
                pad=0.0,
                id_to_obj=id_to_obj,
            )

        if isinstance(result, tuple):
            has_overlap, label_a, label_b = result
            print(f"[DEBUG] Final overlap check: {has_overlap}, labels: {label_a}, {label_b}")

    
    # bounds = compute_scene_bounds(region_objs)
    error_bounds = labeler.calculate_region_bounds(region_building_ids)
    if str(inject_action.get("op", "")).lower() == "scale":
        bounds = dict(clean_bounds)
        print(f"[DEBUG] Region {region_id}: use clean bounds for scale anomaly to keep camera stable")
    else:
        bounds = error_bounds if error_bounds else dict(clean_bounds)
    
    region_max_dim = max(bounds["width"], bounds["depth"])
    label_font_size = region_max_dim * LABEL_SIZE_RATIO

    
    # labeler.clear_all_labels()
    # labeler.add_building_labels(region_building_ids, bounds)
    
    # for obj in bpy.data.objects:
    #     if obj.name.startswith("Label_"):
    #         obj.visible_shadow = False
    #         if hasattr(obj, "cycles_visibility"):
    #             obj.cycles_visibility.shadow = False

    
    
    print(f"[DEBUG] Before rendering error images, all buildings state:")
    for label_num, obj in sorted(id_to_obj.items()):
        if obj:
            world_loc = obj.matrix_world.to_translation()
            print(f"  Building {label_num} ({obj.name}): loc=({world_loc.x:.2f},{world_loc.y:.2f}), rot_z={obj.rotation_euler.z:.4f}")

    
    cam_top = labeler.setup_camera_top_down(bounds, f"Camera_Top_{region_id}")
    cam_iso = labeler.setup_camera_isometric(bounds, f"Camera_Iso_{region_id}")
    fit_ortho_camera_to_objects(
        cam_top,
        region_objs,
        margin_ratio=TOP_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    inflate_ortho_scale(cam_iso, ISO_FIT_MARGIN_RATIO)

    
    top_error_path = os.path.join(output_region_dir, "top_error.png")
    render_top_view_with_labels(
        cam_top, top_error_path, bounds, region_objs,
        id_to_obj, True, label_font_size, region_max_dim
    )

    
    iso_error_path = os.path.join(output_region_dir, "isometric_error.png")
    render_top_view_with_labels(
        cam_iso, iso_error_path, bounds, region_objs,
        id_to_obj, False, label_font_size, region_max_dim
    )

    
    if cam_top and cam_top.name in bpy.data.objects:
        bpy.data.objects.remove(cam_top)
    if cam_iso and cam_iso.name in bpy.data.objects:
        bpy.data.objects.remove(cam_iso)

    
    
    anomaly_buildings = selected_anomaly.get("building_ids", [])
    building_labels = []
    for bid_original in anomaly_buildings:
        for i, bid_map in enumerate(building_ids):
            if bid_map == bid_original:
                building_labels.append(str(i + 1))
                break
        else:
            building_labels.append(bid_original)

    ref_dir = os.path.join(output_region_dir, "ref_images")
    os.makedirs(ref_dir, exist_ok=True)

    
    error_state_transforms = {}
    for bid, (obj, original_matrix) in bid_to_obj.items():
        error_state_transforms[bid] = _snapshot_obj_transform(obj)

    def _restore_error_state_all() -> None:
        for bid_str, snap in error_state_transforms.items():
            obj = bpy.data.objects.get(bid_str)
            if obj:
                _restore_obj_transform(obj, snap)

    def _is_action_clean(action_text: str, target_obj_local: bpy.types.Object, road_objs_local: list) -> tuple[bool, dict | None]:
        parsed = _parse_choice_text(action_text)
        if not parsed or target_obj_local is None:
            return False, None
        _restore_error_state_all()
        ok = _apply_choice(target_obj_local, parsed)
        if not ok:
            return False, None
        bpy.context.view_layer.update()
        status = _get_object_state_status(target_obj_local, region_objs, road_objs_local)
        is_clean = (not status.get("has_bb", True)) and (not status.get("has_br", True)) and (not status.get("is_misaligned", True))
        return is_clean, status

    def _is_valid_distractor(action_text: str, target_obj_local: bpy.types.Object, road_objs_local: list) -> bool:
        """
        A valid distractor must:
        1) be parseable/applicable (status is not None)
        2) NOT fully clean the scene
        """
        is_clean_local, status_local = _is_action_clean(action_text, target_obj_local, road_objs_local)
        if status_local is None:
            return False
        return not is_clean_local

    
    target_label = int(building_labels[0]) if building_labels and str(building_labels[0]).isdigit() else 1
    target_obj_for_fix = id_to_obj.get(target_label)
    road_objs_for_fix = [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and ("road" in obj.name.lower() or "path" in obj.name.lower())
    ]

    corrected_action_text = inject_action.get("reverse_choice_text", "")
    use_exact_scale_inverse_for_correct = False
    use_exact_rotate_inverse_for_correct = False
    if corrected_action_text and target_obj_for_fix is not None:
        clean_now = False
        status_now = None

        inject_op = str(inject_action.get("op", "")).lower()
        if inject_op == "scale":
            _restore_error_state_all()
            if _apply_exact_inverse_scale_if_possible(target_obj_for_fix, inject_action):
                status_now = _get_object_state_status(target_obj_for_fix, region_objs, road_objs_for_fix)
                clean_now = (not status_now.get("has_bb", True)) and (not status_now.get("has_br", True)) and (not status_now.get("is_misaligned", True))
                if clean_now:
                    use_exact_scale_inverse_for_correct = True
        elif inject_op == "rotate":
            _restore_error_state_all()
            if _apply_exact_inverse_rotate_if_possible(target_obj_for_fix, inject_action):
                status_now = _get_object_state_status(target_obj_for_fix, region_objs, road_objs_for_fix)
                clean_now = (not status_now.get("has_bb", True)) and (not status_now.get("has_br", True)) and (not status_now.get("is_misaligned", True))
                if clean_now:
                    use_exact_rotate_inverse_for_correct = True

        if not clean_now:
            clean_now, status_now = _is_action_clean(corrected_action_text, target_obj_for_fix, road_objs_for_fix)
        
        if clean_now and inject_op == "scale":
            expected_dir = None
            expected_pct = None
            try:
                sf_local = float(inject_action.get("scale_factor", 1.0))
                if sf_local > 1.0:
                    expected_dir = "down"
                    expected_pct = (1.0 - 1.0 / sf_local) * 100.0
                elif sf_local < 1.0:
                    expected_dir = "up"
                    expected_pct = (1.0 / sf_local - 1.0) * 100.0
            except Exception:
                expected_dir = None
                expected_pct = None

            friendly_candidates = []
            for pct_try in NICE_SCALE_REVERSE_PCTS + [60, 75, 90]:
                friendly_candidates.append(("down", int(pct_try), f"Scale down building {target_label} by {pct_try}%"))
                friendly_candidates.append(("up", int(pct_try), f"Scale up building {target_label} by {pct_try}%"))

            def _friendly_rank(item):
                cand_dir, cand_pct, _ = item
                dir_penalty = 0
                if expected_dir is not None and cand_dir != expected_dir:
                    dir_penalty = 1
                pct_penalty = 0.0
                if expected_pct is not None:
                    pct_penalty = abs(float(cand_pct) - float(expected_pct))
                return (dir_penalty, pct_penalty)

            friendly_candidates.sort(key=_friendly_rank)

            friendly_found = None
            seen = set()
            for _, _, cand in friendly_candidates:
                if cand in seen:
                    continue
                seen.add(cand)
                cand_clean, _ = _is_action_clean(cand, target_obj_for_fix, road_objs_for_fix)
                if cand_clean:
                    friendly_found = cand
                    break

            if friendly_found:
                corrected_action_text = friendly_found
                use_exact_scale_inverse_for_correct = False
                inject_action["reverse_choice_text"] = friendly_found
                if isinstance(selected_anomaly.get("inject_action"), dict):
                    selected_anomaly["inject_action"]["reverse_choice_text"] = friendly_found
        if not clean_now:
            print(f"[WARN] Region {region_id}: initial correct action not clean, status={status_now}, action={corrected_action_text}")
            candidate_actions = _build_candidate_fix_actions_from_inject(inject_action, str(target_label))

            
            if inject_op == "rotate":
                for deg_try in range(1, 180):
                    candidate_actions.append(f"Rotate building {target_label} clockwise by {deg_try}°")
                    candidate_actions.append(f"Rotate building {target_label} counter-clockwise by {deg_try}°")
            elif inject_op == "scale":
                for pct_try in range(1, 91):
                    candidate_actions.append(f"Scale up building {target_label} by {pct_try}%")
                    candidate_actions.append(f"Scale down building {target_label} by {pct_try}%")
            elif inject_op == "move":
                move_dirs_full = ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]
                move_steps = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
                for d_try in move_dirs_full:
                    for m_try in move_steps:
                        candidate_actions.append(f"Move building {target_label} {d_try} by {m_try} units")

            
            for deg_try in [5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 150]:
                candidate_actions.append(f"Rotate building {target_label} clockwise by {deg_try}°")
                candidate_actions.append(f"Rotate building {target_label} counter-clockwise by {deg_try}°")
            for pct_try in [5, 10, 15, 20, 25, 30, 40, 50]:
                candidate_actions.append(f"Scale up building {target_label} by {pct_try}%")
                candidate_actions.append(f"Scale down building {target_label} by {pct_try}%")

            candidate_actions = list(dict.fromkeys(candidate_actions))
            found_action = None

            max_checks = 320
            checks = 0
            for search_round in range(3):
                if search_round > 0:
                    random.shuffle(candidate_actions)

                for cand in candidate_actions:
                    if checks >= max_checks:
                        break
                    if not cand:
                        continue
                    checks += 1
                    clean_cand, _ = _is_action_clean(cand, target_obj_for_fix, road_objs_for_fix)
                    if clean_cand:
                        found_action = cand
                        break

                if found_action or checks >= max_checks:
                    break

                
                stochastic_candidates = []
                if inject_op == "rotate":
                    for _ in range(48):
                        d = "clockwise" if random.random() < 0.5 else "counter-clockwise"
                        a = random.randint(1, 179)
                        stochastic_candidates.append(f"Rotate building {target_label} {d} by {a}°")
                elif inject_op == "scale":
                    for _ in range(48):
                        d = "up" if random.random() < 0.5 else "down"
                        p = random.choice([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90])
                        stochastic_candidates.append(f"Scale {d} building {target_label} by {p}%")
                else:
                    dirs = ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]
                    for _ in range(48):
                        d = random.choice(dirs)
                        m = round(random.uniform(0.1, 3.0), 1)
                        stochastic_candidates.append(f"Move building {target_label} {d} by {m} units")

                candidate_actions.extend(stochastic_candidates)
                candidate_actions = list(dict.fromkeys(candidate_actions))

            if found_action:
                corrected_action_text = found_action
                use_exact_scale_inverse_for_correct = False
                use_exact_rotate_inverse_for_correct = False
                inject_action["reverse_choice_text"] = found_action
                if isinstance(selected_anomaly.get("inject_action"), dict):
                    selected_anomaly["inject_action"]["reverse_choice_text"] = found_action
                print(f"[INFO] Region {region_id}: regenerated correct action after {checks} checks -> {found_action}")
            else:
                fail_reason = f"failed to regenerate a fully clean correct action after {checks} checks"
                print(f"[WARN] Region {region_id}: {fail_reason}, skip region")
                for bid, (obj, original_matrix) in bid_to_obj.items():
                    restore_object(obj, original_matrix)
                clean_cleanup_temp_roads(temp_road, hidden_states)
                return {"failed": True, "reason": fail_reason}

    
    _restore_error_state_all()

    
    correct_action = corrected_action_text if corrected_action_text else inject_action.get("reverse_choice_text", "")
    distractors = generate_distractors_with_filter(
        selected_anomaly.get("type", "overlap"),
        building_labels[0] if building_labels else "1",
        correct_action,
        region_objs,
        id_to_obj,
        num_distractors=3
    )
    all_options = [correct_action]
    target_bid_for_opts = building_labels[0] if building_labels else "1"
    candidate_pool = list(dict.fromkeys(distractors))

    # Enrich pool with operation-consistent and random candidates (still strictly validated below)
    candidate_pool.extend(_build_candidate_fix_actions_from_inject(inject_action, str(target_bid_for_opts)))
    for deg_try in [5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 150]:
        candidate_pool.append(f"Rotate building {target_bid_for_opts} clockwise by {deg_try}°")
        candidate_pool.append(f"Rotate building {target_bid_for_opts} counter-clockwise by {deg_try}°")
    for pct_try in [5, 10, 15, 20, 25, 30, 40, 50]:
        candidate_pool.append(f"Scale up building {target_bid_for_opts} by {pct_try}%")
        candidate_pool.append(f"Scale down building {target_bid_for_opts} by {pct_try}%")
    for d_try in ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"]:
        for m_try in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            candidate_pool.append(f"Move building {target_bid_for_opts} {d_try} by {m_try} units")

    candidate_pool = list(dict.fromkeys(candidate_pool))
    random.shuffle(candidate_pool)

    for cand in candidate_pool:
        if len(all_options) >= 4:
            break
        if not cand or cand == correct_action or cand in all_options:
            continue
        if _is_valid_distractor(cand, target_obj_for_fix, road_objs_for_fix):
            all_options.append(cand)

    # Last-chance stochastic search, still with strict validation
    max_attempts = 180
    attempt = 0
    while len(all_options) < 4 and attempt < max_attempts:
        attempt += 1
        mode_roll = random.random()
        if mode_roll < 0.34:
            d = random.choice(["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"])
            m = round(random.uniform(0.1, 3.0), 1)
            new_option = f"Move building {target_bid_for_opts} {d} by {m} units"
        elif mode_roll < 0.67:
            direction = "clockwise" if random.random() < 0.5 else "counter-clockwise"
            deg = random.randint(5, 150)
            new_option = f"Rotate building {target_bid_for_opts} {direction} by {deg}°"
        else:
            direction = "up" if random.random() < 0.5 else "down"
            pct = random.choice([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90])
            new_option = f"Scale {direction} building {target_bid_for_opts} by {pct}%"

        if new_option in all_options or new_option == correct_action:
            continue
        if _is_valid_distractor(new_option, target_obj_for_fix, road_objs_for_fix):
            all_options.append(new_option)

    if len(all_options) < 4:
        fail_reason = f"unable to build 3 strictly-valid distractors (got {len(all_options)-1})"
        print(f"[WARN] Region {region_id}: {fail_reason}, skip region")
        for bid, (obj, original_matrix) in bid_to_obj.items():
            restore_object(obj, original_matrix)
        clean_cleanup_temp_roads(temp_road, hidden_states)
        return {"failed": True, "reason": fail_reason}

    # Important: Shuffle options BEFORE rendering so images match the shuffled order
    random.shuffle(all_options)
    # Ensure correct answer is present (it must be, but sanity check)
    if correct_action not in all_options:
        # Should not happen unless list manipulation bug
        print("[ERROR] Correct action lost during shuffle!")
        all_options[0] = correct_action

    
    qa3_bounds = dict(clean_bounds)
    cam_top = labeler.setup_camera_top_down(qa3_bounds, f"Camera_Top_{region_id}")
    fit_ortho_camera_to_objects(
        cam_top,
        region_objs,
        margin_ratio=TOP_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )

    
    option_images = []

    for opt_idx, opt_text in enumerate(all_options):
        
        # Correctly iterate through snapshot items
        for bid_str, snap in error_state_transforms.items():
            obj = bpy.data.objects.get(bid_str)
            if obj:
                _restore_obj_transform(obj, snap)


        if opt_text == correct_action:
            # Applying correct action (reverse) to broken state -> Fixed Control State
            target_label = int(building_labels[0]) if building_labels and building_labels[0].isdigit() else 1
            target_obj = id_to_obj.get(target_label)
            if target_obj:
                applied = False
                if use_exact_rotate_inverse_for_correct:
                    applied = _apply_exact_inverse_rotate_if_possible(target_obj, inject_action)
                if use_exact_scale_inverse_for_correct:
                    applied = _apply_exact_inverse_scale_if_possible(target_obj, inject_action)
                if not applied:
                    parsed = _parse_choice_text(opt_text)
                    if parsed:
                        _apply_choice(target_obj, parsed)
        else:
            parsed = _parse_choice_text(opt_text)
            if parsed:
                target_label = int(building_labels[0]) if building_labels and building_labels[0].isdigit() else 1
                target_obj = id_to_obj.get(target_label)
                if target_obj:
                    _apply_choice(target_obj, parsed)

        
        opt_path = os.path.join(ref_dir, f"option_{opt_idx}.png")
        render_top_view_with_labels(
            cam_top, opt_path, qa3_bounds, region_objs,
            id_to_obj, True, label_font_size, region_max_dim
        )
        option_images.append(os.path.abspath(opt_path))

    
    if cam_top and cam_top.name in bpy.data.objects:
        bpy.data.objects.remove(cam_top)

    
    for bid, (obj, original_matrix) in bid_to_obj.items():
        restore_object(obj, original_matrix)

    
    issue_type_map = {
        "overlap": ISSUE_OVERLAP,
        "road_conflict": ISSUE_ROAD,
        "orientation": ISSUE_ANGLE,
        "scale_small": ISSUE_SCALE,
        "scale_large": ISSUE_SCALE,
    }

    
    anomaly_buildings = selected_anomaly.get("building_ids", [])
    building_labels = []
    for bid_original in anomaly_buildings:
        for i, bid_map in enumerate(building_ids):
            if bid_map == bid_original:
                building_labels.append(str(i + 1))
                break
        else:
            building_labels.append(bid_original)

    
    anomaly_type = selected_anomaly.get("type", "")
    if anomaly_type == "overlap":
        
        overlapped_label = selected_anomaly.get("overlapped_building_label")
        if overlapped_label:
            new_description = f"Building {building_labels[0]} overlaps with Building {overlapped_label}"
        elif len(building_labels) >= 2:
            new_description = f"Buildings {building_labels[0]} and {building_labels[1]} overlap each other"
        else:
            new_description = f"Building {building_labels[0]} overlaps with another building"
    elif anomaly_type == "road_conflict":
        new_description = f"Building {building_labels[0]} overlaps with the road"
    elif anomaly_type == "orientation":
        metric_val = selected_anomaly.get("metric_value", 0)
        new_description = f"Building {building_labels[0]} has an abnormal orientation (angle deviation: {metric_val:.1f}°)"
    elif anomaly_type in ["scale_small", "scale_large"]:
        new_description = f"Building {building_labels[0]} has an abnormal scale"
    else:
        new_description = selected_anomaly.get("description", "")

    issue_meta = {
        "issues": [
            {
                "type": issue_type_map.get(selected_anomaly["type"], selected_anomaly["type"]),
                "buildings": selected_anomaly.get("building_ids", []),
                "building_labels": building_labels,
                "description": new_description,
                "severity": selected_anomaly.get("severity", "medium"),
                "metric_value": selected_anomaly.get("metric_value", 0),
            }
        ]
    }

    inject_action = selected_anomaly.get("inject_action", {})

    
    # Use ERROR images for identification and modification tasks
    top_abs = os.path.abspath(top_error_path)
    iso_abs = os.path.abspath(iso_error_path)

    images_top = [top_abs]
    images_both = [top_abs, iso_abs]

    qa_top_1 = qa1_mcq_what_problem(issue_meta, images=images_top)
    # Pass fixed_options_text to match the rendered images
    qa_top_3 = qa3_fix(issue_meta, inject_action, None, images=images_top, building_objs=region_objs, id_to_obj=id_to_obj, option_images=option_images, fixed_options_text=all_options)
    qa_top_iso = qa2_mcq_what_problem(issue_meta, images=images_both)

    # Cleanup temp resources
    clean_cleanup_temp_roads(temp_road, hidden_states)

    return {
        "region_id": region_id,
        "anomaly": selected_anomaly,
        "is_synthetic": not use_existing,
        "issue_meta": issue_meta,
        "inject_action": inject_action,
        "qa": {"top": [qa_top_1, qa_top_3], "top_isometric": [qa_top_iso]},
    }


def main():
    """Main function"""
    print("=" * 60)
    print(
        f"[RUN] mode={INPUT_MODE}, region={REGION_NAME}, min_region={MIN_REGION}, "
        f"max_regions={MAX_REGIONS}, workers={WORKERS}"
    )
    print(f"[BLEND] {BLEND_PATH}")
    print(f"[REGION_DIR] {REGION_DIR}")
    print(f"[REGION_DATA] {REGION_DATA_PATH}")
    print(f"[BUILDING_REGION_MAP] {BUILDING_REGION_MAP_PATH}")
    print(f"[OUT] {OUTPUT_DIR}")
    print("=" * 60)

    
    is_worker_child = str(os.environ.get("REGION_WORKER_CHILD", "0")).strip().lower() in {"1", "true", "yes"}
    if WORKERS > 1 and not is_worker_child:
        run_parallel_workers()
        return

    validate_required_inputs()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    print("Load region data...")
    region_data = load_region_data()

    print("Load building-to-region mapping...")
    building_region_map = load_building_region_map()
    print(f"[DEBUG] building_region_map size: {len(building_region_map) if isinstance(building_region_map, dict) else 'N/A'}")

    
    all_regions_raw = region_data.get("regions", [])
    all_regions_raw = sorted(all_regions_raw, key=lambda r: int(r.get("region_id", 0)))

    start_idx = max(0, int(MIN_REGION))
    total_available = len(all_regions_raw)
    if start_idx > total_available:
        start_idx = total_available

    if MAX_REGIONS > 0:
        end_idx = min(start_idx + int(MAX_REGIONS), total_available)
        all_regions = all_regions_raw[start_idx:end_idx]
        print(f"[DEBUG] Region list slice index: [{start_idx}, {end_idx})")
    else:
        all_regions = all_regions_raw[start_idx:]
        print(f"[DEBUG] Region list slice index: [{start_idx}, {total_available})")

    selected_region_ids = [int(r.get("region_id", -1)) for r in all_regions]
    print(f"[DEBUG] Selected region_ids ({len(selected_region_ids)}): {selected_region_ids}")
        
    print(f" {len(all_regions)}  (min: {MIN_REGION}, max: {MAX_REGIONS})")

    # Open blend file
    bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)

    render_resolution = (RESOLUTION, RESOLUTION) if RESOLUTION > 0 else _render_resolution_px()

    # Render settings (aligned with task4)
    setup_render(render_resolution)
    force_disable_all_shadows_and_world()

    # Setup labeler for camera + masking
    labeler = BuildingLabeler(
        region_data_path=REGION_DATA_PATH,
        blend_path=BLEND_PATH,
        output_dir=OUTPUT_DIR,
        ortho_scale_factor=1.1,
        label_height_ratio=0.01,
        font_size_ratio=0.07,
        samples=32,
        resolution=render_resolution,
        mask_alpha=0.8,
    )

    # Patch lighting to disable shadows (exactly like reference)
    def _patched_setup_lighting(self):
        scene = bpy.context.scene
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("SunLight", type="SUN"))
        bpy.context.collection.objects.link(sun)
        sun.data.energy = 4.0
        sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
        if hasattr(sun.data, "use_shadow"):
            sun.data.use_shadow = False
        if hasattr(sun.data, "cast_shadow"):
            sun.data.cast_shadow = False

        fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLight", type="AREA"))
        bpy.context.collection.objects.link(fill)
        fill.data.energy = 150.0
        fill.location = (0.0, 0.0, 100.0)
        fill.data.size = 100.0
        if hasattr(fill.data, "use_shadow"):
            fill.data.use_shadow = False
        if hasattr(fill.data, "cast_shadow"):
            fill.data.cast_shadow = False

        scene.cycles.use_shadows = False
        scene.cycles.use_progressive = False
        scene.cycles.use_shadow_highlight = False
        scene.cycles.blur_shadow = 0
        if hasattr(scene.cycles, 'shader_cache'):
            scene.cycles.shader_cache = 0

        for o in bpy.data.objects:
            if hasattr(o, "cycles_visibility"):
                o.cycles_visibility.cast_shadow = False
                o.cycles_visibility.receive_shadow = False
            try:
                o.visible_shadow = False
            except:
                pass
        
        if "World" in bpy.data.worlds:
            world = bpy.data.worlds["World"]
            if world.use_nodes:

                pass

    labeler.setup_lighting = _patched_setup_lighting.__get__(labeler, BuildingLabeler)
    labeler.setup_render()
    labeler.setup_lighting()

    # Apply road material to make roads visible
    if SHOW_ROADS_IN_REGION_RENDER:
        apply_road_material()


    
    total_regions = len(all_regions)
    total_target_qas = total_regions * len(TARGET_QA_ANOMALY_TYPES)
    success_count = 0
    existing_count = 0
    synthetic_count = 0
    failed_regions = []

    all_qa_data = []

    
    for idx, region in enumerate(all_regions):
        region_id = region.get("region_id", idx)
        building_count = region.get("building_count", 0)

        print(f"\n[PROCESS] Region {region_id} ({building_count} buildings) [{idx+1}/{len(all_regions)}]")
        region_success = 0
        for anomaly_type in TARGET_QA_ANOMALY_TYPES:
            print(f"  [TYPE] Region {region_id}: {anomaly_type}")
            try:
                result = generate_qa_for_region(
                    region_id,
                    region,
                    region_data,
                    labeler,
                    OUTPUT_DIR,
                    sample_idx=idx,
                    anomaly_type_override=anomaly_type,
                )

                
                if isinstance(result, dict) and result.get("failed"):
                    failed_regions.append({
                        "region_id": region_id,
                        "anomaly_type": anomaly_type,
                        "reason": result.get("reason", "unknown"),
                    })
                    print(f"    [SKIP] {anomaly_type}: {result.get('reason', 'unknown')}")
                    continue

                qa_data = result
                if qa_data:
                    qa_data["requested_anomaly_type"] = anomaly_type
                    all_qa_data.append(qa_data)
                    success_count += 1
                    region_success += 1
                    if qa_data.get("is_synthetic"):
                        synthetic_count += 1
                    else:
                        existing_count += 1
                    print(f"    [OK] {anomaly_type} generated QA successfully")
                else:
                    failed_regions.append({
                        "region_id": region_id,
                        "anomaly_type": anomaly_type,
                        "reason": "no anomaly generated",
                    })
                    print(f"    [SKIP] {anomaly_type} cannot generate anomaly")

            except Exception as e:
                import traceback
                failed_regions.append({"region_id": region_id, "anomaly_type": anomaly_type, "error": str(e)})
                print(f"    [FAIL] {anomaly_type}: {repr(e)}")
                traceback.print_exc()
                continue

        print(f"  [REGION_SUMMARY] Region {region_id}: {region_success}/{len(TARGET_QA_ANOMALY_TYPES)} QA generated")

    # Save QA data
    output_path = os.path.join(OUTPUT_DIR, f"qa_{REGION_NAME}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa_data, f, ensure_ascii=False, indent=2)

    # Save statistics
    stats = {
        "region": REGION_NAME,
        "total_regions": total_regions,
        "total_target_qas": total_target_qas,
        "target_anomaly_types": TARGET_QA_ANOMALY_TYPES,
        "success_count": success_count,
        "existing_anomaly_count": existing_count,
        "synthetic_anomaly_count": synthetic_count,
        "failed_count": len(failed_regions),
        "failed_regions": failed_regions,
    }

    stats_path = os.path.join(OUTPUT_DIR, f"stats_{REGION_NAME}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Total regions: {total_regions}")
    print(f"  GoalQA: {total_target_qas}")
    print(f"  Success: {success_count}")
    print(f"  : {existing_count}")
    print(f"  : {synthetic_count}")
    print(f"  Failed: {len(failed_regions)}")
    print(f"  QA : {output_path}")
    print(f"  Statistics: {stats_path}")


if __name__ == "__main__":
    main()
