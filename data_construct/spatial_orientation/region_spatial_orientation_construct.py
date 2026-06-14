#!/usr/bin/env python3

import bpy
import random
import math
import json
import os
import sys
import subprocess
import types
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view


PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.expanduser('~/SpatialAct'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'blender_scripts'))

from src.building_labels import BuildingLabeler, is_building as check_is_building, is_road, apply_road_material, is_visible_object

# ============================================================

# ============================================================
import argparse


parser = argparse.ArgumentParser(description="Region-level spatial orientation data generation")
parser.add_argument("--region", type=str, default=None, help="Region name, e.g. region_a, region_b")
parser.add_argument("--max-regions", type=int, default=None, help="Maximum number of regions to process; <=0 means all")
parser.add_argument(
    "--use-clean",
    type=int,
    default=1,
    help="Whether to prioritize clean outputs from preprocess_clean.py (1=yes, 0=no)",
)
args, _ = parser.parse_known_args()


REGION_NAME = args.region if args.region else os.environ.get("REGION_NAME", "default_region")
MAX_REGIONS_ARG = args.max_regions
USE_CLEAN_ARG = int(args.use_clean)


BLEND_PATH = os.environ.get("BLEND_PATH", os.path.join(PROJECT_ROOT, f"osm_scene_0228/{REGION_NAME}_osm_scene_0228/osm_reference.blend"))


REGION_DIR = os.environ.get("REGION_DIR", os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{REGION_NAME}_kmeans"))


OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "benchmark/data/spatial_orientation/"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{REGION_NAME}_regions")


RESOLUTION = int(os.environ.get("RESOLUTION", "0"))
MAX_BUILDINGS_PER_REGION = int(os.environ.get("MAX_BUILDINGS_PER_REGION", "15"))

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
    corner_idx: int | None = None,
) -> dict | None:
    payload = {
        "image_path": image_path,
        "screen_vec": [float(screen_vec.x), float(screen_vec.y)],
        "arrow_color": [int(arrow_color[0]), int(arrow_color[1]), int(arrow_color[2]), int(arrow_color[3])],
        "arrow_size": int(arrow_size),
        "corner_idx": (int(corner_idx) if corner_idx is not None else None),
    }
    helper_script = r"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

def load_font(sz):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, sz)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def norm2(x, y):
    n = (x * x + y * y) ** 0.5
    if n < 1e-8:
        return 0.0, -1.0
    return x / n, y / n

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def mul(a, s):
    return (a[0] * s, a[1] * s)

payload = json.loads(sys.argv[1])
image_path = payload["image_path"]
if not os.path.exists(image_path):
    raise SystemExit(0)
sv = payload.get("screen_vec", [0.0, -1.0])
vx, vy = norm2(float(sv[0]), float(sv[1]))
arrow_color = tuple(int(v) for v in payload.get("arrow_color", [255, 0, 0, 255]))
arrow_size = int(payload.get("arrow_size", 120))

img = Image.open(image_path).convert("RGBA")
draw = ImageDraw.Draw(img)
width, height = img.size
base_w, base_h = width, height
alpha = img.getchannel("A")
pix = img.load()
ext_top = 0
ext_right = 0
expanded = False

margin = max(16, int(arrow_size * 0.20))
panel_w = int(arrow_size * 1.75)
panel_h = int(arrow_size * 1.95)

candidates = [
    (width - margin - panel_w, margin),
    (margin, margin),
    (width - margin - panel_w, height - margin - panel_h),
    (margin, height - margin - panel_h),
]
corner_idx = payload.get("corner_idx", None)

def overlap_ratio(px, py):
    x1 = max(0, int(px))
    y1 = max(0, int(py))
    x2 = min(width, x1 + panel_w)
    y2 = min(height, y1 + panel_h)
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
    x2 = min(width, x1 + panel_w)
    y2 = min(height, y1 + panel_h)
    if x2 <= x1 or y2 <= y1:
        return False
    for yy in range(y1, y2):
        for xx in range(x1, x2):
            r, g, b, a = pix[xx, yy]
            if is_building_pixel(r, g, b, a):
                return True
    return False

if isinstance(corner_idx, int) and 0 <= corner_idx < len(candidates):
    best_xy = candidates[corner_idx]
else:
    best_xy = min(candidates, key=lambda xy: overlap_ratio(xy[0], xy[1]))

x1 = int(best_xy[0]); y1 = int(best_xy[1])
x2 = x1 + panel_w; y2 = y1 + panel_h


if panel_overlaps_buildings(x1, y1):
    old_w, old_h = width, height
    ext_w = panel_w + 2 * margin
    ext_top = max(0, panel_h + 2 * margin - old_h)
    new_w = old_w + ext_w
    new_h = old_h + ext_top
    ext_right = ext_w
    expanded = True

    new_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    new_img.paste(img, (0, ext_top), img)
    img = new_img
    draw = ImageDraw.Draw(img)
    pix = img.load()
    width, height = img.size

    x1 = old_w + max(0, (ext_w - panel_w) // 2)
    y1 = margin

x2 = x1 + panel_w; y2 = y1 + panel_h
draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 220))

cx = x1 + panel_w * 0.63
cy = y1 + panel_h * 0.40
arrow_len = arrow_size * 0.52
head = arrow_size * 0.20
shaft_w = max(3, int(arrow_size * 0.05))

screen_vec = (vx, vy)
perp = (-vy, vx)
tip = add((cx, cy), mul(screen_vec, arrow_len / 2.0))
bottom = sub((cx, cy), mul(screen_vec, arrow_len / 2.0))
wing_base = sub(tip, mul(screen_vec, head))
left_wing = add(wing_base, mul(perp, head * 0.58))
right_wing = sub(wing_base, mul(perp, head * 0.58))

draw.line([bottom, tip], fill=arrow_color, width=shaft_w)
draw.polygon([tip, left_wing, right_wing], fill=arrow_color)

font = load_font(max(16, int(arrow_size * 0.34)))
if font is not None:
    draw.text((x1 + panel_w * 0.20, y1 + panel_h * 0.60), "N", fill=arrow_color, font=font)

img.save(image_path)
print(json.dumps({
    "ok": True,
    "expanded": bool(expanded),
    "ext_top": int(ext_top),
    "ext_right": int(ext_right),
    "base_w": int(base_w),
    "base_h": int(base_h),
    "final_w": int(width),
    "final_h": int(height),
}))
"""
    try:
        cp = subprocess.run(
            ["python3", "-c", helper_script, json.dumps(payload, ensure_ascii=False)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        meta = None
        for line in reversed(cp.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    meta = parsed
                    break
            except Exception:
                continue
        if meta is None:
            meta = {"ok": True, "expanded": False, "ext_top": 0, "ext_right": 0}
        return meta
    except Exception as e:
        print(f"  [draw_north_arrow] Warning: external python fallback failed: {e}")
        return None


def draw_north_arrow(
    image_path: str,
    cam,
    north_world_dir=None,
    arrow_color=(255, 0, 0, 255),
    arrow_size=120,
    corner_idx: int | None = None,
    return_meta: bool = False,
):
    if not os.path.exists(image_path):
        print(f"  [draw_north_arrow] Warning: Image not found: {image_path}")
        if return_meta:
            return None
        return image_path
    if north_world_dir is None:
        north_world_dir = Vector((0.0, 1.0, 0.0))
    screen_vec = _compute_north_screen_vec(cam, north_world_dir)
    meta = _draw_north_arrow_via_system_python(
        image_path,
        screen_vec,
        arrow_color=arrow_color,
        arrow_size=arrow_size,
        corner_idx=corner_idx,
    )
    if return_meta:
        return meta
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


def trim_image_to_alpha_bbox(image_path: str, pad_px: int = 0, return_meta: bool = False):
    if not os.path.exists(image_path):
        return None if return_meta else False
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
out = img.crop((x1, y1, x2, y2))
out.save(image_path)
print(json.dumps({
    "ok": True,
    "crop_x1": int(x1),
    "crop_y1": int(y1),
    "crop_x2": int(x2),
    "crop_y2": int(y2),
    "final_w": int(out.size[0]),
    "final_h": int(out.size[1]),
}))
"""
    try:
        cp = subprocess.run(
            ["python3", "-c", helper_script, json.dumps(payload, ensure_ascii=False)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        meta = None
        for line in reversed(cp.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    meta = parsed
                    break
            except Exception:
                continue
        if return_meta:
            return meta
        return (meta is not None)
    except Exception as e:
        print(f"  [trim] Warning: trim failed for {image_path}: {e}")
        return None if return_meta else False



DISTANCE_MARGIN = 3  


ISO_EXTREME_AXIS_MARGIN = 5


TASK1_WORLD_ANGLE_MARGIN_DEG = 10.0
TASK1_MIN_WORLD_RADIUS = 0.6


TASK4_ANGLE_MARGIN_DEG = 10.0
TASK4_MIN_SCREEN_RADIUS = 0.4

# Task3 switch
ENABLE_TASK3 = False


LABEL_FONT_SIZE = 1.2
LABEL_FONT_SIZE_A = 1.4
LABEL_Z_OFFSET_RATIO = 0.15
LABEL_COLOR = (0.02, 0.02, 0.02, 1.0)
WHITE_FILM_ALPHA = 0.92
TOP_FIT_MARGIN_RATIO = 0.20
ISO_FIT_MARGIN_RATIO = 0.24
MIN_FIT_ORTHO_SCALE = 2.2
TOP_NORTH_ARROW_SIZE = 96
ISO_NORTH_ARROW_SIZE = 104
SHOW_ROADS_IN_REGION_RENDER = os.environ.get("SHOW_ROADS_IN_REGION_RENDER", "1").lower() in ("1", "true", "yes")
DEBUG_CAMERA_FIT = os.environ.get("DEBUG_CAMERA_FIT", "0").lower() in ("1", "true", "yes")
REFRAME_PAD_RATIO = float(os.environ.get("REFRAME_PAD_RATIO", "0.45"))
REFRAME_CONTENT_FILL_RATIO = float(os.environ.get("REFRAME_CONTENT_FILL_RATIO", "1.00"))
FINAL_TRIM_PAD_PX = int(os.environ.get("FINAL_TRIM_PAD_PX", "180"))


def force_disable_all_shadows():
    
    scene = bpy.context.scene

    
    scene.cycles.use_shadows = False
    scene.cycles.use_progressive = False
    scene.cycles.use_shadow_highlight = False
    scene.cycles.blur_shadow = 0

    
    if hasattr(scene.cycles, 'shader_cache'):
        scene.cycles.shader_cache = 0

    
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            if hasattr(obj.data, 'shadow_ray_visibility'):
                obj.data.shadow_ray_visibility = {'CAST_SHADOWS': False}
            if hasattr(obj.data, 'cast_shadow'):
                obj.data.cast_shadow = False

    
    for obj in bpy.data.objects:
        if hasattr(obj, 'cycles_visibility'):
            obj.cycles_visibility.cast_shadow = False
            obj.cycles_visibility.receive_shadow = False

    
    for obj in bpy.data.objects:
        if check_is_building(obj.name):
            if hasattr(obj, 'cycles_visibility'):
                obj.cycles_visibility.cast_shadow = False
                obj.cycles_visibility.receive_shadow = False

    
    for obj in bpy.data.objects:
        if obj.name.startswith('Label_'):
            if hasattr(obj, 'cycles_visibility'):
                obj.cycles_visibility.cast_shadow = False
                obj.cycles_visibility.receive_shadow = False
                obj.cycles_visibility.glossy = False
                obj.cycles_visibility.scatter = False
                obj.cycles_visibility.shadow = False
            
            try:
                obj.visible_shadow = False
            except:
                pass
            try:
                obj.shadow_door = False
            except:
                pass

    
    if "World" in bpy.data.worlds:
        world = bpy.data.worlds["World"]
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()
        links.clear()
        output = nodes.new('ShaderNodeOutputWorld')
        bg = nodes.new('ShaderNodeBackground')
        bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs['Strength'].default_value = 0.0
        links.new(bg.outputs['Background'], output.inputs['Surface'])

    
    for obj in bpy.data.objects:
        if hasattr(obj, 'cycles_visibility'):
            obj.cycles_visibility.cast_shadow = False


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


def fit_ortho_camera_to_objects(
    cam,
    objects: list,
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

    cam_axes = cam.matrix_world.to_3x3()
    cam_right = (cam_axes @ Vector((1.0, 0.0, 0.0))).normalized()
    cam_up = (cam_axes @ Vector((0.0, 1.0, 0.0))).normalized()

    top_like = (
        abs(cam_right.z) < 1e-5 and
        abs(cam_up.z) < 1e-5 and
        abs(abs(cam_right.x) - 1.0) < 1e-3 and
        abs(abs(cam_up.y) - 1.0) < 1e-3
    )

    if top_like:
        wx = [float(p.x) for p in world_points]
        wy = [float(p.y) for p in world_points]
        center_x = 0.5 * (min(wx) + max(wx))
        center_y = 0.5 * (min(wy) + max(wy))
        fit_half_x = 0.5 * (max(wx) - min(wx))
        fit_half_y = 0.5 * (max(wy) - min(wy))

        cam.location.x = center_x
        cam.location.y = center_y
        bpy.context.view_layer.update()

        rx, ry = _render_resolution_px()
        aspect = float(rx) / float(max(1, ry))
        needed_scale_x = 2.0 * fit_half_x
        needed_scale_y = 2.0 * fit_half_y * aspect
        needed_scale = max(needed_scale_x, needed_scale_y)
        cam.data.ortho_scale = max(min_ortho_scale, needed_scale * (1.0 + margin_ratio))
        return

    cam_inv = cam.matrix_world.inverted()
    xs = []
    ys = []
    for p in world_points:
        q = cam_inv @ p
        xs.append(float(q.x))
        ys.append(float(q.y))

    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    cam.location = cam.location + cam_right * center_x + cam_up * center_y
    bpy.context.view_layer.update()

    cam_inv = cam.matrix_world.inverted()
    xs = []
    ys = []
    for p in world_points:
        q = cam_inv @ p
        xs.append(float(q.x))
        ys.append(float(q.y))
    fit_half_x = max(abs(min(xs)), abs(max(xs)))
    fit_half_y = max(abs(min(ys)), abs(max(ys)))

    rx, ry = _render_resolution_px()
    aspect = float(rx) / float(max(1, ry))
    needed_scale_x = 2.0 * fit_half_x
    needed_scale_y = 2.0 * fit_half_y * aspect
    needed_scale = max(needed_scale_x, needed_scale_y)
    cam.data.ortho_scale = max(min_ortho_scale, needed_scale * (1.0 + margin_ratio))

    if DEBUG_CAMERA_FIT:
        print(
            f"[camera_fit] cam={cam.name} needed={needed_scale:.4f} "
            f"fit_half=({fit_half_x:.4f},{fit_half_y:.4f}) aspect={aspect:.3f} "
            f"margin={margin_ratio:.3f} final={cam.data.ortho_scale:.4f}"
        )


def setup_rotated_isometric_camera_from_base(
    labeler: BuildingLabeler,
    bounds: dict,
    base_cam,
    name: str,
    rotate_deg: float,
):
    
    cam = labeler.setup_camera_isometric(bounds, name)
    if base_cam is None or base_cam.type != "CAMERA":
        return cam

    center = Vector((
        float(bounds["center_x"]),
        float(bounds["center_y"]),
        float(bounds["center_z"]),
    ))
    rotate_world = (
        Matrix.Translation(center)
        @ Matrix.Rotation(math.radians(float(rotate_deg)), 4, "Z")
        @ Matrix.Translation(-center)
    )
    cam.matrix_world = rotate_world @ base_cam.matrix_world

    if cam.data and base_cam.data:
        cam.data.type = base_cam.data.type
        cam.data.ortho_scale = float(base_cam.data.ortho_scale)
        cam.data.clip_start = float(base_cam.data.clip_start)
        cam.data.clip_end = float(base_cam.data.clip_end)

    bpy.context.view_layer.update()
    return cam


def create_white_film_material(name: str = "Mask_WhiteFilm") -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND" if WHITE_FILM_ALPHA < 0.995 else "OPAQUE"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    if WHITE_FILM_ALPHA < 0.995:
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = WHITE_FILM_ALPHA
        links.new(transparent.outputs["BSDF"], mix.inputs[1])
        links.new(emission.outputs["Emission"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Surface"])
    else:
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def apply_white_film_to_buildings(building_ids: list[str]) -> None:
    mat = create_white_film_material()
    for building_id in building_ids:
        obj = bpy.data.objects.get(building_id)
        if not obj or obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.visible_shadow = False
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.shadow = False


def create_black_label_material(name: str = "Label_Black_Mat") -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = LABEL_COLOR
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Specular"].default_value = 0.28
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def set_all_labels_black() -> None:
    label_mat = create_black_label_material()
    for obj in bpy.data.objects:
        if not obj.name.startswith("Label_"):
            continue
        if obj.type != "FONT":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(label_mat)
        obj.visible_shadow = False
        if hasattr(obj, "cycles_visibility"):
            obj.cycles_visibility.shadow = False
            obj.cycles_visibility.diffuse = False
            obj.cycles_visibility.glossy = False
            obj.cycles_visibility.ambient_occlusion = False


def get_region_building_objects(building_ids: list[str]) -> list:
    objs = []
    for building_id in building_ids:
        obj = bpy.data.objects.get(building_id)
        if obj and obj.type == "MESH":
            objs.append(obj)
    return objs


def set_road_visibility(visible: bool) -> None:
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not is_road(obj.name):
            continue
        obj.hide_set(not visible)
        obj.hide_render = (not visible)


def projected_building_bbox_pixels(cam, objects: list) -> tuple[int, int, int, int] | None:
    if cam is None or cam.type != "CAMERA":
        return None
    scene = bpy.context.scene
    rx, ry = _render_resolution_px()

    xs = []
    ys = []
    for obj in objects:
        for p in _iter_object_world_vertices(obj):
            ndc = world_to_camera_view(scene, cam, p)
            x = float(ndc.x) * rx
            y = (1.0 - float(ndc.y)) * ry
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return None
    x1 = max(0, min(rx - 1, int(math.floor(min(xs)))))
    y1 = max(0, min(ry - 1, int(math.floor(min(ys)))))
    x2 = max(0, min(rx - 1, int(math.ceil(max(xs)))))
    y2 = max(0, min(ry - 1, int(math.ceil(max(ys)))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _project_world_point_to_pixel(cam, world_point: Vector, rx: int, ry: int) -> tuple[float, float]:
    scene = bpy.context.scene
    ndc = world_to_camera_view(scene, cam, world_point)
    x = float(ndc.x) * float(rx)
    y = (1.0 - float(ndc.y)) * float(ry)
    return x, y


def _project_building_center_pixels(cam, buildings: list[dict], rx: int, ry: int) -> dict[int, tuple[float, float]]:
    out = {}
    for b in buildings:
        label_id = int(b["label_id"])
        c = b["center"]
        world_point = Vector((float(c[0]), float(c[1]), float(c[2])))
        out[label_id] = _project_world_point_to_pixel(cam, world_point, rx, ry)
    return out


def _compute_reframe_crop_bounds_from_bbox(
    image_w: int,
    image_h: int,
    building_bbox: tuple[int, int, int, int] | None,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    if building_bbox is None:
        return (0, 0, int(image_w), int(image_h))

    x1 = max(0, min(image_w - 1, int(building_bbox[0])))
    y1 = max(0, min(image_h - 1, int(building_bbox[1])))
    x2 = max(0, min(image_w - 1, int(building_bbox[2])))
    y2 = max(0, min(image_h - 1, int(building_bbox[3])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    pad_x = int(round(bw * float(pad_ratio)))
    pad_y = int(round(bh * float(pad_ratio)))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_w - 1, x2 + pad_x)
    y2 = min(image_h - 1, y2 + pad_y)
    return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))


# ============================================================

# ============================================================

def load_buildings_from_blend(blend_file: str) -> list:
    
    blender_script = """
import bpy
import json
import math
from mathutils import Vector

buildings = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    if 'osm_buildings' not in obj.name:
        continue

    
    world_matrix = obj.matrix_world
    mesh = obj.data
    if not mesh:
        continue

    vertices = []
    for v in mesh.vertices:
        world_co = world_matrix @ v.co
        vertices.append([world_co.x, world_co.y, world_co.z])

    if not vertices:
        continue

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z

    buildings.append({
        "id": obj.name,
        "center": [center_x, center_y, center_z],
        "min": [min_x, min_y, min_z],
        "max": [max_x, max_y, max_z],
        "width": width,
        "depth": depth,
        "height": height,
    })

print(f"JSON_DATA:{json.dumps(buildings)}")
"""

    import subprocess
    result = subprocess.run(
        ['blender', '-b', blend_file, '--python-expr', blender_script],
        capture_output=True, text=True, timeout=60
    )

    output = result.stdout + result.stderr
    if "JSON_DATA:" in output:
        json_str = output.split("JSON_DATA:")[1].strip()
        
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = json_str[start:end]
            return json.loads(json_str)

    print(f"Warning: Failed to load buildings from blend file, stderr: {result.stderr}")
    return []


def load_region_data(region_file_or_dir: str) -> dict:
    
    if os.path.isdir(region_file_or_dir):
        region_file = os.path.join(region_file_or_dir, "region_data.json")
    else:
        region_file = region_file_or_dir
    with open(region_file, 'r') as f:
        return json.load(f)


def _load_failed_region_ids(failed_regions_path: str) -> set[int]:
    if not os.path.exists(failed_regions_path):
        return set()
    try:
        with open(failed_regions_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {int(x) for x in data}
    except Exception as e:
        print(f"[WARN] Failed to read failed_regions: {failed_regions_path}, err={e}")
    return set()


def resolve_region_inputs(blend_path: str, region_dir: str, use_clean: bool) -> tuple[str, str, set[int], bool]:
    
    clean_blend_path = os.environ.get("CLEAN_BLEND_PATH", blend_path.replace(".blend", "_clean.blend"))
    clean_region_json_path = os.environ.get("CLEAN_REGION_JSON_PATH", os.path.join(region_dir, "region_data_clean.json"))
    raw_region_json_path = os.path.join(region_dir, "region_data.json")
    failed_regions_path = os.environ.get("FAILED_REGIONS_PATH", os.path.join(region_dir, "failed_regions.json"))

    failed_region_ids = _load_failed_region_ids(failed_regions_path)

    if use_clean and os.path.exists(clean_blend_path) and os.path.exists(clean_region_json_path):
        return clean_blend_path, clean_region_json_path, failed_region_ids, True

    if use_clean:
        missing = []
        if not os.path.exists(clean_blend_path):
            missing.append(clean_blend_path)
        if not os.path.exists(clean_region_json_path):
            missing.append(clean_region_json_path)
        print("[WARN] Incomplete clean inputs; falling back to raw scene. Missing files:")
        for p in missing:
            print(f"       - {p}")

    return blend_path, raw_region_json_path, failed_region_ids, False


def get_buildings_in_region(all_buildings: list, building_ids: list) -> list:
    
    id_set = set(building_ids)
    return [b for b in all_buildings if b['id'] in id_set]


# ============================================================

# ============================================================

def _building_world_point(b: dict) -> Vector:
    x, y = b["center"][0], b["center"][1]
    return Vector((float(x), float(y), 0.0))


def _scene_centroid_xy(buildings: list[dict]) -> Vector:
    cx = sum(b["center"][0] for b in buildings) / len(buildings)
    cy = sum(b["center"][1] for b in buildings) / len(buildings)
    return Vector((float(cx), float(cy), 0.0))


def _scene_bbox_center_xy(buildings: list[dict]) -> Vector:
    xs = [float(b["center"][0]) for b in buildings]
    ys = [float(b["center"][1]) for b in buildings]
    return Vector((float((min(xs) + max(xs)) / 2.0), float((min(ys) + max(ys)) / 2.0), 0.0))


def _camera_forward_world(cam: bpy.types.Object) -> Vector:
    m = cam.matrix_world.to_3x3()
    forward = -(m @ Vector((0.0, 0.0, 1.0)))  # local -Z
    return forward.normalized()


def _camera_space_xy(cam: bpy.types.Object, world_point: Vector) -> tuple[float, float]:
    p_cam = cam.matrix_world.inverted() @ world_point
    return float(p_cam.x), float(p_cam.y)


def _wrap_pi(angle: float) -> float:
    while angle >= math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _min_abs_angle_diff(a: float, b: float) -> float:
    return abs(_wrap_pi(a - b))


def _world_dir_8way(theta: float) -> tuple[str, float]:
    sector = math.pi / 4.0
    half = sector / 2.0
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
    best_dir, best_diff = None, 1e9
    for d, c in centers:
        diff = _min_abs_angle_diff(theta, c)
        if diff < best_diff:
            best_dir, best_diff = d, diff
    dist_to_boundary = half - best_diff
    return best_dir, dist_to_boundary


def _screen_dir_8way(theta: float) -> tuple[str, float]:
    sector = math.pi / 4.0
    half = sector / 2.0
    centers = [
        ("right", 0.0),
        ("frontright", math.pi / 4.0),
        ("front", math.pi / 2.0),
        ("frontleft", 3.0 * math.pi / 4.0),
        ("left", math.pi),
        ("backleft", -3.0 * math.pi / 4.0),
        ("back", -math.pi / 2.0),
        ("backright", -math.pi / 4.0),
    ]
    best_dir, best_diff = None, 1e9
    for d, c in centers:
        diff = _min_abs_angle_diff(theta, c)
        if diff < best_diff:
            best_dir, best_diff = d, diff
    dist_to_boundary = half - best_diff
    return best_dir, dist_to_boundary


def _mcq_4_from_pool(correct: str, pool: list[str]) -> tuple[list[str], int]:
    pool = list(dict.fromkeys(pool))
    if correct not in pool:
        pool.append(correct)

    
    
    
    adjacent_map = {
        # world 8-way
        "northeast": ["north", "east"],
        "northwest": ["north", "west"],
        "southeast": ["south", "east"],
        "southwest": ["south", "west"],
        
        "north": ["northeast", "northwest"],
        "south": ["southeast", "southwest"],
        "east": ["northeast", "southeast"],
        "west": ["northwest", "southwest"],
        # screen 8-way
        "frontright": ["front", "right"],
        "frontleft": ["front", "left"],
        "backright": ["back", "right"],
        "backleft": ["back", "left"],
        
        "front": ["frontright", "frontleft"],
        "back": ["backright", "backleft"],
        "right": ["frontright", "backright"],
        "left": ["frontleft", "backleft"],
    }

    
    adjacent_to_exclude = adjacent_map.get(correct, [])

    
    distractors = [x for x in pool if x != correct and x not in adjacent_to_exclude]

    
    if len(distractors) < 3:
        
        distractors = [x for x in pool if x != correct]
        if len(distractors) >= 3:
            distractors = random.sample(distractors, 3)
    else:
        distractors = random.sample(distractors, 3)

    options = [correct] + distractors
    random.shuffle(options)
    return options, options.index(correct)


def _extreme_building_id_and_margin(
    buildings: list[dict],
    cam: bpy.types.Object,
    extreme_type: str,
) -> tuple[int, float]:
    """
    extreme_type in {'rightmost','leftmost','topmost','bottommost'}
    right/left -> camera-space x, top/bottom -> camera-space y
    margin = best - second_best on that axis.
    """
    scored: list[tuple[float, int]] = []
    for b in buildings:
        x, y = _camera_space_xy(cam, _building_world_point(b))
        key = x if extreme_type in ("rightmost", "leftmost") else y
        scored.append((key, int(b["label_id"])))

    scored.sort(key=lambda t: t[0])

    if extreme_type in ("rightmost", "topmost"):
        best = scored[-1]
        second = scored[-2]
        return best[1], float(best[0] - second[0])

    best = scored[0]
    second = scored[1]
    return best[1], float(second[0] - best[0])


def _extreme_building_world_coord(
    buildings: list[dict],
    extreme_type: str,
) -> tuple[int, float]:
    
    scored: list[tuple[float, int]] = []
    for b in buildings:
        world_pos = _building_world_point(b)
        
        key = world_pos.y if extreme_type in ("northmost", "southmost") else world_pos.x
        scored.append((key, int(b["label_id"])))

    scored.sort(key=lambda t: t[0])

    if extreme_type in ("northmost", "eastmost"):
        best = scored[-1]
        second = scored[-2]
        return best[1], float(best[0] - second[0])

    best = scored[0]
    second = scored[1]
    return best[1], float(second[0] - best[0])


def _rotate_screen_xy(x: float, y: float, rot_dir: str) -> tuple[float, float]:
    # screen: x=right, y=front(up)
    
    if rot_dir == "cw90":
        
        return (-y, x)
    if rot_dir == "ccw90":
        
        return (y, -x)
    raise ValueError(rot_dir)


def _task1_ok_with_retry(buildings: list[dict], cam: bpy.types.Object, scene_center: Vector | None = None) -> tuple[bool, str]:
    
    all_directions = ["northmost", "southmost", "eastmost", "westmost"]
    
    directions = all_directions.copy()
    # random.shuffle(directions)

    
    for extreme_type in directions:
        bid, margin = _extreme_building_world_coord(buildings, extreme_type)
        if margin < ISO_EXTREME_AXIS_MARGIN:
            
            continue

        centroid = scene_center if scene_center is not None else _scene_bbox_center_xy(buildings)
        b = next(x for x in buildings if x["label_id"] == bid)
        v = _building_world_point(b) - centroid
        if v.length < TASK1_MIN_WORLD_RADIUS:
            continue

        theta = math.atan2(v.y, v.x)
        _ans, dist_to_boundary = _world_dir_8way(theta)
        if dist_to_boundary >= math.radians(TASK1_WORLD_ANGLE_MARGIN_DEG):
            return True, extreme_type

    return False, None


def _task4_ok_with_retry(buildings: list[dict], cam: bpy.types.Object, scene_center: Vector | None = None) -> tuple[bool, str]:
    
    all_directions = ["northmost", "southmost", "eastmost", "westmost"]
    directions = all_directions.copy()
    random.shuffle(directions)

    centroid = scene_center if scene_center is not None else _scene_bbox_center_xy(buildings)

    for extreme_type in directions:
        bid, margin = _extreme_building_world_coord(buildings, extreme_type)
        if margin < ISO_EXTREME_AXIS_MARGIN:
            continue

        b = next(x for x in buildings if x["label_id"] == bid)
        v = _building_world_point(b) - centroid
        v2 = Vector((float(v.x), float(v.y)))

        if v2.length < TASK4_MIN_SCREEN_RADIUS:
            continue

        ok = True

        for rot_dir in ("cw90", "ccw90"):
            if rot_dir == "cw90":
                new_north = Vector((1.0, 0.0))   
                new_east = Vector((0.0, -1.0))   
            elif rot_dir == "ccw90":
                new_north = Vector((-1.0, 0.0))  
                new_east = Vector((0.0, 1.0))    
            else:
                raise ValueError(rot_dir)

            x_new = float(v2.dot(new_east))
            y_new = float(v2.dot(new_north))

            if math.hypot(x_new, y_new) < TASK4_MIN_SCREEN_RADIUS:
                ok = False
                break

            theta = math.atan2(y_new, x_new)
            _ans, dist_to_boundary = _world_dir_8way(theta)

            if dist_to_boundary < math.radians(TASK4_ANGLE_MARGIN_DEG):
                ok = False
                break

        if ok:
            return True, extreme_type

    return False, None



# ============================================================

# ============================================================

def qa_iso_task1_extreme_building_top_direction_mcq(
    all_buildings: list[dict],
    cam_iso: bpy.types.Object,
    extreme_type: str = None,
    scene_center: Vector | None = None,
    top_image_context: dict | None = None,
) -> dict:
    if extreme_type is None:
        extreme_type = random.choice(("northmost", "southmost", "eastmost", "westmost"))

    target_id, _margin = _extreme_building_world_coord(all_buildings, extreme_type)

    b = next(x for x in all_buildings if x["label_id"] == target_id)

    if top_image_context is not None:
        center_px = top_image_context.get("center_px")
        building_px_map = top_image_context.get("building_px", {})
        bp = building_px_map.get(int(target_id))
        if center_px is not None and bp is not None:
            dx = float(bp[0]) - float(center_px[0])  # east+
            dy = float(center_px[1]) - float(bp[1])  # north+
            theta = math.atan2(dy, dx)
        else:
            centroid = scene_center if scene_center is not None else _scene_bbox_center_xy(all_buildings)
            v = _building_world_point(b) - centroid
            theta = math.atan2(v.y, v.x)
    else:
        centroid = scene_center if scene_center is not None else _scene_bbox_center_xy(all_buildings)
        v = _building_world_point(b) - centroid
        theta = math.atan2(v.y, v.x)
    correct, _dist_to_boundary = _world_dir_8way(theta)

    pool = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    options, correct_idx = _mcq_4_from_pool(correct, pool)
    labels = ["A", "B", "C", "D"]
    choices = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

    extreme_type_desc = {
        "northmost": "the northernmost",
        "southmost": "the southernmost",
        "eastmost": "the easternmost",
        "westmost": "the westernmost",
    }.get(extreme_type, extreme_type)

    q = (
        "There are two images: an isometric image and a top-view image.\n"
        "- In the top-view image, top is North, bottom is South, left is West, and right is East.\n"
        "- In the isometric image, the red arrow indicates the same North direction as the top-view image.\n"
        "- The reference center is the geometric center of the final top-view image (including the North panel).\n"
        "- Building directions are determined by comparing building center points (bounding box centers).\n\n"
        f"Consider {extreme_type_desc} building in the isometric image. "
        "Relative to the reference center, which direction is that building located in the top-view image? "
        "Choose one option.\n\n"
        "Isometric image: <image>\n"
        "Top-view image: <image>\n"
    )

    return {
        "task_type": "top_isometric_direction_consistent",
        "question": q + "\n".join(choices),
        "answer": labels[correct_idx],
        "answer_text": options[correct_idx],
        "images": ["isometric.png", "top.png"],
        "meta": {
            "extreme_type": extreme_type,
            "target_building_id": target_id,
        },
    }



def qa_iso_task2_buildingA_maps_to_top_id(buildingA_label: int) -> dict:
    
    q = (
        "There are two images: an isometric image where only one building is labeled 'A', and a top-view image with numbered buildings. Which numbered building is 'A' in the top-view image? "
        "Answer with one number.\n\n"
        f"Isometric image: <image>\ntop-view image: <image>"
    )
    return {
        "task_type": "top_isometric_A_consistent",
        "question": q,
        "answer": str(buildingA_label),
        "images": ["isometric_A.png", "top.png"],
        "meta": {"buildingA_label": buildingA_label},
    }


def qa_iso_task3_camera_facing_cardinal(cam_iso: bpy.types.Object) -> dict:
    fwd = _camera_forward_world(cam_iso)
    fwd_xy = Vector((fwd.x, fwd.y, 0.0))
    if fwd_xy.length < 1e-6:
        ans = "north"
    else:
        fwd_xy.normalize()
        
        theta = math.atan2(fwd_xy.y, fwd_xy.x)
        ans, _ = _world_dir_8way(theta)

    q = (
        "For the isometric image, which cardinal direction (north/east/south/west) is the camera facing "
        "on the ground plane? Answer with one word."
    )
    return {
        "question": q,
        "answer": ans,
        "images": ["isometric.png"],
        "meta": {"camera_forward_xy": (float(fwd_xy.x), float(fwd_xy.y))},
    }


def qa_iso_task4_extreme_building_after_rot90_mcq(
    all_buildings: list[dict],
    cam_iso: bpy.types.Object,
    rot_dir: str,
    extreme_type: str = None,
    scene_center: Vector | None = None,
    top_image_context: dict | None = None,
) -> dict:
    
    if extreme_type is None:
        extreme_type = random.choice(("northmost", "southmost", "eastmost", "westmost"))

    target_id, _margin = _extreme_building_world_coord(all_buildings, extreme_type)

    b = next(x for x in all_buildings if x["label_id"] == target_id)

    
    centroid = scene_center if scene_center is not None else _scene_bbox_center_xy(all_buildings)
    v = _building_world_point(b) - centroid
    v2 = Vector((float(v.x), float(v.y)))

    
    # world: x=east, y=north
    if rot_dir == "cw90":
        new_north = Vector((1.0, 0.0))   
        new_east = Vector((0.0, -1.0))   
    elif rot_dir == "ccw90":
        new_north = Vector((-1.0, 0.0))  
        new_east = Vector((0.0, 1.0))    
    else:
        raise ValueError(f"Unsupported rot_dir: {rot_dir}")

    
    x_new = float(v2.dot(new_east))
    y_new = float(v2.dot(new_north))

    relative_angle = math.atan2(y_new, x_new)
    correct, _ = _world_dir_8way(relative_angle)

    pool = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    options, correct_idx = _mcq_4_from_pool(correct, pool)
    labels = ["A", "B", "C", "D"]
    choices = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

    rot_text = "CLOCKWISE" if rot_dir == "cw90" else "COUNTER-CLOCKWISE"
    rot_img_name = "isometric_rotated_cw90.png" if rot_dir == "cw90" else "isometric_rotated_ccw90.png"

    extreme_type_desc = {
        "northmost": "the northernmost",
        "southmost": "the southernmost",
        "eastmost": "the easternmost",
        "westmost": "the westernmost",
    }.get(extreme_type, extreme_type)

    q = f"""There is an isometric image of a city scene. In this image, the red arrow indicates the North direction.
Identify {extreme_type_desc} building in the image.

Now imagine that the camera rotates {rot_text} by 90 degrees around the center of the region.
When the camera rotates, the cardinal directions (north, east, south, west) rotate together with the view.

After this rotation, where will that building be located relative to the center of the region in the rotated coordinate frame?\n
Choose one option.\n
Isometric image: <image>\n"""

    return {
        "task_type": "isometric_camera_rotate",
        "question": q + "\n".join(choices),
        "answer": labels[correct_idx],
        "answer_text": options[correct_idx],
        "images": ["isometric.png"],
        "reference_images": [rot_img_name],
        "meta": {
            "extreme_type": extreme_type,
            "target_building_id": target_id,
            "rotation": rot_dir,
            "center_definition": "region center (world bbox center)",
            "x_new": x_new,
            "y_new": y_new,
            "relative_angle_deg": math.degrees(relative_angle),
        },
    }


def build_qa_isometric(
    all_buildings: list[dict],
    cam_iso: bpy.types.Object,
    buildingA_label: int,
    task1_extreme: str = None,
    task4_extreme: str = None,
    scene_center: Vector | None = None,
    top_image_context: dict | None = None,
) -> list[dict]:
    
    qa_top_isometric = [
        qa_iso_task1_extreme_building_top_direction_mcq(
            all_buildings, cam_iso, task1_extreme, scene_center, top_image_context
        ),
        qa_iso_task2_buildingA_maps_to_top_id(buildingA_label),
    ]
    qa_isometric = []
    if ENABLE_TASK3:
        qa_isometric.append(qa_iso_task3_camera_facing_cardinal(cam_iso))
    qa_isometric.append(
        qa_iso_task4_extreme_building_after_rot90_mcq(
            all_buildings, cam_iso, "cw90", task4_extreme, scene_center, top_image_context
        )
    )
    qa_isometric.append(
        qa_iso_task4_extreme_building_after_rot90_mcq(
            all_buildings, cam_iso, "ccw90", task4_extreme, scene_center, top_image_context
        )
    )
    return {
        "top_isometric": qa_top_isometric,
        "isometric": qa_isometric,
    }


def _qa_images_to_abs(qa_items: list[dict], sample_dir: str) -> list[dict]:
    """
    Convert each qa['images'] from filenames to absolute paths under sample_dir.
    Keeps order unchanged.
    """
    for item in qa_items:
        imgs = item.get("images")
        if not imgs:
            continue
        item["images"] = [os.path.abspath(os.path.join(sample_dir, p)) for p in imgs]
    return qa_items


# ============================================================

# ============================================================

def calculate_region_bounds(buildings: list) -> dict:
    
    if not buildings:
        return None

    min_x = min(b["min"][0] for b in buildings)
    max_x = max(b["max"][0] for b in buildings)
    min_y = min(b["min"][1] for b in buildings)
    max_y = max(b["max"][1] for b in buildings)
    min_z = min(b["min"][2] for b in buildings)
    max_z = max(b["max"][2] for b in buildings)

    return {
        "min_x": min_x, "max_x": max_x,
        "min_y": min_y, "max_y": max_y,
        "min_z": min_z, "max_z": max_z,
        "center_x": (min_x + max_x) / 2,
        "center_y": (min_y + max_y) / 2,
        "center_z": (min_z + max_z) / 2,
        "width": max_x - min_x,
        "depth": max_y - min_y,
        "height": max_z - min_z,
    }


def process_region(region_id: int, region_data: dict, all_buildings: list, labeler: BuildingLabeler) -> dict:
    
    building_ids = region_data["building_ids"]
    region_buildings = get_buildings_in_region(all_buildings, building_ids)

    if len(region_buildings) > MAX_BUILDINGS_PER_REGION:
        print(
            f"  Region {region_id}: skipped (building count {len(region_buildings)} exceeds limit {MAX_BUILDINGS_PER_REGION})"
        )
        return None

    if len(region_buildings) < 2:
        print(f"  Region {region_id}: skipped (insufficient buildings: {len(region_buildings)})")
        return None

    
    if len(region_buildings) < 3:
        print(f"  Region {region_id}: skipped (building count {len(region_buildings)} is insufficient to generate QA)")
        return None

    print(f"  Region {region_id}: processing {len(region_buildings)} buildings...")

    
    bounds = calculate_region_bounds(region_buildings)

    
    labeler.set_region_visibility(building_ids)
    set_road_visibility(SHOW_ROADS_IN_REGION_RENDER)

    
    labeler.clear_mask_materials()
    labeler.clear_all_labels()
    apply_white_film_to_buildings(building_ids)
    if SHOW_ROADS_IN_REGION_RENDER:
        apply_road_material()

    
    labeler.add_building_labels(building_ids, bounds)
    set_all_labels_black()

    
    cam_top = labeler.setup_camera_top_down(bounds, 'Camera_Top')
    cam_iso = labeler.setup_camera_isometric(bounds, 'Camera_Iso')
    building_objs = get_region_building_objects(building_ids)

    fit_ortho_camera_to_objects(
        cam_top,
        building_objs,
        margin_ratio=TOP_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    fit_ortho_camera_to_objects(
        cam_iso,
        building_objs,
        margin_ratio=ISO_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    top_building_bbox_px = projected_building_bbox_pixels(cam_top, building_objs)
    iso_building_bbox_px = projected_building_bbox_pixels(cam_iso, building_objs)

    
    simple_buildings = []
    for i, b in enumerate(region_buildings):
        simple_buildings.append({
            "label_id": i + 1,  
            "original_id": b["id"],
            "center": b["center"],
            "pos": (b["center"][0], b["center"][1]),
        })

    
    scene_center = Vector((float(bounds["center_x"]), float(bounds["center_y"]), 0.0))

    task1_ok, task1_extreme = _task1_ok_with_retry(simple_buildings, cam_iso, scene_center)
    if not task1_ok:
        print(f"    Warning: Region {region_id} does not satisfy Task1 constraints, skipping")
        labeler.show_all_visible_objects()
        return None
    task4_ok, task4_extreme = _task4_ok_with_retry(simple_buildings, cam_iso, scene_center)
    if not task4_ok:
        print(f"    Warning: Region {region_id} does not satisfy Task4 constraints, skipping")
        labeler.show_all_visible_objects()
        return None

    
    sample_dir = os.path.join(OUTPUT_DIR, f"region_{region_id:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    
    print(f"    Rendering top view...")
    force_disable_all_shadows()  
    top_path = os.path.join(sample_dir, "top.png")
    render_rx, render_ry = _render_resolution_px()
    labeler.render_view(cam_top, top_path)
    top_reframe_crop = _compute_reframe_crop_bounds_from_bbox(
        render_rx, render_ry, top_building_bbox_px, REFRAME_PAD_RATIO
    )
    reframe_image_focus_buildings(
        top_path,
        arrow_size=TOP_NORTH_ARROW_SIZE,
        reserve_for_north=True,
        building_bbox=top_building_bbox_px,
    )
    top_north_meta = draw_north_arrow(
        top_path,
        cam=cam_top,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=TOP_NORTH_ARROW_SIZE,
        corner_idx=0,
        return_meta=True,
    )
    top_trim_meta = trim_image_to_alpha_bbox(top_path, pad_px=FINAL_TRIM_PAD_PX, return_meta=True)

    
    top_center_context = None
    if isinstance(top_trim_meta, dict):
        projected_centers = _project_building_center_pixels(cam_top, simple_buildings, render_rx, render_ry)
        reframe_x1, reframe_y1 = int(top_reframe_crop[0]), int(top_reframe_crop[1])
        ext_top = int((top_north_meta or {}).get("ext_top", 0))
        trim_x1 = int(top_trim_meta.get("crop_x1", 0))
        trim_y1 = int(top_trim_meta.get("crop_y1", 0))
        final_w = float(top_trim_meta.get("final_w", 0))
        final_h = float(top_trim_meta.get("final_h", 0))
        building_px_final = {}
        for bid, (px0, py0) in projected_centers.items():
            px1 = float(px0) - float(reframe_x1)
            py1 = float(py0) - float(reframe_y1) + float(ext_top)
            px2 = px1 - float(trim_x1)
            py2 = py1 - float(trim_y1)
            building_px_final[int(bid)] = (px2, py2)
        top_center_context = {
            "center_px": (final_w * 0.5, final_h * 0.5),
            "building_px": building_px_final,
        }

    
    labeler.clear_all_labels()

    
    labeler.add_building_labels(building_ids, bounds)
    set_all_labels_black()

    
    print(f"    Rendering isometric view...")
    force_disable_all_shadows()  
    iso_path = os.path.join(sample_dir, "isometric.png")
    labeler.render_view(cam_iso, iso_path)
    reframe_image_focus_buildings(
        iso_path,
        arrow_size=ISO_NORTH_ARROW_SIZE,
        reserve_for_north=True,
        building_bbox=iso_building_bbox_px,
    )

    
    print(f"    Drawing north arrow...")
    draw_north_arrow(
        iso_path,
        cam=cam_iso,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=ISO_NORTH_ARROW_SIZE,
        corner_idx=0,
    )
    trim_image_to_alpha_bbox(iso_path, pad_px=FINAL_TRIM_PAD_PX)

    

    
    labeler.clear_all_labels()

    
    buildingA_label = random.randint(1, len(region_buildings))
    buildingA_id = building_ids[buildingA_label - 1]
    
    buildingA_data = next(b for b in region_buildings if b["id"] == buildingA_id)

    
    bldg_center = buildingA_data["center"]
    bldg_height = buildingA_data["height"]

    
    region_max_dim = max(bounds["width"], bounds["depth"])
    label_font_size = region_max_dim * 0.07

    
    text_curve = bpy.data.curves.new(f"Label_A_curve", type="FONT")
    text_curve.body = "A"
    text_curve.size = label_font_size
    text_curve.align_x = "CENTER"
    text_curve.align_y = "CENTER"
    text_curve.extrude = 0.02
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new(f"Label_A", text_curve)
    bpy.context.collection.objects.link(text_obj)
    
    label_z = bldg_center[2] + bldg_height/2 + region_max_dim * 0.03
    text_obj.location = (bldg_center[0], bldg_center[1], label_z)

    
    text_obj.visible_shadow = False
    if hasattr(text_obj, "cycles_visibility"):
        text_obj.cycles_visibility.shadow = False
        text_obj.cycles_visibility.diffuse = False
        text_obj.cycles_visibility.glossy = False
        text_obj.cycles_visibility.ambient_occlusion = False

    
    billboard = text_obj.constraints.new(type="LOCKED_TRACK")
    billboard.target = cam_iso
    billboard.track_axis = "TRACK_Z"
    billboard.lock_axis = "LOCK_Y"

    lock_roll = text_obj.constraints.new(type="LIMIT_ROTATION")
    lock_roll.owner_space = "LOCAL"
    lock_roll.use_limit_x = False
    lock_roll.use_limit_y = False
    lock_roll.use_limit_z = True
    lock_roll.min_z = 0.0
    lock_roll.max_z = 0.0

    
    mat = create_black_label_material("Label_A_mat")
    text_obj.data.materials.clear()
    text_obj.data.materials.append(mat)

    bpy.context.view_layer.update()
    iso_a_path = os.path.join(sample_dir, "isometric_A.png")
    labeler.render_view(cam_iso, iso_a_path)
    reframe_image_focus_buildings(
        iso_a_path,
        arrow_size=ISO_NORTH_ARROW_SIZE,
        reserve_for_north=True,
        building_bbox=iso_building_bbox_px,
    )
    draw_north_arrow(
        iso_a_path,
        cam=cam_iso,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=ISO_NORTH_ARROW_SIZE,
        corner_idx=0,
    )
    trim_image_to_alpha_bbox(iso_a_path, pad_px=FINAL_TRIM_PAD_PX)

    
    labeler.clear_all_labels()
    labeler.clear_mask_materials()

    
    
    apply_white_film_to_buildings(building_ids)
    
    labeler.add_building_labels(building_ids, bounds)
    set_all_labels_black()

    
    print(f"    Rendering rotated_cw90 view...")
    cam_cw90 = setup_rotated_isometric_camera_from_base(
        labeler=labeler,
        bounds=bounds,
        base_cam=cam_iso,
        name='Camera_CW90',
        rotate_deg=-90.0,
    )
    fit_ortho_camera_to_objects(
        cam_cw90,
        building_objs,
        margin_ratio=ISO_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    force_disable_all_shadows()
    rotated_cw90_path = os.path.join(sample_dir, "isometric_rotated_cw90.png")
    labeler.render_view(cam_cw90, rotated_cw90_path)
    cw_bbox_px = projected_building_bbox_pixels(cam_cw90, building_objs)
    reframe_image_focus_buildings(
        rotated_cw90_path,
        arrow_size=ISO_NORTH_ARROW_SIZE,
        reserve_for_north=True,
        building_bbox=cw_bbox_px,
    )
    
    draw_north_arrow(
        rotated_cw90_path,
        cam=cam_cw90,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=ISO_NORTH_ARROW_SIZE,
        corner_idx=0,
    )
    trim_image_to_alpha_bbox(rotated_cw90_path, pad_px=FINAL_TRIM_PAD_PX)

    
    print(f"    Rendering rotated_ccw90 view...")
    cam_ccw90 = setup_rotated_isometric_camera_from_base(
        labeler=labeler,
        bounds=bounds,
        base_cam=cam_iso,
        name='Camera_CCW90',
        rotate_deg=90.0,
    )
    fit_ortho_camera_to_objects(
        cam_ccw90,
        building_objs,
        margin_ratio=ISO_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    force_disable_all_shadows()
    rotated_ccw90_path = os.path.join(sample_dir, "isometric_rotated_ccw90.png")
    labeler.render_view(cam_ccw90, rotated_ccw90_path)
    ccw_bbox_px = projected_building_bbox_pixels(cam_ccw90, building_objs)
    reframe_image_focus_buildings(
        rotated_ccw90_path,
        arrow_size=ISO_NORTH_ARROW_SIZE,
        reserve_for_north=True,
        building_bbox=ccw_bbox_px,
    )
    
    draw_north_arrow(
        rotated_ccw90_path,
        cam=cam_ccw90,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=ISO_NORTH_ARROW_SIZE,
        corner_idx=0,
    )
    trim_image_to_alpha_bbox(rotated_ccw90_path, pad_px=FINAL_TRIM_PAD_PX)
    labeler.clear_all_labels()
    labeler.clear_mask_materials()

    
    labeler.show_all_visible_objects()

    
    print(f"    Generating QA...")
    qa_groups = build_qa_isometric(
        simple_buildings,
        cam_iso,
        buildingA_label,
        task1_extreme,
        task4_extreme,
        scene_center,
        top_center_context,
    )

    
    qa_groups["top_isometric"] = _qa_images_to_abs(qa_groups["top_isometric"], sample_dir)
    qa_groups["isometric"] = _qa_images_to_abs(qa_groups["isometric"], sample_dir)

    
    for qa in qa_groups["isometric"]:
        if "rotation" in qa.get("meta", {}):
            rot_dir = qa["meta"]["rotation"]
            if rot_dir == "cw90":
                qa["reference_images"] = [os.path.abspath(rotated_cw90_path)]
            elif rot_dir == "ccw90":
                qa["reference_images"] = [os.path.abspath(rotated_ccw90_path)]

    return {
        "region_id": region_id,
        "building_count": len(region_buildings),
        "buildings": simple_buildings,
        "images": {
            "top": os.path.abspath(top_path),
            "isometric": os.path.abspath(iso_path),
            "isometric_A": os.path.abspath(iso_a_path),
        },
        "qa": {
            "top_isometric": qa_groups["top_isometric"],
            "isometric": qa_groups["isometric"],
        },
        "special_refs": {
            "buildingA_id": buildingA_label,
            "enable_task3": ENABLE_TASK3,
            "scene_center_definition": "final top-image geometric center (including north panel)",
            "iso_extreme_axis_margin": ISO_EXTREME_AXIS_MARGIN,
            "task1_world_angle_margin_deg": TASK1_WORLD_ANGLE_MARGIN_DEG,
            "task4_angle_margin_deg": TASK4_ANGLE_MARGIN_DEG,
        },
    }


def get_object_world_bounds_from_data(obj: bpy.types.Object) -> dict:
    
    
    if hasattr(obj, 'bound_box') and obj.bound_box:
        verts = [obj.matrix_world @ Vector(co) for co in obj.bound_box]
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        zs = [v.z for v in verts]
    else:
        
        display_bounds = obj.display_bounds
        if display_bounds:
            verts = [obj.matrix_world @ Vector(co) for co in display_bounds]
            xs = [v.x for v in verts]
            ys = [v.y for v in verts]
            zs = [v.z for v in verts]
        else:
            
            return {
                "min_x": obj.location.x - 1,
                "max_x": obj.location.x + 1,
                "min_y": obj.location.y - 1,
                "max_y": obj.location.y + 1,
                "min_z": obj.location.z,
                "max_z": obj.location.z + 5,
                "center_x": obj.location.x,
                "center_y": obj.location.y,
                "center_z": obj.location.z + 2.5,
                "width": 2,
                "depth": 2,
                "height": 5,
            }

    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "center_z": (min(zs) + max(zs)) / 2,
        "width": max(xs) - min(xs),
        "depth": max(ys) - min(ys),
        "height": max(zs) - min(zs),
    }


def main():
    max_regions = MAX_REGIONS_ARG
    if max_regions is None:
        max_regions = int(os.environ.get("MAX_REGIONS", "0"))
    use_clean = USE_CLEAN_ARG == 1
    if "USE_CLEAN_SCENE" in os.environ:
        use_clean = os.environ.get("USE_CLEAN_SCENE", "1").lower() in ("1", "true", "yes")

    effective_blend_path, effective_region_json_path, failed_region_ids, using_clean = resolve_region_inputs(
        BLEND_PATH,
        REGION_DIR,
        use_clean=use_clean,
    )

    print("=" * 60)
    print(f"Region-level spatial orientation data generation - Region: {REGION_NAME}")
    print("=" * 60)

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    print("\n[1/4] Loading data...")
    print(f"  Region: {REGION_NAME}")
    print(f"  BLEND file: {effective_blend_path}")
    print(f"  Region directory: {REGION_DIR}")
    print(f"  Region JSON: {effective_region_json_path}")
    print(f"  Using clean results: {using_clean}")

    all_buildings = load_buildings_from_blend(effective_blend_path)
    print(f"  Loaded buildings: {len(all_buildings)}")

    region_data = load_region_data(effective_region_json_path)
    regions = region_data.get("regions", [])
    raw_region_count = len(regions)
    if failed_region_ids:
        regions = [r for r in regions if int(r.get("region_id", -1)) not in failed_region_ids]
        print(f"  After filtering failed_regions: {len(regions)}/{raw_region_count} regions")
    if max_regions <= 0:
        regions = regions  
    else:
        regions = regions[:max_regions]  
    print(f"  Region count: {len(regions)}")

    
    print("\n[2/4] Opening BLEND file...")
    bpy.ops.wm.open_mainfile(filepath=effective_blend_path)
    render_resolution = (RESOLUTION, RESOLUTION) if RESOLUTION > 0 else _render_resolution_px()

    
    labeler = BuildingLabeler(
        region_data_path=effective_region_json_path,
        blend_path=effective_blend_path,
        output_dir=OUTPUT_DIR,
        ortho_scale_factor=1.3,  
        label_height_ratio=0.01,
        font_size_ratio=0.07,
        samples=32,
        resolution=render_resolution,
        mask_alpha=0.8,
    )

    
    _original_setup_lighting = labeler.setup_lighting

    def _patched_setup_lighting(self):
        
        scene = bpy.context.scene

        
        for obj in list(bpy.data.objects):
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj)

        
        sun = bpy.data.objects.new("Sun", bpy.data.lights.new("SunLight", type='SUN'))
        bpy.context.collection.objects.link(sun)
        sun.data.energy = 4.0
        sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
        
        if hasattr(sun.data, 'cast_shadow'):
            sun.data.cast_shadow = False

        
        fill_light = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLight", type='AREA'))
        bpy.context.collection.objects.link(fill_light)
        fill_light.data.energy = 150
        fill_light.location = (0, 0, 100)
        fill_light.data.size = 100
        
        if hasattr(fill_light.data, 'cast_shadow'):
            fill_light.data.cast_shadow = False

        
        scene.cycles.use_shadows = False
        scene.cycles.use_progressive = False
        scene.cycles.use_shadow_highlight = False
        scene.cycles.blur_shadow = 0

        
        for obj in bpy.data.objects:
            if hasattr(obj, 'cycles_visibility'):
                obj.cycles_visibility.cast_shadow = False
                obj.cycles_visibility.receive_shadow = False

    labeler.setup_lighting = types.MethodType(_patched_setup_lighting, labeler)

    
    labeler.setup_render()
    labeler.setup_lighting()
    force_disable_all_shadows()  

    
    print("\n[3/4] Processing regions...")
    all_results = []
    valid_count = 0

    for region in regions:
        region_id = region["region_id"]
        result = process_region(region_id, region, all_buildings, labeler)
        if result:
            all_results.append(result)
            valid_count += 1

    print(f"\n  Successfully processed {valid_count}/{len(regions)} regions")

    
    print("\n[4/4] Saving metadata...")
    meta_path = os.path.join(OUTPUT_DIR, "metadata_regions.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Metadata file: {meta_path}")
    print(f"  Valid regions: {valid_count}")


if __name__ == "__main__":
    main()
