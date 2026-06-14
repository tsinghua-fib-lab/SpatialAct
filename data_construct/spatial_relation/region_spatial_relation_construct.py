#!/usr/bin/env python3


import bpy
import random
import math
import json
import os
import sys
import subprocess
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.expanduser('~/SpatialAct'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'blender_scripts'))

import types
from src.building_labels import BuildingLabeler, is_building as check_is_building, is_road, apply_road_material, is_visible_object

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
) -> bool:
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
alpha = img.getchannel("A")
pix = img.load()
content_bbox = alpha.getbbox()
if content_bbox is None:
    content_bbox = (0, 0, width, height)
cx1, cy1, cx2, cy2 = content_bbox

margin = max(16, int(arrow_size * 0.20))
panel_w = int(arrow_size * 1.75)
panel_h = int(arrow_size * 1.95)
y_top = max(margin, min(height - margin - panel_h, int(cy1 + margin)))
y_bottom = max(margin, min(height - margin - panel_h, int(cy2 - panel_h - margin)))
x_left = max(margin, min(width - margin - panel_w, int(cx1 + margin)))
x_right = max(margin, min(width - margin - panel_w, int(cx2 - panel_w - margin)))

candidates = [
    (x_right, y_top),
    (x_left, y_top),
    (x_right, y_bottom),
    (x_left, y_bottom),
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
        print(f"  [draw_north_arrow] Warning: external python fallback failed: {e}")
        return False


def draw_north_arrow(
    image_path: str,
    cam,
    north_world_dir=None,
    arrow_color=(255, 0, 0, 255),
    arrow_size=120,
    corner_idx: int | None = None,
):
    if not os.path.exists(image_path):
        print(f"  [draw_north_arrow] Warning: Image not found: {image_path}")
        return image_path
    if north_world_dir is None:
        north_world_dir = Vector((0.0, 1.0, 0.0))
    screen_vec = _compute_north_screen_vec(cam, north_world_dir)
    _draw_north_arrow_via_system_python(
        image_path,
        screen_vec,
        arrow_color=arrow_color,
        arrow_size=arrow_size,
        corner_idx=corner_idx,
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


import argparse


parser = argparse.ArgumentParser(description="Region-level spatial relation data generation")
parser.add_argument("--region", type=str, default=None, help="Region name")
parser.add_argument("--max-regions", type=int, default=None, help="Maximum number of regions to process; <=0 means all")
parser.add_argument(
    "--use-clean",
    type=int,
    default=1,
    help="Whether to prioritize clean outputs from preprocess_clean.py (1=yes, 0=no)",
)
args, _ = parser.parse_known_args()


REGION_NAME = args.region if args.region else os.environ.get("REGION_NAME")
MAX_REGIONS_ARG = args.max_regions
USE_CLEAN_ARG = int(args.use_clean)


BLEND_PATH = os.environ.get("BLEND_PATH", os.path.join(PROJECT_ROOT, f"osm_scene_0228/{REGION_NAME}_osm_scene_0228/osm_reference.blend"))


REGION_DIR = os.environ.get("REGION_DIR", os.path.join(PROJECT_ROOT, f"benchmark/data_construct/model_process/results/{REGION_NAME}_kmeans"))


OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "benchmark/data/spatial_relation/"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{REGION_NAME}_regions")


RESOLUTION = int(os.environ.get("RESOLUTION", "0"))
MAX_BUILDINGS_PER_REGION = int(os.environ.get("MAX_BUILDINGS_PER_REGION", "15"))


GENERATE_TOP_QA = True
GENERATE_ISO_QA = True


DISTANCE_MARGIN = 5  
DIRECTION_THRESHOLD = 2  

DIRECTION_ANGLE_MARGIN_DEG = 10.0


LABEL_FONT_SIZE = 1.2
LABEL_Z_OFFSET = 0.25
LABEL_COLOR = (0.02, 0.02, 0.02, 1.0)


WHITE_FILM_ALPHA = 0.92
TOP_FIT_MARGIN_RATIO = 0.20
ISO_FIT_MARGIN_RATIO = 0.24
MIN_FIT_ORTHO_SCALE = 2.2
ARROW_CORNER_SCALE_STEP = 1.04
ARROW_CORNER_MAX_STEPS = 8
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


def _arrow_panel_pixel_geometry(arrow_size: int) -> tuple[int, int, int]:
    margin_px = max(16, int(arrow_size * 0.20))
    panel_w_px = int(arrow_size * 1.75)
    panel_h_px = int(arrow_size * 1.95)
    return margin_px, panel_w_px, panel_h_px


def _camera_half_extents(cam) -> tuple[float, float]:
    frame = cam.data.view_frame(scene=bpy.context.scene)
    hx = max(abs(v.x) for v in frame)
    hy = max(abs(v.y) for v in frame)
    return float(hx), float(hy)


def _project_object_boxes_camera(cam, objects) -> list[tuple[float, float, float, float]]:
    boxes = []
    cam_inv = cam.matrix_world.inverted()
    for obj in objects:
        xs = []
        ys = []
        for p in _iter_object_world_vertices(obj):
            q = cam_inv @ p
            xs.append(float(q.x))
            ys.append(float(q.y))
        if xs and ys:
            boxes.append((min(xs), max(xs), min(ys), max(ys)))
    return boxes


def _panel_rect_in_camera(cam, arrow_size: int, corner_idx: int) -> tuple[float, float, float, float]:
    hx, hy = _camera_half_extents(cam)
    rx, ry = _render_resolution_px()
    margin_px, panel_w_px, panel_h_px = _arrow_panel_pixel_geometry(arrow_size)

    px_to_wx = (2.0 * hx) / float(rx)
    px_to_wy = (2.0 * hy) / float(ry)
    margin_x = margin_px * px_to_wx
    margin_y = margin_px * px_to_wy
    panel_w = panel_w_px * px_to_wx
    panel_h = panel_h_px * px_to_wy

    if corner_idx == 0:  # top-right
        x1, x2 = hx - margin_x - panel_w, hx - margin_x
        y1, y2 = hy - margin_y - panel_h, hy - margin_y
    elif corner_idx == 1:  # top-left
        x1, x2 = -hx + margin_x, -hx + margin_x + panel_w
        y1, y2 = hy - margin_y - panel_h, hy - margin_y
    elif corner_idx == 2:  # bottom-right
        x1, x2 = hx - margin_x - panel_w, hx - margin_x
        y1, y2 = -hy + margin_y, -hy + margin_y + panel_h
    else:  # bottom-left
        x1, x2 = -hx + margin_x, -hx + margin_x + panel_w
        y1, y2 = -hy + margin_y, -hy + margin_y + panel_h
    return float(x1), float(x2), float(y1), float(y2)


def _rect_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


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
        if DEBUG_CAMERA_FIT:
            print(
                f"[camera_fit_axes] cam={cam.name} "
                f"right=({cam_right.x:.3f},{cam_right.y:.3f},{cam_right.z:.3f}) "
                f"up=({cam_up.x:.3f},{cam_up.y:.3f},{cam_up.z:.3f}) "
                f"center=({center_x:.3f},{center_y:.3f}) top_like=1"
            )
            print(
                f"[camera_fit] cam={cam.name} needed={needed_scale:.4f} "
                f"fit_half=({fit_half_x:.4f},{fit_half_y:.4f}) aspect={aspect:.3f} "
                f"margin={margin_ratio:.3f} final={cam.data.ortho_scale:.4f}"
            )
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

    if DEBUG_CAMERA_FIT:
        print(
            f"[camera_fit_axes] cam={cam.name} "
            f"right=({cam_right.x:.3f},{cam_right.y:.3f},{cam_right.z:.3f}) "
            f"up=({cam_up.x:.3f},{cam_up.y:.3f},{cam_up.z:.3f}) "
            f"center=({center_x:.3f},{center_y:.3f}) top_like=0"
        )
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


def choose_arrow_corner_and_adapt_view(
    cam,
    objects: list,
    arrow_size: int,
    scale_step: float = ARROW_CORNER_SCALE_STEP,
    max_steps: int = ARROW_CORNER_MAX_STEPS,
) -> int:
    if cam is None or cam.type != "CAMERA" or cam.data.type != "ORTHO":
        return 0
    boxes = _project_object_boxes_camera(cam, objects)
    if not boxes:
        return 0

    def overlap_areas_and_panel():
        vals = []
        panel_areas = []
        for idx in range(4):
            rect = _panel_rect_in_camera(cam, arrow_size, idx)
            vals.append(sum(_rect_overlap_area(rect, b) for b in boxes))
            panel_areas.append(max(0.0, (rect[1] - rect[0]) * (rect[3] - rect[2])))
        return vals, panel_areas

    areas, panel_areas = overlap_areas_and_panel()
    selected = min(range(4), key=lambda i: areas[i])

    panel_area = max(panel_areas[selected], 1e-9)
    overlap_ratio = areas[selected] / panel_area

    
    if overlap_ratio <= 0.18:
        if DEBUG_CAMERA_FIT:
            print(
                f"[arrow_fit] cam={cam.name} corner={selected} overlap_ratio={overlap_ratio:.4f} steps=0 "
                f"scale={cam.data.ortho_scale:.4f}"
            )
        return int(selected)

    step = 0
    while (areas[selected] / panel_area) > 0.05 and step < max_steps:
        cam.data.ortho_scale *= scale_step
        areas, panel_areas = overlap_areas_and_panel()
        panel_area = max(panel_areas[selected], 1e-9)
        step += 1
    if (areas[selected] / panel_area) > 0.05:
        selected = min(range(4), key=lambda i: areas[i])
    if DEBUG_CAMERA_FIT:
        final_ratio = areas[selected] / max(panel_areas[selected], 1e-9)
        print(
            f"[arrow_fit] cam={cam.name} corner={selected} overlap_ratio={final_ratio:.4f} "
            f"steps={step} scale={cam.data.ortho_scale:.4f}"
        )
    return int(selected)


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


def get_direction(a_pos, b_pos, view_type: str) -> str:
    
    dx = a_pos[0] - b_pos[0]
    dy = a_pos[1] - b_pos[1]

    
    theta = math.atan2(dy, dx)

    if view_type == "top":
        direction, _ = _world_dir_8way(theta)
    else:
        direction, _ = _screen_dir_8way(theta)

    return direction


def get_direction_with_confidence(a_pos, b_pos, view_type: str, min_margin_deg: float = 10.0) -> tuple[str, float, bool]:
    
    dx = a_pos[0] - b_pos[0]
    dy = a_pos[1] - b_pos[1]

    
    theta = math.atan2(dy, dx)

    if view_type == "top":
        direction, dist_to_boundary = _world_dir_8way(theta)
    else:
        direction, dist_to_boundary = _screen_dir_8way(theta)

    is_clear = math.degrees(dist_to_boundary) >= min_margin_deg
    return direction, math.degrees(dist_to_boundary), is_clear


def get_buildings_on_side(target_pos, side: str, all_buildings: list, view_type: str) -> list:
    
    tx, ty = target_pos[0], target_pos[1]
    result = []

    for b in all_buildings:
        bx, by = b["center"][0], b["center"][1]
        dx = bx - tx
        dy = by - ty

        if view_type == "top":
            if side == "north" and dy > DIRECTION_THRESHOLD:
                result.append(b["id"])
            elif side == "south" and dy < -DIRECTION_THRESHOLD:
                result.append(b["id"])
            elif side == "east" and dx > DIRECTION_THRESHOLD:
                result.append(b["id"])
            elif side == "west" and dx < -DIRECTION_THRESHOLD:
                result.append(b["id"])
        else:
            if side == "left" and dx < -DIRECTION_THRESHOLD:
                result.append(b["id"])
            elif side == "right" and dx > DIRECTION_THRESHOLD:
                result.append(b["id"])
            elif side == "front" and dy > DIRECTION_THRESHOLD:
                result.append(b["id"])
            elif side == "back" and dy < -DIRECTION_THRESHOLD:
                result.append(b["id"])

    return result


def _direction_pool(view_type: str) -> list:
    
    if view_type == "top":
        singles = ["north", "south", "east", "west"]
        doubles = ["northeast", "northwest", "southeast", "southwest"]
        return singles + doubles
    singles = ["front", "back", "left", "right"]
    doubles = ["frontleft", "frontright", "backleft", "backright"]
    return singles + doubles


def make_mcq_options(correct: str, view_type: str, k: int = 4) -> tuple:
    
    pool = _direction_pool(view_type)

    if correct not in pool:
        correct = random.choice(pool)

    
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

    
    distractors = [d for d in pool if d != correct and d not in adjacent_to_exclude]

    
    if len(distractors) < k - 1:
        raise ValueError(
            f"Not enough non-adjacent distractors: correct={correct}, "
            f"excluded={adjacent_to_exclude}, pool={pool}"
        )
    distractors = random.sample(distractors, k - 1)

    options = [correct] + distractors
    random.shuffle(options)
    correct_idx = options.index(correct)
    return options, correct_idx


def qa_task1_direction(all_buildings: list, view_type: str) -> dict:
    
    if len(all_buildings) < 2:
        return None

    ids = [b["id"] for b in all_buildings]

    
    effective_view_type = "top" if view_type == "isometric" else view_type

    
    for _ in range(100):
        a_id, b_id = random.sample(ids, 2)
        a = next(b for b in all_buildings if b["id"] == a_id)
        b = next(b for b in all_buildings if b["id"] == b_id)

        correct_dir, dist_to_boundary_deg, is_clear = get_direction_with_confidence(
            a["center"], b["center"], effective_view_type, DIRECTION_ANGLE_MARGIN_DEG
        )

        if is_clear:
            if view_type == "top":
                q_text = (
                    "In this top-view image: top is North, bottom is South, left is West, right is East. "
                    "Direction is determined by comparing the center points (bounding box center) of the buildings. "
                    f"Which side is building {a_id} relative to building {b_id}?\n"
                    "Top-view image: <image>"
                )
            else:
                
                q_text = (
                    "In this isometric image, the red arrow points to North. "
                    "Direction is determined by comparing the center points (bounding box center) of the buildings. "
                    f"Which side is building {a_id} relative to building {b_id}?\n"
                    f"Isometric image: <image>"
                )

            options, correct_idx = make_mcq_options(correct_dir, effective_view_type)
            labels = ["A", "B", "C", "D"]
            choices = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

            return {
                "question": q_text + "\n".join(choices),
                "answer": labels[correct_idx],
                "answer_text": options[correct_idx],
                "task_type": f"{view_type}_direction"
            }

    
    return None


def qa_task2_closest_pair(all_buildings: list, view_type: str) -> dict:
    
    if len(all_buildings) < 4:
        return None

    
    pairs = []
    for i in range(len(all_buildings)):
        for j in range(i + 1, len(all_buildings)):
            d = math.dist(all_buildings[i]["center"][:2], all_buildings[j]["center"][:2])
            pairs.append((d, all_buildings[i]["id"], all_buildings[j]["id"]))

    
    pairs.sort(key=lambda x: x[0])

    
    for idx in range(len(pairs)):
        closest = pairs[idx]
        min_d = closest[0]

        
        for jdx in range(len(pairs)):
            if jdx == idx:
                continue
            second = pairs[jdx]
            
            if set([closest[1], closest[2]]) == set([second[1], second[2]]):
                continue
            second_d = second[0]
            
            if second_d - min_d >= DISTANCE_MARGIN:
                a_id, b_id = sorted([closest[1], closest[2]], key=lambda x: int(x))
                return {
                    "question": (
                        "Which two buildings are closest to each other? "
                        "Object positions are determined by the center points (bounding box center) of the buildings. "
                        "Answer with two numbers joined by 'and' in ascending order (x < y), e.g., '2 and 5'.\n"
                        + ("Top-view image: <image>" if view_type == "top" else "Isometric image: <image>")
                    ),
                    "answer": f"{a_id} and {b_id}",
                    "task_type": f"{view_type}_closest"
                }

    return None


def qa_task3_farthest(all_buildings: list, view_type: str) -> dict:
    
    if len(all_buildings) < 3:
        return None

    
    for _ in range(50):
        target_id = random.choice([b["id"] for b in all_buildings])
        target = next(b for b in all_buildings if b["id"] == target_id)

        
        distances = []
        for b in all_buildings:
            if b["id"] == target_id:
                continue
            d = math.dist(target["center"][:2], b["center"][:2])
            distances.append((d, b["id"]))

        if len(distances) < 2:
            continue

        
        distances.sort(key=lambda x: x[0], reverse=True)

        farthest = distances[0]
        second_farthest = distances[1]

        
        if farthest[0] - second_farthest[0] >= DISTANCE_MARGIN:
            return {
                "question": (
                    f"Which building is farthest from building {target_id}? "
                    "Object positions are determined by the center points (bounding box center) of the buildings. "
                    "Answer with one number.\n"
                    + ("Top-view image: <image>" if view_type == "top" else "Isometric image: <image>")
                ),
                "answer": str(farthest[1]),
                "task_type": f"{view_type}_farthest"
            }

    return None


def qa_task4_side_count(all_buildings: list, view_type: str) -> dict:
    
    if len(all_buildings) < 2:
        return None

    target_id = random.choice([b["id"] for b in all_buildings])
    target = next(b for b in all_buildings if b["id"] == target_id)

    
    side = random.choice(["north", "south", "east", "west"])
    side_building_ids = get_buildings_on_side(target["center"], side, all_buildings, "top")

    def _sort_key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))

    side_building_ids = sorted(side_building_ids, key=_sort_key)
    count = len(side_building_ids)

    if view_type == "top":
        question = (
            "In this top-view image: top is North, bottom is South, left is West, right is East. "
            "Direction is determined by comparing the center points (bounding box center) of the buildings. "
            "An object can be on multiple sides at once (e.g., northwest counts as both north and west).\n"
            f"How many buildings are on the {side} side of building {target_id}? Answer with one number.\n"
            "Top-view image: <image>"
        )
    else:
        
        question = (
            "In this isometric image, the red arrow points to North. "
            "Direction is determined by comparing the center points (bounding box center) of the buildings. "
            "An object can be on multiple sides at once (e.g., northwest counts as both north and west).\n"
            f"How many buildings are on the {side} side of building {target_id}? Answer with one number.\n"
            "Isometric image: <image>"
        )

    return {
        "question": question,
        "answer": str(count),
        "task_type": f"{view_type}_side_count",
        "meta": {
            "target_id": target_id,
            "side": side,
            "side_building_ids": side_building_ids,
            "count": count,
        },
    }


def build_qa(all_buildings: list, view_type: str, sample_dir: str) -> list:
    
    qa_items = []

    if view_type == "top" and GENERATE_TOP_QA:
        
        for _ in range(3):
            qa = qa_task1_direction(all_buildings, view_type)
            if qa:
                qa_items.append(qa)
                break

        
        qa = qa_task2_closest_pair(all_buildings, view_type)
        if qa:
            qa_items.append(qa)

        
        qa = qa_task3_farthest(all_buildings, view_type)
        if qa:
            qa_items.append(qa)

        
        for _ in range(3):
            qa = qa_task4_side_count(all_buildings, view_type)
            if qa:
                qa_items.append(qa)
                break

    elif view_type == "isometric" and GENERATE_ISO_QA:
        
        for _ in range(3):
            qa = qa_task1_direction(all_buildings, view_type)
            if qa:
                qa_items.append(qa)
                break

        
        qa = qa_task2_closest_pair(all_buildings, view_type)
        if qa:
            qa_items.append(qa)

        
        qa = qa_task3_farthest(all_buildings, view_type)
        if qa:
            qa_items.append(qa)

        
        for _ in range(3):
            qa = qa_task4_side_count(all_buildings, view_type)
            if qa:
                qa_items.append(qa)
                break

    
    image_filename = "top.png" if view_type == "top" else "isometric.png"
    abs_path = os.path.abspath(os.path.join(sample_dir, image_filename))
    for item in qa_items:
        item["images"] = [abs_path]

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

    print(f"  Region {region_id}: processing {len(region_buildings)} buildings...")

    
    sample_dir = os.path.join(OUTPUT_DIR, f"region_{region_id:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    
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
    if DEBUG_CAMERA_FIT:
        print(f"[reframe_bbox] region={region_id} top={top_building_bbox_px} iso={iso_building_bbox_px}")

    
    simple_buildings = []
    for i, b in enumerate(region_buildings):
        simple_buildings.append({
            "id": str(i + 1),  
            "original_id": b["id"],
            "center": b["center"],
            "pos": (b["center"][0], b["center"][1]),
        })

    
    print(f"    Rendering top view...")
    force_disable_all_shadows()  
    top_path = os.path.join(sample_dir, "top.png")
    labeler.render_view(cam_top, top_path)
    reframe_image_focus_buildings(
        top_path,
        arrow_size=TOP_NORTH_ARROW_SIZE,
        reserve_for_north=True,
    )
    draw_north_arrow(
        top_path,
        cam_top,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=TOP_NORTH_ARROW_SIZE,
        corner_idx=0,
    )
    trim_image_to_alpha_bbox(top_path, pad_px=FINAL_TRIM_PAD_PX)

    
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
    )

    
    print(f"    Drawing north arrow...")
    draw_north_arrow(
        iso_path,
        cam_iso,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=ISO_NORTH_ARROW_SIZE,
        corner_idx=0,
    )
    trim_image_to_alpha_bbox(iso_path, pad_px=FINAL_TRIM_PAD_PX)

    
    labeler.clear_all_labels()
    labeler.clear_mask_materials()

    
    labeler.show_all_visible_objects()

    
    print(f"    Generating QA...")
    qa_top = build_qa(simple_buildings, "top", sample_dir) if GENERATE_TOP_QA else []
    qa_iso = build_qa(simple_buildings, "isometric", sample_dir) if GENERATE_ISO_QA else []

    return {
        "region_id": region_id,
        "building_count": len(region_buildings),
        "buildings": simple_buildings,
        "images": {
            "top": os.path.abspath(top_path),
            "isometric": os.path.abspath(iso_path),
        },
        "qa": {
            "top": qa_top,
            "isometric": qa_iso,
        },
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
    print(f"Region-level spatial relation data generation - Region: {REGION_NAME}")
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
        ortho_scale_factor=1.8,
        label_height_ratio=0.01,
        font_size_ratio=0.07,
        samples=64,
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
    scene = bpy.context.scene
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = 'BOX'
    if hasattr(scene.cycles, "filter_width"):
        scene.cycles.filter_width = 0.5
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
