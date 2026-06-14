#!/usr/bin/env python3

import bpy
import random
import math
import json
import os
import subprocess
import time
from mathutils import Vector

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ============================================================
# Config
# ============================================================

NUM_SAMPLES = int(os.environ.get("SV_NUM_SAMPLES", "60"))
START_SAMPLE_IDX = int(os.environ.get("SV_START_SAMPLE_IDX", "0"))
BUILDINGS_PER_SCENE = 6

SCENE_BOUNDS = 14.0
IN_BOUNDS_MARGIN = 0.8

MIN_GAP = 1.0
MAX_PLACE_ATTEMPTS = 800
MAX_SCENE_RETRY = 300
MAX_ATTEMPT_SECONDS = float(os.environ.get("SV_MAX_ATTEMPT_SECONDS", "180"))
PROGRESS_LOG_INTERVAL_SECONDS = float(os.environ.get("SV_PROGRESS_LOG_INTERVAL_SECONDS", "15"))

RESOLUTION = 1080

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.expanduser("~/SpatialAct/benchmark/data/spatial_visualization/v1.2"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "top_isometric")

LABEL_FONT_SIZE = 1.75
LABEL_Z_OFFSET = 0.35
LABEL_COLOR = (0.02, 0.02, 0.02, 1.0)
MIN_HEIGHT_DIFF = 1.2

NEAR_PAIR_GAP = 0.15
PUSH_OUT_START_INSET = 0.80
PUSH_OUT_STEP = 0.01
PUSH_OUT_MAX_STEPS = 250

TOP_ORTHO_SCALE_FACTOR = 1.14
TOP_CAMERA_EDGE_MARGIN = 2.4
TOP_FIT_MARGIN_RATIO = 0.15
MIN_FIT_ORTHO_SCALE = 2.6
ARROW_CORNER_SCALE_STEP = 1.04
ARROW_CORNER_MAX_STEPS = 40
TOP_NORTH_ARROW_SIZE = 102

COLLISION_PAD = 0.12

# ---- NEW TASK CONFIG ----
MOVE_DIR_POOL = ["E", "W", "N", "S"]
MOVE_DIST_POOL = [2, 3, 4, 5, 6]
MOVE_TASK_TRIES = 220
SWAP_TASK_TRIES = 260

# ---- QA2 collision mode ----
# MUST use AABB precheck + footprint (circle / parts OBB) like rotation-collision script.
QA2_USE_BVH = False

# ---- OVERLAY (SCALE MARKER) ----
SCALE_MARK_LENGTH_M = 1.0
SCALE_BAR_THICKNESS = 0.08   # bar thickness in Y/Z
SCALE_TICK_LEN_Y = 0.45      # tick length along Y (visible in top view)
SCALE_MARK_MARGIN = 1.2
SCALE_MARK_Z_EPS = 0.08
SCALE_LABEL_SIZE = 0.95
SCALE_LABEL_Z_OFF = 0.18
SCALE_PAD_FOR_PLACEMENT = 0.25  # keep scale area free (AABB check)

COLOR_PALETTE = [
    (0.9, 0.3, 0.3),
    (0.3, 0.9, 0.3),
    (0.3, 0.3, 0.9),
    (0.9, 0.9, 0.3),
    (0.9, 0.3, 0.9),
    (0.3, 0.9, 0.9),
    (0.9, 0.5, 0.3),
    (0.5, 0.3, 0.9),
    (0.9, 0.3, 0.5),
    (0.3, 0.5, 0.9),
    (0.5, 0.9, 0.3),
    (0.9, 0.5, 0.5),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Render settings
# ============================================================

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.render.use_compositing = False
scene.render.use_sequencer = False

# ============================================================
# Scene / World / Lights (shadows off)
# ============================================================

def set_black_world() -> None:
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg = nodes.get("Background") or nodes.new(type="ShaderNodeBackground")
    bg.name = "Background"
    out = nodes.get("World Output") or nodes.new(type="ShaderNodeOutputWorld")
    out.name = "World Output"

    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    for link in list(links):
        links.remove(link)
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def remove_object_and_data(obj: bpy.types.Object) -> None:
    if obj is None:
        return

    mesh = obj.data if obj.type == "MESH" else None
    mats = []
    if obj.type == "MESH" and getattr(obj.data, "materials", None):
        mats = [m for m in obj.data.materials if m]

    bpy.data.objects.remove(obj, do_unlink=True)

    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)

    for m in mats:
        if m.users == 0:
            bpy.data.materials.remove(m)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for curve in list(bpy.data.curves):
        bpy.data.curves.remove(curve)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)

    set_black_world()


def _compute_north_screen_vec(cam, north_world_dir: Vector) -> Vector:
    m = cam.matrix_world.to_3x3()
    cam_right = (m @ Vector((1.0, 0.0, 0.0))).normalized()
    cam_up = (m @ Vector((0.0, 1.0, 0.0))).normalized()

    d = north_world_dir.normalized()
    screen_x = d.dot(cam_right)
    screen_y = d.dot(cam_up)

    screen_vec = Vector((screen_x, -screen_y))
    if screen_vec.length < 1e-6:
        return Vector((0.0, -1.0))
    screen_vec.normalize()
    return screen_vec


def _draw_north_arrow_via_system_python(
    image_path: str,
    screen_vec: Vector,
    arrow_color=(255, 0, 0, 255),
    arrow_size=150,
    corner_idx: int | None = None,
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
arrow_size = int(payload.get("arrow_size", 150))
unit_bar_color = tuple(int(v) for v in payload.get("unit_bar_color", [255, 255, 255, 255]))

img = Image.open(image_path).convert("RGBA")
draw = ImageDraw.Draw(img)
width, height = img.size
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

alpha = img.getchannel("A")
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
    best_idx, best_xy, best_ratio = 0, candidates[0], 1e9
    for i, (px, py) in enumerate(candidates):
        r = overlap_ratio(px, py)
        if r < best_ratio - 1e-9:
            best_idx, best_xy, best_ratio = i, (px, py), r

x1 = int(best_xy[0])
y1 = int(best_xy[1])
x2 = x1 + panel_w
y2 = y1 + panel_h

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

x2 = x1 + panel_w
y2 = y1 + panel_h
panel_fill = (0, 0, 0, 220)
draw.rectangle([x1, y1, x2, y2], fill=panel_fill)

layout_shift = -0.10 * panel_h
cx = x1 + panel_w * 0.63
cy = y1 + panel_h * 0.34 + layout_shift
arrow_length = arrow_size * 0.52
arrow_head_size = arrow_size * 0.20
shaft_width = max(3, int(arrow_size * 0.05))

screen_vec = (vx, vy)
perp = (-vy, vx)
tip = add((cx, cy), mul(screen_vec, arrow_length / 2.0))
bottom = sub((cx, cy), mul(screen_vec, arrow_length / 2.0))
wing_base = sub(tip, mul(screen_vec, arrow_head_size))
left_wing = add(wing_base, mul(perp, arrow_head_size * 0.58))
right_wing = sub(wing_base, mul(perp, arrow_head_size * 0.58))

draw.line([bottom, tip], fill=arrow_color, width=shaft_width)
draw.polygon([tip, left_wing, right_wing], fill=arrow_color)

font_size = max(16, int(arrow_size * 0.34))
font = load_font(font_size)
if font is not None:
    tx = x1 + panel_w * 0.20
    ty = y1 + panel_h * 0.50 + layout_shift
    draw.text((tx, ty), "N", fill=arrow_color, font=font)

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

        lbl_size = max(16, int(arrow_size * 0.24))
        lbl_font = load_font(lbl_size)
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
        print(f"  [draw_north_arrow] Warning: external python fallback failed: {e}")
        return False


def _make_black_background_transparent(image_path: str, black_thresh: int = 2) -> bool:
    payload = {
        "image_path": image_path,
        "black_thresh": int(black_thresh),
    }
    helper_script = r"""
import json, os, sys
from PIL import Image

payload = json.loads(sys.argv[1])
image_path = payload["image_path"]
if not os.path.exists(image_path):
    raise SystemExit(0)
th = int(payload.get("black_thresh", 2))

img = Image.open(image_path).convert("RGBA")
px = img.load()
w, h = img.size
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        if r <= th and g <= th and b <= th:
            px[x, y] = (r, g, b, 0)
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
        print(f"  [transparent_bg] Warning: background post-process failed: {e}")
        return False


def draw_north_arrow(
    image_path: str,
    cam,
    north_world_dir=None,
    arrow_color=(255, 0, 0, 255),
    arrow_size=150,
    corner_idx: int | None = None,
):
    if not os.path.exists(image_path):
        print(f"  [draw_north_arrow] Warning: Image not found: {image_path}")
        return image_path

    if north_world_dir is None:
        north_world_dir = Vector((0.0, 1.0, 0.0))

    screen_vec = _compute_north_screen_vec(cam, north_world_dir)
    unit_bar_px = _unit_bar_length_px(cam, unit_world=SCALE_MARK_LENGTH_M)

    try:
        if _draw_north_arrow_via_system_python(
            image_path,
            screen_vec,
            arrow_color=arrow_color,
            arrow_size=arrow_size,
            corner_idx=corner_idx,
            unit_bar_px=unit_bar_px,
            unit_bar_text=f"{SCALE_MARK_LENGTH_M:g} unit",
        ):
            return image_path

        print(f"  [draw_north_arrow] Warning: external python fallback failed for {image_path}")
        return image_path
    except Exception as e:
        print(f"  [draw_north_arrow] Error drawing arrow on {image_path}: {e}")
        return image_path


def setup_lighting() -> None:
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("SunLight", type="SUN"))
    bpy.context.collection.objects.link(sun)
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
    sun.data.use_shadow = False

    fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLightData", type="AREA"))
    bpy.context.collection.objects.link(fill)
    fill.data.energy = 150.0
    fill.location = (0.0, 0.0, 60.0)
    fill.data.size = 80.0
    fill.data.use_shadow = False


def create_dark_ground() -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=250.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "Ground"

    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0

    plane.data.materials.clear()
    plane.data.materials.append(mat)

    # Keep ground invisible in final render to preserve transparent background.
    plane.hide_render = True
    if hasattr(plane, "visible_camera"):
        plane.visible_camera = False

    plane.visible_shadow = False
    if hasattr(plane, "cycles_visibility"):
        plane.cycles_visibility.camera = False
        plane.cycles_visibility.shadow = False

    return plane

# ============================================================
# Bounds / Camera
# ============================================================

def get_object_world_bounds(obj: bpy.types.Object) -> dict:
    world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [v.x for v in world_verts]
    ys = [v.y for v in world_verts]
    zs = [v.z for v in world_verts]
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


def get_scene_bounds(building_objs: list[bpy.types.Object]) -> dict:
    bounds_list = [get_object_world_bounds(o) for o in building_objs if o and o.type == "MESH"]
    all_min_x = min(b["min_x"] for b in bounds_list)
    all_max_x = max(b["max_x"] for b in bounds_list)
    all_min_y = min(b["min_y"] for b in bounds_list)
    all_max_y = max(b["max_y"] for b in bounds_list)
    all_min_z = min(b["min_z"] for b in bounds_list)
    all_max_z = max(b["max_z"] for b in bounds_list)
    return {
        "min_x": all_min_x,
        "max_x": all_max_x,
        "min_y": all_min_y,
        "max_y": all_max_y,
        "min_z": all_min_z,
        "max_z": all_max_z,
        "center_x": (all_min_x + all_max_x) / 2,
        "center_y": (all_min_y + all_max_y) / 2,
        "center_z": (all_min_z + all_max_z) / 2,
        "width": all_max_x - all_min_x,
        "depth": all_max_y - all_min_y,
        "height": all_max_z - all_min_z,
    }


def create_ortho_camera(name: str) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "ORTHO"
    cam_data.lens = 35
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    return cam_obj


def setup_camera_top(bounds: dict) -> bpy.types.Object:
    cx, cy = bounds["center_x"], bounds["center_y"]
    max_dim = max(bounds["width"], bounds["depth"], 2.0)
    cam = create_ortho_camera("Camera_Top")
    cam.location = (cx, cy, bounds["max_z"] + max_dim * 0.65 + 4.5)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.ortho_scale = max_dim * TOP_ORTHO_SCALE_FACTOR + TOP_CAMERA_EDGE_MARGIN
    return cam


def _iter_object_world_vertices(obj: bpy.types.Object):
    if obj is None or obj.type != "MESH":
        return
    for v in obj.data.vertices:
        yield obj.matrix_world @ v.co


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

    cam_inv = cam.matrix_world.inverted()
    xs = []
    ys = []
    for p in world_points:
        q = cam_inv @ p
        xs.append(float(q.x))
        ys.append(float(q.y))

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)

    cam_axes = cam.matrix_world.to_3x3()
    cam_right = (cam_axes @ Vector((1.0, 0.0, 0.0))).normalized()
    cam_up = (cam_axes @ Vector((0.0, 1.0, 0.0))).normalized()
    cam.location = cam.location + cam_right * center_x + cam_up * center_y

    cam_inv = cam.matrix_world.inverted()
    xs = []
    ys = []
    for p in world_points:
        q = cam_inv @ p
        xs.append(float(q.x))
        ys.append(float(q.y))

    fit_half_x = max(abs(min(xs)), abs(max(xs)))
    fit_half_y = max(abs(min(ys)), abs(max(ys)))

    frame = cam.data.view_frame(scene=scene)
    frame_half_x = max(abs(v.x) for v in frame)
    frame_half_y = max(abs(v.y) for v in frame)

    if frame_half_x < 1e-8 or frame_half_y < 1e-8:
        fit_half = max(fit_half_x, fit_half_y)
        needed_scale = 2.0 * fit_half
    else:
        scale_from_x = cam.data.ortho_scale * (fit_half_x / frame_half_x)
        scale_from_y = cam.data.ortho_scale * (fit_half_y / frame_half_y)
        needed_scale = max(scale_from_x, scale_from_y)

    cam.data.ortho_scale = max(min_ortho_scale, needed_scale * (1.0 + margin_ratio))


def _render_resolution_px() -> tuple[int, int]:
    rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)
    return max(1, rx), max(1, ry)


def _arrow_panel_pixel_geometry(cam: bpy.types.Object, arrow_size: int) -> tuple[int, int, int]:
    margin_px = max(16, int(arrow_size * 0.20))
    unit_bar_px = _unit_bar_length_px(cam, unit_world=SCALE_MARK_LENGTH_M)
    bar_need_px = max(0, int(math.ceil(unit_bar_px))) + int(arrow_size * 0.56)
    panel_w_px = max(int(arrow_size * 1.75), bar_need_px)
    panel_h_px = int(arrow_size * 2.25)
    return margin_px, panel_w_px, panel_h_px


def _camera_half_extents(cam: bpy.types.Object) -> tuple[float, float]:
    frame = cam.data.view_frame(scene=scene)
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


def _project_object_boxes_camera(
    cam: bpy.types.Object,
    objects: list[bpy.types.Object],
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
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


def _panel_rect_in_camera(
    cam: bpy.types.Object,
    arrow_size: int,
    corner_idx: int,
) -> tuple[float, float, float, float]:
    hx, hy = _camera_half_extents(cam)
    rx, ry = _render_resolution_px()
    margin_px, panel_w_px, panel_h_px = _arrow_panel_pixel_geometry(cam, arrow_size)

    px_to_wx = (2.0 * hx) / float(rx)
    px_to_wy = (2.0 * hy) / float(ry)
    margin_x = margin_px * px_to_wx
    margin_y = margin_px * px_to_wy
    panel_w = panel_w_px * px_to_wx
    panel_h = panel_h_px * px_to_wy

    if corner_idx == 0:
        x1, x2 = hx - margin_x - panel_w, hx - margin_x
        y1, y2 = hy - margin_y - panel_h, hy - margin_y
    elif corner_idx == 1:
        x1, x2 = -hx + margin_x, -hx + margin_x + panel_w
        y1, y2 = hy - margin_y - panel_h, hy - margin_y
    elif corner_idx == 2:
        x1, x2 = hx - margin_x - panel_w, hx - margin_x
        y1, y2 = -hy + margin_y, -hy + margin_y + panel_h
    else:
        x1, x2 = -hx + margin_x, -hx + margin_x + panel_w
        y1, y2 = -hy + margin_y, -hy + margin_y + panel_h
    return float(x1), float(x2), float(y1), float(y2)


def _rect_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def choose_arrow_corner_and_adapt_view(
    cam: bpy.types.Object,
    objects: list[bpy.types.Object],
    arrow_size: int,
    scale_step: float = ARROW_CORNER_SCALE_STEP,
    max_steps: int = ARROW_CORNER_MAX_STEPS,
) -> int:
    if cam is None or cam.type != "CAMERA" or cam.data.type != "ORTHO":
        return 0

    boxes = _project_object_boxes_camera(cam, objects)
    if not boxes:
        return 0

    def overlap_areas():
        vals = []
        for idx in range(4):
            rect = _panel_rect_in_camera(cam, arrow_size, idx)
            vals.append(sum(_rect_overlap_area(rect, b) for b in boxes))
        return vals

    areas = overlap_areas()
    selected = min(range(4), key=lambda i: areas[i])

    step = 0
    while areas[selected] > 1e-8 and step < max_steps:
        cam.data.ortho_scale *= scale_step
        areas = overlap_areas()
        step += 1

    if areas[selected] > 1e-8:
        return -1
    return int(selected)

# ============================================================
# Labels / Overlay materials
# ============================================================

def create_label_material(mat_name: str) -> bpy.types.Material:
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = LABEL_COLOR
    bsdf.inputs["Specular"].default_value = 0.35
    bsdf.inputs["Roughness"].default_value = 0.28
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def clear_labels_only() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Label_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for curve in list(bpy.data.curves):
        if curve.name.startswith("Label_"):
            bpy.data.curves.remove(curve)


def clear_overlays_only() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Scale_") or obj.name.startswith("Overlay_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for curve in list(bpy.data.curves):
        if curve.name.startswith("Scale_") or curve.name.startswith("Overlay_"):
            bpy.data.curves.remove(curve)


def add_label(building_obj: bpy.types.Object, label_text: str, cam: bpy.types.Object) -> None:
    bnd = get_object_world_bounds(building_obj)

    text_curve = bpy.data.curves.new(f"Label_{label_text}_curve", type="FONT")
    text_curve.body = str(label_text)
    text_curve.size = LABEL_FONT_SIZE
    text_curve.align_x = "CENTER"
    text_curve.align_y = "CENTER"
    text_curve.extrude = 0.02
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new(f"Label_{label_text}", text_curve)
    bpy.context.collection.objects.link(text_obj)
    text_obj.location = (bnd["center_x"], bnd["center_y"], bnd["max_z"] + LABEL_Z_OFFSET)

    billboard = text_obj.constraints.new(type="LOCKED_TRACK")
    billboard.target = cam
    billboard.track_axis = "TRACK_Z"
    billboard.lock_axis = "LOCK_Y"

    lock_roll = text_obj.constraints.new(type="LIMIT_ROTATION")
    lock_roll.owner_space = "LOCAL"
    lock_roll.use_limit_z = True
    lock_roll.min_z = 0.0
    lock_roll.max_z = 0.0

    mat = create_label_material(f"Label_{label_text}_mat")
    text_obj.data.materials.clear()
    text_obj.data.materials.append(mat)

    text_obj.visible_shadow = False
    if hasattr(text_obj, "cycles_visibility"):
        text_obj.cycles_visibility.shadow = False
        text_obj.cycles_visibility.diffuse = False
        text_obj.cycles_visibility.glossy = False
        text_obj.cycles_visibility.ambient_occlusion = False

    bpy.context.view_layer.update()


def _rect_overlaps_any_building(
    rect_min_x: float,
    rect_max_x: float,
    rect_min_y: float,
    rect_max_y: float,
    building_objs: list[bpy.types.Object],
    pad: float,
) -> bool:
    for o in building_objs:
        b = get_object_world_bounds(o)
        if not (
            (rect_max_x + pad) < (b["min_x"] - pad) or
            (rect_min_x - pad) > (b["max_x"] + pad) or
            (rect_max_y + pad) < (b["min_y"] - pad) or
            (rect_min_y - pad) > (b["max_y"] + pad)
        ):
            return True
    return False


def _pick_scale_marker_position(bounds: dict, building_objs: list[bpy.types.Object]) -> tuple[float, float]:
    """
    Prefer bottom-right empty area first, then other corners/edges.
    """
    w = SCALE_MARK_LENGTH_M + 1.2
    h = SCALE_TICK_LEN_Y + 1.0

    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])

    candidates = [
        (max_x - SCALE_MARK_MARGIN - w, min_y + SCALE_MARK_MARGIN),  # bottom-right (preferred)
        (max_x - SCALE_MARK_MARGIN - w, max_y - SCALE_MARK_MARGIN - h),  # top-right
        (min_x + SCALE_MARK_MARGIN, min_y + SCALE_MARK_MARGIN),  # bottom-left
        (min_x + SCALE_MARK_MARGIN, max_y - SCALE_MARK_MARGIN - h),  # top-left
        (max_x - SCALE_MARK_MARGIN - w, (min_y + max_y) * 0.5 - h * 0.5),
        (min_x + SCALE_MARK_MARGIN, (min_y + max_y) * 0.5 - h * 0.5),
    ]

    for (x0, y0) in candidates:
        if not _rect_overlaps_any_building(
            x0, x0 + w, y0, y0 + h, building_objs, pad=SCALE_PAD_FOR_PLACEMENT
        ):
            return float(x0), float(y0)

    return float(max_x - SCALE_MARK_MARGIN - w), float(min_y + SCALE_MARK_MARGIN)


def add_scale_marker(bounds: dict, building_objs: list[bpy.types.Object], cam: bpy.types.Object, length_m: float = 1.0) -> None:
    """
    Scale marker in XY plane:
    - bar along X
    - two ticks extend along Y (so they are visible in top view as vertical ticks)
    """
    x0, y0 = _pick_scale_marker_position(bounds, building_objs)
    z = float(bounds["max_z"]) + SCALE_MARK_Z_EPS

    mat = create_label_material("Scale_Mark_Mat")
    bar_y = y0 + SCALE_TICK_LEN_Y * 0.5

    created = []

    # Horizontal bar
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0 + length_m / 2.0, bar_y, z))
    bar = bpy.context.active_object
    bar.name = "Scale_Bar"
    bar.dimensions = (float(length_m), float(SCALE_BAR_THICKNESS), float(SCALE_BAR_THICKNESS))
    bar.data.materials.clear()
    bar.data.materials.append(mat)
    created.append(bar)

    # Ticks (VISIBLE in top view): extend along Y, thin in X/Z
    for name, x in [("Scale_Tick_L", x0), ("Scale_Tick_R", x0 + length_m)]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, bar_y, z))
        t = bpy.context.active_object
        t.name = name
        t.dimensions = (float(SCALE_BAR_THICKNESS), float(SCALE_TICK_LEN_Y), float(SCALE_BAR_THICKNESS))
        t.data.materials.clear()
        t.data.materials.append(mat)
        created.append(t)

    # Disable shadows/visibility for all marker meshes
    for o in created:
        o.visible_shadow = False
        if hasattr(o, "cycles_visibility"):
            o.cycles_visibility.shadow = False
            o.cycles_visibility.diffuse = False
            o.cycles_visibility.glossy = False
            o.cycles_visibility.ambient_occlusion = False

    # Text label
    text_curve = bpy.data.curves.new("Scale_Label_curve", type="FONT")
    text_curve.body = f"{length_m:g}m"
    text_curve.size = SCALE_LABEL_SIZE
    text_curve.align_x = "LEFT"
    text_curve.align_y = "CENTER"
    text_curve.extrude = 0.02
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new("Scale_Label", text_curve)
    bpy.context.collection.objects.link(text_obj)
    text_obj.location = (x0 + length_m + 0.35, bar_y, z + SCALE_LABEL_Z_OFF)
    text_obj.data.materials.clear()
    text_obj.data.materials.append(create_label_material("Scale_Label_Mat"))

    billboard = text_obj.constraints.new(type="LOCKED_TRACK")
    billboard.target = cam
    billboard.track_axis = "TRACK_Z"
    billboard.lock_axis = "LOCK_Y"

    lock_roll = text_obj.constraints.new(type="LIMIT_ROTATION")
    lock_roll.owner_space = "LOCAL"
    lock_roll.use_limit_z = True
    lock_roll.min_z = 0.0
    lock_roll.max_z = 0.0

    text_obj.visible_shadow = False
    if hasattr(text_obj, "cycles_visibility"):
        text_obj.cycles_visibility.shadow = False
        text_obj.cycles_visibility.diffuse = False
        text_obj.cycles_visibility.glossy = False
        text_obj.cycles_visibility.ambient_occlusion = False

    bpy.context.view_layer.update()

# ============================================================
# Materials / Colors
# ============================================================

def pick_palette_color() -> tuple[float, float, float]:
    return random.choice(COLOR_PALETTE)


def _apply_material(obj: bpy.types.Object, rgb: tuple[float, float, float], name: str) -> None:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# ============================================================
# Origin/footprint fix for complex shapes
# ============================================================

def _set_origin_to_geometry_and_center_xy(obj: bpy.types.Object, x: float, y: float) -> tuple[float, float]:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    old_z = float(obj.location.z)

    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.view_layer.update()

    before = obj.location.copy()
    dx = float(x) - float(before.x)
    dy = float(y) - float(before.y)

    obj.location = (float(x), float(y), old_z)
    bpy.context.view_layer.update()

    return dx, dy


def _world_xy_to_local_xy(obj: bpy.types.Object, wx: float, wy: float) -> tuple[float, float]:
    inv = obj.matrix_world.inverted()
    lp = inv @ Vector((float(wx), float(wy), 0.0))
    return float(lp.x), float(lp.y)

# ============================================================
# Collision geometry (2D OBB parts) for overlap checks
# ============================================================

def _vec2(v: Vector) -> Vector:
    return Vector((float(v.x), float(v.y)))


def _normalize2(v: Vector) -> Vector:
    l = math.hypot(v.x, v.y)
    if l < 1e-9:
        return Vector((1.0, 0.0))
    return Vector((v.x / l, v.y / l))


def _obb_axes_from_obj(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    m = obj.matrix_world.to_3x3()
    ax = m @ Vector((1.0, 0.0, 0.0))
    ay = m @ Vector((0.0, 1.0, 0.0))
    ax2 = _normalize2(_vec2(ax))
    ay2 = _normalize2(_vec2(ay))
    return ax2, ay2


def _project_obb(center: Vector, ax: Vector, ay: Vector, hx: float, hy: float, axis: Vector) -> tuple[float, float]:
    dots = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            p = center + ax * (sx * hx) + ay * (sy * hy)
            dots.append(p.dot(axis))
    return min(dots), max(dots)


def _obb_overlap(
    center_a: Vector, ax_a: Vector, ay_a: Vector, hx_a: float, hy_a: float,
    center_b: Vector, ax_b: Vector, ay_b: Vector, hx_b: float, hy_b: float
) -> bool:
    axes = [ax_a, ay_a, ax_b, ay_b]
    for axis in axes:
        axis = _normalize2(axis)
        a0, a1 = _project_obb(center_a, ax_a, ay_a, hx_a, hy_a, axis)
        b0, b1 = _project_obb(center_b, ax_b, ay_b, hx_b, hy_b, axis)
        if a1 < b0 or b1 < a0:
            return False
    return True


def _circle_rect_overlap(circle_c: Vector, r: float, rect_c: Vector, ax: Vector, ay: Vector, hx: float, hy: float) -> bool:
    d = circle_c - rect_c
    lx = d.dot(ax)
    ly = d.dot(ay)
    cx = max(-hx, min(hx, lx))
    cy = max(-hy, min(hy, ly))
    closest = rect_c + ax * cx + ay * cy
    return (circle_c - closest).length <= r


def _build_obb_part_world(
    obj: bpy.types.Object, local_center: Vector, hx: float, hy: float
) -> tuple[Vector, Vector, Vector, float, float]:
    wc = obj.matrix_world @ Vector((local_center.x, local_center.y, 0.0))
    center2 = _vec2(wc)
    ax, ay = _obb_axes_from_obj(obj)
    return center2, ax, ay, float(hx), float(hy)


def _unique_local_xy_from_mesh(obj: bpy.types.Object, ndigits: int = 8) -> list[Vector]:
    pts: list[Vector] = []
    seen: set[tuple[float, float]] = set()
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "vertices"):
        return pts
    for v in mesh.vertices:
        key = (round(float(v.co.x), ndigits), round(float(v.co.y), ndigits))
        if key in seen:
            continue
        seen.add(key)
        pts.append(Vector((float(v.co.x), float(v.co.y))))
    return pts


def _convex_hull_2d(points: list[Vector]) -> list[Vector]:
    if len(points) <= 1:
        return points[:]

    pts = sorted(((float(p.x), float(p.y)) for p in points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [Vector((x, y)) for x, y in hull]


def _polygon_world_points(obj: bpy.types.Object, local_poly: list[dict | tuple | list]) -> list[Vector]:
    out: list[Vector] = []
    for p in local_poly:
        if isinstance(p, dict):
            lx, ly = float(p["x"]), float(p["y"])
        else:
            lx, ly = float(p[0]), float(p[1])
        wp = obj.matrix_world @ Vector((lx, ly, 0.0))
        out.append(_vec2(wp))
    return out


def _poly_axes(poly: list[Vector]) -> list[Vector]:
    axes: list[Vector] = []
    n = len(poly)
    if n < 2:
        return axes
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        axis = _normalize2(Vector((-e.y, e.x)))
        axes.append(axis)
    return axes


def _project_points_on_axis(points: list[Vector], axis: Vector) -> tuple[float, float]:
    vals = [p.dot(axis) for p in points]
    return min(vals), max(vals)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def _rect_corners(center: Vector, ax: Vector, ay: Vector, hx: float, hy: float) -> list[Vector]:
    return [
        center + ax * (-hx) + ay * (-hy),
        center + ax * (+hx) + ay * (-hy),
        center + ax * (+hx) + ay * (+hy),
        center + ax * (-hx) + ay * (+hy),
    ]


def _poly_poly_overlap(poly_a: list[Vector], poly_b: list[Vector]) -> bool:
    axes = _poly_axes(poly_a) + _poly_axes(poly_b)
    for axis in axes:
        a0, a1 = _project_points_on_axis(poly_a, axis)
        b0, b1 = _project_points_on_axis(poly_b, axis)
        if not _interval_overlap(a0, a1, b0, b1):
            return False
    return True


def _poly_rect_overlap(poly: list[Vector], rect_c: Vector, ax: Vector, ay: Vector, hx: float, hy: float) -> bool:
    rect = _rect_corners(rect_c, ax, ay, hx, hy)
    return _poly_poly_overlap(poly, rect)


def _poly_circle_overlap(poly: list[Vector], circle_c: Vector, r: float) -> bool:
    if not poly:
        return False

    axes = _poly_axes(poly)
    closest = min(poly, key=lambda p: (p - circle_c).length_squared)
    v = circle_c - closest
    if v.length > 1e-9:
        axes.append(_normalize2(v))

    for axis in axes:
        p0, p1 = _project_points_on_axis(poly, axis)
        c = circle_c.dot(axis)
        c0, c1 = c - r, c + r
        if not _interval_overlap(p0, p1, c0, c1):
            return False
    return True


def aabb_overlap_xy(obj_a, obj_b, inflate=0.0) -> bool:
    aa = get_object_world_bounds(obj_a)
    bb = get_object_world_bounds(obj_b)
    inf = float(inflate)
    return not (
        (aa["max_x"] + inf) < (bb["min_x"] - inf) or
        (aa["min_x"] - inf) > (bb["max_x"] + inf) or
        (aa["max_y"] + inf) < (bb["min_y"] - inf) or
        (aa["min_y"] - inf) > (bb["max_y"] + inf)
    )


def buildings_overlap_2d(
    a_obj: bpy.types.Object, a_meta: dict,
    b_obj: bpy.types.Object, b_meta: dict,
    inflate: float = 0.0
) -> bool:
    a_parts = a_meta.get("footprint_parts", [])
    a_circle = a_meta.get("footprint_circle", None)
    a_poly_local = a_meta.get("footprint_polygon", None)

    b_parts = b_meta.get("footprint_parts", [])
    b_circle = b_meta.get("footprint_circle", None)
    b_poly_local = b_meta.get("footprint_polygon", None)

    inflate = float(inflate)
    use_precise_poly = (abs(inflate) < 1e-12)

    a_poly = _polygon_world_points(a_obj, a_poly_local) if (use_precise_poly and a_poly_local) else None
    b_poly = _polygon_world_points(b_obj, b_poly_local) if (use_precise_poly and b_poly_local) else None

    if a_poly and b_poly:
        return _poly_poly_overlap(a_poly, b_poly)

    if a_poly and b_circle:
        bc = _vec2(b_obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        return _poly_circle_overlap(a_poly, bc, float(b_circle))

    if b_poly and a_circle:
        ac = _vec2(a_obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        return _poly_circle_overlap(b_poly, ac, float(a_circle))

    if a_poly and b_parts:
        for p in b_parts:
            c2, ax, ay, hx, hy = _build_obb_part_world(
                b_obj, Vector((p["cx"], p["cy"])), p["hx"], p["hy"]
            )
            if _poly_rect_overlap(a_poly, c2, ax, ay, hx, hy):
                return True
        return False

    if b_poly and a_parts:
        for p in a_parts:
            c2, ax, ay, hx, hy = _build_obb_part_world(
                a_obj, Vector((p["cx"], p["cy"])), p["hx"], p["hy"]
            )
            if _poly_rect_overlap(b_poly, c2, ax, ay, hx, hy):
                return True
        return False

    if a_circle and b_circle:
        ac = _vec2(a_obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        bc = _vec2(b_obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        return (ac - bc).length <= (float(a_circle) + inflate) + (float(b_circle) + inflate)

    if a_circle and b_parts:
        ac = _vec2(a_obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        ar = float(a_circle) + inflate
        for p in b_parts:
            c2, ax, ay, hx, hy = _build_obb_part_world(
                b_obj, Vector((p["cx"], p["cy"])), p["hx"] + inflate, p["hy"] + inflate
            )
            if _circle_rect_overlap(ac, ar, c2, ax, ay, hx, hy):
                return True
        return False

    if b_circle and a_parts:
        bc = _vec2(b_obj.matrix_world @ Vector((0.0, 0.0, 0.0)))
        br = float(b_circle) + inflate
        for p in a_parts:
            c2, ax, ay, hx, hy = _build_obb_part_world(
                a_obj, Vector((p["cx"], p["cy"])), p["hx"] + inflate, p["hy"] + inflate
            )
            if _circle_rect_overlap(bc, br, c2, ax, ay, hx, hy):
                return True
        return False

    if a_parts and b_parts:
        for pa in a_parts:
            ca, axa, aya, hxa, hya = _build_obb_part_world(
                a_obj, Vector((pa["cx"], pa["cy"])), pa["hx"] + inflate, pa["hy"] + inflate
            )
            for pb in b_parts:
                cb, axb, ayb, hxb, hyb = _build_obb_part_world(
                    b_obj, Vector((pb["cx"], pb["cy"])), pb["hx"] + inflate, pb["hy"] + inflate
                )
                if _obb_overlap(ca, axa, aya, hxa, hya, cb, axb, ayb, hxb, hyb):
                    return True
        return False

    # fallback AABB if no footprint info
    aa = get_object_world_bounds(a_obj)
    bb = get_object_world_bounds(b_obj)
    return not (
        (aa["max_x"] + inflate) < (bb["min_x"] - inflate) or
        (aa["min_x"] - inflate) > (bb["max_x"] + inflate) or
        (aa["max_y"] + inflate) < (bb["min_y"] - inflate) or
        (aa["min_y"] - inflate) > (bb["max_y"] + inflate)
    )

# ============================================================
# Scene-wide collision (ALL pairs) using AABB precheck + footprint
# ============================================================

def _any_collision_in_scene(objs: list[bpy.types.Object], metas: list[dict]) -> bool:
    n = len(objs)
    for i in range(n):
        for j in range(i + 1, n):
            if not aabb_overlap_xy(objs[i], objs[j], inflate=0.0):
                continue
            if buildings_overlap_2d(objs[i], metas[i], objs[j], metas[j], inflate=0.0):
                return True
    return False

# ============================================================
# Building generation
# ============================================================

def _in_bounds_center(x: float, y: float, half_w: float, half_d: float) -> bool:
    return (
        (-SCENE_BOUNDS + half_w + IN_BOUNDS_MARGIN <= x <= SCENE_BOUNDS - half_w - IN_BOUNDS_MARGIN) and
        (-SCENE_BOUNDS + half_d + IN_BOUNDS_MARGIN <= y <= SCENE_BOUNDS - half_d - IN_BOUNDS_MARGIN)
    )


def sample_position(existing: list[dict]) -> tuple[float, float]:
    for _ in range(MAX_PLACE_ATTEMPTS):
        x = random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)
        y = random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)
        ok = True
        for b in existing:
            dx = abs(x - b["pos"][0])
            dy = abs(y - b["pos"][1])
            if dx < (b["half_w"] + 3.5 + MIN_GAP) and dy < (b["half_d"] + 3.5 + MIN_GAP):
                ok = False
                break
        if ok:
            return x, y
    return random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS), random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)


def validate_position_with_gap(x: float, y: float, new_meta: dict, existing: list[dict], gap: float) -> bool:
    new_hw = new_meta["half_w"]
    new_hd = new_meta["half_d"]
    for b in existing:
        dx = abs(x - b["pos"][0])
        dy = abs(y - b["pos"][1])
        min_dx = b["half_w"] + new_hw + gap
        min_dy = b["half_d"] + new_hd + gap
        if dx < min_dx and dy < min_dy:
            return False
    return True


def _make_complex_shape_L_or_U(x: float, y: float, h: float) -> tuple[bpy.types.Object, dict]:
    shape = random.choice(["L_SHAPE", "U_SHAPE"])

    if shape == "L_SHAPE":
        wx = random.uniform(3.5, 7.0)
        wy = random.uniform(3.5, 7.0)
        arm = random.uniform(0.28, 0.45)
        world_parts: list[tuple[float, float, float, float]] = []

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj1 = bpy.context.active_object
        obj1.dimensions = (wx, wy * arm, h)
        world_parts.append((x, y, wx / 2.0, (wy * arm) / 2.0))

        offx = wx * random.uniform(0.15, 0.35)
        offy = wy * random.uniform(0.15, 0.35)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + offx, y + offy, h / 2))
        obj2 = bpy.context.active_object
        obj2.dimensions = (wx * arm, wy, h)
        world_parts.append((x + offx, y + offy, (wx * arm) / 2.0, wy / 2.0))

        obj1.select_set(True)
        obj2.select_set(True)
        bpy.context.view_layer.objects.active = obj1
        bpy.ops.object.join()
        obj = bpy.context.active_object

        dx, dy = _set_origin_to_geometry_and_center_xy(obj, x, y)

        parts: list[dict] = []
        for wx_c, wy_c, hx, hy in world_parts:
            cx, cy = _world_xy_to_local_xy(obj, wx_c + dx, wy_c + dy)
            parts.append({"cx": cx, "cy": cy, "hx": float(hx), "hy": float(hy)})

        half_w = max(p["hx"] + abs(p["cx"]) for p in parts)
        half_d = max(p["hy"] + abs(p["cy"]) for p in parts)
    else:
        wx = random.uniform(4.0, 8.0)
        wy = random.uniform(4.0, 8.0)
        wall = random.uniform(0.22, 0.38)
        world_parts: list[tuple[float, float, float, float]] = []

        offx_l = -wx * 0.35
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + offx_l, y, h / 2))
        left = bpy.context.active_object
        left.dimensions = (wx * wall, wy, h)
        world_parts.append((x + offx_l, y, (wx * wall) / 2.0, wy / 2.0))

        offx_r = wx * 0.35
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + offx_r, y, h / 2))
        right = bpy.context.active_object
        right.dimensions = (wx * wall, wy, h)
        world_parts.append((x + offx_r, y, (wx * wall) / 2.0, wy / 2.0))

        offy_b = -wy * 0.35
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y + offy_b, h / 2))
        bottom = bpy.context.active_object
        bottom.dimensions = (wx, wy * wall, h)
        world_parts.append((x, y + offy_b, wx / 2.0, (wy * wall) / 2.0))

        left.select_set(True)
        right.select_set(True)
        bottom.select_set(True)
        bpy.context.view_layer.objects.active = left
        bpy.ops.object.join()
        obj = bpy.context.active_object

        dx, dy = _set_origin_to_geometry_and_center_xy(obj, x, y)

        parts: list[dict] = []
        for wx_c, wy_c, hx, hy in world_parts:
            cx, cy = _world_xy_to_local_xy(obj, wx_c + dx, wy_c + dy)
            parts.append({"cx": cx, "cy": cy, "hx": float(hx), "hy": float(hy)})

        half_w = max(p["hx"] + abs(p["cx"]) for p in parts)
        half_d = max(p["hy"] + abs(p["cy"]) for p in parts)

    rgb = pick_palette_color()
    _apply_material(obj, rgb, "Mat_complex")

    obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False

    meta = {
        "shape": shape,
        "footprint_parts": parts,
        "footprint_circle": None,
        "half_w": float(half_w),
        "half_d": float(half_d),
    }
    return obj, meta


def sample_simple_building_spec(height: float) -> dict:
    shape = random.choice(["CUBE", "CUBOID", "CYLINDER", "PRISM", "SPHERE"])

    if shape == "CUBE":
        w = random.uniform(2.2, 6.0)
        return {"shape": shape, "height": float(height), "w": float(w)}
    if shape == "CUBOID":
        wx = random.uniform(2.0, 7.0)
        wy = random.uniform(2.0, 7.0)
        return {"shape": shape, "height": float(height), "wx": float(wx), "wy": float(wy)}
    if shape in ("CYLINDER", "PRISM"):
        w = random.uniform(2.2, 6.0)
        return {"shape": shape, "height": float(height), "w": float(w)}
    w = random.uniform(2.2, 6.0)
    return {"shape": "SPHERE", "height": float(w), "w": float(w)}


def spec_half_extents(spec: dict) -> tuple[float, float]:
    shape = spec["shape"]
    if shape == "CUBE":
        hw = spec["w"] / 2.0
        return float(hw), float(hw)
    if shape == "CUBOID":
        return float(spec["wx"] / 2.0), float(spec["wy"] / 2.0)
    if shape in ("CYLINDER", "PRISM", "SPHERE"):
        r = spec["w"] / 2.0
        return float(r), float(r)
    raise ValueError(f"Unknown shape in spec: {shape}")


def create_building_from_spec(idx: int, x: float, y: float, spec: dict) -> tuple[bpy.types.Object, dict]:
    shape = spec["shape"]
    h = float(spec["height"])
    poly_local = None

    if shape == "CUBE":
        w = float(spec["w"])
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (w, w, h)
        parts = [{"cx": 0.0, "cy": 0.0, "hx": w / 2.0, "hy": w / 2.0}]
        circle = None
    elif shape == "CUBOID":
        wx = float(spec["wx"])
        wy = float(spec["wy"])
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (wx, wy, h)
        parts = [{"cx": 0.0, "cy": 0.0, "hx": wx / 2.0, "hy": wy / 2.0}]
        circle = None
    elif shape == "CYLINDER":
        w = float(spec["w"])
        bpy.ops.mesh.primitive_cylinder_add(radius=w / 2, depth=h, location=(x, y, h / 2))
        obj = bpy.context.active_object
        parts = []
        circle = w / 2.0
    elif shape == "PRISM":
        w = float(spec["w"])
        bpy.ops.mesh.primitive_cylinder_add(radius=w / 2, depth=h, vertices=3, location=(x, y, h / 2))
        obj = bpy.context.active_object
        parts = []
        circle = w / 2.0
        poly_local = _convex_hull_2d(_unique_local_xy_from_mesh(obj))
    elif shape == "SPHERE":
        w = float(spec["w"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=w / 2, location=(x, y, w / 2))
        obj = bpy.context.active_object
        h = w
        parts = []
        circle = w / 2.0
    else:
        raise ValueError(f"Unknown shape: {shape}")

    obj.name = f"B_{idx}"
    rgb = pick_palette_color()
    _apply_material(obj, rgb, f"Mat_{idx}")

    obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False

    bnd = get_object_world_bounds(obj)
    half_w = 0.5 * float(bnd["width"])
    half_d = 0.5 * float(bnd["depth"])

    meta = {
        "id": idx,
        "pos": (float(x), float(y)),
        "height": float(h),
        "shape": shape,
        "color": rgb,
        "footprint_parts": parts,
        "footprint_circle": circle,
        "footprint_polygon": ([{"x": float(p.x), "y": float(p.y)} for p in poly_local] if poly_local else None),
        "half_w": float(half_w),
        "half_d": float(half_d),
        "spec": spec,
    }
    return obj, meta


def create_building(
    idx: int,
    x: float,
    y: float,
    height: float,
    force_complex: bool = False,
    spec: dict | None = None
) -> tuple[bpy.types.Object, dict]:
    if force_complex:
        obj, cm = _make_complex_shape_L_or_U(x, y, float(height))
        obj.name = f"B_{idx}"
        bnd = get_object_world_bounds(obj)
        half_w = 0.5 * float(bnd["width"])
        half_d = 0.5 * float(bnd["depth"])
        return obj, {
            "id": idx,
            "pos": (float(x), float(y)),
            "height": float(height),
            "shape": cm["shape"],
            "color": None,
            "footprint_parts": cm["footprint_parts"],
            "footprint_circle": cm["footprint_circle"],
            "footprint_polygon": None,
            "half_w": float(half_w),
            "half_d": float(half_d),
        }

    if spec is not None:
        return create_building_from_spec(idx, x, y, spec)

    spec2 = sample_simple_building_spec(height=float(height))
    return create_building_from_spec(idx, x, y, spec2)

# ============================================================
# Rendering helpers
# ============================================================

def render_view(cam: bpy.types.Object, output_path: str) -> None:
    scene.camera = cam
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    _make_black_background_transparent(output_path, black_thresh=2)


def render_view_with_north(
    cam: bpy.types.Object,
    output_path: str,
    north_world_dir: Vector,
    arrow_size: int,
    corner_idx: int,
) -> None:
    render_view(cam, output_path)
    draw_north_arrow(
        output_path,
        cam=cam,
        north_world_dir=north_world_dir,
        arrow_size=arrow_size,
        corner_idx=corner_idx,
    )


def render_ref_top(
    cam: bpy.types.Object,
    output_path: str,
    bounds: dict,
    building_objs: list[bpy.types.Object],
    id_to_obj: dict[int, bpy.types.Object],
    with_scale_marker: bool,
    arrow_corner_idx: int,
) -> None:
    clear_labels_only()
    clear_overlays_only()
    _ = with_scale_marker
    _ = bounds
    _ = building_objs

    for idx in sorted(id_to_obj.keys()):
        add_label(id_to_obj[idx], str(idx), cam)

    bpy.context.view_layer.update()
    render_view_with_north(
        cam,
        output_path,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        arrow_size=TOP_NORTH_ARROW_SIZE,
        corner_idx=arrow_corner_idx,
    )


# ============================================================
# Relations (placement uses AABB + footprint)
# ============================================================

def relation_2d_with_pad(
    a_obj: bpy.types.Object,
    a_meta: dict,
    b_obj: bpy.types.Object,
    b_meta: dict,
    pad: float,
) -> str:
    pad = float(pad)

    if not aabb_overlap_xy(a_obj, b_obj, inflate=pad):
        return "SAFE"
    if buildings_overlap_2d(a_obj, a_meta, b_obj, b_meta, inflate=0.0):
        return "COLLIDE"
    if buildings_overlap_2d(a_obj, a_meta, b_obj, b_meta, inflate=pad):
        return "NEAR"
    return "SAFE"


def scene_relation_with_pad(
    new_obj: bpy.types.Object,
    new_meta: dict,
    existing_objs: list[bpy.types.Object],
    existing_meta: list[dict],
    pad: float,
) -> str:
    pad = float(pad)
    for o, m in zip(existing_objs, existing_meta):
        if not aabb_overlap_xy(new_obj, o, inflate=pad):
            continue
        if buildings_overlap_2d(new_obj, new_meta, o, m, inflate=0.0):
            return "COLLIDE"
        if buildings_overlap_2d(new_obj, new_meta, o, m, inflate=pad):
            return "NEAR"
    return "SAFE"

# ============================================================
# Near-target placement (support SAFE or forced COLLIDE)
# ============================================================

def _sample_near_target_position_push_out(
    tx: float,
    ty: float,
    target_obj: bpy.types.Object,
    target_meta: dict,
    other_id: int,
    other_spec: dict,
    want_collision_with_target: bool,
    trials: int = 320,
    deadline_ts: float | None = None,
    progress_hook=None,
) -> tuple[float, float] | None:
    other_hw, other_hd = spec_half_extents(other_spec)
    base_r = max(target_meta["half_w"], target_meta["half_d"]) + max(other_hw, other_hd) + NEAR_PAIR_GAP
    inset = PUSH_OUT_START_INSET if not want_collision_with_target else (PUSH_OUT_START_INSET + 0.25)

    for t_idx in range(trials):
        if deadline_ts is not None and time.monotonic() > deadline_ts:
            return None
        if progress_hook is not None and (t_idx % 40 == 0):
            progress_hook("near_target_search", t_idx, trials)

        ang = random.uniform(0.0, 2.0 * math.pi)
        r = max(0.0, base_r - inset)

        for s_idx in range(PUSH_OUT_MAX_STEPS):
            if deadline_ts is not None and (s_idx % 16 == 0) and time.monotonic() > deadline_ts:
                return None
            nx = float(tx + r * math.cos(ang))
            ny = float(ty + r * math.sin(ang))

            if not _in_bounds_center(nx, ny, other_hw, other_hd):
                r += PUSH_OUT_STEP
                continue

            tmp_obj, tmp_meta = create_building(other_id, nx, ny, other_spec["height"], force_complex=False, spec=other_spec)
            try:
                rel = relation_2d_with_pad(target_obj, target_meta, tmp_obj, tmp_meta, pad=COLLISION_PAD)
                if want_collision_with_target:
                    if rel == "COLLIDE":
                        return nx, ny
                else:
                    if rel == "SAFE":
                        return nx, ny
            finally:
                remove_object_and_data(tmp_obj)

            r += PUSH_OUT_STEP

    return None

# ============================================================
# Tasks
# ============================================================

def _move_dir_to_vec(dir_token: str) -> tuple[float, float]:
    return {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}[dir_token]


def _move_dir_to_words_en(dir_token: str) -> str:
    return {"E": "East", "W": "West", "N": "North", "S": "South"}[dir_token]


def qa_task_move_collision_yesno(
    sample_dir: str,
    move_id: int,
    other_id: int,
    dir_token: str,
    dist_m: int,
    will_collide: bool,
) -> dict:
    dir_en = _move_dir_to_words_en(dir_token)
    q = (
        "There is one image: a top-view image.\n"
        "Directions in the image: North=up, South=down, East=right, West=left.\n"
        f"If object {move_id} is moved {dir_en} by {dist_m} units, "
        f"will it overlap (collide) with object {other_id}? Answer with Yes or No.\n\n"
        "Top-view image: <image>\n"
        "Note: the image includes a 1 unit scale bar for reference.\n"
    )
    return {
        "question": q,
        "answer": "Yes" if will_collide else "No",
        "images": [os.path.abspath(os.path.join(sample_dir, "top_move.png"))],
        "meta": {
            "move_id": move_id,
            "other_id": other_id,
            "direction": dir_en,
            "dist_m": int(dist_m),
            "mapping": {"North": "up", "South": "down", "East": "right", "West": "left"},
            "adjacent": "nearest_neighbor",
        },
    }


def qa_task_swap_collision_yesno(
    sample_dir: str,
    a_id: int,
    b_id: int,
    before_collide: bool,
    after_collide: bool,
) -> dict:
    q = (
        "There is one image: a top-view image.\n"
        f"If we swap the positions of object {a_id} and object {b_id}, "
        "will there be any overlap (collision) in the scene AFTER the swap? Answer with Yes or No.\n\n"
        "Top-view image: <image>"
    )
    return {
        "question": q,
        "answer": "Yes" if after_collide else "No",
        "images": [os.path.abspath(os.path.join(sample_dir, "top_swap.png"))],
        "meta": {
            "swap_a": a_id,
            "swap_b": b_id,
            "before_collide": before_collide,
            "after_collide": after_collide,
            "case": "collide_to_safe" if (before_collide and not after_collide) else "safe_to_collide",
            "collision_mode": "2D_footprint_AABB_precheck",
        },
    }

# ============================================================
# Scene utilities
# ============================================================

def _pick_heights(existing: list[dict]) -> float:
    used = [b["height"] for b in existing]
    h = random.uniform(2.0, 10.0)
    for _ in range(80):
        if all(abs(h - uh) >= MIN_HEIGHT_DIFF for uh in used):
            return h
        h = random.uniform(2.0, 10.0)
    return h


def _nearest_neighbor_id(ids: list[int], id_to_obj: dict[int, bpy.types.Object], pivot_id: int) -> int:
    p = id_to_obj[pivot_id].location
    best = None
    best_d = 1e18
    for i in ids:
        if i == pivot_id:
            continue
        q = id_to_obj[i].location
        d = (p.x - q.x) ** 2 + (p.y - q.y) ** 2
        if d < best_d:
            best_d = d
            best = i
    return int(best) if best is not None else int(random.choice([i for i in ids if i != pivot_id]))


def _collision_pairs_in_scene(objs: list[bpy.types.Object], metas: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    n = len(objs)
    for i in range(n):
        for j in range(i + 1, n):
            if not aabb_overlap_xy(objs[i], objs[j], inflate=0.0):
                continue
            if buildings_overlap_2d(objs[i], metas[i], objs[j], metas[j], inflate=0.0):
                ida = int(metas[i]["id"])
                idb = int(metas[j]["id"])
                pairs.append((min(ida, idb), max(ida, idb)))
    return pairs


# ============================================================
# Sample generation
# ============================================================

def generate_sample(sample_idx: int) -> dict:
    """
    Alternation:
    - QA1 desired answer: sample_idx even -> collide, odd -> not collide
    - QA2 desired transition:
        even  -> before False, after True   (safe_to_collide)
        odd   -> before True,  after False  (collide_to_safe)
    """
    desired_move_collide = (sample_idx % 2 == 0)
    desired_before_collide = (sample_idx % 2 == 1)
    desired_after_collide = not desired_before_collide

    def _mk_attempt_logger(sid: int, att: int):
        start_ts = time.monotonic()
        last_log_ts = start_ts
        deadline_ts = (start_ts + MAX_ATTEMPT_SECONDS) if MAX_ATTEMPT_SECONDS > 0 else None

        def log(stage: str, force: bool = False) -> None:
            nonlocal last_log_ts
            now = time.monotonic()
            if force or (now - last_log_ts) >= PROGRESS_LOG_INTERVAL_SECONDS:
                print(
                    f"[sample {sid}] attempt {att} {stage} elapsed={now - start_ts:.1f}s",
                    flush=True,
                )
                last_log_ts = now

        def timed_out(stage: str) -> bool:
            if deadline_ts is None:
                return False
            now = time.monotonic()
            if now > deadline_ts:
                print(
                    f"[sample {sid}] attempt {att} timeout at {stage} "
                    f"elapsed={now - start_ts:.1f}s (limit={MAX_ATTEMPT_SECONDS:.1f}s)",
                    flush=True,
                )
                return True
            return False

        return log, timed_out, deadline_ts

    for attempt in range(MAX_SCENE_RETRY):
        print(f"[sample {sample_idx}] attempt {attempt} start", flush=True)
        log_attempt, is_timeout, deadline_ts = _mk_attempt_logger(sample_idx, attempt)

        clear_scene()
        setup_lighting()

        building_objs: list[bpy.types.Object] = []
        building_meta: list[dict] = []
        id_to_obj: dict[int, bpy.types.Object] = {}

        ids = list(range(1, BUILDINGS_PER_SCENE + 1))
        target_id = random.choice(ids)
        other_id = random.choice([i for i in ids if i != target_id])

        # (1) Place TARGET first (complex)
        placed_target = False
        meta_t = None
        for place_try in range(MAX_PLACE_ATTEMPTS):
            if place_try % 80 == 0:
                log_attempt(f"placing_target try={place_try}")
            if is_timeout("placing_target"):
                break
            h_t = _pick_heights(building_meta)
            x, y = random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS), random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)

            obj_t, meta_t_candidate = create_building(target_id, x, y, h_t, force_complex=True)

            if not _in_bounds_center(x, y, meta_t_candidate["half_w"], meta_t_candidate["half_d"]):
                remove_object_and_data(obj_t)
                continue

            if validate_position_with_gap(x, y, meta_t_candidate, building_meta, MIN_GAP):
                rel0 = scene_relation_with_pad(obj_t, meta_t_candidate, building_objs, building_meta, pad=0)
                relp = scene_relation_with_pad(obj_t, meta_t_candidate, building_objs, building_meta, pad=COLLISION_PAD)
                if rel0 == "SAFE" and relp == "SAFE":
                    building_objs.append(obj_t)
                    building_meta.append(meta_t_candidate)
                    id_to_obj[target_id] = obj_t
                    placed_target = True
                    meta_t = meta_t_candidate
                    break

            remove_object_and_data(obj_t)

        if not placed_target or meta_t is None:
            log_attempt("target_not_placed -> retry", force=True)
            continue

        tx, ty = meta_t["pos"]
        target_obj = id_to_obj[target_id]

        # (2) Place OTHER near target:
        want_initial_collision = desired_before_collide
        placed_other = False

        def _near_progress(stage: str, t_idx: int, t_all: int) -> None:
            if t_idx % 80 == 0:
                log_attempt(f"{stage} trial={t_idx}/{t_all}")

        for place_try in range(MAX_PLACE_ATTEMPTS):
            if place_try % 60 == 0:
                log_attempt(f"placing_other try={place_try}")
            if is_timeout("placing_other"):
                break
            h_o = _pick_heights(building_meta)
            other_spec = sample_simple_building_spec(height=float(h_o))

            pos = _sample_near_target_position_push_out(
                tx=tx,
                ty=ty,
                target_obj=target_obj,
                target_meta=meta_t,
                other_id=other_id,
                other_spec=other_spec,
                want_collision_with_target=want_initial_collision,
                trials=360,
                deadline_ts=deadline_ts,
                progress_hook=_near_progress,
            )
            if pos is None:
                continue
            ox, oy = pos

            obj_o, meta_o = create_building(other_id, ox, oy, other_spec["height"], force_complex=False, spec=other_spec)

            try:
                if not _in_bounds_center(ox, oy, meta_o["half_w"], meta_o["half_d"]):
                    continue

                rel_to_target = relation_2d_with_pad(target_obj, meta_t, obj_o, meta_o, pad=COLLISION_PAD)
                if want_initial_collision:
                    if rel_to_target != "COLLIDE":
                        continue
                else:
                    if rel_to_target != "SAFE":
                        continue

                building_objs.append(obj_o)
                building_meta.append(meta_o)
                id_to_obj[other_id] = obj_o
                placed_other = True
                break
            finally:
                if not placed_other:
                    remove_object_and_data(obj_o)

        if not placed_other:
            log_attempt("other_not_placed -> retry", force=True)
            continue

        # (3) Place remaining buildings normally; must be SAFE vs scene under pad-aware.
        ok_scene = True
        for i in ids:
            if i in (target_id, other_id):
                continue

            placed = False
            for place_try in range(MAX_PLACE_ATTEMPTS):
                if place_try % 80 == 0:
                    log_attempt(f"placing_remaining obj={i} try={place_try}")
                if is_timeout(f"placing_remaining_obj_{i}"):
                    break
                h = _pick_heights(building_meta)
                x, y = sample_position(building_meta)
                obj, meta = create_building(i, x, y, h, force_complex=False)

                if not _in_bounds_center(x, y, meta["half_w"], meta["half_d"]):
                    remove_object_and_data(obj)
                    continue

                if validate_position_with_gap(x, y, meta, building_meta, MIN_GAP):
                    rel0 = scene_relation_with_pad(obj, meta, building_objs, building_meta, pad=0.0)
                    relp = scene_relation_with_pad(obj, meta, building_objs, building_meta, pad=COLLISION_PAD)
                    if rel0 == "SAFE" and relp == "SAFE":

                        building_objs.append(obj)
                        building_meta.append(meta)
                        id_to_obj[i] = obj
                        placed = True
                        break

                remove_object_and_data(obj)

            if not placed:
                ok_scene = False
                break

        if not ok_scene:
            log_attempt("remaining_not_placed -> retry", force=True)
            continue

        bpy.context.view_layer.update()

        # Ensure initial collision state matches desired_before_collide (ALL pairs).
        before_pairs = sorted(_collision_pairs_in_scene(building_objs, building_meta))
        before_collide = (len(before_pairs) > 0)

        if before_collide != desired_before_collide:
            log_attempt("before_collide_mismatch -> retry", force=True)
            continue

        # If before=True is expected (collide_to_safe case), enforce collision on A-B pair only
        if desired_before_collide:
            ab = (min(target_id, other_id), max(target_id, other_id))
            if sorted(before_pairs) != [ab]:
                log_attempt("before_pairs_not_clean_flip_case -> retry", force=True)
                continue

        # (4) Camera
        bounds = get_scene_bounds(building_objs)
        cam_top = setup_camera_top(bounds)
        bpy.context.view_layer.update()
        fit_ortho_camera_to_objects(
            cam_top,
            building_objs,
            margin_ratio=TOP_FIT_MARGIN_RATIO,
            min_ortho_scale=MIN_FIT_ORTHO_SCALE,
        )
        top_arrow_corner = choose_arrow_corner_and_adapt_view(
            cam_top,
            building_objs,
            arrow_size=TOP_NORTH_ARROW_SIZE,
        )
        if top_arrow_corner < 0:
            log_attempt("arrow_corner_failed -> retry", force=True)
            continue

        # (5) Output dirs
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        ref_dir = os.path.join(sample_dir, "ref_images")
        os.makedirs(ref_dir, exist_ok=True)


        # (6a) Render top_swap.png (labels only; north panel includes 1-unit bar)
        clear_labels_only()
        clear_overlays_only()
        for idx in range(1, BUILDINGS_PER_SCENE + 1):
            add_label(id_to_obj[idx], str(idx), cam_top)
        bpy.context.view_layer.update()
        render_view_with_north(
            cam_top,
            os.path.join(sample_dir, "top_swap.png"),
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=top_arrow_corner,
        )

        # (6b) Render top_move.png (labels only; 1-unit bar is inside north panel)
        clear_labels_only()
        clear_overlays_only()
        for idx in range(1, BUILDINGS_PER_SCENE + 1):
            add_label(id_to_obj[idx], str(idx), cam_top)
        bpy.context.view_layer.update()
        render_view_with_north(
            cam_top,
            os.path.join(sample_dir, "top_move.png"),
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=top_arrow_corner,
        )

        id_to_meta = {m["id"]: m for m in building_meta}

        # ========================================================
        # QA1: Move, nearest neighbor, enforce collide/non-collide split
        # ========================================================
        move_task = None
        for move_try in range(MOVE_TASK_TRIES):
            if move_try % 40 == 0:
                log_attempt(f"qa1_search try={move_try}")
            if is_timeout("qa1_search"):
                break
            move_id = random.choice(ids)
            other_adj = _nearest_neighbor_id(ids, id_to_obj, move_id)

            obj_m = id_to_obj[move_id]
            meta_m = id_to_meta[move_id]
            obj_o = id_to_obj[other_adj]
            meta_o = id_to_meta[other_adj]

            dirs = MOVE_DIR_POOL[:]
            dists = MOVE_DIST_POOL[:]
            random.shuffle(dirs)
            random.shuffle(dists)

            old_loc = obj_m.location.copy()
            found = False
            try:
                for dir_token in dirs:
                    vx, vy = _move_dir_to_vec(dir_token)
                    for dist_m in dists:
                        nx = float(old_loc.x + vx * dist_m)
                        ny = float(old_loc.y + vy * dist_m)
                        if not _in_bounds_center(nx, ny, meta_m["half_w"], meta_m["half_d"]):
                            continue

                        obj_m.location = (nx, ny, old_loc.z)
                        bpy.context.view_layer.update()

                        rel = relation_2d_with_pad(obj_m, meta_m, obj_o, meta_o, pad=COLLISION_PAD)
                        if rel == "NEAR":
                            continue
                        will_collide = (rel == "COLLIDE")
                        if will_collide != desired_move_collide:
                            continue

                        move_task = qa_task_move_collision_yesno(
                            sample_dir=sample_dir,
                            move_id=move_id,
                            other_id=other_adj,
                            dir_token=dir_token,
                            dist_m=int(dist_m),
                            will_collide=will_collide,
                        )
                        move_ref = (move_id, nx, ny)

                        found = True
                        break
                    if found:
                        break
            finally:
                obj_m.location = old_loc
                bpy.context.view_layer.update()

            if found:
                break

        if move_task is None:
            print(f"[sample {sample_idx}] attempt {attempt} FAIL move_task", flush=True)
            continue

        # --- Render QA1 moved scene (after move) to ref_images ---
        if move_ref is not None:
            m_id, mx, my = move_ref
            obj_m = id_to_obj[m_id]
            old_loc = obj_m.location.copy()
            try:
                obj_m.location = (float(mx), float(my), old_loc.z)
                bpy.context.view_layer.update()
                render_ref_top(
                    cam=cam_top,
                    output_path=os.path.join(ref_dir, "qa1_move_after.png"),
                    bounds=bounds,
                    building_objs=building_objs,
                    id_to_obj=id_to_obj,
                    with_scale_marker=True,   # legacy arg; scale bar now in north panel
                    arrow_corner_idx=top_arrow_corner,
                )
            finally:
                obj_m.location = old_loc
                bpy.context.view_layer.update()

        # QA2: Deterministic swap on (target_id, other_id)
        # Constructed to guarantee state flip
        # ========================================================
        swap_ref = None  # (a_id, b_id)
        swap_task = None
        before_pairs = _collision_pairs_in_scene(building_objs, building_meta)
        before_collide = (len(before_pairs) > 0)

        for swap_try in range(SWAP_TASK_TRIES):
            if swap_try % 50 == 0:
                log_attempt(f"qa2_search try={swap_try}")
            if is_timeout("qa2_search"):
                break
            a_id, b_id = random.sample(ids, 2)
            obj_a, obj_b = id_to_obj[a_id], id_to_obj[b_id]
            loc_a, loc_b = obj_a.location.copy(), obj_b.location.copy()

            try:
                obj_a.location = (loc_b.x, loc_b.y, loc_a.z)
                obj_b.location = (loc_a.x, loc_a.y, loc_b.z)
                bpy.context.view_layer.update()

                after_pairs = _collision_pairs_in_scene(building_objs, building_meta)
                after_collide = (len(after_pairs) > 0)

            finally:
                obj_a.location = loc_a
                obj_b.location = loc_b
                bpy.context.view_layer.update()

            if after_collide != desired_after_collide:
                continue
            if after_collide == before_collide:
                continue

            # Optional: if you want cleaner state flips, add extra constraints:
            # - collide_to_safe: after_pairs must be empty
            # - safe_to_collide: ideally only one pair in after_pairs (closer to desired data)
            if desired_before_collide and desired_after_collide is False:
                if len(after_pairs) != 0:
                    continue
            if (not desired_before_collide) and desired_after_collide:
                if len(after_pairs) < 1:
                    continue

            swap_task = qa_task_swap_collision_yesno(sample_dir, a_id, b_id, before_collide, after_collide)
            swap_ref = (a_id, b_id)

            break

        if swap_task is None:
            log_attempt("swap_task_not_found -> retry", force=True)
            continue
        
        # --- Render QA2 swapped scene (after swap) to ref_images ---
        if swap_ref is not None:
            a_id, b_id = swap_ref
            obj_a = id_to_obj[a_id]
            obj_b = id_to_obj[b_id]
            loc_a = obj_a.location.copy()
            loc_b = obj_b.location.copy()
            try:
                obj_a.location = (loc_b.x, loc_b.y, loc_a.z)
                obj_b.location = (loc_a.x, loc_a.y, loc_b.z)
                bpy.context.view_layer.update()
                render_ref_top(
                    cam=cam_top,
                    output_path=os.path.join(ref_dir, "qa2_swap_after.png"),
                    bounds=bounds,
                    building_objs=building_objs,
                    id_to_obj=id_to_obj,
                    with_scale_marker=False,  # legacy arg; scale bar now in north panel
                    arrow_corner_idx=top_arrow_corner,
                )
            finally:
                obj_a.location = loc_a
                obj_b.location = loc_b
                bpy.context.view_layer.update()

        log_attempt("attempt_success", force=True)
        return {
            "sample_id": sample_idx,
            "attempt": attempt,
            "bounds": bounds,
            "buildings": building_meta,
            "qa": {"top": [move_task, swap_task]},
            "special_refs": {
                "scale_marker_m": SCALE_MARK_LENGTH_M,
                "scale_marker_unit": SCALE_MARK_LENGTH_M,
                "scale_marker_in_north_panel": True,
                "qa1_move_adjacent_pair": True,
                "qa1_balanced_collide": desired_move_collide,
                "qa2_state_flip": True,
                "qa2_desired_before": before_collide,
                "qa2_desired_after": desired_after_collide,
                "qa2_collision_mode": "2D_footprint_AABB_precheck",
            },
            "camera_params": {
                "top": {"ortho_scale": cam_top.data.ortho_scale, "location": tuple(cam_top.location)},
            },
        }

    raise RuntimeError(
        f"Failed to generate sample {sample_idx} after {MAX_SCENE_RETRY} retries. "
        f"Try widening SCENE_BOUNDS or reducing IN_BOUNDS_MARGIN / MIN_GAP."
    )

# ============================================================
# Main
# ============================================================

def main() -> None:
    all_metadata = []
    end_idx = START_SAMPLE_IDX + NUM_SAMPLES
    print(
        f"Generating samples in range [{START_SAMPLE_IDX}, {end_idx})",
        flush=True,
    )
    for s in range(START_SAMPLE_IDX, end_idx):
        all_metadata.append(generate_sample(s))

    meta_path = os.path.join(OUTPUT_DIR, "metadata_top_move_swap_collision.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
