#!/usr/bin/env python3


import bpy
import random
import math
import json
import os
import re
import subprocess
from mathutils import Vector
from shapely.geometry import Polygon

# ============================================================
# Config
# ============================================================
NUM_SAMPLES = 120
BUILDINGS_PER_SCENE = 5

SCENE_BOUNDS = 14.0
MIN_GAP = 1
MAX_PLACE_ATTEMPTS = 600
MAX_SCENE_RETRY = 800

RESOLUTION = 1080
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.expanduser("~/SpatialAct/benchmark/data/error_mode_clean/v1.2"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "top_isometric")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_PALETTE = [
    (0.9, 0.3, 0.3), (0.3, 0.9, 0.3), (0.3, 0.3, 0.9),
    (0.9, 0.9, 0.3), (0.9, 0.3, 0.9), (0.3, 0.9, 0.9),
    (0.9, 0.5, 0.3), (0.5, 0.3, 0.9), (0.9, 0.3, 0.5),
    (0.3, 0.5, 0.9), (0.5, 0.9, 0.3), (0.9, 0.5, 0.5),
]

LABEL_FONT_SIZE = 1.75
LABEL_Z_OFFSET = 0.35
LABEL_COLOR = (0.02, 0.02, 0.02, 1.0)

TOP_ORTHO_SCALE_FACTOR = 1.08
ISO_ORTHO_SCALE_FACTOR = 1.22
TOP_CAMERA_EDGE_MARGIN = 1.2
ISO_CAMERA_EDGE_MARGIN = 2.1
ISO_HEIGHT_COMPENSATION = 0.45
TOP_FIT_MARGIN_RATIO = 0.08
ISO_FIT_MARGIN_RATIO = 0.10
MIN_FIT_ORTHO_SCALE = 2.2
ARROW_CORNER_SCALE_STEP = 1.04
ARROW_CORNER_MAX_STEPS = 40
TOP_NORTH_ARROW_SIZE = 108
ISO_NORTH_ARROW_SIZE = 118
ARROW_PANEL_SAFE_GAP_PX = 8
ARROW_PANEL_OVERLAP_EPS = 1e-10

# ============================================================
# Road config (3x3 grid NO holes)
# ============================================================
ROAD_WIDTH = 0.3
ROAD_ELEVATION = 0.02
ROAD_THICKNESS = 0.06  # slab thickness (box)


BUILDING_BASE_Z = ROAD_ELEVATION + ROAD_THICKNESS + 0.001

ROAD_COLOR = (1.0, 0.95, 0.05, 1.0)     # bright yellow
ROAD_EMISSION_STRENGTH = 6.0
ROAD_ROUGHNESS = 1.0

# placement: don't place buildings on roads (2D precheck margin)
ROAD_AVOID_MARGIN = 0.4

# ============================================================
# 2D overlap thresholds (IMPORTANT)
# ============================================================

OVERLAP_AREA_THRESHOLD = 0.05  # m^2- increased to make strong overlap easier
OVERLAP_NUM_EPS = 1e-9

# ============================================================
# Angle anomaly settings (near an intersection)
# ============================================================
ANGLE_ANOMALY_DEG = 12.0
ANGLE_ROTATE_CHOICES_DEG = [30.0, 45.0]
ANGLE_INTERSECTION_RADIUS = 2.6
ANGLE_SQUARE_ASPECT_TOL = 1.5

ANGLE_CLEAR_FROM_ROAD = 0.20
ANGLE_JITTER = 0.20

ANGLE_ELIGIBLE_SHAPES = {"CUBE", "CUBOID", "L_SHAPE", "U_SHAPE"}

# ============================================================
# Move choices for QA3 (move issues)
# ============================================================
MOVE_DIRS = ["North", "South", "East", "West"]
MOVE_DISTS_M = [2.0, 3.0, 4.0, 5.0]

DIR_TO_VEC = {
    "East":  (1.0, 0.0),
    "West":  (-1.0, 0.0),
    "North": (0.0, 1.0),
    "South": (0.0, -1.0),
}

# Issues
ISSUE_OVERLAP = "overlap"
ISSUE_ANGLE = "angle_anomaly"
ISSUE_ROAD = "road_overlap"

# ============================================================
# Render settings
# ============================================================
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = True
scene.render.use_compositing = False
scene.render.use_sequencer = False

# ============================================================
# Scale marker config (top view only)
# ============================================================
SCALE_MARK_LENGTH_M = 1.0
SCALE_BAR_THICKNESS = 0.08
SCALE_TICK_LEN_Y = 0.45
SCALE_MARK_MARGIN = 1.2
SCALE_MARK_Z_EPS = 0.08
SCALE_LABEL_SIZE = 0.95
SCALE_LABEL_Z_OFF = 0.18
SCALE_PAD_FOR_PLACEMENT = 0.25

# ============================================================
# Scene / materials / lights
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

    curve = obj.data if obj.type in {"FONT", "CURVE"} else None

    bpy.data.objects.remove(obj, do_unlink=True)

    if mesh and getattr(mesh, "users", 0) == 0:
        bpy.data.meshes.remove(mesh)
    if curve and getattr(curve, "users", 0) == 0 and curve.name in bpy.data.curves:
        bpy.data.curves.remove(curve)

    for m in mats:
        if m and getattr(m, "users", 0) == 0 and m.name in bpy.data.materials:
            bpy.data.materials.remove(m)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        if light.users == 0:
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

margin = max(16, int(arrow_size * 0.20))
unit_bar_px = payload.get("unit_bar_px", None)
unit_bar_text = str(payload.get("unit_bar_text", "1 unit"))
bar_need_px = 0
if unit_bar_px is not None:
    try:
        u = float(unit_bar_px)
        if u < 0.0:
            u = 0.0
        bar_need_px = int(u + 0.999) + int(arrow_size * 0.32)
    except Exception:
        bar_need_px = 0
panel_w = max(int(arrow_size * 1.55), bar_need_px)
panel_h = int(arrow_size * 1.90)

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

panel_fill = (0, 0, 0, 220)
draw.rectangle([x1, y1, x2, y2], fill=panel_fill)

cx = x1 + panel_w * 0.63
cy = y1 + panel_h * 0.34
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
    ty = y1 + panel_h * 0.50
    draw.text((tx, ty), "N", fill=arrow_color, font=font)

if unit_bar_px is not None:
    try:
        bar_len = max(1.0, float(unit_bar_px))
        bar_y = y1 + panel_h * 0.84
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
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, 0.0))
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
    plane.hide_render = True
    if hasattr(plane, "visible_camera"):
        plane.visible_camera = False
    plane.visible_shadow = False
    if hasattr(plane, "cycles_visibility"):
        plane.cycles_visibility.camera = False
        plane.cycles_visibility.shadow = False
    return plane

# ============================================================
# Bounds / cameras
# ============================================================
def get_object_world_bounds(obj: bpy.types.Object) -> dict:
    if not obj or obj.type != "MESH" or not obj.data:
        return {
            "min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0,
            "center_x": 0, "center_y": 0, "center_z": 0, "width": 0, "depth": 0, "height": 0,
        }
    world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [v.x for v in world_verts]
    ys = [v.y for v in world_verts]
    zs = [v.z for v in world_verts]
    return {
        "min_x": min(xs), "max_x": max(xs),
        "min_y": min(ys), "max_y": max(ys),
        "min_z": min(zs), "max_z": max(zs),
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "center_z": (min(zs) + max(zs)) / 2,
        "width": max(xs) - min(xs),
        "depth": max(ys) - min(ys),
        "height": max(zs) - min(zs),
    }


def get_scene_bounds(mesh_objs: list[bpy.types.Object]) -> dict:
    bounds_list = [get_object_world_bounds(o) for o in mesh_objs if o and o.type == "MESH"]
    all_min_x = min(b["min_x"] for b in bounds_list)
    all_max_x = max(b["max_x"] for b in bounds_list)
    all_min_y = min(b["min_y"] for b in bounds_list)
    all_max_y = max(b["max_y"] for b in bounds_list)
    all_min_z = min(b["min_z"] for b in bounds_list)
    all_max_z = max(b["max_z"] for b in bounds_list)
    return {
        "min_x": all_min_x, "max_x": all_max_x,
        "min_y": all_min_y, "max_y": all_max_y,
        "min_z": all_min_z, "max_z": all_max_z,
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


def setup_camera_top(bounds: dict, ortho_scale_factor: float = TOP_ORTHO_SCALE_FACTOR) -> bpy.types.Object:
    cx, cy = bounds["center_x"], bounds["center_y"]
    max_dim = max(bounds["width"], bounds["depth"], 2.0)
    cam = create_ortho_camera("Camera_Top")
    cam.location = (cx, cy, bounds["max_z"] + max_dim * 0.65 + 4.5)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.ortho_scale = max_dim * ortho_scale_factor + TOP_CAMERA_EDGE_MARGIN
    return cam


def setup_camera_isometric(bounds: dict, ortho_scale_factor: float = ISO_ORTHO_SCALE_FACTOR) -> bpy.types.Object:
    cx, cy, cz = bounds["center_x"], bounds["center_y"], bounds["center_z"]
    base_dim = max(bounds["width"], bounds["depth"], 2.0)
    frame_dim = base_dim + bounds["height"] * ISO_HEIGHT_COMPENSATION
    cam = create_ortho_camera("Camera_Iso")
    cam.location = (cx + frame_dim * 0.42, cy - frame_dim * 0.42, bounds["max_z"] + frame_dim * 0.78)
    direction = Vector((cx - cam.location.x, cy - cam.location.y, cz - cam.location.z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = frame_dim * ortho_scale_factor + ISO_CAMERA_EDGE_MARGIN
    bpy.context.view_layer.update()
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


def _arrow_panel_pixel_geometry(cam: bpy.types.Object, arrow_size: int) -> tuple[int, int, int]:
    margin_px = max(16, int(arrow_size * 0.20))
    unit_bar_px = _unit_bar_length_px(cam, unit_world=SCALE_MARK_LENGTH_M)
    bar_need_px = max(0, int(math.ceil(unit_bar_px))) + int(arrow_size * 0.32)
    panel_w_px = max(int(arrow_size * 1.55), bar_need_px)
    panel_h_px = int(arrow_size * 1.90)
    return margin_px, panel_w_px, panel_h_px


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


def _inflate_rect_by_pixels(
    cam: bpy.types.Object,
    rect: tuple[float, float, float, float],
    pad_px: int,
) -> tuple[float, float, float, float]:
    if pad_px <= 0:
        return rect
    hx, hy = _camera_half_extents(cam)
    rx, ry = _render_resolution_px()
    pad_x = (2.0 * hx) * (float(pad_px) / float(rx))
    pad_y = (2.0 * hy) * (float(pad_px) / float(ry))
    x1, x2, y1, y2 = rect
    return (x1 - pad_x, x2 + pad_x, y1 - pad_y, y2 + pad_y)


def _panel_overlap_area_with_objects(
    cam: bpy.types.Object,
    boxes: list[tuple[float, float, float, float]],
    arrow_size: int,
    corner_idx: int,
    safe_gap_px: int = ARROW_PANEL_SAFE_GAP_PX,
) -> float:
    rect = _panel_rect_in_camera(cam, arrow_size, corner_idx)
    rect = _inflate_rect_by_pixels(cam, rect, safe_gap_px)
    return sum(_rect_overlap_area(rect, b) for b in boxes)


def panel_overlaps_objects(
    cam: bpy.types.Object,
    objects: list[bpy.types.Object],
    arrow_size: int,
    corner_idx: int,
    safe_gap_px: int = ARROW_PANEL_SAFE_GAP_PX,
    overlap_eps: float = ARROW_PANEL_OVERLAP_EPS,
) -> bool:
    boxes = _project_object_boxes_camera(cam, objects)
    if not boxes:
        return False
    area = _panel_overlap_area_with_objects(
        cam=cam,
        boxes=boxes,
        arrow_size=arrow_size,
        corner_idx=corner_idx,
        safe_gap_px=safe_gap_px,
    )
    return area > float(overlap_eps)


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

    def overlap_areas() -> list[float]:
        vals = []
        for idx in range(4):
            vals.append(
                _panel_overlap_area_with_objects(
                    cam=cam,
                    boxes=boxes,
                    arrow_size=arrow_size,
                    corner_idx=idx,
                    safe_gap_px=ARROW_PANEL_SAFE_GAP_PX,
                )
            )
        return vals

    areas = overlap_areas()
    selected = min(range(4), key=lambda i: areas[i])

    step = 0
    while areas[selected] > ARROW_PANEL_OVERLAP_EPS and step < max_steps:
        cam.data.ortho_scale *= scale_step
        areas = overlap_areas()
        selected = min(range(4), key=lambda i: areas[i])
        step += 1

    if areas[selected] > ARROW_PANEL_OVERLAP_EPS:
        return -1
    return int(selected)


def get_center_xy(obj: bpy.types.Object) -> tuple[float, float]:
    return float(obj.location.x), float(obj.location.y)


# ============================================================
# Labels / marker materials
# ============================================================
def create_emission_material(mat_name: str, rgba=(1, 1, 1, 1), strength=3.0) -> bpy.types.Material:
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba
    em.inputs["Strength"].default_value = float(strength)
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


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


def add_label(building_obj: bpy.types.Object, label_text: str, cam: bpy.types.Object) -> None:
    bnd = get_object_world_bounds(building_obj)
    cx, cy = get_center_xy(building_obj)

    text_curve = bpy.data.curves.new(f"Label_{label_text}_curve", type="FONT")
    text_curve.body = str(label_text)
    text_curve.size = LABEL_FONT_SIZE
    text_curve.align_x = "CENTER"
    text_curve.align_y = "CENTER"
    text_curve.extrude = 0.02
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new(f"Label_{label_text}", text_curve)
    bpy.context.collection.objects.link(text_obj)

    # IMPORTANT: label anchored to origin-based center, not AABB center
    text_obj.location = (cx, cy, float(bnd["max_z"]) + LABEL_Z_OFFSET)

    text_obj.visible_shadow = False
    if hasattr(text_obj, "cycles_visibility"):
        text_obj.cycles_visibility.shadow = False
        text_obj.cycles_visibility.diffuse = False
        text_obj.cycles_visibility.glossy = False
        text_obj.cycles_visibility.ambient_occlusion = False

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

    bpy.context.view_layer.update()


def clear_labels_only() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Label_") or obj.name.startswith("Scale_"):
            remove_object_and_data(obj)
    for curve in list(bpy.data.curves):
        if curve.name.startswith("Label_") or curve.name.startswith("Scale_"):
            if curve.users == 0:
                bpy.data.curves.remove(curve)


def _rect_overlaps_any_building(rect_min_x, rect_max_x, rect_min_y, rect_max_y, building_objs, pad) -> bool:
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
    w = SCALE_MARK_LENGTH_M + 1.2
    h = SCALE_TICK_LEN_Y + 1.0

    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])

    candidates = [
        (max_x - SCALE_MARK_MARGIN - w, min_y + SCALE_MARK_MARGIN),
        (max_x - SCALE_MARK_MARGIN - w, max_y - SCALE_MARK_MARGIN - h),
        (min_x + SCALE_MARK_MARGIN, min_y + SCALE_MARK_MARGIN),
        (min_x + SCALE_MARK_MARGIN, max_y - SCALE_MARK_MARGIN - h),
    ]
    for (x0, y0) in candidates:
        if not _rect_overlaps_any_building(x0, x0 + w, y0, y0 + h, building_objs, pad=SCALE_PAD_FOR_PLACEMENT):
            return float(x0), float(y0)
    return float(max_x - SCALE_MARK_MARGIN - w), float(min_y + SCALE_MARK_MARGIN)


def add_scale_marker(bounds: dict, building_objs: list[bpy.types.Object], cam: bpy.types.Object, length_m: float = 1.0) -> None:
    x0, y0 = _pick_scale_marker_position(bounds, building_objs)
    z = float(bounds["max_z"]) + SCALE_MARK_Z_EPS

    mat = create_emission_material("Scale_Mark_Mat", rgba=(1, 1, 1, 1), strength=3.0)
    bar_y = y0 + SCALE_TICK_LEN_Y * 0.5

    created = []
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0 + length_m / 2.0, bar_y, z))
    bar = bpy.context.active_object
    bar.name = "Scale_Bar"
    bar.dimensions = (float(length_m), float(SCALE_BAR_THICKNESS), float(SCALE_BAR_THICKNESS))
    bar.data.materials.clear()
    bar.data.materials.append(mat)
    created.append(bar)

    for name, x in [("Scale_Tick_L", x0), ("Scale_Tick_R", x0 + length_m)]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, bar_y, z))
        t = bpy.context.active_object
        t.name = name
        t.dimensions = (float(SCALE_BAR_THICKNESS), float(SCALE_TICK_LEN_Y), float(SCALE_BAR_THICKNESS))
        t.data.materials.clear()
        t.data.materials.append(mat)
        created.append(t)

    for o in created:
        o.visible_shadow = False
        if hasattr(o, "cycles_visibility"):
            o.cycles_visibility.shadow = False
            o.cycles_visibility.diffuse = False
            o.cycles_visibility.glossy = False
            o.cycles_visibility.ambient_occlusion = False

    text_curve = bpy.data.curves.new("Scale_Label_curve", type="FONT")
    text_curve.body = f"{length_m:g} unit"
    text_curve.size = SCALE_LABEL_SIZE
    text_curve.align_x = "LEFT"
    text_curve.align_y = "CENTER"
    text_curve.extrude = 0.02
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new("Scale_Label", text_curve)
    bpy.context.collection.objects.link(text_obj)
    text_obj.location = (x0 + length_m + 0.35, bar_y, z + SCALE_LABEL_Z_OFF)
    text_obj.data.materials.clear()
    text_obj.data.materials.append(create_emission_material("Scale_Label_Mat", rgba=(1, 1, 1, 1), strength=3.0))

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
# Roads 
# ============================================================
def create_road_material(name="RoadMat") -> bpy.types.Material:
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = ROAD_COLOR
        if "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = ROAD_COLOR
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = ROAD_EMISSION_STRENGTH
        bsdf.inputs["Roughness"].default_value = ROAD_ROUGHNESS
    return mat


def add_road_slab(x0, x1, y0, y1, z=ROAD_ELEVATION, name="Road") -> bpy.types.Object:
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    sx = abs(x1 - x0)
    sy = abs(y1 - y0)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, z + ROAD_THICKNESS * 0.5))
    obj = bpy.context.active_object
    obj.name = name
    obj.dimensions = (sx, sy, ROAD_THICKNESS)

    mat = create_road_material()
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False
    return obj


def build_road_network() -> tuple[list[bpy.types.Object], list[dict], list[float], list[float]]:
    L = SCENE_BOUNDS * 1.05
    gap = L * 0.6
    xs = [-gap, 0.0, gap]
    ys = [-gap, 0.0, gap]

    road_objs: list[bpy.types.Object] = []
    rects: list[dict] = []

    for i, x in enumerate(xs):
        x0, x1 = x - ROAD_WIDTH / 2, x + ROAD_WIDTH / 2
        y0, y1 = -L, L
        road_objs.append(add_road_slab(x0, x1, y0, y1, z=ROAD_ELEVATION, name=f"Road_V_{i}"))
        rects.append({"id": f"V_{i}", "x0": x0, "x1": x1, "y0": y0, "y1": y1})

    for j, y in enumerate(ys):
        x0, x1 = -L, L
        y0, y1 = y - ROAD_WIDTH / 2, y + ROAD_WIDTH / 2
        road_objs.append(add_road_slab(x0, x1, y0, y1, z=ROAD_ELEVATION + 0.001, name=f"Road_H_{j}"))
        rects.append({"id": f"H_{j}", "x0": x0, "x1": x1, "y0": y0, "y1": y1})

    return road_objs, rects, xs, ys

# ============================================================
# ============================================================
def _rect_intersect(a: dict, b: dict) -> bool:
    return not (a["x1"] <= b["x0"] or a["x0"] >= b["x1"] or a["y1"] <= b["y0"] or a["y0"] >= b["y1"])


def _point_to_rect_dist(px, py, r) -> float:
    dx = max(r["x0"] - px, 0.0, px - r["x1"])
    dy = max(r["y0"] - py, 0.0, py - r["y1"])
    return math.hypot(dx, dy)


def nearest_road_rect(x: float, y: float, road_rects: list[dict]) -> tuple[dict | None, float]:
    if not road_rects:
        return None, 1e9
    best_rr, best_d = None, 1e9
    for rr in road_rects:
        d = _point_to_rect_dist(x, y, rr)
        if d < best_d:
            best_d = d
            best_rr = rr
    return best_rr, best_d


def _yaw_deg(obj: bpy.types.Object) -> float:
    return float(math.degrees(float(obj.rotation_euler.z)) % 360.0)


def _min_angle_diff_deg(a: float, targets: list[float]) -> float:
    best = 1e9
    for t in targets:
        d = abs((a - t + 180) % 360 - 180)
        best = min(best, d)
    return best

def align_building_to_nearest_road(obj: bpy.types.Object, road_rects: list[dict]) -> None:
    cx, cy = get_center_xy(obj)
    rr, _ = nearest_road_rect(cx, cy, road_rects)
    if rr is None:
        return

    w = abs(rr["x1"] - rr["x0"])
    h = abs(rr["y1"] - rr["y0"])
    vertical = (h >= w)
    yaw = 0.0 if vertical else random.choice([90.0, 270.0])

    obj.rotation_euler.z = math.radians(yaw)
    bpy.context.view_layer.update()


def sample_position() -> tuple[float, float]:
    return random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS), random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)


def validate_position_with_roads_2d_precheck(
    x: float,
    y: float,
    new_meta: dict,
    existing: list[dict],
    road_rects: list[dict],
) -> bool:
    new_hw = new_meta["half_w"]
    new_hd = new_meta["half_d"]

    for b in existing:
        dx = abs(x - b["pos"][0])
        dy = abs(y - b["pos"][1])
        if dx < (b["half_w"] + new_hw + MIN_GAP) and dy < (b["half_d"] + new_hd + MIN_GAP):
            return False

    brect = {"x0": x - new_hw, "x1": x + new_hw, "y0": y - new_hd, "y1": y + new_hd}
    for rr in road_rects:
        rr_exp = {
            "x0": rr["x0"] - ROAD_AVOID_MARGIN,
            "x1": rr["x1"] + ROAD_AVOID_MARGIN,
            "y0": rr["y0"] - ROAD_AVOID_MARGIN,
            "y1": rr["y1"] + ROAD_AVOID_MARGIN,
        }
        if _rect_intersect(brect, rr_exp):
            return False

    return True

# ============================================================

# ============================================================
def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _poly_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _inside(p, a, b) -> bool:
    return _cross(a, b, p) >= -OVERLAP_NUM_EPS


def _line_intersection(p1, p2, a1, a2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a1
    x4, y4 = a2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) <= 1e-12:
        return p2
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (px, py)


def _clip_convex(subject: list[tuple[float, float]], clipper: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = subject[:]
    if len(out) < 3 or len(clipper) < 3:
        return []
    for i in range(len(clipper)):
        inp = out
        out = []
        A = clipper[i]
        B = clipper[(i + 1) % len(clipper)]
        if not inp:
            break
        S = inp[-1]
        for E in inp:
            if _inside(E, A, B):
                if not _inside(S, A, B):
                    out.append(_line_intersection(S, E, A, B))
                out.append(E)
            elif _inside(S, A, B):
                out.append(_line_intersection(S, E, A, B))
            S = E
    return out


def _aabb2(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def _aabb2_intersect(ax0, ax1, ay0, ay1, bx0, bx1, by0, by1) -> bool:
    return not (ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1)


def _poly_intersection_area_convex(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ax0, ax1, ay0, ay1 = _aabb2(a)
    bx0, bx1, by0, by1 = _aabb2(b)
    if not _aabb2_intersect(ax0, ax1, ay0, ay1, bx0, bx1, by0, by1):
        return 0.0
    inter = _clip_convex(a, b)
    if not inter:
        return 0.0
    return _poly_area(inter)


def _rect_poly(rr: dict, margin: float = 0.0) -> list[tuple[float, float]]:
    x0 = float(rr["x0"] - margin)
    x1 = float(rr["x1"] + margin)
    y0 = float(rr["y0"] - margin)
    y1 = float(rr["y1"] + margin)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _get_footprint_hull(obj: bpy.types.Object) -> list[tuple[float, float]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    try:
        mw = obj_eval.matrix_world
        pts = []
        for v in mesh.vertices:
            p = mw @ v.co
            pts.append((float(p.x), float(p.y)))
        return _convex_hull_2d(pts)
    finally:
        obj_eval.to_mesh_clear()

# ============================================================
# Buildings
# ============================================================
def pick_palette_color() -> tuple[float, float, float]:
    return random.choice(COLOR_PALETTE)


def _set_origin_to_geometry_and_center_xy(obj: bpy.types.Object, x: float, y: float) -> None:
    if obj is None:
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    old_z = float(obj.location.z)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.view_layer.update()
    obj.location = (float(x), float(y), old_z)
    bpy.context.view_layer.update()


def create_building(idx: int, x: float, y: float) -> tuple[bpy.types.Object, dict]:
    shape = random.choice(["CUBE", "CUBOID", "CYLINDER", "PRISM", "SPHERE", "L_SHAPE", "U_SHAPE"])
    h = random.uniform(2.0, 10.0)
    w = random.uniform(2.4, 6.0)

    base_z = float(BUILDING_BASE_Z)

    if shape == "CUBE":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, base_z + h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (w, w, h)

    elif shape == "CUBOID":
        wx = random.uniform(2.0, 6.0)
        wy = random.uniform(2.0, 6.0)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, base_z + h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (wx, wy, h)

    elif shape == "CYLINDER":
        bpy.ops.mesh.primitive_cylinder_add(radius=w / 2, depth=h, location=(x, y, base_z + h / 2))
        obj = bpy.context.active_object

    elif shape == "PRISM":
        bpy.ops.mesh.primitive_cylinder_add(radius=w / 2, depth=h, vertices=3, location=(x, y, base_z + h / 2))
        obj = bpy.context.active_object

    elif shape == "L_SHAPE":
        wx = random.uniform(2.0, 6.0)
        wy = random.uniform(2.0, 6.0)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, base_z + h / 2))
        obj1 = bpy.context.active_object
        obj1.dimensions = (wx, wy * 0.4, h)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + wx * 0.30, y + wy * 0.30, base_z + h / 2))
        obj2 = bpy.context.active_object
        obj2.dimensions = (wx * 0.4, wy, h)

        obj1.select_set(True)
        obj2.select_set(True)
        bpy.context.view_layer.objects.active = obj1
        bpy.ops.object.join()
        obj = bpy.context.active_object
        _set_origin_to_geometry_and_center_xy(obj, x, y)

    elif shape == "U_SHAPE":
        wx = random.uniform(2.0, 6.0)
        wy = random.uniform(2.0, 6.0)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x - wx * 0.35, y, base_z + h / 2))
        left = bpy.context.active_object
        left.dimensions = (wx * 0.3, wy, h)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + wx * 0.35, y, base_z + h / 2))
        right = bpy.context.active_object
        right.dimensions = (wx * 0.3, wy, h)

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y - wy * 0.35, base_z + h / 2))
        bottom = bpy.context.active_object
        bottom.dimensions = (wx, wy * 0.3, h)

        left.select_set(True)
        right.select_set(True)
        bottom.select_set(True)
        bpy.context.view_layer.objects.active = left
        bpy.ops.object.join()
        obj = bpy.context.active_object
        _set_origin_to_geometry_and_center_xy(obj, x, y)

    else:  # SPHERE
        bpy.ops.mesh.primitive_uv_sphere_add(radius=w / 2, location=(x, y, base_z + w / 2))
        obj = bpy.context.active_object
        h = w

    obj.name = f"B_{idx}"

    rgb = pick_palette_color()
    mat = bpy.data.materials.new(f"Mat_{idx}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False

    bpy.context.view_layer.update()
    bnd = get_object_world_bounds(obj)
    half_w = 0.5 * float(bnd["width"])
    half_d = 0.5 * float(bnd["depth"])

    obj["shape"] = shape

    meta = {
        "id": idx,
        "pos": (float(obj.location.x), float(obj.location.y)),
        "yaw_deg": float(_yaw_deg(obj)),
        "height": float(h),
        "shape": shape,
        "color": rgb,
        "half_w": float(half_w),
        "half_d": float(half_d),
    }
    return obj, meta


def refresh_building_meta(building_meta: list[dict], building_objs: list[bpy.types.Object]) -> list[dict]:
    obj_by_id = {int(o.name.split("_")[1]): o for o in building_objs if o.name.startswith("B_")}
    out = []
    for m in building_meta:
        bid = m["id"]
        obj = obj_by_id.get(bid)
        mm = dict(m)
        if obj:
            mm["pos"] = (float(obj.location.x), float(obj.location.y))
            mm["yaw_deg"] = float(_yaw_deg(obj))
        out.append(mm)
    return out


def _shape(obj: bpy.types.Object) -> str:
    return str(obj.get("shape", ""))

# ============================================================
# ============================================================
def detect_overlaps_buildings_2d(
    building_objs: list[bpy.types.Object],
    area_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    strong = []
    weak = []

    bs = [o for o in building_objs if o and o.type == "MESH" and o.name.startswith("B_")]
    hulls = {}
    for b in bs:
        hid = int(b.name.split("_")[1])
        hulls[hid] = _get_footprint_hull(b)

    ids = sorted(hulls.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            ha = hulls[a_id]
            hb = hulls[b_id]
            area = _poly_intersection_area_convex(ha, hb)
            if area <= OVERLAP_NUM_EPS:
                continue
            if area > area_threshold:
                strong.append((a_id, b_id, float(area)))
            else:
                weak.append((a_id, b_id, float(area)))

    return strong, weak


def detect_overlaps_building_road_2d(
    building_objs: list[bpy.types.Object],
    road_rects: list[dict],
    area_threshold: float,
) -> tuple[list[tuple[int, str, float]], list[tuple[int, str, float]]]:
    strong = []
    weak = []

    roads_poly = []
    for rr in road_rects:
        rpoly_pts = _rect_poly(rr, margin=0.0)
        if len(rpoly_pts) < 3:
            continue
        try:
            rpoly = Polygon(rpoly_pts)
            if not rpoly.is_valid:
                rpoly = rpoly.buffer(0)
            if rpoly.is_empty:
                continue
            roads_poly.append((rr["id"], rpoly))
        except Exception:
            continue

    for b in building_objs:
        if not b or b.type != "MESH" or not b.name.startswith("B_"):
            continue
        bid = int(b.name.split("_")[1])
        hull = _get_footprint_hull(b)
        if len(hull) < 3:
            continue
        try:
            bpoly = Polygon(hull)
            if not bpoly.is_valid:
                bpoly = bpoly.buffer(0)
            if bpoly.is_empty:
                continue
        except Exception:
            continue

        for rid, rpoly in roads_poly:
            if not bpoly.intersects(rpoly):
                continue
            try:
                area = float(bpoly.intersection(rpoly).area)
            except Exception:
                continue
            if area <= OVERLAP_NUM_EPS:
                continue
            if area > area_threshold:
                strong.append((bid, str(rid), float(area)))
                break
            weak.append((bid, str(rid), float(area)))
            break

    return strong, weak

# ============================================================
# Angle anomaly detection
# ============================================================
def _nearest_intersection_distance(x: float, y: float, xs: list[float], ys: list[float]) -> tuple[float, tuple[float, float]]:
    best_d = 1e18
    best_p = (0.0, 0.0)
    for xi in xs:
        for yi in ys:
            d = math.hypot(float(x - xi), float(y - yi))
            if d < best_d:
                best_d = d
                best_p = (float(xi), float(yi))
    return float(best_d), best_p


def _is_square_like(obj: bpy.types.Object) -> bool:
    shape = obj.get("shape", "")
    if shape not in {"CUBE", "CUBOID", "L_SHAPE", "U_SHAPE"}:
        return False

    bnd = get_object_world_bounds(obj)
    w = max(1e-6, float(bnd["width"]))
    d = max(1e-6, float(bnd["depth"]))
    ratio = max(w, d) / min(w, d)
    return ratio <= float(ANGLE_SQUARE_ASPECT_TOL)


def detect_angle_anomaly_buildings(building_objs: list[bpy.types.Object], xs: list[float], ys: list[float]) -> list[int]:
    bad: list[int] = []
    for obj in building_objs:
        if not obj or obj.type != "MESH" or not obj.name.startswith("B_"):
            continue
        if not _is_square_like(obj):
            continue

        cx, cy = get_center_xy(obj)
        dist, _p = _nearest_intersection_distance(cx, cy, xs, ys)
        if dist > float(ANGLE_INTERSECTION_RADIUS):
            continue

        yaw = _yaw_deg(obj)
        diff = _min_angle_diff_deg(yaw, [0.0, 90.0, 180.0, 270.0])
        if diff > float(ANGLE_ANOMALY_DEG):
            bad.append(int(obj.name.split("_")[1]))
    return sorted(set(bad))


def _eligible_for_angle(obj: bpy.types.Object) -> bool:
    return _shape(obj) in ANGLE_ELIGIBLE_SHAPES

# ============================================================
# Evaluate issues
# ============================================================
def has_any_weak_overlap(details: dict) -> bool:
    return (len(details.get("bb_overlap_weak", [])) > 0) or (len(details.get("road_overlap_weak", [])) > 0)


def evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys):
    bb_strong, bb_weak = detect_overlaps_buildings_2d(building_objs, area_threshold=OVERLAP_AREA_THRESHOLD)
    br_strong, br_weak = detect_overlaps_building_road_2d(building_objs, road_rects, area_threshold=OVERLAP_AREA_THRESHOLD)
    bad_angle = detect_angle_anomaly_buildings(building_objs, xs, ys)

    flags = {
        ISSUE_OVERLAP: len(bb_strong) > 0,
        ISSUE_ROAD: len(br_strong) > 0,
        ISSUE_ANGLE: len(bad_angle) > 0,
    }
    details = {
        "bb_overlap_strong": bb_strong,
        "bb_overlap_weak": bb_weak,
        "road_overlap_strong": br_strong,
        "road_overlap_weak": br_weak,
        "angle_anomaly_buildings": bad_angle,
    }
    # print("flags:", flags)
    # print("details:", details)
    return flags, details


def exactly_one_issue(flags: dict) -> bool:
    return sum(1 for v in flags.values() if v) == 1


def normalize_issue_meta_from_detection(flags: dict, details: dict) -> dict:
    if flags.get(ISSUE_OVERLAP) and details["bb_overlap_strong"]:
        a, b, _area = details["bb_overlap_strong"][0]
        return {"issues": [{"type": ISSUE_OVERLAP, "buildings": [int(a), int(b)]}]}
    if flags.get(ISSUE_ROAD) and details["road_overlap_strong"]:
        bid, _rid, _area = details["road_overlap_strong"][0]
        return {"issues": [{"type": ISSUE_ROAD, "buildings": [int(bid)]}]}
    if flags.get(ISSUE_ANGLE) and details["angle_anomaly_buildings"]:
        return {"issues": [{"type": ISSUE_ANGLE, "buildings": [int(details["angle_anomaly_buildings"][0])]}]}
    return {"issues": [{"type": "unknown", "buildings": []}]}

# ============================================================
# QA
# ============================================================
def _snapshot_obj_transform(obj: bpy.types.Object):
    return (obj.location.copy(), obj.rotation_euler.copy(), obj.dimensions.copy())


def _restore_obj_transform(obj: bpy.types.Object, snap):
    loc, rot, dim = snap
    obj.location = loc
    obj.rotation_euler = rot
    obj.dimensions = dim
    bpy.context.view_layer.update()


def _parse_choice_text(choice_text: str) -> dict | None:
    s = choice_text.strip()

    m = re.match(r"^Move object\s+(\d+)\s+(North|South|East|West)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s*unit$", s, re.IGNORECASE)
    if m:
        bid = int(m.group(1))
        d = m.group(2).capitalize()
        dist = float(m.group(3))
        return {"op": "move", "id": bid, "dir": d, "dist": dist}

    r = re.match(r"^Rotate object\s+(\d+)\s+(clockwise|counter-clockwise)\s+by\s+([0-9]+(?:\.[0-9]+)?)°$", s, re.IGNORECASE)
    if r:
        bid = int(r.group(1))
        cw = (r.group(2).lower() == "clockwise")
        deg = float(r.group(3))
        return {"op": "rotate", "id": bid, "clockwise": cw, "deg": deg}

    return None


def _apply_parsed_choice(parsed: dict, building_by_id: dict[int, bpy.types.Object]) -> bool:
    obj = building_by_id.get(int(parsed["id"]))
    if obj is None:
        return False

    if parsed["op"] == "move":
        dir_name = parsed["dir"]
        dist = float(parsed["dist"])
        vx, vy = DIR_TO_VEC[dir_name]
        obj.location.x += vx * dist
        obj.location.y += vy * dist
        bpy.context.view_layer.update()
        return True

    if parsed["op"] == "rotate":
        deg = float(parsed["deg"])
        clockwise = bool(parsed["clockwise"])
        sgn = -1.0 if clockwise else 1.0
        obj.rotation_euler.z += sgn * math.radians(deg)
        bpy.context.view_layer.update()
        return True

    return False


def _choice_makes_scene_clean(choice_text, building_objs, road_objs, road_rects, xs, ys) -> bool:
    parsed = _parse_choice_text(choice_text)
    if parsed is None:
        return False

    building_by_id = {int(o.name.split("_")[1]): o for o in building_objs if o.name.startswith("B_")}
    obj = building_by_id.get(int(parsed["id"]))
    if obj is None:
        return False

    snap = _snapshot_obj_transform(obj)

    ok = _apply_parsed_choice(parsed, building_by_id)
    if not ok:
        _restore_obj_transform(obj, snap)
        return False

    flags, details = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)
    is_clean = (not any(flags.values())) and (not has_any_weak_overlap(details))

    _restore_obj_transform(obj, snap)
    return is_clean


def _filter_distractors_unique_fix(correct_text, candidate_texts, building_objs, road_objs, road_rects, xs, ys, need=3, max_rounds=80) -> list[str]:
    out: list[str] = []
    seen = set([correct_text])

    for t in candidate_texts:
        if t in seen:
            continue
        seen.add(t)
        if not _choice_makes_scene_clean(t, building_objs, road_objs, road_rects, xs, ys):
            out.append(t)
        if len(out) >= need:
            return out[:need]

    rounds = 0
    while len(out) < need and rounds < max_rounds:
        rounds += 1
        bid = int(re.search(r"\d+", correct_text).group()) if re.search(r"\d+", correct_text) else 1

        if random.random() < 0.6:
            d = random.choice(MOVE_DIRS)
            m = random.choice(MOVE_DISTS_M)
            cand = f"Move object {bid} {d} by {m:g} unit"
        else:
            deg = float(random.choice(ANGLE_ROTATE_CHOICES_DEG))
            cand = f"Rotate object {bid} {'clockwise' if random.random()<0.5 else 'counter-clockwise'} by {deg:g}°"

        if cand in seen:
            continue
        seen.add(cand)

        if not _choice_makes_scene_clean(cand, building_objs, road_objs, road_rects, xs, ys):
            out.append(cand)

    return out[:need]


def qa1_mcq_what_problem(issue_meta: dict, images: list[str]) -> dict:
    t = issue_meta["issues"][0]["type"] if issue_meta.get("issues") else None
    mapping = {ISSUE_OVERLAP: "A", ISSUE_ANGLE: "B", ISSUE_ROAD: "C"}
    choices = [
        "A. Objects overlap each other",
        "B. An object near a boundary intersection is rotated at an abnormal angle (not aligned with boundaries)",
        "C. An object overlaps the boundary network",
    ]
    q = (
        "You are viewing a top-down view of a 3D scene containing:\n"
        "- Multiple colored rectangular/polygonal objects of varying heights\n"
        "- A yellow boundary network in a 3x3 grid pattern (vertical and horizontal boundaries)\n"
        "- The objects are labeled with numbers (1, 2, 3, etc.)\n"
        "Question: Examine the scene carefully and identify what problem exists.\n"
        "Top-view image: <image>"
    )
    return {"question": q + "\n" + "\n".join(choices), "answer": mapping.get(t), "task_type": "top_error_identify", "meta": issue_meta, "images": images}


def qa2_mcq_what_problem(issue_meta: dict, images: list[str]) -> dict:
    t = issue_meta["issues"][0]["type"] if issue_meta.get("issues") else None
    mapping = {ISSUE_OVERLAP: "A", ISSUE_ANGLE: "B", ISSUE_ROAD: "C"}
    choices = [
        "A. Objects overlap each other",
        "B. An object near a boundary intersection is rotated at an abnormal angle (not aligned with boundaries)",
        "C. An object overlaps the boundary network",
    ]
    q = (
        "You are viewing two images of a 3D scene: a top-down view and an isometric view.\n"
        "The scene contains:\n"
        "- Multiple colored rectangular/polygonal objects of varying heights\n"
        "- A yellow boundary network in a 3x3 grid pattern (vertical and horizontal boundaries)\n"
        "- The objects are labeled with numbers (1, 2, 3, etc.)\n"
        "Question: Examine both views carefully and identify what problem exists.\n"
        "Top-view image: <image>\n"
        "Isometric-view image: <image>"
    )
    return {"question": q + "\n" + "\n".join(choices), "answer": mapping.get(t, "A"), "task_type": "top_isometric_error_identify", "meta": issue_meta, "images": images}


def qa3_fix(issue_meta, action, images, building_objs, road_objs, road_rects, xs, ys) -> dict:
    issue = issue_meta["issues"][0]
    t = issue["type"]

    if t == ISSUE_OVERLAP:
        a, b = issue["buildings"]
        known = f"A problem is detected in the image: Objects {a} and {b} overlap each other."
    elif t == ISSUE_ROAD:
        b = issue["buildings"][0]
        known = f"A problem is detected in the image: Object {b} overlaps the boundary network."
    elif t == ISSUE_ANGLE:
        b = issue["buildings"][0]
        known = f"A problem is detected in the image: Object {b} has an angle anomaly near a boundary intersection."
    else:
        known = "A problem is detected in the image: Unknown."


    correct = action["reverse_choice_text"]
    raw_distractors = [x for x in action.get("distractors", []) if isinstance(x, str)]

    if not _choice_makes_scene_clean(correct, building_objs, road_objs, road_rects, xs, ys):
        print("[WARN] QA3 correct choice does NOT clean the scene. Injection may not be reversible or weak-overlap exists.")

    distractors = _filter_distractors_unique_fix(
        correct_text=correct,
        candidate_texts=raw_distractors,
        building_objs=building_objs,
        road_objs=road_objs,
        road_rects=road_rects,
        xs=xs,
        ys=ys,
        need=3,
    )

    # Generate fallback distractors with collision detection
    bid = action['id']
    fallback_attempts = 0
    while len(distractors) < 3 and fallback_attempts < 20:
        fallback_attempts += 1
        rand_dir = random.choice(MOVE_DIRS)
        rand_dist = random.choice(MOVE_DISTS_M)
        cand = f"Move object {bid} {rand_dir} by {rand_dist:g} unit"
        # Only add if it doesn't solve the problem
        if cand not in distractors and not _choice_makes_scene_clean(cand, building_objs, road_objs, road_rects, xs, ys):
            distractors.append(cand)

    choices_text = [correct] + distractors[:3]
    random.shuffle(choices_text)

    letters = ["A", "B", "C", "D"]
    choices = [f"{letters[i]}. {choices_text[i]}" for i in range(4)]
    ans_letter = letters[choices_text.index(correct)]

    q = (
        f"{known}\n"
        "Choose ONE action to fix it in ONE step. The fix should resolve the issue without introducing new problems.\n\n"
        "Reference information:\n"
        "- Directions in the image: North=up, South=down, East=right, West=left.\n"
        "- The top-view image includes a 1 unit scale bar inside the North indicator panel.\n"
        "Top-view image: <image>"
    )

    return {
        "question": q + "\n" + "\n".join(choices),
        "answer": ans_letter,
        "task_type": "top_error_modify",
        "meta": {"issue_meta": issue_meta, "inject_action": action, "verified_unique_fix": True},
        "images": images,
    }


def build_new_qa_formats(sample_dir, issue_meta, action, building_objs, road_objs, road_rects, xs, ys) -> dict:
    top_abs = os.path.abspath(os.path.join(sample_dir, "top.png"))
    iso_abs = os.path.abspath(os.path.join(sample_dir, "isometric.png"))

    qa_top_1 = qa1_mcq_what_problem(issue_meta, images=[top_abs])
    qa_top_3 = qa3_fix(issue_meta, action, images=[top_abs], building_objs=building_objs, road_objs=road_objs, road_rects=road_rects, xs=xs, ys=ys)
    qa_top_iso = qa2_mcq_what_problem(issue_meta, images=[top_abs, iso_abs])

    return {"top": [qa_top_1, qa_top_3], "top_isometric": [qa_top_iso]}

# ============================================================
# Inject ONE-step error (Route-1) + verification
# ============================================================
def _apply_move(obj: bpy.types.Object, dir_name: str, dist_m: float) -> None:
    vx, vy = DIR_TO_VEC[dir_name]
    obj.location.x += float(vx * dist_m)
    obj.location.y += float(vy * dist_m)
    bpy.context.view_layer.update()


def _apply_rotate(obj: bpy.types.Object, clockwise: bool, deg: float) -> None:
    sgn = -1.0 if clockwise else 1.0
    obj.rotation_euler.z += float(sgn * math.radians(deg))
    bpy.context.view_layer.update()


def _dir_opposite(d: str) -> str:
    return {"North": "South", "South": "North", "East": "West", "West": "East"}[d]


def _reverse_action(action: dict, building_objs_by_id: dict[int, bpy.types.Object]) -> None:
    obj = building_objs_by_id.get(int(action["id"]))
    if obj is None:
        return
    if action["op"] == "move":
        _apply_move(obj, action["reverse_dir"], float(action["dist_m"]))
    elif action["op"] == "rotate":
        _apply_rotate(obj, clockwise=action["reverse_clockwise"], deg=float(action["deg"]))


def _ensure_square_building(obj: bpy.types.Object, target_size: float | None = None) -> None:
    bpy.context.view_layer.update()
    bnd = get_object_world_bounds(obj)
    h = float(bnd["height"])
    s = max(float(bnd["width"]), float(bnd["depth"])) if target_size is None else float(target_size)
    obj.dimensions = (s, s, max(0.2, h))
    bpy.context.view_layer.update()


def _attempt_inject_overlap(building_objs, road_objs, road_rects, xs, ys) -> dict | None:
    bs = [o for o in building_objs if o.name.startswith("B_")]
    if len(bs) < 2:
        return None

    for _ in range(80):
        a, b = random.sample(bs, 2)
        bid_a = int(a.name.split("_")[1])
        bid_b = int(b.name.split("_")[1])

        ax, ay = get_center_xy(a)
        bx, by = get_center_xy(b)
        dx = bx - ax
        dy = by - ay

        ba = get_object_world_bounds(a)
        bb = get_object_world_bounds(b)

        # move a towards b
        if abs(dx) >= abs(dy):
            dir_name = "East" if dx > 0 else "West"
            gap = abs(dx) - (ba["width"] / 2 + bb["width"] / 2)
        else:
            dir_name = "North" if dy > 0 else "South"
            gap = abs(dy) - (ba["depth"] / 2 + bb["depth"] / 2)

        # Increase move distance to ensure strong overlap
        # When gap <= 0 (already touching/overlapping), add more distance
        # When gap > 0, move MORE than the gap to create actual overlap
        if gap <= 0:
            dist = random.uniform(0.5, 1.2)  # increased from 0.3-0.8
        else:
            # Move 100-150% of gap to ensure overlap (was 60-95%)
            dist = gap * random.uniform(1.0, 1.5)

        loc0 = a.location.copy()
        rot0 = a.rotation_euler.copy()

        _apply_move(a, dir_name, dist)

        
        align_building_to_nearest_road(a, road_rects)

        flags, details = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)

        
        bb_strong = details.get("bb_overlap_strong", [])
        bb_weak = details.get("bb_overlap_weak", [])
        print(f"[inject_debug] Attempt {_+1}: bid_a={bid_a}, bid_b={bid_b}, dir={dir_name}, dist={dist:.2f}")
        print(f"[inject_debug]   bb_strong={bb_strong}, bb_weak={bb_weak}, flags={flags}")

        # Check for weak overlaps 
        if not flags[ISSUE_OVERLAP] and (len(bb_weak) > 0 or len(details.get("road_overlap_weak", [])) > 0):
            print(f"[inject_debug]   -> Failed: weak overlap but no strong overlap")
            a.location = loc0
            a.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        if (not flags[ISSUE_OVERLAP]) or flags[ISSUE_ROAD] or flags[ISSUE_ANGLE]:
            print(f"[inject_debug]   -> Failed: flags check failed (overlap={flags[ISSUE_OVERLAP]}, road={flags[ISSUE_ROAD]}, angle={flags[ISSUE_ANGLE]})")
            a.location = loc0
            a.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        pairs = details.get("bb_overlap_strong", [])
        if len(pairs) != 1:
            print(f"[inject_debug]   -> Failed: {len(pairs)} strong pairs, need exactly 1")
            a.location = loc0
            a.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        p = pairs[0]
        if set([int(p[0]), int(p[1])]) != set([bid_a, bid_b]):
            print(f"[inject_debug]   -> Failed: pair mismatch {p} vs {[bid_a, bid_b]}")
            a.location = loc0
            a.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        print(f"[inject_debug]   -> SUCCESS!")

        dist_exec = float(dist)
        dist_text = round(dist_exec, 1)
        rev_dir = _dir_opposite(dir_name)

        return {
            "op": "move",
            "id": bid_a,
            "dir": dir_name,
            "dist_m": dist_exec,
            "reverse_dir": rev_dir,
            "reverse_choice_text": f"Move object {bid_a} {rev_dir} by {dist_text:g} unit",
            "distractors": [
                f"Move object {bid_a} {dir_name} by {dist_text:g} unit",
                f"Move object {bid_a} {random.choice(MOVE_DIRS)} by {random.choice(MOVE_DISTS_M):g} unit",
                f"Rotate object {bid_a} clockwise by {random.choice(ANGLE_ROTATE_CHOICES_DEG):g}°",
                f"Rotate object {bid_a} counter-clockwise by {random.choice(ANGLE_ROTATE_CHOICES_DEG):g}°",
            ],
        }

    return None



def _attempt_inject_road_overlap(building_objs, road_objs, road_rects, xs, ys) -> dict | None:
    bs = [o for o in building_objs if o.name.startswith("B_")]
    if not bs:
        return None

    for _ in range(80):
        obj = random.choice(bs)
        bid = int(obj.name.split("_")[1])

        rr = next((r for r in road_rects if r["id"] == "V_1"), road_rects[0])
        road_x = (rr["x0"] + rr["x1"]) * 0.5

        loc0 = obj.location.copy()
        rot0 = obj.rotation_euler.copy()

        cx, _cy = get_center_xy(obj)
        dir_name = "East" if road_x > cx else "West"
        dist = abs(float(road_x - cx))

        # IMPORTANT: align CENTER (origin) to road_x
        obj.location.x += float(road_x - cx)
        bpy.context.view_layer.update()

        align_building_to_nearest_road(obj, road_rects)

        flags, details = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)

        if has_any_weak_overlap(details):
            obj.location = loc0
            obj.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        if (not flags[ISSUE_ROAD]) or flags[ISSUE_OVERLAP] or flags[ISSUE_ANGLE]:
            obj.location = loc0
            obj.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        bads = details.get("road_overlap_strong", [])
        if len(bads) != 1 or int(bads[0][0]) != bid:
            obj.location = loc0
            obj.rotation_euler = rot0
            bpy.context.view_layer.update()
            continue

        rev_dir = _dir_opposite(dir_name)
        # Use mix of rotate and different-direction moves (not opposite, which would fix the issue)
        other_dirs = [d for d in MOVE_DIRS if d != rev_dir]
        return {
            "op": "move",
            "id": bid,
            "dir": dir_name,
            "dist_m": float(dist),
            "reverse_dir": rev_dir,
            "reverse_choice_text": f"Move object {bid} {rev_dir} by {round(dist, 1):g} unit",
            "distractors": [
                f"Move object {bid} {random.choice(other_dirs)} by {random.choice(MOVE_DISTS_M):g} unit",
                f"Rotate object {bid} clockwise by {random.choice(ANGLE_ROTATE_CHOICES_DEG):g}°",
                f"Move object {bid} {random.choice(other_dirs)} by {random.choice(MOVE_DISTS_M):g} unit",
                f"Rotate object {bid} counter-clockwise by {random.choice(ANGLE_ROTATE_CHOICES_DEG):g}°",
            ],
        }

    return None


def _pick_intersection_near_center(xs, ys) -> tuple[float, float]:
    # center intersection should be (0,0) in this grid
    # choose nearest to (0,0)
    best = (xs[0], ys[0])
    best_d = 1e18
    for xi in xs:
        for yi in ys:
            d = abs(xi) + abs(yi)
            if d < best_d:
                best_d = d
                best = (xi, yi)
    return float(best[0]), float(best[1])


def _attempt_place_near_intersection_without_roads(obj, road_objs, xi, yi, road_rects) -> bool:
    """
    Place obj near (xi, yi) within ANGLE_INTERSECTION_RADIUS,
    while ensuring NO road overlap in 2D (strong or weak).
    """

    bpy.context.view_layer.update()
    bnd = get_object_world_bounds(obj)

    
    s = max(float(bnd["width"]), float(bnd["depth"]))
    half_extent = 0.5 * s * math.sqrt(2.0)
    min_off = (ROAD_WIDTH / 2.0) + half_extent + float(ANGLE_CLEAR_FROM_ROAD)

    if min_off >= float(ANGLE_INTERSECTION_RADIUS) - 0.05:
        return False

    loc0 = obj.location.copy()

    for _ in range(120):
        r = random.uniform(min_off, float(ANGLE_INTERSECTION_RADIUS) - 0.05)
        ang = random.uniform(0, 2 * math.pi)
        x = xi + r * math.cos(ang)
        y = yi + r * math.sin(ang)

        obj.location.x = float(x)
        obj.location.y = float(y)
        bpy.context.view_layer.update()

        strong, weak = detect_overlaps_building_road_2d([obj], road_rects, area_threshold=OVERLAP_AREA_THRESHOLD)
        if (not strong) and (not weak):
            return True

    obj.location = loc0
    bpy.context.view_layer.update()
    return False


def _attempt_inject_angle(building_objs, road_objs, road_rects, xs, ys, max_attempts: int = 10) -> dict | None:
    """
    Place a square-like building near an intersection and rotate 30/45 degrees.
    Must keep: building overlap = False, road overlap = False.
    Must be reversible to clean.
    """
    bs = [o for o in building_objs if o.name.startswith("B_")]
    if not bs:
        print("[angle_debug] No buildings found")
        return None

    
    cands = [o for o in bs if _eligible_for_angle(o)]
    print(f"[angle_debug] Found {len(cands)} eligible buildings for angle injection out of {len(bs)} total")
    if not cands:
        return None

    for _attempt in range(max_attempts):
        obj = random.choice(cands)
        bid = int(obj.name.split("_")[1])

        loc0 = obj.location.copy()
        rot0 = obj.rotation_euler.copy()
        dim0 = obj.dimensions.copy()

        # make it square footprint + align first
        
        _ensure_square_building(obj, target_size=random.uniform(1.0, 1.6))

        align_building_to_nearest_road(obj, road_rects)

        xi, yi = _pick_intersection_near_center(xs, ys)

        ok_place = _attempt_place_near_intersection_without_roads(obj, road_objs, xi, yi, road_rects)
        if not ok_place:
            print(f"[angle_debug]   -> Failed: could not place near intersection (building too large?)")
            # revert
            obj.location = loc0
            obj.rotation_euler = rot0
            obj.dimensions = dim0
            bpy.context.view_layer.update()
            continue 

        # rotate by 30 or 45 deg (one-step)
        deg = float(random.choice(ANGLE_ROTATE_CHOICES_DEG))
        clockwise = random.choice([True, False])
        _apply_rotate(obj, clockwise=clockwise, deg=deg)

        cx, cy = get_center_xy(obj)
        dist_int, _p = _nearest_intersection_distance(cx, cy, xs, ys)
        yaw = _yaw_deg(obj)
        diff = _min_angle_diff_deg(yaw, [0.0, 90.0, 180.0, 270.0])
        print(f"[angle_debug] bid={bid}, deg={deg}, cw={clockwise}, dist_int={dist_int:.2f}, yaw={yaw:.1f}, diff={diff:.1f}")

        # check must be angle only
        flags, details = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)
        print(f"[angle_debug]   flags={flags}, angle_bldgs={details.get('angle_anomaly_buildings', [])}")

        if not flags[ISSUE_ANGLE]:
            print(f"[angle_debug]   -> Failed: no angle detected")
            # revert
            obj.location = loc0
            obj.rotation_euler = rot0
            obj.dimensions = dim0
            bpy.context.view_layer.update()
            continue 
        if flags[ISSUE_OVERLAP] or flags[ISSUE_ROAD]:
            print(f"[angle_debug]   -> Failed: overlap or road overlap")
            # revert
            obj.location = loc0
            obj.rotation_euler = rot0
            obj.dimensions = dim0
            bpy.context.view_layer.update()
            continue 

        print(f"[angle_debug]   -> SUCCESS!")

        
        rev_clockwise = (not clockwise)
        rev_text = f"Rotate object {bid} {'clockwise' if rev_clockwise else 'counter-clockwise'} by {deg:g}°"

        action = {
            "op": "rotate",
            "id": bid,
            "clockwise": bool(clockwise),
            "deg": float(deg),
            "reverse_clockwise": bool(rev_clockwise),
            "reverse_choice_text": rev_text,
            "distractors": [
                f"Rotate object {bid} {'clockwise' if clockwise else 'counter-clockwise'} by {deg:g}°",
                f"Move object {bid} {random.choice(MOVE_DIRS)} by {random.choice(MOVE_DISTS_M):g} unit",
                f"Move object {bid} {random.choice(MOVE_DIRS)} by {random.choice(MOVE_DISTS_M):g} unit",
                f"Rotate object {bid} {'clockwise' if random.choice([True, False]) else 'counter-clockwise'} by {random.choice(ANGLE_ROTATE_CHOICES_DEG):g}°",
            ],
        }
        return action

    
    return None



def make_problem_scene_route1(building_objs, road_objs, road_rects, xs, ys, target_type: str | None = None) -> dict | None:
    building_by_id = {int(o.name.split("_")[1]): o for o in building_objs if o.name.startswith("B_")}

    if target_type is not None:
        inject_order = [target_type]
    else:
        inject_order = [ISSUE_OVERLAP, ISSUE_ROAD, ISSUE_ANGLE]
        random.shuffle(inject_order)

    for typ in inject_order:
        snap = {}
        for o in building_objs:
            if o.name.startswith("B_"):
                snap[o.name] = (o.location.copy(), o.rotation_euler.copy(), o.dimensions.copy())

        if typ == ISSUE_OVERLAP:
            action = _attempt_inject_overlap(building_objs, road_objs, road_rects, xs, ys)
        elif typ == ISSUE_ROAD:
            action = _attempt_inject_road_overlap(building_objs, road_objs, road_rects, xs, ys)
        else:
            action = _attempt_inject_angle(building_objs, road_objs, road_rects, xs, ys)

        if action is None:
            for o in building_objs:
                if o.name in snap:
                    loc, rot, dim = snap[o.name]
                    o.location, o.rotation_euler, o.dimensions = loc, rot, dim
            bpy.context.view_layer.update()
            continue

        flags, details = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)
        if (not exactly_one_issue(flags)) or has_any_weak_overlap(details):
            for o in building_objs:
                if o.name in snap:
                    loc, rot, dim = snap[o.name]
                    o.location, o.rotation_euler, o.dimensions = loc, rot, dim
            bpy.context.view_layer.update()
            continue

        # reverse and verify clean (no issue + no weak)
        _reverse_action(action, building_by_id)
        flags2, details2 = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)
        is_clean_again = (not any(flags2.values())) and (not has_any_weak_overlap(details2))

        # re-apply injection to keep problem scene
        # move: ONLY move (do NOT align -> do NOT change rotation)
        # rotate: apply rotation (this is the angle anomaly)
        if action["op"] == "move":
            _apply_move(building_by_id[action["id"]], action["dir"], float(action["dist_m"]))
        else:
            _apply_rotate(building_by_id[action["id"]], clockwise=action["clockwise"], deg=float(action["deg"]))

        bpy.context.view_layer.update()

        if not is_clean_again:
            for o in building_objs:
                if o.name in snap:
                    loc, rot, dim = snap[o.name]
                    o.location, o.rotation_euler, o.dimensions = loc, rot, dim
            bpy.context.view_layer.update()
            continue

        return action

    return None



# ============================================================
# Render
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


# ============================================================
# Sample generation (Route-1)
# ============================================================
def generate_sample(sample_idx: int) -> dict:
    fail = {"place_fail": 0, "clean_not_ok": 0, "inject_fail": 0, "post_not_unique": 0, "weak_overlap": 0}

    for attempt in range(MAX_SCENE_RETRY):
        clear_scene()
        setup_lighting()
        create_dark_ground()

        road_objs, road_rects, xs, ys = build_road_network()

        building_objs: list[bpy.types.Object] = []
        building_meta: list[dict] = []

        # ---------- 1) generate CLEAN scene ----------
        ok_scene = True
        for i in range(1, BUILDINGS_PER_SCENE + 1):
            placed = False
            for _ in range(MAX_PLACE_ATTEMPTS):
                x, y = sample_position()
                obj, meta = create_building(i, x, y)

                if validate_position_with_roads_2d_precheck(x, y, meta, building_meta, road_rects):
                    align_building_to_nearest_road(obj, road_rects)

                    # placement: forbid road overlap (strong/weak)
                    s_r, w_r = detect_overlaps_building_road_2d([obj], road_rects, area_threshold=OVERLAP_AREA_THRESHOLD)
                    if s_r or w_r:
                        remove_object_and_data(obj)
                        continue

                    # placement: forbid building overlap with existing (strong/weak)
                    s_b, w_b = detect_overlaps_buildings_2d(building_objs + [obj], area_threshold=OVERLAP_AREA_THRESHOLD)
                    if s_b or w_b:
                        remove_object_and_data(obj)
                        continue

                    building_objs.append(obj)
                    building_meta.append(meta)
                    placed = True
                    break

                remove_object_and_data(obj)

            if not placed:
                ok_scene = False
                break

        if not ok_scene or len(building_meta) != BUILDINGS_PER_SCENE:
            fail["place_fail"] += 1
            if attempt % 50 == 0:
                print(f"[sample {sample_idx}] attempt={attempt} place_fail={fail['place_fail']}")
            continue

        # strict clean check
        flags_clean, details_clean = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)
        if any(flags_clean.values()):
            fail["clean_not_ok"] += 1
            if attempt % 50 == 0:
                print(f"[sample {sample_idx}] attempt={attempt} clean_not_ok={fail['clean_not_ok']} flags={flags_clean} details={details_clean}")
            continue

        if has_any_weak_overlap(details_clean):
            fail["weak_overlap"] += 1
            if attempt % 50 == 0:
                print(f"[sample {sample_idx}] attempt={attempt} weak_overlap(clean)={fail['weak_overlap']} details={details_clean}")
            continue

        # ---------- 2) setup cameras ----------
        bounds_all = get_scene_bounds(building_objs + road_objs)
        cam_top = setup_camera_top(bounds_all, ortho_scale_factor=TOP_ORTHO_SCALE_FACTOR)
        cam_iso = setup_camera_isometric(bounds_all, ortho_scale_factor=ISO_ORTHO_SCALE_FACTOR)
        bpy.context.view_layer.update()

        # ---------- 3) inject ONE issue ----------
        cycle = [ISSUE_OVERLAP, ISSUE_ROAD, ISSUE_ANGLE]
        target_type = cycle[sample_idx % 3]
        # print(f"[inject_debug] Attempting to inject issue {target_type}")
        action = make_problem_scene_route1(building_objs, road_objs, road_rects, xs, ys, target_type=target_type)
        # if action is None:
        #     print(f"[inject_debug] Injection failed for issue type: {target_type}")

        if action is None:
            fail["inject_fail"] += 1
            if attempt % 50 == 0:
                print(f"[sample {sample_idx}] attempt={attempt} inject_fail={fail['inject_fail']}")
            continue

        # ---------- 4) post-check ----------
        flags, details = evaluate_issue_flags(building_objs, road_objs, road_rects, xs, ys)
        if not exactly_one_issue(flags):
            fail["post_not_unique"] += 1
            if attempt % 50 == 0:
                print(f"[sample {sample_idx}] attempt={attempt} post_not_unique={fail['post_not_unique']} flags={flags} details={details}")
            continue

        # Allow weak overlaps if we have exactly one strong overlap (our intentional issue)
        # This is more lenient than rejecting ANY weak overlap
        bb_strong = details.get("bb_overlap_strong", [])
        road_strong = details.get("road_overlap_strong", [])
        if not (len(bb_strong) > 0 or len(road_strong) > 0):
            if has_any_weak_overlap(details):
                fail["weak_overlap"] += 1
                if attempt % 50 == 0:
                    print(f"[sample {sample_idx}] attempt={attempt} weak_overlap(after inject)={fail['weak_overlap']} details={details}")
                continue

        issue_meta = normalize_issue_meta_from_detection(flags, details)
        building_meta_updated = refresh_building_meta(building_meta, building_objs)

        # ---------- 5) final view fit (after issue injection) ----------
        bpy.context.view_layer.update()
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
        top_arrow_corner = choose_arrow_corner_and_adapt_view(
            cam_top,
            building_objs,
            arrow_size=TOP_NORTH_ARROW_SIZE,
        )
        iso_arrow_corner = choose_arrow_corner_and_adapt_view(
            cam_iso,
            building_objs,
            arrow_size=ISO_NORTH_ARROW_SIZE,
        )
        if top_arrow_corner < 0 or iso_arrow_corner < 0:
            continue
        # Hard constraint: North panel (N + 1 unit bar) must not overlap any object.
        if panel_overlaps_objects(cam_top, building_objs, TOP_NORTH_ARROW_SIZE, top_arrow_corner):
            continue
        if panel_overlaps_objects(cam_iso, building_objs, ISO_NORTH_ARROW_SIZE, iso_arrow_corner):
            continue

        # ---------- 6) output ----------
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            add_label(obj, idx, cam_top)
        bpy.context.view_layer.update()
        top_path = os.path.join(sample_dir, "top.png")
        render_view_with_north(
            cam_top,
            top_path,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=top_arrow_corner,
        )

        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            add_label(obj, idx, cam_iso)
        bpy.context.view_layer.update()
        iso_path = os.path.join(sample_dir, "isometric.png")
        render_view_with_north(
            cam_iso,
            iso_path,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=ISO_NORTH_ARROW_SIZE,
            corner_idx=iso_arrow_corner,
        )

        qa = build_new_qa_formats(sample_dir, issue_meta, action, building_objs, road_objs, road_rects, xs, ys)

        return {
            "sample_id": sample_idx,
            "attempt": attempt,
            "images": {"top": os.path.abspath(top_path), "isometric": os.path.abspath(iso_path)},
            "bounds": bounds_all,
            "roads": road_rects,
            "buildings": building_meta_updated,
            "issue_meta": issue_meta,
            "inject_action": action,
            "issue_detected_debug": details,
            "qa": qa,
            "camera_params": {
                "top": {"ortho_scale": cam_top.data.ortho_scale, "location": tuple(cam_top.location)},
                "isometric": {"ortho_scale": cam_iso.data.ortho_scale, "location": tuple(cam_iso.location)},
            },
            
            "overlap_thresholds": {"overlap_area_threshold": OVERLAP_AREA_THRESHOLD},
            "route": "route1_clean_then_inject_then_reverse_validate_2d_overlap_area",
        }

    raise RuntimeError(
        f"Failed to generate sample {sample_idx} after {MAX_SCENE_RETRY} retries. "
        f"Stats: {fail}. Try: increase SCENE_BOUNDS / reduce MIN_GAP / reduce ROAD_WIDTH / reduce ROAD_AVOID_MARGIN / tune OVERLAP_AREA_THRESHOLD."
    )



# ============================================================
# Main (resume + continue on failure + incremental metadata write)
# ============================================================
def main():
    meta_path = os.path.join(OUTPUT_DIR, "metadata_top_isometric_road_qa.json")
    fail_path = os.path.join(OUTPUT_DIR, "failed_samples.json")

    all_metadata = []
    failed = []

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
        except Exception:
            all_metadata = []

    start_idx = len(all_metadata)
    print(f"[Resume] already have {start_idx} samples in metadata, will start from {start_idx}")

    for s in range(start_idx, NUM_SAMPLES):
        try:
            m = generate_sample(s)
            all_metadata.append(m)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)

            print(f"[OK] sample_{s:03d} written. total={len(all_metadata)}")

        except Exception as e:
            print(f"[FAIL] sample_{s:03d}: {repr(e)}")
            failed.append({"sample_id": s, "error": repr(e)})

            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)

            continue

    print(f"Done! Output: {OUTPUT_DIR}")
    print(f"Metadata: {meta_path}")
    if failed:
        print(f"Failed samples logged at: {fail_path}")


if __name__ == "__main__":
    main()
