#!/usr/bin/env python3


import bpy
import random
import math
import json
import os
import sys
import subprocess


try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, arrow drawing will be skipped")
from mathutils import Vector

# ============================================================
# Config
# ============================================================

NUM_SAMPLES = 100
BUILDINGS_PER_SCENE = 5

SCENE_BOUNDS = 12.0
MIN_GAP = 2.0
MAX_PLACE_ATTEMPTS = 500

RESOLUTION = 1080

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.expanduser("~/SpatialAct/benchmark/data/spatial_orientation/v1.2"))
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "top_isometric")

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

LABEL_FONT_SIZE = 1.75
LABEL_FONT_SIZE_A = 1.75
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
ARROW_CORNER_MAX_STEPS = 24

TOP_NORTH_ARROW_SIZE = 108
ISO_NORTH_ARROW_SIZE = 118

MIN_HEIGHT_DIFF = 1.5  

DISTANCE_MARGIN = 3
MAX_SCENE_RETRY = 200

# Task2 style (all buildings same color)
SAME_COLOR_RGB = (0.65, 0.65, 0.65)

# Extreme selection margin in isometric camera-space (avoid ambiguous rightmost/leftmost/topmost/bottommost)
ISO_EXTREME_AXIS_MARGIN = 0.35

# Task1 (world 8-way) clarity
TASK1_WORLD_ANGLE_MARGIN_DEG = 10.0
TASK1_MIN_WORLD_RADIUS = 0.6

# Task4 (screen 8-way after rotation) clarity
TASK4_ANGLE_MARGIN_DEG = 10.0
TASK4_MIN_SCREEN_RADIUS = 0.4

# Task3 switch
ENABLE_TASK3 = False

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


def _load_bold_font(font_size: int):
    if not PIL_AVAILABLE:
        return None
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


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
arrow_size = int(payload.get("arrow_size", 150))

img = Image.open(image_path).convert("RGBA")
draw = ImageDraw.Draw(img)
width, height = img.size

margin = max(16, int(arrow_size * 0.20))
panel_w = int(arrow_size * 1.35)
panel_h = int(arrow_size * 1.35)

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
cy = y1 + panel_h * 0.40
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
    ty = y1 + panel_h * 0.60
    draw.text((tx, ty), "N", fill=arrow_color, font=font)
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

    try:
        if _draw_north_arrow_via_system_python(
            image_path,
            screen_vec,
            arrow_color=arrow_color,
            arrow_size=arrow_size,
            corner_idx=corner_idx,
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
    sun.data.use_shadow = False  # shadows off

    fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLightData", type="AREA"))
    bpy.context.collection.objects.link(fill)
    fill.data.energy = 150.0
    fill.location = (0.0, 0.0, 60.0)
    fill.data.size = 80.0
    fill.data.use_shadow = False  # shadows off


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

    # Keep ground invisible in final render to preserve transparent background.
    plane.hide_render = True
    if hasattr(plane, "visible_camera"):
        plane.visible_camera = False

    # shadows off (match your reference style)
    plane.visible_shadow = False
    if hasattr(plane, "cycles_visibility"):
        plane.cycles_visibility.camera = False
        plane.cycles_visibility.shadow = False

    return plane


# ============================================================
# Bounds / Cameras
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


def setup_camera_top(bounds: dict, ortho_scale_factor: float = TOP_ORTHO_SCALE_FACTOR) -> bpy.types.Object:
    cx, cy = bounds["center_x"], bounds["center_y"]
    max_dim = max(bounds["width"], bounds["depth"], 2.0)
    cam = create_ortho_camera("Camera_Top")
    cam.location = (cx, cy, bounds["max_z"] + max_dim * 0.65 + 4.5)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.ortho_scale = max_dim * ortho_scale_factor + TOP_CAMERA_EDGE_MARGIN
    return cam


def setup_camera_isometric(
    bounds: dict,
    name: str = "Camera_Iso",
    ortho_scale_factor: float = ISO_ORTHO_SCALE_FACTOR,
) -> bpy.types.Object:
    cx, cy, cz = bounds["center_x"], bounds["center_y"], bounds["center_z"]
    base_dim = max(bounds["width"], bounds["depth"], 2.0)
    frame_dim = base_dim + bounds["height"] * ISO_HEIGHT_COMPENSATION
    cam = create_ortho_camera(name)
    cam.location = (cx + frame_dim * 0.42, cy - frame_dim * 0.42, bounds["max_z"] + frame_dim * 0.78)
    direction = Vector((cx - cam.location.x, cy - cam.location.y, cz - cam.location.z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = frame_dim * ortho_scale_factor + ISO_CAMERA_EDGE_MARGIN
    return cam


# ============================================================
# Colors / Labels (labels already shadow-safe)
# ============================================================


def pick_palette_color() -> tuple[float, float, float]:
    return random.choice(COLOR_PALETTE)


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


def _arrow_panel_pixel_geometry(arrow_size: int) -> tuple[int, int, int]:
    margin_px = max(16, int(arrow_size * 0.20))
    panel_w_px = int(arrow_size * 1.35)
    panel_h_px = int(arrow_size * 1.35)
    return margin_px, panel_w_px, panel_h_px


def _camera_half_extents(cam: bpy.types.Object) -> tuple[float, float]:
    frame = cam.data.view_frame(scene=scene)
    hx = max(abs(v.x) for v in frame)
    hy = max(abs(v.y) for v in frame)
    return float(hx), float(hy)


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
    margin_px, panel_w_px, panel_h_px = _arrow_panel_pixel_geometry(arrow_size)

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
        selected = min(range(4), key=lambda i: areas[i])
    return int(selected)


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


def add_label(building_obj: bpy.types.Object, label_text: str, cam: bpy.types.Object, size: float | None = None) -> None:
    bnd = get_object_world_bounds(building_obj)

    text_curve = bpy.data.curves.new(f"Label_{label_text}_curve", type="FONT")
    text_curve.body = str(label_text)
    text_curve.size = float(size if size is not None else LABEL_FONT_SIZE)
    text_curve.align_x = "CENTER"
    text_curve.align_y = "CENTER"
    text_curve.extrude = 0.02
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new(f"Label_{label_text}", text_curve)
    bpy.context.collection.objects.link(text_obj)
    text_obj.location = (bnd["center_x"], bnd["center_y"], bnd["max_z"] + LABEL_Z_OFFSET)

    # shadows off for text too
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


def set_buildings_same_color(building_objs: list[bpy.types.Object], rgb: tuple[float, float, float]) -> None:
    for obj in building_objs:
        if not obj.data.materials:
            continue
        mat = obj.data.materials[0]
        if not mat or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)


def clear_labels_only() -> None:
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Label_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for curve in list(bpy.data.curves):
        if curve.name.startswith("Label_"):
            bpy.data.curves.remove(curve)


# ============================================================
# Building generation (AABB + MIN_GAP) + shadows OFF
# ============================================================


def create_building(idx: int, x: float, y: float, height: float) -> tuple[bpy.types.Object, dict]:
    shape = random.choice(["CUBE", "CUBOID", "CYLINDER", "PRISM", "SPHERE", "L_SHAPE", "U_SHAPE"])
    h = height
    w = random.uniform(2.4, 6.0)

    if shape == "CUBE":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (w, w, h)
        footprint_r = 0.5 * max(obj.dimensions.x, obj.dimensions.y)
        half_w = float(obj.dimensions.x) / 2.0
        half_d = float(obj.dimensions.y) / 2.0

    elif shape == "CUBOID":
        wx = random.uniform(2.0, 6.0)
        wy = random.uniform(2.0, 6.0)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (wx, wy, h)
        footprint_r = 0.5 * max(obj.dimensions.x, obj.dimensions.y)
        half_w = float(obj.dimensions.x) / 2.0
        half_d = float(obj.dimensions.y) / 2.0

    elif shape == "CYLINDER":
        bpy.ops.mesh.primitive_cylinder_add(radius=w / 2, depth=h, location=(x, y, h / 2))
        obj = bpy.context.active_object
        footprint_r = w / 2
        half_w = float(footprint_r)
        half_d = float(footprint_r)

    elif shape == "PRISM":
        bpy.ops.mesh.primitive_cylinder_add(radius=w / 2, depth=h, vertices=3, location=(x, y, h / 2))
        obj = bpy.context.active_object
        footprint_r = w / 2
        half_w = float(footprint_r)
        half_d = float(footprint_r)

    elif shape == "L_SHAPE":
        wx = random.uniform(2.0, 6.0)
        wy = random.uniform(2.0, 6.0)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj1 = bpy.context.active_object
        obj1.dimensions = (wx, wy * 0.4, h)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + wx * 0.3, y + wy * 0.3, h / 2))
        obj2 = bpy.context.active_object
        obj2.dimensions = (wx * 0.4, wy, h)
        obj1.select_set(True)
        bpy.context.view_layer.objects.active = obj1
        bpy.ops.object.join()
        obj = bpy.context.active_object
        half_w = wx / 2 + wx * 0.3
        half_d = wy / 2 + wy * 0.3
        footprint_r = max(half_w, half_d)

    elif shape == "U_SHAPE":
        wx = random.uniform(2.0, 6.0)
        wy = random.uniform(2.0, 6.0)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x - wx * 0.35, y, h / 2))
        left = bpy.context.active_object
        left.dimensions = (wx * 0.3, wy, h)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x + wx * 0.35, y, h / 2))
        right = bpy.context.active_object
        right.dimensions = (wx * 0.3, wy, h)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y - wy * 0.35, h / 2))
        bottom = bpy.context.active_object
        bottom.dimensions = (wx, wy * 0.3, h)
        left.select_set(True)
        right.select_set(True)
        bottom.select_set(True)
        bpy.context.view_layer.objects.active = left
        bpy.ops.object.join()
        obj = bpy.context.active_object
        half_w = wx / 2 + wx * 0.35
        half_d = wy / 2 + wy * 0.35
        footprint_r = max(half_w, half_d)

    else:  # SPHERE
        bpy.ops.mesh.primitive_uv_sphere_add(radius=w / 2, location=(x, y, w / 2))
        obj = bpy.context.active_object
        h = w
        footprint_r = w / 2
        half_w = float(footprint_r)
        half_d = float(footprint_r)

    obj.name = f"B_{idx}"

    rgb = pick_palette_color()
    mat = bpy.data.materials.new(f"Mat_{idx}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # shadows off (match your reference style)
    obj.visible_shadow = False
    if hasattr(obj, "cycles_visibility"):
        obj.cycles_visibility.shadow = False

    meta = {
        "id": idx,
        "pos": (x, y),
        "height": float(h),
        "shape": shape,
        "color": rgb,
        "footprint_r": float(footprint_r),
        "half_w": float(half_w),
        "half_d": float(half_d),
    }
    return obj, meta


def sample_position(existing: list[dict]) -> tuple[float, float]:
    MAX_HALF = 2.2
    for _ in range(MAX_PLACE_ATTEMPTS):
        x = random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)
        y = random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)

        ok = True
        for b in existing:
            dx = abs(x - b["pos"][0])
            dy = abs(y - b["pos"][1])
            if dx < (b["half_w"] + MAX_HALF + MIN_GAP) and dy < (b["half_d"] + MAX_HALF + MIN_GAP):
                ok = False
                break
        if ok:
            return x, y

    return random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS), random.uniform(-SCENE_BOUNDS, SCENE_BOUNDS)


def validate_position(x: float, y: float, new_meta: dict, existing: list[dict]) -> bool:
    new_hw = new_meta["half_w"]
    new_hd = new_meta["half_d"]

    for b in existing:
        dx = abs(x - b["pos"][0])
        dy = abs(y - b["pos"][1])
        min_dx = b["half_w"] + new_hw + MIN_GAP
        min_dy = b["half_d"] + new_hd + MIN_GAP
        if dx < min_dx and dy < min_dy:
            return False
    return True


# ============================================================
# Scene filters
# ============================================================


def _pairwise_center_distances(buildings: list[dict]) -> list[tuple[float, int, int]]:
    dists: list[tuple[float, int, int]] = []
    for i in range(len(buildings)):
        for j in range(i + 1, len(buildings)):
            a, b = buildings[i], buildings[j]
            d = math.dist(a["pos"], b["pos"])
            dists.append((d, a["id"], b["id"]))
    dists.sort(key=lambda x: x[0])
    return dists


def _task2_scene_ok(buildings: list[dict]) -> bool:
    dists = _pairwise_center_distances(buildings)
    if len(dists) < 2:
        return False
    return (dists[1][0] - dists[0][0]) >= DISTANCE_MARGIN


# ============================================================
# Isometric QA helpers
# ============================================================


def _building_world_point(b: dict) -> Vector:
    x, y = b["pos"]
    return Vector((float(x), float(y), 0.0))


def _scene_centroid_xy(buildings: list[dict]) -> Vector:
    cx = sum(b["pos"][0] for b in buildings) / len(buildings)
    cy = sum(b["pos"][1] for b in buildings) / len(buildings)
    return Vector((float(cx), float(cy), 0.0))


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
        scored.append((key, int(b["id"])))

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
        scored.append((key, int(b["id"])))

    scored.sort(key=lambda t: t[0])

    if extreme_type in ("northmost", "eastmost"):
        best = scored[-1]
        second = scored[-2]
        return best[1], float(best[0] - second[0])

    best = scored[0]
    second = scored[1]
    return best[1], float(second[0] - best[0])


def _world_extreme_candidates(buildings: list[dict]) -> list[tuple[str, int, float]]:
    cands: list[tuple[str, int, float]] = []
    for extreme_type in ("northmost", "southmost", "eastmost", "westmost"):
        bid, margin = _extreme_building_world_coord(buildings, extreme_type)
        cands.append((extreme_type, bid, float(margin)))
    return cands


def _pick_task1_extreme_for_qa(buildings: list[dict]) -> tuple[str, int, float] | None:
    centroid = _scene_centroid_xy(buildings)
    cands = _world_extreme_candidates(buildings)
    random.shuffle(cands)

    for extreme_type, bid, margin in cands:
        if margin < ISO_EXTREME_AXIS_MARGIN:
            continue

        b = next(x for x in buildings if x["id"] == bid)
        v = _building_world_point(b) - centroid
        if v.length < TASK1_MIN_WORLD_RADIUS:
            continue

        theta = math.atan2(v.y, v.x)
        _ans, dist_to_boundary = _world_dir_8way(theta)
        if dist_to_boundary >= math.radians(TASK1_WORLD_ANGLE_MARGIN_DEG):
            return extreme_type, bid, margin
    return None


def _pick_task4_extreme_for_qa(buildings: list[dict]) -> tuple[str, int, float] | None:
    centroid = _scene_centroid_xy(buildings)
    cands = _world_extreme_candidates(buildings)
    random.shuffle(cands)

    for extreme_type, bid, margin in cands:
        if margin < ISO_EXTREME_AXIS_MARGIN:
            continue

        b = next(x for x in buildings if x["id"] == bid)
        v = _building_world_point(b) - centroid
        v2 = Vector((float(v.x), float(v.y)))

        if v2.length < TASK4_MIN_SCREEN_RADIUS:
            continue

        ok = True
        for rot_dir in ("cw90", "ccw90"):
            if rot_dir == "cw90":
                new_north = Vector((1.0, 0.0))
                new_east = Vector((0.0, -1.0))
            else:
                new_north = Vector((-1.0, 0.0))
                new_east = Vector((0.0, 1.0))

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
            return extreme_type, bid, margin
    return None


def _rotate_screen_xy(x: float, y: float, rot_dir: str) -> tuple[float, float]:
    # screen: x=right, y=front(up)
    
    if rot_dir == "cw90":
        
        return (-y, x)
    if rot_dir == "ccw90":
        
        return (y, -x)
    raise ValueError(rot_dir)


def _task1_ok(buildings: list[dict], cam: bpy.types.Object) -> bool:
    _ = cam
    return _pick_task1_extreme_for_qa(buildings) is not None


def _task4_ok(buildings: list[dict], cam: bpy.types.Object) -> bool:
    _ = cam
    return _pick_task4_extreme_for_qa(buildings) is not None


# ============================================================
# Isometric QA tasks
# ============================================================


def qa_iso_task1_extreme_building_top_direction_mcq(
    all_buildings: list[dict],
    cam_iso: bpy.types.Object,
    extreme_pick: tuple[str, int, float] | None = None,
) -> dict:
    _ = cam_iso
    if extreme_pick is None:
        picked = _pick_task1_extreme_for_qa(all_buildings)
        if picked is None:
            raise ValueError("No valid task1 extreme with sufficient margin to second extreme.")
        extreme_type, target_id, margin = picked
    else:
        extreme_type, target_id, margin = extreme_pick

    centroid = _scene_centroid_xy(all_buildings)
    b = next(x for x in all_buildings if x["id"] == target_id)
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
        "- Object directions are determined by comparing object center points (bounding box centers).\n\n"
        f"Consider {extreme_type_desc} object in the isometric image. "
        "Relative to the scene center, which direction is that object located in the top-view image? "
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
        "meta": {"extreme_type": extreme_type, "target_building_id": target_id, "extreme_margin": float(margin)},
    }


def qa_iso_task2_buildingA_maps_to_top_id(buildingA_id: int) -> dict:
    q = (
        "There are two images: an isometric image where only one object is labeled 'A', and a top-view image with numbered objects. Which numbered object is 'A' in the top-view image? "
        "Answer with one number.\n\n"
        f"Isometric image: <image>\ntop-view image: <image>"
    )
    return {
        "task_type": "top_isometric_A_consistent",
        "question": q,
        "answer": str(buildingA_id),
        "images": ["isometric_A.png", "top.png"],
        "meta": {"buildingA_id": buildingA_id},
    }


def qa_iso_task3_camera_facing_cardinal(cam_iso: bpy.types.Object) -> dict:
    fwd = _camera_forward_world(cam_iso)
    fwd_xy = Vector((fwd.x, fwd.y, 0.0))
    if fwd_xy.length < 1e-6:
        ans = "north"
    else:
        fwd_xy.normalize()
        if abs(fwd_xy.x) >= abs(fwd_xy.y):
            ans = "east" if fwd_xy.x > 0 else "west"
        else:
            ans = "north" if fwd_xy.y > 0 else "south"

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
    rot_dir: str,  # 'cw90' or 'ccw90'
    extreme_pick: tuple[str, int, float] | None = None,
) -> dict:
    _ = cam_iso
    if extreme_pick is None:
        picked = _pick_task4_extreme_for_qa(all_buildings)
        if picked is None:
            raise ValueError("No valid task4 extreme with sufficient margin to second extreme.")
        extreme_type, target_id, margin = picked
    else:
        extreme_type, target_id, margin = extreme_pick

    centroid = _scene_centroid_xy(all_buildings)
    b = next(x for x in all_buildings if x["id"] == target_id)

    
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
    correct, _dist_to_boundary = _world_dir_8way(relative_angle)

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
Identify {extreme_type_desc} object in the image.

Now imagine that the camera rotates {rot_text} by 90 degrees around the scene center.
When the camera rotates, the cardinal directions (north, east, south, west) rotate together with the view.

After this rotation, where will that object be located relative to the scene center in the rotated coordinate frame?

Choose one option.

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
            "extreme_margin": float(margin),
            "rotation": rot_dir,
            "x_new": x_new,
            "y_new": y_new,
            "relative_angle_deg": math.degrees(relative_angle),
        },
    }


def build_qa_isometric(
    all_buildings: list[dict],
    cam_iso: bpy.types.Object,
    buildingA_id: int,
    task1_extreme_pick: tuple[str, int, float],
    task4_extreme_pick: tuple[str, int, float],
) -> list[dict]:
    qa_top_isometric = [
        qa_iso_task1_extreme_building_top_direction_mcq(all_buildings, cam_iso, extreme_pick=task1_extreme_pick),
        qa_iso_task2_buildingA_maps_to_top_id(buildingA_id),
    ]
    qa_isometric = []
    if ENABLE_TASK3:
        qa_isometric.append(qa_iso_task3_camera_facing_cardinal(cam_iso))
    qa_isometric.append(
        qa_iso_task4_extreme_building_after_rot90_mcq(
            all_buildings, cam_iso, "cw90", extreme_pick=task4_extreme_pick
        )
    )
    qa_isometric.append(
        qa_iso_task4_extreme_building_after_rot90_mcq(
            all_buildings, cam_iso, "ccw90", extreme_pick=task4_extreme_pick
        )
    )
    return {
        "top_isometric": qa_top_isometric,
        "isometric": qa_isometric,
    }


# ============================================================
# Rendering
# ============================================================

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


def render_view(cam: bpy.types.Object, output_path: str) -> None:
    scene.camera = cam
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    _make_black_background_transparent(output_path, black_thresh=2)


# ============================================================
# Sample generation
# ============================================================


def generate_sample(sample_idx: int) -> dict:
    for attempt in range(MAX_SCENE_RETRY):
        clear_scene()
        setup_lighting()

        building_objs: list[bpy.types.Object] = []
        building_meta: list[dict] = []

        ok_scene = True
        for i in range(1, BUILDINGS_PER_SCENE + 1):
            x, y = sample_position(building_meta)

            placed = False
            for _ in range(MAX_PLACE_ATTEMPTS):
                used_heights = [b["height"] for b in building_meta]
                h = random.uniform(2.0, 10.0)
                for _ in range(50):
                    if all(abs(h - uh) >= MIN_HEIGHT_DIFF for uh in used_heights):
                        break
                    h = random.uniform(2.0, 10.0)

                obj, meta = create_building(i, x, y, h)
                if validate_position(x, y, meta, building_meta):
                    building_objs.append(obj)
                    building_meta.append(meta)
                    placed = True
                    break

                bpy.data.objects.remove(obj, do_unlink=True)
                x, y = sample_position(building_meta)

            if not placed:
                ok_scene = False
                break

        if (not ok_scene) or len(building_meta) != BUILDINGS_PER_SCENE:
            continue

        if not _task2_scene_ok(building_meta):
            continue

        bounds = get_scene_bounds(building_objs)
        cam_top = setup_camera_top(bounds, ortho_scale_factor=TOP_ORTHO_SCALE_FACTOR)
        cam_iso = setup_camera_isometric(bounds, ortho_scale_factor=ISO_ORTHO_SCALE_FACTOR)
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

        # QA clarity filters
        if not _task1_ok(building_meta, cam_iso):
            continue
        if not _task4_ok(building_meta, cam_iso):
            continue
        task1_extreme_pick = _pick_task1_extreme_for_qa(building_meta)
        if task1_extreme_pick is None:
            continue
        task4_extreme_pick = _pick_task4_extreme_for_qa(building_meta)
        if task4_extreme_pick is None:
            continue

        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # --- TOP (numeric labels) ---
        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            add_label(obj, str(idx), cam_top)
        bpy.context.view_layer.update()
        top_path = os.path.join(sample_dir, "top.png")
        render_view(cam_top, top_path)
        draw_north_arrow(
            top_path,
            cam=cam_top,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=top_arrow_corner,
        )

        # --- ISOMETRIC (numeric labels) ---
        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            add_label(obj, str(idx), cam_iso)
        bpy.context.view_layer.update()
        iso_path = os.path.join(sample_dir, "isometric.png")
        render_view(cam_iso, iso_path)
        draw_north_arrow(
            iso_path,
            cam=cam_iso,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=ISO_NORTH_ARROW_SIZE,
            corner_idx=iso_arrow_corner,
        )

        
        cx, cy = bounds["center_x"], bounds["center_y"]
        max_dim = max(bounds["width"], bounds["depth"], 2.0)

        
        cam_cw90 = setup_camera_isometric(bounds, 'Camera_CW90')
        cam_cw90.rotation_euler = (cam_iso.rotation_euler[0],
                                    cam_iso.rotation_euler[1],
                                    cam_iso.rotation_euler[2] - math.radians(90))
        cam_cw90.location = (cx + max_dim * 0.35, cy - max_dim * 0.35, bounds["max_z"] + max_dim * 0.9)
        fit_ortho_camera_to_objects(
            cam_cw90,
            building_objs,
            margin_ratio=ISO_FIT_MARGIN_RATIO,
            min_ortho_scale=MIN_FIT_ORTHO_SCALE,
        )
        cw90_arrow_corner = choose_arrow_corner_and_adapt_view(
            cam_cw90,
            building_objs,
            arrow_size=ISO_NORTH_ARROW_SIZE,
        )
        force_disable_all_shadows()
        rotated_cw90_path = os.path.join(sample_dir, "isometric_rotated_cw90.png")
        render_view(cam_cw90, rotated_cw90_path)
        draw_north_arrow(
            rotated_cw90_path,
            cam=cam_cw90,
            north_world_dir=Vector((1.0, 0.0, 0.0)),
            arrow_size=ISO_NORTH_ARROW_SIZE,
            corner_idx=cw90_arrow_corner,
        )

        
        cam_ccw90 = setup_camera_isometric(bounds, 'Camera_CCW90')
        cam_ccw90.rotation_euler = (cam_iso.rotation_euler[0],
                                     cam_iso.rotation_euler[1],
                                     cam_iso.rotation_euler[2] + math.radians(90))
        cam_ccw90.location = (cx - max_dim * 0.35, cy + max_dim * 0.35, bounds["max_z"] + max_dim * 0.9)
        fit_ortho_camera_to_objects(
            cam_ccw90,
            building_objs,
            margin_ratio=ISO_FIT_MARGIN_RATIO,
            min_ortho_scale=MIN_FIT_ORTHO_SCALE,
        )
        ccw90_arrow_corner = choose_arrow_corner_and_adapt_view(
            cam_ccw90,
            building_objs,
            arrow_size=ISO_NORTH_ARROW_SIZE,
        )
        force_disable_all_shadows()
        rotated_ccw90_path = os.path.join(sample_dir, "isometric_rotated_ccw90.png")
        render_view(cam_ccw90, rotated_ccw90_path)
        draw_north_arrow(
            rotated_ccw90_path,
            cam=cam_ccw90,
            north_world_dir=Vector((-1.0, 0.0, 0.0)),
            arrow_size=ISO_NORTH_ARROW_SIZE,
            corner_idx=ccw90_arrow_corner,
        )


        # --- ISOMETRIC_A (all same color, ONLY label A) ---
        buildingA_id = random.randint(1, BUILDINGS_PER_SCENE)
        set_buildings_same_color(building_objs, SAME_COLOR_RGB)
        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            if idx == buildingA_id:
                add_label(obj, "A", cam_iso, size = LABEL_FONT_SIZE_A)
        bpy.context.view_layer.update()
        iso_a_path = os.path.join(sample_dir, "isometric_A.png")
        render_view(cam_iso, iso_a_path)
        draw_north_arrow(
            iso_a_path,
            cam=cam_iso,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=ISO_NORTH_ARROW_SIZE,
            corner_idx=iso_arrow_corner,
        )

        qa_groups = build_qa_isometric(
            building_meta,
            cam_iso,
            buildingA_id,
            task1_extreme_pick=task1_extreme_pick,
            task4_extreme_pick=task4_extreme_pick,
        )

        # Convert QA images to absolute paths (keeps order unchanged)
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
            "sample_id": sample_idx,
            "attempt": attempt,
            "images": {
                "top": os.path.abspath(top_path),
                "isometric": os.path.abspath(iso_path),
                "isometric_A": os.path.abspath(iso_a_path),
            },
            "bounds": bounds,
            "buildings": building_meta,
            "qa": {
                "top_isometric": qa_groups["top_isometric"],
                "isometric": qa_groups["isometric"],
            },
            "special_refs": {
                "buildingA_id": buildingA_id,
                "task1_extreme_pick": {
                    "extreme_type": task1_extreme_pick[0],
                    "target_building_id": int(task1_extreme_pick[1]),
                    "margin_vs_second": float(task1_extreme_pick[2]),
                },
                "task4_extreme_pick": {
                    "extreme_type": task4_extreme_pick[0],
                    "target_building_id": int(task4_extreme_pick[1]),
                    "margin_vs_second": float(task4_extreme_pick[2]),
                },
                "enable_task3": ENABLE_TASK3,
                "iso_extreme_axis_margin": ISO_EXTREME_AXIS_MARGIN,
                "task1_world_angle_margin_deg": TASK1_WORLD_ANGLE_MARGIN_DEG,
                "task4_angle_margin_deg": TASK4_ANGLE_MARGIN_DEG,
            },
            "camera_params": {
                "top": {"ortho_scale": cam_top.data.ortho_scale, "location": tuple(cam_top.location)},
                "isometric": {"ortho_scale": cam_iso.data.ortho_scale, "location": tuple(cam_iso.location)},
            },
        }

    raise RuntimeError(
        f"Failed to generate sample {sample_idx} after {MAX_SCENE_RETRY} retries. "
        "Try reducing margins or increasing SCENE_BOUNDS."
    )


def main() -> None:
    all_metadata = []
    for s in range(NUM_SAMPLES):
        all_metadata.append(generate_sample(s))

    meta_path = os.path.join(OUTPUT_DIR, "metadata_top_isometric.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
