#!/usr/bin/env python3
import bpy
import random
import math
import json
import os
import subprocess
from mathutils import Vector

# PIL for drawing North arrow
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


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


def _draw_north_arrow_with_pil(
    image_path: str,
    screen_vec: Vector,
    arrow_color=(255, 0, 0, 255),
    arrow_size=150,
):
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    width, _ = img.size
    margin = max(24, int(arrow_size * 0.26))
    panel_w = int(arrow_size * 1.7)
    panel_h = int(arrow_size * 1.7)

    x1 = width - margin - panel_w
    y1 = margin
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    radius = max(8, int(arrow_size * 0.18))
    panel_fill = (16, 16, 16, 110)
    panel_outline = (245, 245, 245, 120)
    panel_width = max(2, int(arrow_size * 0.03))
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=panel_fill, outline=panel_outline, width=panel_width)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=panel_fill, outline=panel_outline, width=panel_width)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 + arrow_size * 0.11
    arrow_length = arrow_size * 0.9
    arrow_head_size = arrow_size * 0.32
    shaft_width = max(5, int(arrow_size * 0.08))
    outline_width = shaft_width + max(2, int(arrow_size * 0.04))

    tip = Vector((cx, cy)) + screen_vec * (arrow_length / 2.0)
    bottom = Vector((cx, cy)) - screen_vec * (arrow_length / 2.0)

    perp = Vector((-screen_vec.y, screen_vec.x))
    wing_base = tip - screen_vec * arrow_head_size
    left_wing = wing_base + perp * (arrow_head_size * 0.58)
    right_wing = wing_base - perp * (arrow_head_size * 0.58)

    def as_xy(v):
        return (float(v.x), float(v.y))

    
    draw.line([as_xy(bottom), as_xy(tip)], fill=(255, 255, 255, 235), width=outline_width)
    draw.line([as_xy(bottom), as_xy(tip)], fill=arrow_color, width=shaft_width)
    draw.polygon([as_xy(tip), as_xy(left_wing), as_xy(right_wing)], fill=(255, 255, 255, 235))
    inner_left = wing_base + perp * (arrow_head_size * 0.44)
    inner_right = wing_base - perp * (arrow_head_size * 0.44)
    draw.polygon([as_xy(tip), as_xy(inner_left), as_xy(inner_right)], fill=arrow_color)

    
    n_gap = arrow_size * 0.30
    n_center = Vector((cx, cy)) - screen_vec * (arrow_length / 2.0 + n_gap)
    font_size = max(20, int(arrow_size * 0.42))
    font = _load_bold_font(font_size)
    if font:
        
        tx = float(n_center.x) - font_size * 0.34
        ty = float(n_center.y) - font_size * 0.56
        
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + ox, ty + oy), "N", fill=(255, 255, 255, 235), font=font)
        draw.text((tx, ty), "N", fill=arrow_color, font=font)
    else:
        n_size = arrow_size * 0.30
        ux = perp
        uy = screen_vec

        def n_pt(local_x, local_y):
            p = n_center + ux * local_x + uy * local_y
            return (float(p.x), float(p.y))

        half = n_size / 2.0
        left_top = n_pt(-half, half)
        left_bottom = n_pt(-half, -half)
        right_top = n_pt(half, half)
        right_bottom = n_pt(half, -half)

        draw.line([left_top, left_bottom], fill=arrow_color, width=max(4, int(arrow_size * 0.05)))
        draw.line([left_top, right_bottom], fill=arrow_color, width=max(4, int(arrow_size * 0.05)))
        draw.line([right_top, right_bottom], fill=arrow_color, width=max(4, int(arrow_size * 0.05)))

    img.save(image_path)
    return image_path


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
width, _ = img.size

margin = max(16, int(arrow_size * 0.20))
panel_w = int(arrow_size * 1.35)
panel_h = int(arrow_size * 1.35)

height = img.size[1]
alpha = img.getchannel("A")
candidates = [
    (width - margin - panel_w, margin),                 # top-right
    (margin, margin),                                   # top-left
    (width - margin - panel_w, height - margin - panel_h),  # bottom-right
    (margin, height - margin - panel_h),                # bottom-left
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
            print("  [draw_north_arrow] Arrow drawn successfully (external python fallback)")
            return image_path

        
        print(f"  [draw_north_arrow] Warning: external python fallback failed for {image_path}")
        return image_path
    except Exception as e:
        print(f"  [draw_north_arrow] Error drawing arrow on {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return image_path


# ============================================================

# ============================================================

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "100"))
START_SAMPLE_IDX = int(os.environ.get("START_SAMPLE_IDX", "0"))
BUILDINGS_PER_SCENE = 5

SCENE_BOUNDS = 12.0        
MIN_GAP = 2.0              
MAX_PLACE_ATTEMPTS = 500

RESOLUTION = 1080

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.expanduser("~/SpatialAct/benchmark/data/spatial_relation/v1.2"))
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
LABEL_Z_OFFSET  = 0.35   
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

DISTANCE_MARGIN = 3     
MAX_SCENE_RETRY = 200         




if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================

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

# ============================================================

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


def setup_lighting() -> None:
    
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("SunLight", type="SUN"))
    bpy.context.collection.objects.link(sun)
    sun.data.energy = 4.0
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    
    fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLightData", type="AREA"))
    bpy.context.collection.objects.link(fill)
    fill.data.energy = 150.0
    fill.location = (0.0, 0.0, 60.0)
    fill.data.size = 80.0


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
    return plane


# ============================================================

# ============================================================

def get_object_world_bounds(obj: bpy.types.Object) -> dict:
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


def get_scene_bounds(building_objs: list[bpy.types.Object]) -> dict:
    bounds_list = [get_object_world_bounds(o) for o in building_objs if o and o.type == "MESH"]
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


def setup_camera_top(bounds: dict, ortho_scale_factor: float = 1.25) -> bpy.types.Object:
    cx, cy = bounds["center_x"], bounds["center_y"]
    max_dim = max(bounds["width"], bounds["depth"], 2.0)

    cam = create_ortho_camera("Camera_Top")
    cam.location = (cx, cy, bounds["max_z"] + max_dim * 0.65 + 4.5)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.ortho_scale = max_dim * ortho_scale_factor + TOP_CAMERA_EDGE_MARGIN
    return cam


def setup_camera_isometric(bounds: dict, ortho_scale_factor: float = 1.25) -> bpy.types.Object:
    cx, cy, cz = bounds["center_x"], bounds["center_y"], bounds["center_z"]
    base_dim = max(bounds["width"], bounds["depth"], 2.0)
    frame_dim = base_dim + bounds["height"] * ISO_HEIGHT_COMPENSATION

    cam = create_ortho_camera("Camera_Iso")
    cam.location = (cx + frame_dim * 0.42, cy - frame_dim * 0.42, bounds["max_z"] + frame_dim * 0.78)
    direction = Vector((cx - cam.location.x, cy - cam.location.y, cz - cam.location.z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.ortho_scale = frame_dim * ortho_scale_factor + ISO_CAMERA_EDGE_MARGIN
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

    if corner_idx == 0:      # top-right
        x1, x2 = hx - margin_x - panel_w, hx - margin_x
        y1, y2 = hy - margin_y - panel_h, hy - margin_y
    elif corner_idx == 1:    # top-left
        x1, x2 = -hx + margin_x, -hx + margin_x + panel_w
        y1, y2 = hy - margin_y - panel_h, hy - margin_y
    elif corner_idx == 2:    # bottom-right
        x1, x2 = hx - margin_x - panel_w, hx - margin_x
        y1, y2 = -hy + margin_y, -hy + margin_y + panel_h
    else:                    # bottom-left
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


# ============================================================

# ============================================================

def pick_palette_color() -> tuple[float, float, float]:
    return random.choice(COLOR_PALETTE)


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


def add_label(building_obj: bpy.types.Object, idx: int, cam: bpy.types.Object) -> None:
    bnd = get_object_world_bounds(building_obj)

    text_curve = bpy.data.curves.new(f"Label_{idx}_curve", type="FONT")
    text_curve.body = str(idx)
    text_curve.size = LABEL_FONT_SIZE
    text_curve.align_x = "CENTER"
    text_curve.align_y = "CENTER"

    
    text_curve.extrude = 0.03
    text_curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new(f"Label_{idx}", text_curve)
    bpy.context.collection.objects.link(text_obj)

    text_obj.location = (bnd["center_x"], bnd["center_y"], bnd["max_z"] + LABEL_Z_OFFSET)

    
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

    lock_roll.use_limit_x = False
    lock_roll.use_limit_y = False
    lock_roll.use_limit_z = True

    lock_roll.min_z = 0.0
    lock_roll.max_z = 0.0


    mat = create_label_material(f"Label_{idx}_mat")
    text_obj.data.materials.clear()
    text_obj.data.materials.append(mat)

    
    bpy.context.view_layer.update()


# ============================================================

# ============================================================

def create_building(idx: int, x: float, y: float) -> tuple[bpy.types.Object, dict]:
    shape = random.choice(["CUBE", "CUBOID", "CYLINDER", "PRISM", "SPHERE", "L_SHAPE", "U_SHAPE"])
    h = random.uniform(2.0, 10.0)
    w = random.uniform(2.4, 6)

    if shape == "CUBE":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
        obj = bpy.context.active_object
        obj.dimensions = (w, w, h)
        footprint_r = 0.5 * max(obj.dimensions.x, obj.dimensions.y)
        half_w = float(obj.dimensions.x) / 2.0
        half_d = float(obj.dimensions.y) / 2.0

    elif shape == "CUBOID":
        wx = random.uniform(2.0, 6)
        wy = random.uniform(2.0, 6)
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
        
        wx = random.uniform(2.0, 6)
        wy = random.uniform(2.0, 6)
        
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
        
        wx = random.uniform(2.0, 6)
        wy = random.uniform(2.0, 6)
        
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

# ============================================================

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


def _world_dir_8way(theta: float) -> str:
    
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
    return best_dir


def _world_dir_8way_with_margin(theta: float) -> tuple[str, float]:
    
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


def _screen_dir_8way(theta: float) -> str:
    
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
    return best_dir


def _screen_dir_8way_with_margin(theta: float) -> tuple[str, float]:
    
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


# ============================================================

# ============================================================

def get_direction(a: dict, b: dict, view_type: str) -> str:
    
    dx = a["pos"][0] - b["pos"][0]
    dy = a["pos"][1] - b["pos"][1]

    
    theta = math.atan2(dy, dx)

    if view_type == "top":
        return _world_dir_8way(theta)
    else:
        return _screen_dir_8way(theta)


def get_direction_with_confidence(a: dict, b: dict, view_type: str, min_margin_deg: float = 10.0) -> tuple[str, float, bool]:
    
    dx = a["pos"][0] - b["pos"][0]
    dy = a["pos"][1] - b["pos"][1]

    
    theta = math.atan2(dy, dx)

    if view_type == "top":
        direction, dist_to_boundary = _world_dir_8way_with_margin(theta)
    else:
        direction, dist_to_boundary = _screen_dir_8way_with_margin(theta)

    is_clear = math.degrees(dist_to_boundary) >= min_margin_deg
    return direction, dist_to_boundary, is_clear


def get_buildings_on_side(target_id: int, side: str, all_buildings: list[dict], view_type: str) -> list[int]:
    target = next((b for b in all_buildings if b["id"] == target_id), None)
    if not target:
        return []

    threshold = 0.8
    res = []
    tx, ty = target["pos"]

    for b in all_buildings:
        if b["id"] == target_id:
            continue
        bx, by = b["pos"]
        dx = bx - tx
        dy = by - ty

        if view_type == "top":
            # top-view: north/south/east/west
            if side == "north" and dy > threshold:
                res.append(b["id"])
            elif side == "south" and dy < -threshold:
                res.append(b["id"])
            elif side == "east" and dx > threshold:
                res.append(b["id"])
            elif side == "west" and dx < -threshold:
                res.append(b["id"])
        else:
            # isometric: front/back/left/right (screen-based)
            if side == "left" and dx < -threshold:
                res.append(b["id"])
            elif side == "right" and dx > threshold:
                res.append(b["id"])
            elif side == "front" and dy > threshold:
                res.append(b["id"])
            elif side == "back" and dy < -threshold:
                res.append(b["id"])

    return res



def _direction_pool(view_type: str) -> list[str]:
    
    if view_type == "top":
        singles = ["north", "south", "east", "west"]
        doubles = ["northeast", "northwest", "southeast", "southwest"]
        return singles + doubles
    
    singles = ["front", "back", "left", "right"]
    doubles = ["frontleft", "frontright", "backleft", "backright"]
    return singles + doubles

def _make_mcq_options_task1(correct: str, view_type: str, k: int = 4) -> tuple[list[str], int]:
    pool = _direction_pool(view_type)

    
    
    if correct not in pool:
        
        correct = random.choice(pool)

    
    adjacent_map = {
        # world 8-way
        "northeast": ["north", "east"],
        "northwest": ["north", "west"],
        "southeast": ["south", "east"],
        "southwest": ["south", "west"],
        # screen 8-way
        "frontright": ["front", "right"],
        "frontleft": ["front", "left"],
        "backright": ["back", "right"],
        "backleft": ["back", "left"],
    }

    
    adjacent_to_exclude = adjacent_map.get(correct, [])

    
    distractors = [d for d in pool if d != correct and d not in adjacent_to_exclude]

    
    if len(distractors) < k - 1:
        distractors = [d for d in pool if d != correct]
        if len(distractors) >= k - 1:
            distractors = random.sample(distractors, k - 1)
        else:
            raise ValueError("Not enough distractors to build MCQ.")

    options = [correct] + random.sample(distractors, k - 1)
    random.shuffle(options)
    correct_idx = options.index(correct)
    return options, correct_idx



DIRECTION_ANGLE_MARGIN_DEG = 10.0


def qa_task1(all_buildings: list[dict], view_type: str) -> dict:
    
    ids = [b["id"] for b in all_buildings]

    
    effective_view_type = "top" if view_type == "isometric" else view_type

    
    for _ in range(100):
        a_id, b_id = random.sample(ids, 2)
        a = next(x for x in all_buildings if x["id"] == a_id)
        b = next(x for x in all_buildings if x["id"] == b_id)

        correct_dir, dist_to_boundary, is_clear = get_direction_with_confidence(a, b, effective_view_type, DIRECTION_ANGLE_MARGIN_DEG)

        if is_clear:
            if view_type == "top":
                q_text = (
                    "In this top-view image: top is North, bottom is South, left is West, right is East. "
                    "Direction is determined by comparing the center points (bounding box center) of the objects. "
                    f"Which side is object {a_id} relative to object {b_id}?\n"
                    "Top-view image: <image>"
                )
            else:
                
                q_text = (
                    "In this isometric image, the red arrow points to North. "
                    "Direction is determined by comparing the center points (bounding box center) of the objects. "
                    f"Which side is object {a_id} relative to object {b_id}?\n"
                    "Isometric image: <image>"
                )

            options, correct_idx = _make_mcq_options_task1(correct_dir, effective_view_type, k=4)
            labels = ["A", "B", "C", "D"]
            choices = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

            return {
                "question": q_text + "\n" + "\n".join(choices),
                "choices": choices,
                "answer": labels[correct_idx],
                "answer_text": options[correct_idx],
                "task_type": f"{view_type}_direction"
            }

    
    return None


def _pairwise_center_distances(buildings: list[dict]) -> list[tuple[float, int, int]]:
    """Return sorted list of (dist, id1, id2) by center distance."""
    dists: list[tuple[float, int, int]] = []
    for i in range(len(buildings)):
        for j in range(i + 1, len(buildings)):
            a, b = buildings[i], buildings[j]
            d = math.dist(a["pos"], b["pos"])
            dists.append((d, a["id"], b["id"]))
    dists.sort(key=lambda x: x[0])
    return dists


def _task2_scene_ok(buildings: list[dict]) -> bool:
    """Enforce task2 distance tiers so 'closest pair' is unambiguous and not too close."""
    dists = _pairwise_center_distances(buildings)
    if len(dists) < 2:
        return False
    min_d = dists[0][0]
    second_d = dists[1][0]
    # if min_d < TASK2_MIN_PAIR_DIST:
    #     return False
    if (second_d - min_d) < DISTANCE_MARGIN:
        return False
    return True



def qa_task2(all_buildings: list[dict], view_type: str) -> dict:
    min_d, pair = 1e9, (None, None)
    for i in range(len(all_buildings)):
        for j in range(i + 1, len(all_buildings)):
            d = math.dist(all_buildings[i]["pos"], all_buildings[j]["pos"])
            if d < min_d:
                min_d = d
                pair = (all_buildings[i]["id"], all_buildings[j]["id"])
    a_id, b_id = pair
    if a_id > b_id:
        a_id, b_id = b_id, a_id
    image_tag = "Top-view image: <image>" if view_type == "top" else "Isometric image: <image>"
    return {
        "question": (
            "Which two objects are closest to each other? "
            "Object positions are determined by the center points (bounding box center) of the buildings. "
            "Answer with two numbers joined by 'and' in ascending order (x < y), e.g., '2 and 5'.\n"
            + image_tag
        ),
        "answer": f"{a_id} and {b_id}",
        "task_type": f"{view_type}_closest"
    }


def _task3_target_ok(buildings: list[dict], target_id: int) -> tuple[bool, dict]:
    """
    Return (ok, info).
    ok=True means farthest and second-farthest from target_id have enough margin.
    """
    target = next((b for b in buildings if b["id"] == target_id), None)
    if not target:
        return False, {}

    dlist = []
    for b in buildings:
        if b["id"] == target_id:
            continue
        dlist.append((math.dist(target["pos"], b["pos"]), b["id"]))

    if len(dlist) < 2:
        return False, {}

    dlist.sort(key=lambda x: x[0], reverse=True)
    far_d, far_id = dlist[0]
    second_d, second_id = dlist[1]

    ok = (far_d - second_d) >= DISTANCE_MARGIN
    return ok, {
        "target": target_id,
        "farthest": far_id,
        "farthest_dist": far_d,
        "second_farthest": second_id,
        "second_farthest_dist": second_d,
        "margin": far_d - second_d,
    }


def qa_task3(all_buildings: list[dict], view_type: str) -> dict:
    ids = [b["id"] for b in all_buildings]

    for _ in range(MAX_SCENE_RETRY):
        target_id = random.choice(ids)
        ok, info = _task3_target_ok(all_buildings, target_id)
        if not ok:
            continue

        image_tag = "Top-view image: <image>" if view_type == "top" else "Isometric image: <image>"
        return {
            "question": (
                f"Which object is farthest from object {target_id}? "
                "Object positions are determined by the center points (bounding box center) of the buildings. "
                "Answer with one number.\n"
                + image_tag
            ),
            "answer": f"{info['farthest']}",
            "task_type": f"{view_type}_farthest",
            "meta": info,  
        }

    
    target_id = random.choice(ids)
    target = next(b for b in all_buildings if b["id"] == target_id)
    max_d, far_id = -1.0, None
    for b in all_buildings:
        if b["id"] == target_id:
            continue
        d = math.dist(target["pos"], b["pos"])
        if d > max_d:
            max_d, far_id = d, b["id"]

    image_tag = "Top-view image: <image>" if view_type == "top" else "Isometric image: <image>"
    return {
        "question": (
            f"Which object is farthest from object {target_id}? "
            "Object positions are determined by the center points (bounding box center) of the buildings. "
            "Answer with one number.\n"
            + image_tag
        ),
        "answer": f"{far_id}",
        "task_type": f"{view_type}_farthest"
    }



def qa_task4(all_buildings: list[dict], view_type: str) -> dict:
    target_id = random.choice([b["id"] for b in all_buildings])

    
    side = random.choice(["north", "south", "east", "west"])
    count = len(get_buildings_on_side(target_id, side, all_buildings, view_type="top"))

    if view_type == "top":
        question = (
            "In this top-view image: top is North, bottom is South, left is West, right is East. "
            "Direction is determined by comparing the center points (bounding box center) of the objects. "
            "An object can be on multiple sides at once (e.g., northwest counts as both north and west).\n"
            f"How many objects are on the {side} side of object {target_id}? Answer with one number.\n"
            "Top-view image: <image>"
        )
    else:
        
        question = (
            "In this isometric image, the red arrow points to North. "
            "Direction is determined by comparing the center points (bounding box center) of the objects. "
            "An object can be on multiple sides at once (e.g., northeast counts as both north and east).\n"
            f"How many objects are on the {side} side of object {target_id}? Answer with one number.\n"
            "Isometric image: <image>"
        )

    return {"question": question, "answer": str(count), "task_type": f"{view_type}_side_count"}



def build_qa(all_buildings: list[dict], view_type: str, sample_dir: str) -> list[dict]:
    if view_type == "top":
        qa_items = [
            qa_task1(all_buildings, view_type),
            qa_task2(all_buildings, view_type),
            qa_task3(all_buildings, view_type),
            qa_task4(all_buildings, view_type),
        ]
        
        qa_items = [q for q in qa_items if q is not None]
        return _attach_images_abs(qa_items, sample_dir, "top.png")
    elif view_type == "isometric":
        qa_items = [
            qa_task1(all_buildings, view_type),
            qa_task2(all_buildings, view_type),
            qa_task3(all_buildings, view_type),
            qa_task4(all_buildings, view_type),
        ]
        
        qa_items = [q for q in qa_items if q is not None]
        return _attach_images_abs(qa_items, sample_dir, "isometric.png")



# ============================================================

# ============================================================

def _attach_images_abs(qa_items: list[dict], sample_dir: str, image_filename: str) -> list[dict]:
    """
    Attach images=[ABS_PATH] to each QA item.
    Keeps order stable (single image).
    """
    abs_path = os.path.abspath(os.path.join(sample_dir, image_filename))
    for item in qa_items:
        item["images"] = [abs_path]
    return qa_items

def render_view(cam: bpy.types.Object, output_path: str) -> None:
    scene.camera = cam
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


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
                obj, meta = create_building(i, x, y)
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

        def clear_labels_only():
            for obj in list(bpy.data.objects):
                if obj.name.startswith("Label_"):
                    bpy.data.objects.remove(obj, do_unlink=True)
            for curve in list(bpy.data.curves):
                if curve.name.startswith("Label_"):
                    bpy.data.curves.remove(curve)

        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # --- TOP ---
        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            add_label(obj, idx, cam_top)
        bpy.context.view_layer.update()
        top_path = os.path.join(sample_dir, "top.png")
        render_view(cam_top, top_path)
        draw_north_arrow(
            top_path,
            cam_top,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=top_arrow_corner,
        )

        # --- ISOMETRIC ---
        clear_labels_only()
        for idx, obj in enumerate(building_objs, start=1):
            add_label(obj, idx, cam_iso)
        bpy.context.view_layer.update()
        iso_path = os.path.join(sample_dir, "isometric.png")
        render_view(cam_iso, iso_path)

        
        draw_north_arrow(
            iso_path,
            cam_iso,
            north_world_dir=Vector((0.0, 1.0, 0.0)),
            arrow_size=ISO_NORTH_ARROW_SIZE,
            corner_idx=iso_arrow_corner,
        )

        # metadata
        return {
            "sample_id": sample_idx,
            "attempt": attempt,
            "images": {
                "top": os.path.abspath(top_path),
                "isometric": os.path.abspath(iso_path),
            },
            "bounds": bounds,
            "buildings": building_meta,
            "qa": {
                "top": build_qa(building_meta, "top", sample_dir),
                "isometric": build_qa(building_meta, "isometric", sample_dir),
            },
            "camera_params": {
                "top": {"ortho_scale": cam_top.data.ortho_scale, "location": tuple(cam_top.location)},
                "isometric": {"ortho_scale": cam_iso.data.ortho_scale, "location": tuple(cam_iso.location)},
            },
        }

    raise RuntimeError(
        f"Failed to generate sample {sample_idx} after {MAX_SCENE_RETRY} retries. "
        "Try reducing DISTANCE_MARGIN / enabling fewer constraints / increasing SCENE_BOUNDS."
    )


def main():
    all_metadata = []
    for s in range(START_SAMPLE_IDX, START_SAMPLE_IDX + NUM_SAMPLES):
        all_metadata.append(generate_sample(s))

    meta_path = os.path.join(OUTPUT_DIR, "metadata_top_isometric.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
