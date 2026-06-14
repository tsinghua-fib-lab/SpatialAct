#!/usr/bin/env python3
"""
 Blender  src/  blender_scripts/


- BLENDER_REGION_DIR:  labels.json
- BLENDER_OUTPUT_DIR: iter_N  region
- BLENDER_CUMULATIVE_PATH: JSON,  consolidated_actions
- BLENDER_INPUT_BLEND: blend
- GLTF_PATH:  blend  gLTF
- BLENDER_OUTPUT_TOP:  top  top.png
- BLENDER_OUTPUT_ISO:  isometric  isometric.png
- BLENDER_OUTPUT_BLEND:  blend 
"""

import json
import importlib.util
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector, Matrix

PROJECT_ROOT = "SpatialAct"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.building_labels import BuildingLabeler, apply_road_material, create_road_material

SCALE_MARK_LENGTH_M = 20.0
SCALE_BAR_THICKNESS = 2
SCALE_TICK_THICKNESS = 1.0
SCALE_TICK_LEN_Y = 3.5
SCALE_MARK_MARGIN = 1.2
SCALE_MARK_Z_EPS = 0.5
SCALE_LABEL_SIZE = 10.0
SCALE_LABEL_Z_OFF = 2.0
LABEL_SIZE_RATIO = 0.06
TOP_FIT_MARGIN_RATIO = float(os.environ.get("TOP_FIT_MARGIN_RATIO", "0.40"))
ISO_FIT_MARGIN_RATIO = float(os.environ.get("ISO_FIT_MARGIN_RATIO", "0.40"))
MIN_FIT_ORTHO_SCALE = float(os.environ.get("MIN_FIT_ORTHO_SCALE", "2.2"))
REFRAME_PAD_RATIO = float(os.environ.get("REFRAME_PAD_RATIO", "0.45"))
REFRAME_CONTENT_FILL_RATIO = float(os.environ.get("REFRAME_CONTENT_FILL_RATIO", "1.00"))
FINAL_TRIM_PAD_PX = int(os.environ.get("FINAL_TRIM_PAD_PX", "180"))
WHITE_FILM_ALPHA = 0.92
TOP_NORTH_ARROW_SIZE = int(os.environ.get("TOP_NORTH_ARROW_SIZE", "96"))
FIXED_ISOMETRIC_MODE = "isometric_north_ur"
INSTANCE_NAME_RE = re.compile(r"^\d+_")
_INDOOR_BASE = None
_MOVE_NORTH_WORLD_XY: tuple[float, float] = (0.0, 1.0)
_MOVE_EAST_WORLD_XY: tuple[float, float] = (1.0, 0.0)


def setup_render(res_x: int = 1920, res_y: int = 1080):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True


def force_disable_all_shadows_and_world():
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


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _vec3_to_list(v):
    return [float(v.x), float(v.y), float(v.z)]


def _serialize_render_ctx(render_ctx: dict) -> dict:
    return {
        "center": _vec3_to_list(render_ctx["center"]),
        "radius": float(render_ctx["radius"]),
        "extent": _vec3_to_list(render_ctx["extent"]),
        "target_points": [_vec3_to_list(p) for p in (render_ctx.get("target_points") or [])],
        "up_vec": _vec3_to_list(render_ctx["up_vec"]),
        "h1_vec": _vec3_to_list(render_ctx["h1_vec"]),
        "h2_vec": _vec3_to_list(render_ctx["h2_vec"]),
    }


def _deserialize_render_ctx(base_mod, payload: dict) -> dict:
    VectorCls = base_mod.Vector
    return {
        "center": VectorCls(tuple(payload["center"])),
        "radius": float(payload["radius"]),
        "extent": VectorCls(tuple(payload["extent"])),
        "target_points": [VectorCls(tuple(p)) for p in (payload.get("target_points") or [])],
        "up_vec": VectorCls(tuple(payload["up_vec"])),
        "h1_vec": VectorCls(tuple(payload["h1_vec"])),
        "h2_vec": VectorCls(tuple(payload["h2_vec"])),
    }


def _clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.cameras:
        if block.users == 0:
            bpy.data.cameras.remove(block)


def _is_road(name: str) -> bool:
    token = name.lower()
    return ("road" in token) or ("path" in token)


def clean_get_region_bounds(building_objs, buffer=10.0):
    if not building_objs:
        return {'min_x': -1e6, 'max_x': 1e6, 'min_y': -1e6, 'max_y': 1e6}

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for obj in building_objs:
        if obj is None:
            continue
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


def clean_is_face_in_bounds(face_verts, bounds, buffer=5.0):
    if not face_verts:
        return False
    cx = sum(v[0] for v in face_verts) / len(face_verts)
    cy = sum(v[1] for v in face_verts) / len(face_verts)
    return (bounds['min_x'] - buffer <= cx <= bounds['max_x'] + buffer and
            bounds['min_y'] - buffer <= cy <= bounds['max_y'] + buffer)


def clean_extract_valid_road_faces(road_obj, region_bounds, buffer=5.0):
    valid_faces = []
    if road_obj.type != 'MESH':
        return []
    mesh = road_obj.data
    mw = road_obj.matrix_world
    world_verts = [mw @ v.co for v in mesh.vertices]
    for poly in mesh.polygons:
        face_vs = [world_verts[i] for i in poly.vertices]
        coords_2d = [(v.x, v.y) for v in face_vs]
        if clean_is_face_in_bounds(coords_2d, region_bounds, buffer):
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

        valid_items = clean_extract_valid_road_faces(r_obj, bounds, buffer=5.0)

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
                try:
                    bpy.data.meshes.remove(mesh)
                except Exception:
                    pass
        except Exception:
            pass

    for obj_name, (hr, hv) in hidden_states.items():
        obj = bpy.data.objects.get(obj_name)
        if obj:
            try:
                obj.hide_render = hr
                obj.hide_viewport = hv
            except Exception:
                pass


def _get_bounds(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    return {
        "min_x": min(xs), "max_x": max(xs),
        "min_y": min(ys), "max_y": max(ys),
        "min_z": min(zs), "max_z": max(zs),
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "center_z": (min(zs) + max(zs)) / 2,
    }


def _iter_object_world_vertices(obj: bpy.types.Object):
    if obj is None or obj.type != "MESH" or obj.data is None:
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


def snapshot_building_material_slots(building_ids: list[str]) -> dict[str, list]:
    snapshot = {}
    for bid in building_ids:
        obj = bpy.data.objects.get(bid)
        if not obj or obj.type != "MESH" or not hasattr(obj.data, "materials"):
            continue
        snapshot[bid] = [m for m in obj.data.materials]
    return snapshot


def restore_building_material_slots(snapshot: dict[str, list]) -> int:
    restored = 0
    for bid, mats in (snapshot or {}).items():
        obj = bpy.data.objects.get(bid)
        if not obj or obj.type != "MESH" or not hasattr(obj.data, "materials"):
            continue
        obj.data.materials.clear()
        for mat in mats:
            if mat is not None:
                obj.data.materials.append(mat)
        restored += 1
    return restored


def _render(cam, out_path):
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


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


def merge_bbox_pixels(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    )


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
        print(f"[reframe] Warning: reframing failed for {image_path}: {e}")
        return False


def trim_image_to_alpha_bbox(image_path: str, pad_px: int = 0):
    if not os.path.exists(image_path):
        return False
    payload = {"image_path": image_path, "pad_px": int(max(0, pad_px))}
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
        print(f"[trim] Warning: trim failed for {image_path}: {e}")
        return False


def render_view(
    cam: bpy.types.Object,
    output_path: str,
    add_north: bool = True,
    north_world_dir: Vector = Vector((0.0, 1.0, 0.0)),
    add_unit_bar: bool = False,
    building_bbox: tuple[int, int, int, int] | None = None,
    reframe: bool = True,
    trim_to_alpha: bool = True,
) -> None:
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    if add_north:
        if reframe:
            reframe_image_focus_buildings(
                output_path,
                arrow_size=TOP_NORTH_ARROW_SIZE,
                reserve_for_north=True,
                building_bbox=building_bbox,
            )
        draw_north_arrow(
            output_path,
            cam=cam,
            north_world_dir=north_world_dir,
            arrow_size=TOP_NORTH_ARROW_SIZE,
            corner_idx=None,
            add_unit_bar=add_unit_bar,
        )
        if trim_to_alpha:
            trim_image_to_alpha_bbox(output_path, pad_px=FINAL_TRIM_PAD_PX)


def world_bounds_from_obj(obj: bpy.types.Object) -> dict:
    mw = obj.matrix_world
    if obj.type == "MESH" and obj.data and hasattr(obj.data, "vertices") and obj.data.vertices:
        verts = [mw @ Vector(v.co) for v in obj.data.vertices]
    elif hasattr(obj, "bound_box") and obj.bound_box:
        verts = [mw @ Vector(co) for co in obj.bound_box]
    else:
        return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0, "center_x": 0, "center_y": 0}

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
    }


def compute_region_bounds_from_ids(building_ids: list[str], pad_ratio: float = 0.15) -> dict | None:
    bounds_list = []
    for bid in building_ids:
        obj = bpy.data.objects.get(bid)
        if not obj or obj.type != "MESH":
            continue
        bounds_list.append(world_bounds_from_obj(obj))
    if not bounds_list:
        return None

    min_x = min(b["min_x"] for b in bounds_list)
    max_x = max(b["max_x"] for b in bounds_list)
    min_y = min(b["min_y"] for b in bounds_list)
    max_y = max(b["max_y"] for b in bounds_list)
    min_z = min(b["min_z"] for b in bounds_list)
    max_z = max(b["max_z"] for b in bounds_list)

    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z
    pad = max(width, depth) * max(0.0, float(pad_ratio))

    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad
    width = max_x - min_x
    depth = max_y - min_y

    return {
        "min_x": min_x, "max_x": max_x,
        "min_y": min_y, "max_y": max_y,
        "min_z": min_z, "max_z": max_z,
        "center_x": (min_x + max_x) * 0.5,
        "center_y": (min_y + max_y) * 0.5,
        "center_z": (min_z + max_z) * 0.5,
        "width": width,
        "depth": depth,
        "height": height,
    }


def create_label_material(name: str, strength: float = 5.0):
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
    bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Specular"].default_value = 0.28
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def clear_labels_only():
    to_remove = []
    for obj in bpy.data.objects:
        if obj.type != "FONT":
            continue
        if obj.name == "Scale_Label":
            continue
        if obj.get("qa_dynamic_label", False):
            to_remove.append(obj)
            continue
        if obj.name.startswith("Label_"):
            to_remove.append(obj)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


def clear_overlays_only():
    to_remove = [o for o in bpy.data.objects if o.name.startswith("Scale_")]
    for obj in to_remove:
        bpy.data.objects.remove(obj)


def rect_overlaps_any_building(rect_min_x, rect_max_x, rect_min_y, rect_max_y, objs, pad: float) -> bool:
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
    w = SCALE_MARK_LENGTH_M + 1.2
    h = SCALE_TICK_LEN_Y + 1.0

    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])

    candidates = [
        (min_x + SCALE_MARK_MARGIN, max_y - SCALE_MARK_MARGIN - h),
        (min_x + SCALE_MARK_MARGIN, min_y + SCALE_MARK_MARGIN),
        (max_x - SCALE_MARK_MARGIN - w, max_y - SCALE_MARK_MARGIN - h),
        (max_x - SCALE_MARK_MARGIN - w, min_y + SCALE_MARK_MARGIN),
        (min_x + SCALE_MARK_MARGIN, (min_y + max_y) * 0.5 - h * 0.5),
        (max_x - SCALE_MARK_MARGIN - w, (min_y + max_y) * 0.5 - h * 0.5),
    ]

    for (x0, y0) in candidates:
        if not rect_overlaps_any_building(x0, x0 + w, y0, y0 + h, building_objs, pad=0.25):
            return float(x0), float(y0)
    return float(max_x - SCALE_MARK_MARGIN - w), float(min_y + SCALE_MARK_MARGIN)


def add_scale_marker(bounds: dict, building_objs: list, cam: bpy.types.Object, length_m: float = 1.0) -> None:
    x0, y0 = pick_scale_marker_position(bounds, building_objs)
    z = float(bounds["max_z"]) + SCALE_MARK_Z_EPS
    bar_y = y0 + SCALE_TICK_LEN_Y * 0.5

    mat = create_label_material("Scale_Mark_Mat", strength=10.0)
    created = []

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

    for o in created:
        o.hide_render = False
        o.visible_shadow = False
        if hasattr(o, "cycles_visibility"):
            o.cycles_visibility.shadow = False
            o.cycles_visibility.diffuse = False
            o.cycles_visibility.glossy = False
            o.cycles_visibility.ambient_occlusion = False

    curve = bpy.data.curves.new("Scale_Label_curve", type="FONT")
    curve.body = "1unit"
    curve.size = float(SCALE_LABEL_SIZE)
    curve.align_x = "LEFT"
    curve.align_y = "CENTER"
    curve.extrude = 0.2
    curve.fill_mode = "BOTH"

    text_obj = bpy.data.objects.new("Scale_Label", curve)
    bpy.context.collection.objects.link(text_obj)
    text_obj.location = (float(x0 + length_m + 4.0), float(bar_y), float(z + SCALE_LABEL_Z_OFF))
    text_obj.data.materials.clear()
    text_obj.data.materials.append(create_label_material("Scale_Label_Mat", strength=5.0))

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


def add_label(obj: bpy.types.Object, label_text: str, font_size: float, z_top: float):
    b = world_bounds_from_obj(obj)
    cx = b["center_x"]
    cy = b["center_y"]

    curve = bpy.data.curves.new(f"Label_{label_text}_curve", type="FONT")
    curve.body = str(label_text)
    curve.size = float(font_size)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.fill_mode = "BOTH"

    t = bpy.data.objects.new(f"Label_{label_text}", curve)
    t["qa_dynamic_label"] = True
    bpy.context.collection.objects.link(t)
    t.location = (float(cx), float(cy), float(z_top))
    t.rotation_euler = (0, 0, 0)

    t.data.materials.clear()
    t.data.materials.append(create_black_label_material())

    t.hide_render = False
    t.visible_shadow = False
    if hasattr(t, "cycles_visibility"):
        t.cycles_visibility.shadow = False
        t.cycles_visibility.diffuse = False
        t.cycles_visibility.glossy = False
        t.cycles_visibility.ambient_occlusion = False
        t.cycles_visibility.cast_shadow = False
        t.cycles_visibility.receive_shadow = False

    bpy.context.view_layer.update()


def rebuild_labels_for_current_objects(id_to_obj: dict, font_size: float) -> None:
    bpy.context.view_layer.update()
    clear_labels_only()
    for label_id, obj in sorted(id_to_obj.items()):
        if obj and obj.type == "MESH":
            b_bounds = world_bounds_from_obj(obj)
            z_top = b_bounds["max_z"] + font_size * 0.3
            add_label(obj=obj, label_text=str(label_id), font_size=font_size, z_top=z_top)
    bpy.context.view_layer.update()


def render_top_view_with_labels(
    cam: bpy.types.Object,
    output_path: str,
    bounds: dict,
    building_objs: list,
    id_to_obj: dict,
    with_scale_marker: bool,
    add_unit_bar: bool,
    north_world_dir: Vector,
    label_font_size: float,
    min_building_bbox: tuple[int, int, int, int] | None = None,
    reframe: bool = True,
    trim_to_alpha: bool = True,
) -> None:
    clear_overlays_only()
    rebuild_labels_for_current_objects(id_to_obj=id_to_obj, font_size=label_font_size)
    if with_scale_marker:
        add_scale_marker(bounds, building_objs, cam, length_m=SCALE_MARK_LENGTH_M)
    bpy.context.view_layer.update()
    bbox_px = projected_building_bbox_pixels(cam, building_objs)
    bbox_px = merge_bbox_pixels(bbox_px, min_building_bbox)
    render_view(
        cam=cam,
        output_path=output_path,
        add_north=True,
        north_world_dir=north_world_dir,
        add_unit_bar=add_unit_bar,
        building_bbox=bbox_px,
        reframe=reframe,
        trim_to_alpha=trim_to_alpha,
    )


def _parse_action(action_str: str):
    text = action_str.strip()
    move_match = re.match(r'Move\s*\(\s*(\w+)\s*,\s*([-\d.]+)\s*\)', text, re.IGNORECASE)
    if move_match:
        return ('Move', move_match.group(1).capitalize(), float(move_match.group(2)))
    rotate_match = re.match(r'Rotate\s*\(\s*([-\d.]+)\s*\)', text, re.IGNORECASE)
    if rotate_match:
        return ('Rotate', float(rotate_match.group(1)))
    scale_match = re.match(r'Scale\s*\(\s*([\d.]+)\s*\)', text, re.IGNORECASE)
    if scale_match:
        return ('Scale', float(scale_match.group(1)))
    return None


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _rotation_from_state(state):
    if not isinstance(state, dict):
        return 0.0
    for key in ("rotation_z_deg", "rotation_euler_z_deg", "rotation_y_deg"):
        if key in state:
            return _safe_float(state.get(key), 0.0)
    return 0.0


def _scale_from_state(state):
    if not isinstance(state, dict):
        return (1.0, 1.0, 1.0)
    s = state.get("scale")
    if isinstance(s, (list, tuple)) and len(s) >= 3:
        return (_safe_float(s[0], 1.0), _safe_float(s[1], 1.0), _safe_float(s[2], 1.0))
    if isinstance(s, (int, float)):
        f = _safe_float(s, 1.0)
        return (f, f, f)
    return (1.0, 1.0, 1.0)


def _position_from_state(state):
    if not isinstance(state, dict):
        return (0.0, 0.0, 0.0)
    p = state.get("position")
    if isinstance(p, (list, tuple)) and len(p) >= 3:
        return (_safe_float(p[0], 0.0), _safe_float(p[1], 0.0), _safe_float(p[2], 0.0))
    return (0.0, 0.0, 0.0)


def _normalize_direction(direction: str) -> str:
    d = (direction or "").strip().lower().replace("-", "").replace("_", "")
    alias = {
        "n": "north",
        "u": "up",
        "s": "south",
        "d": "down",
        "e": "east",
        "w": "west",
        "ne": "northeast",
        "nw": "northwest",
        "se": "southeast",
        "sw": "southwest",
    }
    return alias.get(d, d)


def _set_move_basis_from_north(north_world_dir):
    """Align Up/Down/Left/Right with rendered N marker direction in world XY."""
    global _MOVE_NORTH_WORLD_XY, _MOVE_EAST_WORLD_XY
    try:
        nx = float(getattr(north_world_dir, "x", 0.0))
        ny = float(getattr(north_world_dir, "y", 1.0))
    except Exception:
        nx, ny = 0.0, 1.0
    n = math.hypot(nx, ny)
    if n < 1e-9:
        nx, ny = 0.0, 1.0
    else:
        nx, ny = nx / n, ny / n
    # Right-hand +90deg from north in XY gives east.
    ex, ey = ny, -nx
    en = math.hypot(ex, ey)
    if en < 1e-9:
        ex, ey = 1.0, 0.0
    else:
        ex, ey = ex / en, ey / en
    _MOVE_NORTH_WORLD_XY = (nx, ny)
    _MOVE_EAST_WORLD_XY = (ex, ey)


def _move_vec(direction: str) -> tuple[float, float, float]:
    nd = _normalize_direction(direction)
    nx, ny = _MOVE_NORTH_WORLD_XY
    ex, ey = _MOVE_EAST_WORLD_XY

    if nd in {"up", "north"}:
        return (nx, ny, 0.0)
    if nd in {"down", "south"}:
        return (-nx, -ny, 0.0)
    if nd in {"right", "east"}:
        return (ex, ey, 0.0)
    if nd in {"left", "west"}:
        return (-ex, -ey, 0.0)
    if nd == "northeast":
        return ((nx + ex) / math.sqrt(2.0), (ny + ey) / math.sqrt(2.0), 0.0)
    if nd == "northwest":
        return ((nx - ex) / math.sqrt(2.0), (ny - ey) / math.sqrt(2.0), 0.0)
    if nd == "southeast":
        return ((-nx + ex) / math.sqrt(2.0), (-ny + ey) / math.sqrt(2.0), 0.0)
    if nd == "southwest":
        return ((-nx - ex) / math.sqrt(2.0), (-ny - ey) / math.sqrt(2.0), 0.0)
    return (0.0, 0.0, 0.0)


def _move(obj, direction, dist):
    vec = _move_vec(direction)
    obj.matrix_world.translation.x += vec[0] * dist
    obj.matrix_world.translation.y += vec[1] * dist
    obj.matrix_world.translation.z += vec[2] * dist


def _rotate(obj, angle):
    bounds = _get_bounds(obj)
    center = Vector((bounds['center_x'], bounds['center_y'], bounds['center_z']))
    angle_rad = math.radians(-angle)
    rot = Matrix.Rotation(angle_rad, 4, 'Z')
    obj.matrix_world = Matrix.Translation(center) @ rot @ Matrix.Translation(-center) @ obj.matrix_world


def _scale(obj, factor):
    if obj.get('_scale_applied', False):
        return
    bounds = _get_bounds(obj)
    center = Vector((bounds['center_x'], bounds['center_y'], bounds['center_z']))
    f = float(factor)
    scl = Matrix.Diagonal((f, f, f, 1.0))
    obj.matrix_world = Matrix.Translation(center) @ scl @ Matrix.Translation(-center) @ obj.matrix_world
    obj['_scale_applied'] = True


def _matrix_flatten(obj):
    mw = obj.matrix_world
    return [float(mw[r][c]) for r in range(4) for c in range(4)]


def _matrix_diff_max(a, b):
    if len(a) != len(b):
        return float("inf")
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _resolve_label_target_groups(labels_map, strict_exact: bool = False):
    mesh_objs = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    by_name = {obj.name: obj for obj in mesh_objs}
    out = {}
    missing = []

    def _try_exact_or_blender_copy(name_token: str):
        obj_hit = by_name.get(name_token)
        if obj_hit is not None:
            return [obj_hit]
        # Blender duplicate naming fallback: "<name>.001"
        copy_candidates = [o for o in mesh_objs if o.name.startswith(f"{name_token}.")]
        if copy_candidates:
            return sorted(copy_candidates, key=lambda x: x.name)
        return []

    for bid_str, raw_target in (labels_map or {}).items():
        bid = str(bid_str)
        token = str(raw_target or "").strip()
        targets = []

        if strict_exact:
            if token:
                targets = _try_exact_or_blender_copy(token)
            if not targets:
                missing.append({"id": bid, "target": token})
        else:
            obj = by_name.get(token)
            if obj is not None:
                targets = [obj]
            else:
                instance_key = None
                if token.isdigit():
                    instance_key = str(int(token))
                elif token.lower().startswith("instance:"):
                    part = token.split(":", 1)[1].strip()
                    if part.isdigit():
                        instance_key = str(int(part))
                if instance_key is not None:
                    prefix = f"{instance_key}_"
                    targets = [o for o in mesh_objs if o.name.startswith(prefix)]
                if (not targets) and token:
                    prefix = f"{token}_"
                    targets = [o for o in mesh_objs if o.name.startswith(prefix)]

                if (not targets) and token:
                    # Blender duplicate naming fallback: "<name>.001"
                    targets = _try_exact_or_blender_copy(token)

        uniq = []
        seen = set()
        for o in sorted(targets, key=lambda x: x.name):
            if o.name in seen:
                continue
            seen.add(o.name)
            uniq.append(o)
        if uniq:
            out[bid] = uniq

    if strict_exact and missing:
        examples = [obj.name for obj in sorted(mesh_objs, key=lambda x: x.name)[:30]]
        raise RuntimeError(
            "region  labels "
            f" missing={missing[:20]}; mesh_examples={examples}"
        )
    return out


def _hide_indoor_occluder_shells():
    hidden = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if INSTANCE_NAME_RE.match(str(obj.name)):
            continue
        name_l = str(obj.name).lower()
        b = world_bounds_from_obj(obj)
        if not b:
            continue
        w = float(b["max_x"] - b["min_x"])
        d = float(b["max_y"] - b["min_y"])
        h = float(b["max_z"] - b["min_z"])
        is_wall_like_name = any(
            k in name_l for k in ("wall", "ceiling", "doorframe", "door_frame", "windowframe", "window_frame", "structure")
        )
        is_large_vertical_shell = (h > 1.6 and max(w, d) > 1.2)
        if is_wall_like_name or is_large_vertical_shell:
            obj.hide_render = True
            obj.hide_viewport = True
            hidden.append(obj.name)
    return hidden


def _load_indoor_construct_base():
    global _INDOOR_BASE
    if _INDOOR_BASE is not None:
        return _INDOOR_BASE

    base_path = os.path.join(
        PROJECT_ROOT,
        "benchmark/data_construct/task567_error_mode/indoor_error_mode_construct.py",
    )
    spec = importlib.util.spec_from_file_location("indoor_error_mode_construct_runtime", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load indoor constructor base script: {base_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _INDOOR_BASE = mod
    return mod


def _extract_instance_index_from_obj_name(name: str) -> str:
    m = re.match(r"^(\d+)_", str(name or ""))
    if not m:
        return ""
    return str(int(m.group(1)))


def _guess_group_category(objs) -> str:
    for obj in objs:
        m = re.match(r"^\d+_([^@/]+)", str(getattr(obj, "name", "")))
        if m:
            return str(m.group(1)).replace("_", " ").strip().lower()
    return "unknown"


def _build_indoor_capture_entries(labels_map, label_target_groups):
    id_to_entry = {}
    label_map_by_oid = {}
    for bid_str in sorted(label_target_groups.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        objs = label_target_groups.get(bid_str) or []
        if not objs:
            continue
        raw_target = str((labels_map or {}).get(bid_str, "")).strip()
        oid = raw_target
        if not oid:
            oid = _extract_instance_index_from_obj_name(objs[0].name)
        if not oid:
            oid = str(bid_str)
        if oid in id_to_entry:
            oid = f"{oid}__lb{bid_str}"

        try:
            inst_idx = int(oid) if str(oid).isdigit() else -1
        except Exception:
            inst_idx = -1

        id_to_entry[oid] = {
            "id": str(oid),
            "instance_index": inst_idx,
            "category": _guess_group_category(objs),
            "group": list(objs),
        }
        try:
            label_map_by_oid[str(oid)] = int(bid_str)
        except Exception:
            continue

    return id_to_entry, label_map_by_oid


def _load_or_build_indoor_view_anchor(region_dir: str, base_mod, id_to_entry: dict):
    anchor_path = os.path.join(region_dir, "indoor_view_anchor.json")
    if os.path.exists(anchor_path):
        try:
            payload = _load_json(anchor_path)
            render_ctx = _deserialize_render_ctx(base_mod, payload["render_ctx"])
            north = base_mod.Vector(tuple(payload["north_world_dir"]))
            return {
                "render_ctx": render_ctx,
                "north_world_dir": north,
                "anchor_path": anchor_path,
                "source": "cached",
            }
        except Exception:
            pass

    render_ctx = base_mod.dc_utils.compute_render_frame(base_mod._all_mesh_objects())
    north = base_mod.dc_utils.canonical_north_world(render_ctx)
    payload = {
        "render_ctx": _serialize_render_ctx(render_ctx),
        "north_world_dir": _vec3_to_list(north),
    }
    _write_json(anchor_path, payload)
    return {
        "render_ctx": render_ctx,
        "north_world_dir": north,
        "anchor_path": anchor_path,
        "source": "new",
    }


def _render_indoor_with_constructor_style(
    output_dir: str,
    output_top: str,
    output_iso: str,
    gltf_path: str,
    id_to_entry: dict,
    label_map_by_oid: dict,
    view_anchor: dict | None = None,
    glb_name_hint: str = "",
):
    base = _load_indoor_construct_base()

    if not id_to_entry or not label_map_by_oid:
        raise RuntimeError("indoor capture entries ")

    base.set_render_and_world()

    groups_for_bounds = {
        idx: entry.get("group", [])
        for idx, entry in enumerate(id_to_entry.values())
    }
    scene_bounds = base._scene_bounds_from_groups(groups_for_bounds)
    scene_center = Vector((scene_bounds["center_x"], scene_bounds["center_y"], scene_bounds["center_z"]))
    scene_radius = max(float(scene_bounds["width"]), float(scene_bounds["depth"]), 1.0)
    base.setup_lighting(scene_center, scene_radius)

    wall_path = None
    wall_bounds = None
    resolved_glb_name = ""
    glb_candidates: list[str] = []
    if glb_name_hint:
        glb_candidates.append(os.path.basename(str(glb_name_hint)))
    if gltf_path:
        glb_candidates.append(os.path.basename(str(gltf_path)))

    for cand in glb_candidates:
        cand = str(cand or "").strip()
        if not cand or (not cand.lower().endswith(".glb")):
            continue
        try:
            c_wall = base.build_wall_path(cand, base.DEFAULT_LAYOUT_ROOT)
        except Exception:
            c_wall = None
        if c_wall is None or (not os.path.exists(str(c_wall))):
            continue
        wall_path = c_wall
        resolved_glb_name = cand
        break

    if wall_path is not None:
        wall_bounds = base._import_wall_bounds(wall_path)
        base._cap_wall_height(wall_bounds, top_margin=base.WALL_TOP_MARGIN)

    if view_anchor and view_anchor.get("render_ctx") is not None:
        render_ctx = view_anchor["render_ctx"]
    else:
        render_ctx = base.dc_utils.compute_render_frame(base._all_mesh_objects())
    cam_top, cam_top_data = base.create_preview_camera("TopCamIter")
    cam_iso, cam_iso_data = base.create_preview_camera("IsoCamIter")
    if view_anchor and view_anchor.get("north_world_dir") is not None:
        north_world_dir = view_anchor["north_world_dir"]
    else:
        north_world_dir = base.dc_utils.canonical_north_world(render_ctx)
    base._set_move_basis_from_north(north_world_dir)
    base.setup_camera_for_mode(cam_top, cam_top_data, render_ctx, "top")
    base.setup_camera_for_mode(cam_iso, cam_iso_data, render_ctx, "isometric")

    out_top_path = Path(output_dir) / output_top
    out_iso_path = Path(output_dir) / output_iso
    raw_top = Path(output_dir) / f".{Path(output_top).stem}_raw.png"
    raw_iso = Path(output_dir) / f".{Path(output_iso).stem}_raw.png"

    try:
        top_abs = base._capture_labeled(
            cam_top,
            render_ctx,
            "top",
            id_to_entry,
            label_map_by_oid,
            raw_top,
            out_top_path,
            wall_bounds=wall_bounds,
            wall_alpha=base.TOP_WALL_ALPHA,
            north_world_dir=north_world_dir,
        )
        iso_abs = base._capture_labeled(
            cam_iso,
            render_ctx,
            FIXED_ISOMETRIC_MODE,
            id_to_entry,
            label_map_by_oid,
            raw_iso,
            out_iso_path,
            wall_bounds=wall_bounds,
            wall_alpha=base.ISO_WALL_ALPHA,
            north_world_dir=north_world_dir,
        )
    finally:
        for p in (raw_top, raw_iso):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    return {
        "top_path": top_abs,
        "isometric_path": iso_abs,
        "resolved_glb_name": resolved_glb_name,
        "wall_path": (str(wall_path) if wall_path is not None else ""),
        "wall_bounds_available": bool(wall_bounds),
    }


def _group_center(objs):
    xs, ys, zs = [], [], []
    for obj in objs:
        if obj is None:
            continue
        try:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        except Exception:
            corners = []
        if corners:
            xs.extend(float(v.x) for v in corners)
            ys.extend(float(v.y) for v in corners)
            zs.extend(float(v.z) for v in corners)
        else:
            loc = obj.matrix_world.translation
            xs.append(float(loc.x))
            ys.append(float(loc.y))
            zs.append(float(loc.z))
    if not xs:
        return Vector((0.0, 0.0, 0.0))
    return Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5))


def _move_group(objs, direction, dist):
    vec = _move_vec(direction)
    dx, dy, dz = float(vec[0]) * float(dist), float(vec[1]) * float(dist), float(vec[2]) * float(dist)
    for obj in objs:
        obj.matrix_world.translation.x += dx
        obj.matrix_world.translation.y += dy
        obj.matrix_world.translation.z += dz


def _rotate_group(objs, angle):
    center = _group_center(objs)
    angle_rad = math.radians(-float(angle))
    rot = Matrix.Rotation(angle_rad, 4, 'Z')
    trans = Matrix.Translation(center) @ rot @ Matrix.Translation(-center)
    for obj in objs:
        obj.matrix_world = trans @ obj.matrix_world


def _scale_group(objs, factor):
    center = _group_center(objs)
    f = float(factor)
    scl = Matrix.Diagonal((f, f, f, 1.0))
    trans = Matrix.Translation(center) @ scl @ Matrix.Translation(-center)
    for obj in objs:
        obj.matrix_world = trans @ obj.matrix_world


def _apply_semantic_ops(objs, ops):
    """
    Apply structured meta ops directly:
    - translate: absolute target position (after.position)
    - rotate: delta = after.rot_z - before.rot_z
    - scale: absolute target scale (after.scale)
    """
    applied = []
    for op in (ops or []):
        kind = str((op or {}).get("kind", "") or "").strip().lower()
        before = (op or {}).get("before") or {}
        after = (op or {}).get("after") or {}

        if kind == "translate":
            tx, ty, tz = _position_from_state(after)
            c = _group_center(objs)
            dx, dy, dz = float(tx - c.x), float(ty - c.y), float(tz - c.z)
            for obj in objs:
                obj.matrix_world.translation.x += dx
                obj.matrix_world.translation.y += dy
                obj.matrix_world.translation.z += dz
            applied.append(
                {
                    "type": "translate_abs",
                    "target_after_position": [float(tx), float(ty), float(tz)],
                    "delta_applied": [float(dx), float(dy), float(dz)],
                }
            )
            continue

        if kind == "rotate":
            d = _rotation_from_state(after) - _rotation_from_state(before)
            _rotate_group(objs, d)
            applied.append(
                {
                    "type": "rotate_delta",
                    "delta_deg": float(d),
                    "angle_applied_blender": float(-d),
                }
            )
            continue

        if kind == "scale":
            sx, sy, sz = _scale_from_state(after)
            for obj in objs:
                loc, rot_q, _sca = obj.matrix_world.decompose()
                rot_m = rot_q.to_euler("XYZ").to_matrix().to_4x4()
                scl_m = Matrix.Diagonal((float(sx), float(sy), float(sz), 1.0))
                obj.matrix_world = Matrix.Translation(loc) @ rot_m @ scl_m
            applied.append(
                {
                    "type": "scale_abs",
                    "target_after_scale": [float(sx), float(sy), float(sz)],
                }
            )
            continue

        applied.append({"type": "unknown", "raw": op})

    return applied


def _apply_actions(labels_map, consolidated_actions, unit_scale: float):
    target_groups = _resolve_label_target_groups(labels_map)
    logs = []
    all_before = {}
    for bid_str, objs in target_groups.items():
        all_before[bid_str] = {
            'obj_names': [obj.name for obj in objs],
            'matrices': {obj.name: _matrix_flatten(obj) for obj in objs},
        }

    target_ids = {str(item.get('id', '')) for item in consolidated_actions}

    for item in consolidated_actions:
        bid = str(item.get('id', ''))
        objs = target_groups.get(bid, [])
        entry = {
            "id": item.get("id"),
            "obj_names": [obj.name for obj in objs],
            "requested_actions": item.get("action", []),
            "requested_ops": item.get("ops", []),
            "applied_actions": [],
            "skipped_reason": None,
        }
        if not objs:
            entry["skipped_reason"] = "id_not_in_labels"
            logs.append(entry)
            continue
        ops = item.get("ops", [])
        if isinstance(ops, list) and len(ops) > 0:
            before_c = _group_center(objs)
            before_loc = [float(before_c.x), float(before_c.y), float(before_c.z)]
            applied = _apply_semantic_ops(objs, ops)
            after_c = _group_center(objs)
            after_loc = [float(after_c.x), float(after_c.y), float(after_c.z)]
            for a in applied:
                a["before_loc"] = before_loc
                a["after_loc"] = after_loc
                entry["applied_actions"].append(a)
        else:
            for text in item.get('action', []):
                parsed = _parse_action(text)
                if not parsed:
                    entry["applied_actions"].append({"raw": text, "status": "unparsed"})
                    continue
                before_c = _group_center(objs)
                before_loc = [float(before_c.x), float(before_c.y), float(before_c.z)]
                if parsed[0] == 'Move':
                    _, direction, dist = parsed
                    _move_group(objs, direction, dist)
                    entry["applied_actions"].append(
                        {
                            "raw": text,
                            "type": "Move",
                            "direction": direction,
                            "distance_input": float(dist),
                            "distance_applied": float(dist),
                        }
                    )
                elif parsed[0] == 'Rotate':
                    _, angle = parsed
                    _rotate_group(objs, angle)
                    entry["applied_actions"].append(
                        {
                            "raw": text,
                            "type": "Rotate",
                            "angle_input": float(angle),
                            "angle_applied_blender": float(-angle),
                        }
                    )
                elif parsed[0] == 'Scale':
                    _, factor = parsed
                    _scale_group(objs, factor)
                    entry["applied_actions"].append(
                        {
                            "raw": text,
                            "type": "Scale",
                            "factor": float(factor),
                        }
                    )
                after_c = _group_center(objs)
                after_loc = [float(after_c.x), float(after_c.y), float(after_c.z)]
                entry["applied_actions"][-1]["before_loc"] = before_loc
                entry["applied_actions"][-1]["after_loc"] = after_loc
        logs.append(entry)
    bpy.context.view_layer.update()

    all_after = {}
    for bid_str, objs in target_groups.items():
        all_after[bid_str] = {
            'obj_names': [obj.name for obj in objs],
            'matrices': {obj.name: _matrix_flatten(obj) for obj in objs},
        }

    unexpected_changed_ids = []
    for bid_str, before in all_before.items():
        after = all_after.get(bid_str)
        if not after:
            continue
        max_delta = 0.0
        for obj_name, b_mat in before.get('matrices', {}).items():
            a_mat = (after.get('matrices') or {}).get(obj_name)
            if a_mat is None:
                continue
            max_delta = max(max_delta, _matrix_diff_max(b_mat, a_mat))
        if max_delta > 1e-6 and bid_str not in target_ids:
            unexpected_changed_ids.append(
                {
                    'id': int(bid_str) if bid_str.isdigit() else bid_str,
                    'obj_names': before.get('obj_names', []),
                    'max_matrix_delta': float(max_delta),
                }
            )

    return {
        'per_action_logs': logs,
        'target_ids': sorted(int(i) for i in target_ids if str(i).isdigit()),
        'unexpected_changed_ids': unexpected_changed_ids,
    }


def main():
    region_dir = os.environ.get('BLENDER_REGION_DIR', '')
    output_dir = os.environ.get('BLENDER_OUTPUT_DIR', '')
    cumulative_path = os.environ.get('BLENDER_CUMULATIVE_PATH', '')
    input_blend = os.environ.get('BLENDER_INPUT_BLEND', '')
    gltf_path = os.environ.get('GLTF_PATH', '')
    output_top = os.environ.get('BLENDER_OUTPUT_TOP', 'top.png')
    output_iso = os.environ.get('BLENDER_OUTPUT_ISO', 'isometric.png')
    output_blend = os.environ.get('BLENDER_OUTPUT_BLEND', '')
    skip_image_render = str(os.environ.get("BLENDER_SKIP_IMAGE_RENDER", "0") or "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    unit_scale = float(os.environ.get('BLENDER_UNIT_SCALE', '1.0') or '1.0')
    scene_type = str(os.environ.get('BLENDER_SCENE_TYPE', 'region') or 'region').strip().lower()
    if scene_type not in {"region", "indoor"}:
        scene_type = "region"
    region_mode = str(os.environ.get("BLENDER_REGION_MODE", "basic") or "basic").strip().lower()
    if region_mode not in {"basic", "complex", "indoor"}:
        region_mode = "basic"
    action_log_path = os.environ.get('BLENDER_ACTION_LOG_PATH', '')
    region_info_path = os.path.join(region_dir, "region_info.json") if region_dir else ""
    region_info = {}
    if region_info_path and os.path.exists(region_info_path):
        try:
            region_info = _load_json(region_info_path)
        except Exception:
            region_info = {}

    if not region_dir or not os.path.exists(region_dir):
        raise RuntimeError(f"BLENDER_REGION_DIR : {region_dir}")
    if not output_dir:
        raise RuntimeError("BLENDER_OUTPUT_DIR ")
    if not cumulative_path or not os.path.exists(cumulative_path):
        raise RuntimeError(f"BLENDER_CUMULATIVE_PATH : {cumulative_path}")

    os.makedirs(output_dir, exist_ok=True)

    if input_blend and os.path.exists(input_blend):
        bpy.ops.wm.open_mainfile(filepath=input_blend)
    else:
        if not gltf_path or not os.path.exists(gltf_path):
            raise RuntimeError(" BLENDER_INPUT_BLEND GLTF_PATH")
        _clear_scene()
        bpy.ops.import_scene.gltf(filepath=gltf_path)

    labels_path = os.path.join(region_dir, 'labels.json')
    labels_map = _load_json(labels_path)
    label_target_groups = _resolve_label_target_groups(
        labels_map,
        strict_exact=(scene_type == "region"),
    )
    building_ids = sorted({obj.name for objs in label_target_groups.values() for obj in objs})
    all_instance_mesh_ids = sorted(
        obj.name
        for obj in bpy.data.objects
        if obj.type == "MESH" and re.match(r"^\d+_", str(obj.name))
    )
    render_scope_ids = list(building_ids)
    if scene_type == "indoor" and all_instance_mesh_ids:
        render_scope_ids = all_instance_mesh_ids
    reference_bounds = compute_region_bounds_from_ids(render_scope_ids or building_ids, pad_ratio=0.20)

    render_id_to_obj = {}
    for label_id_str, objs in label_target_groups.items():
        if not objs:
            continue
        try:
            render_id_to_obj[int(label_id_str)] = objs[0]
        except Exception:
            pass

    id_to_entry = {}
    label_map_by_oid = {}
    view_anchor = None
    if scene_type == "indoor":
        id_to_entry, label_map_by_oid = _build_indoor_capture_entries(labels_map, label_target_groups)
        base_mod = _load_indoor_construct_base()
        view_anchor = _load_or_build_indoor_view_anchor(region_dir, base_mod, id_to_entry)
        if view_anchor and view_anchor.get("north_world_dir") is not None:
            _set_move_basis_from_north(view_anchor["north_world_dir"])

    cumulative = _load_json(cumulative_path)
    consolidated = cumulative.get('consolidated_actions', [])
    action_audit = _apply_actions(labels_map, consolidated, unit_scale=unit_scale)

    if action_audit.get('unexpected_changed_ids'):
        raise RuntimeError(
            f": {action_audit['unexpected_changed_ids']}"
        )

    if scene_type == "region" and skip_image_render:
        if action_log_path:
            os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
            with open(action_log_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'unit_scale': unit_scale,
                        'scene_type': scene_type,
                        'region_mode': region_mode,
                        'skip_image_render': True,
                        'consolidated_action_count': len(consolidated),
                        'target_ids': action_audit.get('target_ids', []),
                        'unexpected_changed_ids': action_audit.get('unexpected_changed_ids', []),
                        'logs': action_audit.get('per_action_logs', []),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        if output_blend:
            os.makedirs(os.path.dirname(output_blend), exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=output_blend, copy=True)
        return

    if scene_type == "indoor":
        if skip_image_render:
            if action_log_path:
                os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
                with open(action_log_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {
                            'unit_scale': unit_scale,
                            'scene_type': scene_type,
                            'render_style': 'constructor_capture_labeled',
                            'skip_image_render': True,
                            'consolidated_action_count': len(consolidated),
                            'target_ids': action_audit.get('target_ids', []),
                            'unexpected_changed_ids': action_audit.get('unexpected_changed_ids', []),
                            'logs': action_audit.get('per_action_logs', []),
                            'view_anchor_source': (view_anchor or {}).get('source', ''),
                            'view_anchor_path': (view_anchor or {}).get('anchor_path', ''),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            if output_blend:
                os.makedirs(os.path.dirname(output_blend), exist_ok=True)
                bpy.ops.wm.save_as_mainfile(filepath=output_blend, copy=True)
            return

        glb_name_hint = ""
        try:
            glb_name_hint = str((region_info or {}).get("glb_name", "") or "").strip()
        except Exception:
            glb_name_hint = ""
        if not glb_name_hint:
            try:
                source_scene_path = str(((region_info.get("source") or {}).get("source_scene_path", "") or "")).strip()
            except Exception:
                source_scene_path = ""
            if source_scene_path:
                glb_name_hint = os.path.basename(source_scene_path)

        render_meta = _render_indoor_with_constructor_style(
            output_dir=output_dir,
            output_top=output_top,
            output_iso=output_iso,
            gltf_path=gltf_path,
            id_to_entry=id_to_entry,
            label_map_by_oid=label_map_by_oid,
            view_anchor=view_anchor,
            glb_name_hint=glb_name_hint,
        )

        if action_log_path:
            os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
            with open(action_log_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'unit_scale': unit_scale,
                        'scene_type': scene_type,
                        'render_style': 'constructor_capture_labeled',
                        'consolidated_action_count': len(consolidated),
                        'target_ids': action_audit.get('target_ids', []),
                        'unexpected_changed_ids': action_audit.get('unexpected_changed_ids', []),
                        'logs': action_audit.get('per_action_logs', []),
                        'capture_meta': render_meta,
                        'view_anchor_source': (view_anchor or {}).get('source', ''),
                        'view_anchor_path': (view_anchor or {}).get('anchor_path', ''),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if output_blend:
            os.makedirs(os.path.dirname(output_blend), exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=output_blend, copy=True)
        return

    setup_render(1920, 1080)
    force_disable_all_shadows_and_world()

    labeler = BuildingLabeler(
        region_data_path="",
        blend_path=input_blend if input_blend and os.path.exists(input_blend) else None,
        output_dir=output_dir,
        ortho_scale_factor=1.8,
        label_height_ratio=0.01,
        font_size_ratio=0.06,
        samples=64,
        resolution=(1920, 1080),
        mask_alpha=0.8,
    )

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
            except Exception:
                pass

    labeler.setup_lighting = types.MethodType(_patched_setup_lighting, labeler)
    labeler.setup_render()
    labeler.setup_lighting()

    temp_road, road_hidden_states = clean_create_max_region_roads(
        region_id=int(os.path.basename(region_dir).split("_")[-1]),
        building_ids=render_scope_ids,
    )
    original_building_materials = snapshot_building_material_slots(render_scope_ids)
    labeler.set_region_visibility(render_scope_ids)
    if temp_road:
        temp_road.hide_render = False
        temp_road.hide_viewport = False
    labeler.clear_mask_materials()
    labeler.clear_all_labels()
    if region_mode == "complex":
        restore_building_material_slots(original_building_materials)
    else:
        labeler.apply_mask_to_buildings(render_scope_ids)
        apply_white_film_to_buildings(render_scope_ids)
    apply_road_material()

    bounds_before = labeler.calculate_region_bounds(render_scope_ids or building_ids)
    if not bounds_before:
        raise RuntimeError("")

    bounds = reference_bounds or bounds_before
    
    region_max_dim = max(bounds_before["width"], bounds_before["depth"])
    label_font_size = region_max_dim * LABEL_SIZE_RATIO

    if action_log_path:
        os.makedirs(os.path.dirname(action_log_path), exist_ok=True)
        with open(action_log_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'unit_scale': unit_scale,
                    'scene_type': scene_type,
                    'region_mode': region_mode,
                    'consolidated_action_count': len(consolidated),
                    'target_ids': action_audit.get('target_ids', []),
                    'unexpected_changed_ids': action_audit.get('unexpected_changed_ids', []),
                    'logs': action_audit.get('per_action_logs', []),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    labeler.clear_all_labels()

    #  top /
    init_top_bbox_px = None
    init_top_ortho_scale = None
    region_objs_for_camera = [bpy.data.objects.get(bid) for bid in (render_scope_ids or building_ids) if bpy.data.objects.get(bid)]
    region_objs_for_camera = [obj for obj in region_objs_for_camera if obj]
    cam_top_ref = labeler.setup_camera_top_down(bounds, "CameraTopRefStandalone")
    fit_ortho_camera_to_objects(
        cam_top_ref,
        region_objs_for_camera,
        margin_ratio=TOP_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    init_top_bbox_px = projected_building_bbox_pixels(cam_top_ref, region_objs_for_camera)
    init_top_ortho_scale = float(cam_top_ref.data.ortho_scale)
    if cam_top_ref and cam_top_ref.name in bpy.data.objects:
        bpy.data.objects.remove(cam_top_ref)

    cam_top = labeler.setup_camera_top_down(bounds, "CameraTopStandalone")
    cam_iso = labeler.setup_camera_isometric(bounds, "CameraIsoStandalone")

    building_objs = [bpy.data.objects.get(bid) for bid in (render_scope_ids or building_ids) if bpy.data.objects.get(bid)]
    building_objs = [obj for obj in building_objs if obj]

    fit_ortho_camera_to_objects(
        cam_top,
        region_objs_for_camera,
        margin_ratio=TOP_FIT_MARGIN_RATIO,
        min_ortho_scale=MIN_FIT_ORTHO_SCALE,
    )
    if init_top_ortho_scale is not None:
        cam_top.data.ortho_scale = max(float(cam_top.data.ortho_scale), float(init_top_ortho_scale))
        bpy.context.view_layer.update()
    inflate_ortho_scale(cam_iso, ISO_FIT_MARGIN_RATIO)

    render_top_view_with_labels(
        cam=cam_top,
        output_path=os.path.join(output_dir, output_top),
        bounds=bounds,
        building_objs=building_objs,
        id_to_obj=render_id_to_obj,
        with_scale_marker=False,
        add_unit_bar=True,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        label_font_size=label_font_size,
        min_building_bbox=init_top_bbox_px,
        trim_to_alpha=False,
    )
    render_top_view_with_labels(
        cam=cam_iso,
        output_path=os.path.join(output_dir, output_iso),
        bounds=bounds,
        building_objs=building_objs,
        id_to_obj=render_id_to_obj,
        with_scale_marker=False,
        add_unit_bar=True,
        north_world_dir=Vector((0.0, 1.0, 0.0)),
        label_font_size=label_font_size,
        trim_to_alpha=True,
    )

    if output_blend:
        os.makedirs(os.path.dirname(output_blend), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=output_blend, copy=True)

    labeler.clear_all_labels()
    labeler.clear_mask_materials()
    clean_cleanup_temp_roads(temp_road, road_hidden_states)

if __name__ == '__main__':
    main()
