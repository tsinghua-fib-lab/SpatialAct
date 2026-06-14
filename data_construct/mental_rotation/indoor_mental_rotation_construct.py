#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path.home() / "SpatialAct")))
BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", str(PROJECT_ROOT / "blender-3.2.2-linux-x64/blender")))

INTERNSCENES_ROOT = Path(os.environ.get("INTERNSCENES_ROOT", str(Path.home() / "InternScenes")))
SHARED_UTILS_PATH = PROJECT_ROOT / "benchmark/data_construct/utils.py"

_utils_spec = importlib.util.spec_from_file_location("dc_render_utils_task3", SHARED_UTILS_PATH)
if _utils_spec is None or _utils_spec.loader is None:
	raise RuntimeError(f"Cannot load shared utils from {SHARED_UTILS_PATH}")
dc_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(dc_utils)

try:
	import bpy
	from mathutils import Vector, Matrix
	from bpy_extras.object_utils import world_to_camera_view
except ModuleNotFoundError:
	script_path = str(Path(__file__).resolve())
	cmd = [str(BLENDER_BIN), "--background", "--python", script_path, "--"] + sys.argv[1:]
	res = subprocess.run(cmd, env=os.environ.copy(), check=False)
	raise SystemExit(res.returncode)

try:
	from PIL import Image, ImageDraw, ImageFont
	PIL_AVAILABLE = True
except ImportError:
	PIL_AVAILABLE = False


DEFAULT_GLB_DIR = INTERNSCENES_ROOT / "scenes/glb_files_wall_complex-10-15_clean_keep"
DEFAULT_IMAGES_DIR = INTERNSCENES_ROOT / "scenes/images"
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark/data/mental_rotation/indoor_scenes_complex-10-15"
TMP_ROOT_DIR = PROJECT_ROOT / "tmp" / Path(__file__).stem

RESOLUTION_X = 1080
RESOLUTION_Y = 720
SAMPLES = 64
DYNAMIC_RESOLUTION = True
BASE_LONG_EDGE = 1280
MIN_SHORT_EDGE = 720

ROT_MCQ_K = 4
ROT_IMG_MCQ_K = 4
ROT_ANGLE_POOL = [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]
ROT_DIR_POOL = ["cw", "ccw"]

COLLISION_PAD = 0.12
MIN_OBJECTS = 4
KEEP_SCENE_VIEWS = False
KEEP_RAW_VIEWS = False

CAMERA_TOP_DIST_SCALE = 2.9
CAMERA_ISO_DIST_SCALE = 3.2
CAMERA_FIT_MARGIN = 1.10
CAMERA_SAFETY_SCALE = 1.03
FIXED_ISOMETRIC_MODE = "isometric_north_ur"
TOP_WALL_ALPHA = 1.0
ISO_WALL_ALPHA = 0.55
LABEL_RADIUS = 13
LABEL_FONT_SIZE = 16

# Target-shape / visibility filters
TOP_NEAR_ROUND_RATIO_THRESH = 0.88
ISO_MIN_BBOX_PX = 900.0
ISO_MIN_AREA_RATIO = 0.004
QA3_ISO_RETRY_MAX_OBJECTS = 8
TOP_MIN_BBOX_PX = 700.0
TOP_MIN_AREA_RATIO = 0.0012


def parse_args() -> argparse.Namespace:
	argv = sys.argv[1:]
	if "--" in argv:
		argv = argv[argv.index("--") + 1 :]

	parser = argparse.ArgumentParser(description="Construct indoor mental rotation QA")
	parser.add_argument("--glb-dir", type=Path, default=DEFAULT_GLB_DIR)
	parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
	parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--seed", type=int, default=30)
	parser.add_argument("--max-scenes", type=int, default=0, help="0 means all")
	parser.add_argument(
		"--region",
		type=str,
		default="",
		help="Filter scenes by region keyword (match against scene_name and glb filename, case-insensitive)",
	)
	return parser.parse_args(argv)


def _parse_instance_index(name: str):
	m = re.match(r"^(\d+)_", name)
	if not m:
		return None
	try:
		return int(m.group(1))
	except Exception:
		return None


def convert_layout_to_objects(layout_info: list) -> list[dict]:
	objects = []
	for idx, obj in enumerate(layout_info):
		if not isinstance(obj, dict):
			continue
		bbox = obj.get("bbox", [])
		if "id" not in obj or len(bbox) < 3:
			continue
		objects.append(
			{
				"id": str(obj["id"]),
				"instance_index": idx,
				"category": str(obj.get("category", "unknown")),
				"center_world": [float(bbox[0]), float(bbox[1]), float(bbox[2])],
			}
		)
	return objects


def assign_independent_labels(objects: list[dict]) -> dict[str, int]:
	def scene_id_key(obj):
		sid = str(obj.get("id", ""))
		try:
			return (0, int(sid))
		except ValueError:
			return (1, sid)

	sorted_objs = sorted(objects, key=scene_id_key)
	return {obj["id"]: idx for idx, obj in enumerate(sorted_objs, start=1)}


def _load_font(size: int):
	if not PIL_AVAILABLE:
		return None
	if "bpy" in sys.modules:
		return ImageFont.load_default()
	try:
		return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
	except Exception:
		return ImageFont.load_default()


def _ensure_north_text_sprite() -> Path | None:
	if not PIL_AVAILABLE:
		return None
	if NORTH_TEXT_CACHE.exists():
		return NORTH_TEXT_CACHE

	NORTH_TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
	py_bin = PROJECT_ROOT / ".venv/bin/python"
	if not py_bin.exists():
		return None

	code = (
		"from PIL import Image, ImageDraw, ImageFont;"
		"img=Image.new('RGBA',(120,36),(0,0,0,0));"
		"d=ImageDraw.Draw(img);"
		"f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18);"
		"d.text((0,0),'NORTH',fill=(255,0,0,255),font=f);"
		f"img.save(r'{str(NORTH_TEXT_CACHE)}')"
	)
	try:
		subprocess.run([str(py_bin), "-c", code], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except Exception:
		return None
	return NORTH_TEXT_CACHE if NORTH_TEXT_CACHE.exists() else None


def _draw_north_indicator(drawer, north_vec: tuple[float, float], img=None):
	drawer.rectangle([16, 16, 190, 88], fill=(0, 0, 0))
	cx, cy = 48, 52
	arrow_len = 28
	head_len = 10
	vx, vy = north_vec
	n = math.sqrt(vx * vx + vy * vy)
	if n < 1e-8:
		vx, vy = 0.0, -1.0
	else:
		vx, vy = vx / n, vy / n
	tip = (cx + vx * arrow_len, cy + vy * arrow_len)
	bot = (cx - vx * arrow_len * 0.6, cy - vy * arrow_len * 0.6)
	drawer.line([bot, tip], fill=(255, 0, 0), width=5)
	perp = (-vy, vx)
	wing = (tip[0] - vx * head_len, tip[1] - vy * head_len)
	left = (wing[0] + perp[0] * (head_len * 0.6), wing[1] + perp[1] * (head_len * 0.6))
	right = (wing[0] - perp[0] * (head_len * 0.6), wing[1] - perp[1] * (head_len * 0.6))
	drawer.polygon([tip, left, right], fill=(255, 0, 0))
	if img is not None:
		sprite_path = _ensure_north_text_sprite()
		if sprite_path is not None and sprite_path.exists():
			try:
				sprite = Image.open(sprite_path).convert("RGBA")
				img.paste(sprite, (78, 40), sprite)
				return
			except Exception:
				pass
	drawer.text((78, 40), "NORTH", fill=(255, 0, 0), font=_load_font(18))


def _draw_center_text_bold(draw, cx: int, cy: int, txt: str, font):
	for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1)):
		try:
			draw.text((cx + dx, cy + dy), txt, fill=(255, 255, 255), font=font, anchor="mm")
		except TypeError:
			bbox = draw.textbbox((0, 0), txt, font=font)
			draw.text((cx + dx - (bbox[2] - bbox[0]) // 2, cy + dy - (bbox[3] - bbox[1]) // 2), txt, fill=(255, 255, 255), font=font)


def _north_screen_vector(cam, width: int, height: int) -> tuple[float, float]:
	north_world_dir = Vector((0.0, 1.0, 0.0))
	m = cam.matrix_world.to_3x3()
	cam_right = (m @ Vector((1.0, 0.0, 0.0))).normalized()
	cam_up = (m @ Vector((0.0, 1.0, 0.0))).normalized()

	d = north_world_dir.normalized()
	screen_x = float(d.dot(cam_right))
	screen_y = float(-d.dot(cam_up))
	norm = math.sqrt(screen_x * screen_x + screen_y * screen_y)
	if norm > 1e-8:
		screen_x /= norm
		screen_y /= norm
		return (screen_x, screen_y)
	return (0.0, -1.0)


def _project_group_centers(id_to_entry: dict[str, dict], object_ids: list[str], cam, width: int, height: int) -> dict[str, tuple[int, int]]:
	scene = bpy.context.scene
	out: dict[str, tuple[int, int]] = {}
	for oid in object_ids:
		entry = id_to_entry.get(oid)
		if not entry:
			continue
		b = _group_bounds(entry["group"])
		world_c = Vector((b["center_x"], b["center_y"], b["center_z"]))
		v = world_to_camera_view(scene, cam, world_c)
		if v.z < 0:
			continue
		px = int(round(v.x * width))
		py = int(round((1.0 - v.y) * height))
		out[oid] = (px, py)
	return out


def _annotate_numbers(
	base_image: Path,
	out_image: Path,
	label_map: dict[str, int],
	centers: dict[str, tuple[int, int]],
	north_vec: tuple[float, float],
	px_per_unit: float,
):
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	w, h = img.size
	for oid, label in label_map.items():
		if oid not in centers:
			continue
		cx, cy = centers[oid]
		cx = max(14, min(w - 14, cx))
		cy = max(14, min(h - 14, cy))
		r = LABEL_RADIUS
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 60), outline=(255, 255, 255), width=1)
		txt = str(label)
		font = _load_font(LABEL_FONT_SIZE)
		_draw_center_text_bold(draw, cx, cy, txt, font)
	img = dc_utils.compose_compact_with_left_strip(img, north_vec, max(14.0, px_per_unit))
	img.save(out_image)
	return str(out_image.resolve())


def set_render_and_world():
	scene = bpy.context.scene
	scene.render.engine = "CYCLES"
	scene.cycles.device = "CPU"
	scene.cycles.samples = SAMPLES
	scene.render.resolution_x = RESOLUTION_X
	scene.render.resolution_y = RESOLUTION_Y
	scene.render.resolution_percentage = 100
	scene.render.image_settings.file_format = "PNG"

	if scene.world is None:
		scene.world = bpy.data.worlds.new("World")
	world = scene.world
	world.use_nodes = True
	bg = world.node_tree.nodes.get("Background")
	if bg:
		bg.inputs[1].default_value = 2.4
	scene.view_settings.exposure = 0.8


def setup_lighting(scene_center: Vector, scene_radius: float):
	for obj in list(bpy.data.objects):
		if obj.type == "LIGHT":
			bpy.data.objects.remove(obj, do_unlink=True)

	sun = bpy.data.objects.new("Sun", bpy.data.lights.new("SunData", type="SUN"))
	bpy.context.collection.objects.link(sun)
	sun.location = scene_center + Vector((scene_radius * 0.6, -scene_radius * 0.5, scene_radius * 3.2))
	sun.rotation_euler = (math.radians(65), 0.0, math.radians(35))
	sun.data.energy = 6.0

	fill = bpy.data.objects.new("Fill", bpy.data.lights.new("FillData", type="AREA"))
	bpy.context.collection.objects.link(fill)
	fill.location = scene_center + Vector((0.0, 0.0, scene_radius * 2.2))
	fill.rotation_euler = (math.radians(90), 0.0, 0.0)
	fill.data.energy = 1400.0
	fill.data.size = max(scene_radius * 1.6, 3.5)

	side_light_offsets = [
		Vector((scene_radius * 1.3, 0.0, scene_radius * 1.4)),
		Vector((-scene_radius * 1.3, 0.0, scene_radius * 1.4)),
		Vector((0.0, scene_radius * 1.3, scene_radius * 1.4)),
		Vector((0.0, -scene_radius * 1.3, scene_radius * 1.4)),
	]
	for i, ofs in enumerate(side_light_offsets):
		data = bpy.data.lights.new(name=f"PreviewSideFill_{i}", type="AREA")
		obj = bpy.data.objects.new(name=f"PreviewSideFill_{i}", object_data=data)
		bpy.context.collection.objects.link(obj)
		obj.location = scene_center + ofs
		obj.rotation_euler = (scene_center - obj.location).to_track_quat("-Z", "Y").to_euler()
		data.energy = 450.0
		data.size = max(scene_radius * 0.7, 2.0)


def _all_mesh_objects() -> list:
	return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def collect_instance_groups(max_instance_idx: int) -> dict[int, list]:
	groups: dict[int, list] = {}
	for obj in _all_mesh_objects():
		idx = _parse_instance_index(obj.name)
		if idx is None:
			continue
		if idx < 0 or idx >= max_instance_idx:
			continue
		groups.setdefault(idx, []).append(obj)
	return groups


def _group_bounds(group_objs: list) -> dict:
	pts = []
	for obj in group_objs:
		for c in obj.bound_box:
			pts.append(obj.matrix_world @ Vector(c))
	xs = [p.x for p in pts]
	ys = [p.y for p in pts]
	zs = [p.z for p in pts]
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


def _group_center_xy(group_objs: list) -> Vector:
	b = _group_bounds(group_objs)
	return Vector((b["center_x"], b["center_y"], 0.0))


def _is_near_round_top(group_objs: list, ratio_thresh: float = TOP_NEAR_ROUND_RATIO_THRESH) -> bool:
	b = _group_bounds(group_objs)
	w = max(float(b["width"]), 1e-6)
	d = max(float(b["depth"]), 1e-6)
	r = min(w, d) / max(w, d)
	return r >= ratio_thresh


def _project_group_bbox_metrics(group_objs: list, cam, width: int, height: int):
	scene = bpy.context.scene
	xs = []
	ys = []
	for obj in group_objs:
		for c in obj.bound_box:
			wc = obj.matrix_world @ Vector(c)
			v = world_to_camera_view(scene, cam, wc)
			if v.z <= 1e-6:
				continue
			xs.append(float(v.x) * width)
			ys.append((1.0 - float(v.y)) * height)
	if len(xs) < 4 or len(ys) < 4:
		return 0.0, 0.0
	min_x, max_x = max(0.0, min(xs)), min(float(width), max(xs))
	min_y, max_y = max(0.0, min(ys)), min(float(height), max(ys))
	bw = max(0.0, max_x - min_x)
	bh = max(0.0, max_y - min_y)
	area = bw * bh
	img_area = max(1.0, float(width) * float(height))
	return area, area / img_area


def _is_group_center_visible(group_objs: list, cam) -> bool:
	scene = bpy.context.scene
	depsgraph = bpy.context.evaluated_depsgraph_get()
	b = _group_bounds(group_objs)
	target = Vector((b["center_x"], b["center_y"], b["center_z"]))
	origin = cam.matrix_world.translation
	vec = target - origin
	dist = vec.length
	if dist < 1e-6:
		return True
	direction = vec.normalized()
	hit, _loc, _normal, _face_idx, hit_obj, _hit_m = scene.ray_cast(depsgraph, origin, direction, distance=dist + 1e-4)
	if not hit or hit_obj is None:
		# If ray cast misses due precision/eval differences, keep as visible.
		return True
	group_names = {o.name for o in group_objs}
	if hit_obj.name in group_names:
		return True
	try:
		if hit_obj.original is not None and hit_obj.original.name in group_names:
			return True
	except Exception:
		pass
	hit_name = str(getattr(hit_obj, "name", "")).lower()
	if any(tok in hit_name for tok in ("wall", "floor", "ceiling", "window", "door", "frame", "geometry_")):
		# Rendering uses wall alpha blending, so structure hits should not
		# invalidate visibility prefilter.
		return True
	return False


def _isometric_group_visibility_ok(group_objs: list, cam, width: int, height: int) -> bool:
	area_px, area_ratio = _project_group_bbox_metrics(group_objs, cam, width, height)
	if area_px < ISO_MIN_BBOX_PX or area_ratio < ISO_MIN_AREA_RATIO:
		return False
	return _is_group_center_visible(group_objs, cam)


def _top_group_visibility_ok(group_objs: list, cam, width: int, height: int) -> bool:
	area_px, area_ratio = _project_group_bbox_metrics(group_objs, cam, width, height)
	if area_px < TOP_MIN_BBOX_PX or area_ratio < TOP_MIN_AREA_RATIO:
		return False
	return _is_group_center_visible(group_objs, cam)


def _scene_bounds_from_groups(groups: dict[int, list]) -> dict:
	all_bounds = [_group_bounds(v) for v in groups.values() if v]
	min_x = min(b["min_x"] for b in all_bounds)
	max_x = max(b["max_x"] for b in all_bounds)
	min_y = min(b["min_y"] for b in all_bounds)
	max_y = max(b["max_y"] for b in all_bounds)
	min_z = min(b["min_z"] for b in all_bounds)
	max_z = max(b["max_z"] for b in all_bounds)
	return {
		"min_x": min_x,
		"max_x": max_x,
		"min_y": min_y,
		"max_y": max_y,
		"min_z": min_z,
		"max_z": max_z,
		"center_x": (min_x + max_x) / 2,
		"center_y": (min_y + max_y) / 2,
		"center_z": (min_z + max_z) / 2,
		"width": max_x - min_x,
		"depth": max_y - min_y,
		"height": max_z - min_z,
	}


def _compute_render_frame(mesh_objs: list) -> dict:
	frame_meshes = [o for o in mesh_objs if not o.hide_render]
	if len(frame_meshes) == 0:
		frame_meshes = mesh_objs

	min_corner = Vector((float("inf"), float("inf"), float("inf")))
	max_corner = Vector((-float("inf"), -float("inf"), -float("inf")))
	target_points = []
	for obj in frame_meshes:
		for c in obj.bound_box:
			w = obj.matrix_world @ Vector(c)
			min_corner.x = min(min_corner.x, w.x)
			min_corner.y = min(min_corner.y, w.y)
			min_corner.z = min(min_corner.z, w.z)
			max_corner.x = max(max_corner.x, w.x)
			max_corner.y = max(max_corner.y, w.y)
			max_corner.z = max(max_corner.z, w.z)
			target_points.append(w)

	center = (min_corner + max_corner) * 0.5
	extent = max_corner - min_corner
	radius = max(extent.length * 0.5, 1.0)

	axis_vectors = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))]
	vertical_axis = 2
	horizontal_axes = [i for i in [0, 1, 2] if i != vertical_axis]
	h1_idx, h2_idx = horizontal_axes[0], horizontal_axes[1]

	return {
		"center": center,
		"radius": radius,
		"extent": extent,
		"target_points": target_points,
		"up_vec": axis_vectors[vertical_axis],
		"h1_vec": axis_vectors[h1_idx],
		"h2_vec": axis_vectors[h2_idx],
	}


def create_preview_camera(name: str = "PreviewCamera"):
	cam_data = bpy.data.cameras.new(name)
	cam_obj = bpy.data.objects.new(name, cam_data)
	bpy.context.collection.objects.link(cam_obj)
	return cam_obj, cam_data


def fit_camera_to_points(cam_obj, cam_data, center, start_loc, points, margin=CAMERA_FIT_MARGIN, max_iter=16, safety_scale=CAMERA_SAFETY_SCALE):
	if len(points) == 0:
		return

	view_from_center = start_loc - center
	if view_from_center.length < 1e-6:
		view_from_center = Vector((1.0, -1.0, 0.8))
	view_from_center.normalize()

	base_dist = max((start_loc - center).length, 0.5)
	dist = base_dist

	def points_fit(test_dist):
		cam_obj.location = center + view_from_center * test_dist
		bpy.context.view_layer.update()

		tan_x = math.tan(cam_data.angle_x * 0.5)
		tan_y = math.tan(cam_data.angle_y * 0.5)

		for p in points:
			pc = cam_obj.matrix_world.inverted() @ p
			depth = -pc.z
			if depth <= 1e-4:
				return False
			if abs(pc.x) > depth * tan_x / margin:
				return False
			if abs(pc.y) > depth * tan_y / margin:
				return False
		return True

	while not points_fit(dist):
		dist *= 1.18
		if dist > base_dist * 12.0:
			break

	low = max(base_dist * 0.6, 0.2)
	high = dist
	for _ in range(max_iter):
		mid = 0.5 * (low + high)
		if points_fit(mid):
			high = mid
		else:
			low = mid

	cam_obj.location = center + view_from_center * (high * safety_scale)
	bpy.context.view_layer.update()


def _dynamic_resolution_for_mode(render_ctx: dict, render_mode: str) -> tuple[int, int]:
	if not DYNAMIC_RESOLUTION:
		return RESOLUTION_X, RESOLUTION_Y

	ext = render_ctx["extent"]
	span_x = max(float(ext.x), 1e-4)
	span_y = max(float(ext.y), 1e-4)
	span_z = max(float(ext.z), 1e-4)
	if render_mode == "top":
		raw_aspect = span_x / span_y
		aspect = max(raw_aspect, 1.0 / raw_aspect)  # enforce landscape
	else:
		horiz = max(span_x, span_y)
		vert = max(0.6 * span_z + 0.55 * min(span_x, span_y), 1e-4)
		aspect = max(horiz / vert, 1.0)

	long_edge = max(512, BASE_LONG_EDGE)
	short_edge = int(round(long_edge / max(aspect, 1e-4)))
	short_edge = max(480, min(long_edge, max(MIN_SHORT_EDGE, short_edge)))
	return int(long_edge), int(short_edge)


def setup_camera_for_mode(cam_obj, cam_data, render_ctx: dict, render_mode: str):
	center = render_ctx["center"]
	radius = render_ctx["radius"]
	extent = render_ctx["extent"]
	h1_vec = render_ctx["h1_vec"]
	h2_vec = render_ctx["h2_vec"]
	up_vec = render_ctx["up_vec"]
	target_points = render_ctx["target_points"]

	if render_mode == "top":
		vertical_distance = radius * CAMERA_TOP_DIST_SCALE
		best_loc = center + up_vec * vertical_distance
	elif render_mode in {"isometric", "isometric_cw90", "isometric_ccw90"}:
		iso_dir = (h1_vec * 1.0 - h2_vec * 1.0 + up_vec * 1.0).normalized()
		if render_mode == "isometric_cw90":
			iso_dir = (Matrix.Rotation(-math.pi * 0.5, 4, up_vec) @ iso_dir).normalized()
		elif render_mode == "isometric_ccw90":
			iso_dir = (Matrix.Rotation(math.pi * 0.5, 4, up_vec) @ iso_dir).normalized()
		best_loc = center + iso_dir * (radius * CAMERA_ISO_DIST_SCALE)
	else:
		raise ValueError(f"Unsupported render mode: {render_mode}")

	if render_mode == "top":
		cam_obj.location = best_loc
		top_yaw = (math.pi * 0.5) if extent.y > extent.x else 0.0
		cam_obj.rotation_euler = (0.0, 0.0, top_yaw)
		cam_data.type = "ORTHO"
		aspect = max(
			bpy.context.scene.render.resolution_x / max(bpy.context.scene.render.resolution_y, 1),
			1e-6,
		)
		span_x = max(float(extent.x), 1.0)
		span_y = max(float(extent.y), 1.0)
		cam_data.ortho_scale = max(span_x * CAMERA_FIT_MARGIN, span_y * CAMERA_FIT_MARGIN * aspect) * CAMERA_SAFETY_SCALE
	else:
		cam_obj.location = best_loc
		direction = center - cam_obj.location
		cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
		cam_data.type = "PERSP"
		cam_data.lens = 32
		fit_camera_to_points(cam_obj, cam_data, center, best_loc, target_points)
	cam_data.clip_start = 0.1
	cam_data.clip_end = max(radius * 20, 1000)


def _capture_camera_state(cam_obj, cam_data) -> dict:
	scene = bpy.context.scene
	return {
		"location": cam_obj.location.copy(),
		"rotation_euler": cam_obj.rotation_euler.copy(),
		"type": cam_data.type,
		"lens": float(cam_data.lens),
		"ortho_scale": float(getattr(cam_data, "ortho_scale", 1.0)),
		"clip_start": float(cam_data.clip_start),
		"clip_end": float(cam_data.clip_end),
		"res_x": int(scene.render.resolution_x),
		"res_y": int(scene.render.resolution_y),
	}


def _apply_camera_state(cam_obj, cam_data, state: dict):
	scene = bpy.context.scene
	cam_obj.location = state["location"].copy()
	cam_obj.rotation_euler = state["rotation_euler"].copy()
	cam_data.type = state["type"]
	cam_data.lens = state["lens"]
	cam_data.ortho_scale = state.get("ortho_scale", cam_data.ortho_scale)
	cam_data.clip_start = state["clip_start"]
	cam_data.clip_end = state["clip_end"]
	scene.render.resolution_x = state.get("res_x", scene.render.resolution_x)
	scene.render.resolution_y = state.get("res_y", scene.render.resolution_y)
	bpy.context.view_layer.update()


def render_view(cam, output_path: str):
	scene = bpy.context.scene
	scene.camera = cam
	scene.render.filepath = output_path
	bpy.ops.render.render(write_still=True)


def _rotation_matrix_z_around_point(point_xy: Vector, angle_deg: float) -> Matrix:
	angle_rad = math.radians(angle_deg)
	p = Vector((point_xy.x, point_xy.y, 0.0))
	return Matrix.Translation(p) @ Matrix.Rotation(angle_rad, 4, "Z") @ Matrix.Translation(-p)


def _rotate_group_around_point(group_objs: list, point_xy: Vector, angle_deg: float) -> dict:
	orig = {obj.name: obj.matrix_world.copy() for obj in group_objs}
	mat = _rotation_matrix_z_around_point(point_xy, angle_deg)
	for obj in group_objs:
		obj.matrix_world = mat @ obj.matrix_world
	bpy.context.view_layer.update()
	return orig


def _restore_group_matrices(group_objs: list, orig: dict):
	for obj in group_objs:
		if obj.name in orig:
			obj.matrix_world = orig[obj.name]
	bpy.context.view_layer.update()


def aabb_overlap_xy(bounds_a: dict, bounds_b: dict, inflate: float = 0.0) -> bool:
	a_min_x = bounds_a["min_x"] - inflate
	a_max_x = bounds_a["max_x"] + inflate
	a_min_y = bounds_a["min_y"] - inflate
	a_max_y = bounds_a["max_y"] + inflate
	b_min_x = bounds_b["min_x"] - inflate
	b_max_x = bounds_b["max_x"] + inflate
	b_min_y = bounds_b["min_y"] - inflate
	b_max_y = bounds_b["max_y"] + inflate

	if a_max_x < b_min_x or a_min_x > b_max_x:
		return False
	if a_max_y < b_min_y or a_min_y > b_max_y:
		return False
	return True


def relation_2d_with_pad(target_group: list, other_group: list, pad: float) -> str:
	b_t = _group_bounds(target_group)
	b_o = _group_bounds(other_group)
	if not aabb_overlap_xy(b_t, b_o, inflate=pad):
		return "SAFE"
	if aabb_overlap_xy(b_t, b_o, inflate=0.0):
		return "COLLIDE"
	return "NEAR"


def _rotate_dir_to_signed_deg(direction: str, deg: int) -> int:
	return -deg if direction == "cw" else deg


def _dir_token_to_words(direction: str) -> str:
	return "clockwise" if direction == "cw" else "counterclockwise"


def _relation_after_rotation(target_group: list, other_group: list, pivot_xy: Vector, signed_deg: float, pad: float) -> str:
	orig = _rotate_group_around_point(target_group, pivot_xy, signed_deg)
	try:
		return relation_2d_with_pad(target_group, other_group, pad=pad)
	finally:
		_restore_group_matrices(target_group, orig)


def _task2_status_map(target_group: list, other_group: list, pivot_xy: Vector, direction: str) -> dict:
	status = {}
	for deg in ROT_ANGLE_POOL:
		signed = _rotate_dir_to_signed_deg(direction, deg)
		status[deg] = _relation_after_rotation(target_group, other_group, pivot_xy, signed_deg=signed, pad=COLLISION_PAD)
	return status


def _ensure_task2_unique_safe_for_direction(target_group: list, other_group: list, pivot_xy: Vector, direction: str):
	status = _task2_status_map(target_group, other_group, pivot_xy, direction)
	safe = [d for d, rel in status.items() if rel == "SAFE"]
	collide = [d for d, rel in status.items() if rel == "COLLIDE"]
	if len(safe) < 1 or len(collide) < (ROT_MCQ_K - 1):
		return None
	correct_deg = random.choice(safe)
	options = [correct_deg] + random.sample(collide, ROT_MCQ_K - 1)
	return options, correct_deg


def _pick_task2_direction_and_options(target_group: list, other_group: list, pivot_xy: Vector):
	dirs = ROT_DIR_POOL[:]
	random.shuffle(dirs)
	for d in dirs:
		res = _ensure_task2_unique_safe_for_direction(target_group, other_group, pivot_xy, d)
		if res is not None:
			options_deg, correct_deg = res
			return d, options_deg, correct_deg
	return None


def _abs_images(sample_dir: Path, names: list[str]) -> list[str]:
	return [str((sample_dir / n).resolve()) for n in names]


def qa_task1_overlap_yesno(sample_dir: Path, target_id: int, other_id: int, direction: str, angle_deg: int, top_image_rel: str = "top.png") -> dict:
	dir_words = _dir_token_to_words(direction)
	q = (
		"There is one image: a top-view image.\n"
		f"If object {target_id} is rotated {dir_words} by {angle_deg} degrees around its own center, "
		f"will it overlap with object {other_id}? Answer with Yes or No.\n\n"
		"Top-view image: <image>"
	)
	return {
		"question": q,
		"answer": "<to_fill>",
		"task_type": "top_building_rotate_overlap",
		"images": _abs_images(sample_dir, [top_image_rel]),
		"meta": {"target_id": target_id, "other_id": other_id, "direction": dir_words, "angle_deg": angle_deg},
	}


def qa_task2_avoid_collision_angle_mcq(sample_dir: Path, target_id: int, other_id: int, direction: str, options_deg: list[int], correct_deg: int, top_image_rel: str = "top.png") -> dict:
	dir_words = _dir_token_to_words(direction)
	labels = ["A", "B", "C", "D"]
	opts = options_deg[:]
	random.shuffle(opts)
	correct_idx = opts.index(correct_deg)
	choices = [f"{labels[i]}. {opts[i]} degrees" for i in range(len(opts))]
	q = (
		"There is one image: a top-view image.\n"
		f"Rotate object {target_id} {dir_words} by one of the angles below (around its own center). "
		f"Which rotation angle avoids overlap with object {other_id}? Choose one option.\n\n"
		"Top-view image: <image>"
	)
	return {
		"question": q + "\n" + "\n".join(choices),
		"answer": labels[correct_idx],
		"answer_text": f"{correct_deg} degrees",
		"task_type": "top_building_rotate_avoid_overlap",
		"images": _abs_images(sample_dir, [top_image_rel]),
		"meta": {"target_id": target_id, "other_id": other_id, "direction": dir_words, "options_deg": opts, "correct_deg": correct_deg},
	}


def qa_task3_rotation_image_mcq(sample_dir: Path, target_id: int, direction: str, query_deg: int, option_files: list[str], correct_file: str, view: str = "top", origin_image_rel: str | None = None) -> dict:
	dir_words = _dir_token_to_words(direction)
	labels = ["A", "B", "C", "D"]
	files = option_files[:]
	random.shuffle(files)
	correct_idx = files.index(correct_file)

	if view == "top":
		q = (
			"There are five images: one original top-view image and four rotated top-view options.\n"
			f"Object {target_id} is rotated {dir_words} by {query_deg} degrees around its center.\n"
			"Which image matches this rotation? Choose one option.\n\n"
			"Original top-view image: <image>\n"
			"A: <image>\n"
			"B: <image>\n"
			"C: <image>\n"
			"D: <image>"
		)
		origin_rel = origin_image_rel or "top.png"
		img_list = [str((sample_dir / origin_rel).resolve())] + [str((sample_dir / f).resolve()) for f in files]
		task_type = "top_building_rotate"
	else:
		q = (
			"There are five images: one original isometric-view image and four option images.\n"
			f"In the original image, Object {target_id} is rotated {dir_words} by {query_deg} degrees around its center.\n"
			"Which option image shows the correct result after this rotation?\n"
			"Choose one option.\n\n"
			"Original image: <image>\n"
			"A: <image>\n"
			"B: <image>\n"
			"C: <image>\n"
			"D: <image>"
		)
		origin_rel = origin_image_rel or "isometric.png"
		img_list = [str((sample_dir / origin_rel).resolve())] + [str((sample_dir / f).resolve()) for f in files]
		task_type = "isometric_building_rotate"

	return {
		"question": q,
		"answer": labels[correct_idx],
		"answer_text": Path(correct_file).name,
		"task_type": task_type,
		"images": img_list,
		"meta": {
			"target_id": target_id,
			"direction": dir_words,
			"query_deg": query_deg,
			"files": files,
			"correct_file": correct_file,
			"view": view,
		},
	}


def qa_task4_rotation_position_mcq(sample_dir: Path, target_id: int, direction: str, query_deg: int, option_files: list[str], correct_file: str, view: str = "top", origin_image_rel: str | None = None) -> dict:
	dir_words = _dir_token_to_words(direction)
	labels = ["A", "B", "C", "D"]
	files = option_files[:]
	random.shuffle(files)
	correct_idx = files.index(correct_file)

	if view == "top":
		orig_rel = origin_image_rel or "top.png"
		orig_img = str((sample_dir / orig_rel).resolve())
		q = (
			"The first image is the original top-view of the indoor scene.\n"
			f"In the original image, Object {target_id} is rotated {dir_words} by {query_deg} degrees around the center of the scene.\n"
			"Which option shows the correct result after this rotation?\n"
			"Answer with A, B, C, or D.\n\n"
			"Original image: <image>\n"
			"A: <image>\n"
			"B: <image>\n"
			"C: <image>\n"
			"D: <image>"
		)
		task_type = "top_region_building_rotate"
	else:
		orig_rel = origin_image_rel or "isometric.png"
		orig_img = str((sample_dir / orig_rel).resolve())
		q = (
			"The first image is the original isometric view of the indoor scene.\n"
			f"In the original image, Object {target_id} is rotated {dir_words} by {query_deg} degrees around the center of the scene.\n"
			"Which option shows the correct result after this rotation?\n"
			"Answer with A, B, C, or D.\n\n"
			"Original image: <image>\n"
			"A: <image>\n"
			"B: <image>\n"
			"C: <image>\n"
			"D: <image>"
		)
		task_type = "isometric_region_building_rotate"

	return {
		"question": q,
		"answer": labels[correct_idx],
		"answer_text": Path(correct_file).name,
		"task_type": task_type,
		"images": [orig_img] + [str((sample_dir / f).resolve()) for f in files],
		"meta": {
			"target_id": target_id,
			"direction": dir_words,
			"query_deg": query_deg,
			"files": files,
			"correct_file": correct_file,
			"view": view,
			"rotation_type": "around_region_center",
		},
	}


def _build_logical_objects(layout_objects: list[dict], groups: dict[int, list]) -> list[dict]:
	logical = []
	for obj in layout_objects:
		idx = obj["instance_index"]
		g = groups.get(idx)
		if not g:
			continue
		b = _group_bounds(g)
		logical.append(
			{
				"id": obj["id"],
				"instance_index": idx,
				"group": g,
				"center": (b["center_x"], b["center_y"], b["center_z"]),
				"height": b["height"],
			}
		)
	return logical


def _pick_unique_angles_for_qa4(direction: str, target_center_xy: Vector, scene_center_xy: Vector):
	if (target_center_xy - scene_center_xy).length < 0.2:
		return None
	for _ in range(100):
		candidate = random.sample(ROT_ANGLE_POOL, ROT_IMG_MCQ_K)
		positions = []
		for deg in candidate:
			signed = _rotate_dir_to_signed_deg(direction, deg)
			angle_rad = math.radians(signed)
			rel = target_center_xy - scene_center_xy
			new_x = scene_center_xy.x + rel.x * math.cos(angle_rad) - rel.y * math.sin(angle_rad)
			new_y = scene_center_xy.y + rel.x * math.sin(angle_rad) + rel.y * math.cos(angle_rad)
			positions.append((round(new_x, 4), round(new_y, 4)))
		if len(set(positions)) == ROT_IMG_MCQ_K:
			return candidate
	return None


def process_scene(glb_name: str, info: dict, glb_dir: Path, output_dir: Path) -> dict | None:
	layout_info = info.get("layout_info", [])
	scene_name = info.get("scene_name", glb_name)
	layout_objects = convert_layout_to_objects(layout_info)
	if len(layout_objects) < MIN_OBJECTS:
		return None

	glb_path = glb_dir / glb_name
	if not glb_path.exists():
		return None

	bpy.ops.wm.read_factory_settings(use_empty=True)
	bpy.ops.import_scene.gltf(filepath=str(glb_path))

	groups = collect_instance_groups(max_instance_idx=len(layout_objects))
	logical = _build_logical_objects(layout_objects, groups)
	if len(logical) < MIN_OBJECTS:
		return None

	dc_utils.set_render_and_world(SAMPLES, RESOLUTION_X, RESOLUTION_Y, transparent_bg=True)
	bounds = _scene_bounds_from_groups(groups)
	scene_center = Vector((bounds["center_x"], bounds["center_y"], 0.0))
	scene_radius = max(bounds["width"], bounds["depth"], 1.0)
	dc_utils.setup_lighting(Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"])), scene_radius)
	render_ctx = dc_utils.compute_render_frame(_all_mesh_objects())
	wall_targets = dc_utils._resolve_wall_targets(_all_mesh_objects())
	north_world_dir = dc_utils.canonical_north_world(render_ctx)
	cam_obj, cam_data = dc_utils.create_preview_camera()
	camera_states = {}
	for state_mode, render_mode in (("top", "top"), ("isometric", FIXED_ISOMETRIC_MODE)):
		dyn_w, dyn_h = dc_utils.dynamic_resolution_for_mode(render_ctx, render_mode, BASE_LONG_EDGE, MIN_SHORT_EDGE)
		bpy.context.scene.render.resolution_x = dyn_w
		bpy.context.scene.render.resolution_y = dyn_h
		dc_utils.setup_camera_for_mode(
			cam_obj,
			cam_data,
			render_ctx,
			render_mode,
			top_dist_scale=CAMERA_TOP_DIST_SCALE,
			iso_dist_scale=CAMERA_ISO_DIST_SCALE,
			fit_margin=CAMERA_FIT_MARGIN,
			fit_safety=CAMERA_SAFETY_SCALE,
		)
		camera_states[state_mode] = dc_utils.capture_camera_state(cam_obj, cam_data)

	id_to_entry = {e["id"]: e for e in logical}
	valid_ids = [e["id"] for e in logical]
	# Per-view visibility prefilter for involved objects.
	dc_utils.apply_camera_state(cam_obj, cam_data, camera_states["top"])
	rw_top = int(bpy.context.scene.render.resolution_x)
	rh_top = int(bpy.context.scene.render.resolution_y)
	top_visible_ids = [oid for oid in valid_ids if _top_group_visibility_ok(id_to_entry[oid]["group"], cam_obj, rw_top, rh_top)]
	dc_utils.apply_camera_state(cam_obj, cam_data, camera_states["isometric"])
	rw_iso_all = int(bpy.context.scene.render.resolution_x)
	rh_iso_all = int(bpy.context.scene.render.resolution_y)
	iso_visible_ids = [oid for oid in valid_ids if _isometric_group_visibility_ok(id_to_entry[oid]["group"], cam_obj, rw_iso_all, rh_iso_all)]
	valid_ids = [oid for oid in valid_ids if oid in top_visible_ids and oid in iso_visible_ids]
	if len(valid_ids) < MIN_OBJECTS:
		return None

	id_to_entry = {oid: id_to_entry[oid] for oid in valid_ids}
	label_map = assign_independent_labels([{"id": oid} for oid in valid_ids])
	if len(valid_ids) < MIN_OBJECTS:
		return None

	# Filter out near-round targets for all top-involved QA (qa1/2/3/4).
	non_round_ids = [oid for oid in valid_ids if not _is_near_round_top(id_to_entry[oid]["group"])]
	if len(non_round_ids) < 1:
		return None
	target_id = random.choice(non_round_ids)
	other_id = random.choice([x for x in valid_ids if x != target_id])
	target_entry = id_to_entry[target_id]
	other_entry = id_to_entry[other_id]
	target_group = target_entry["group"]
	other_group = other_entry["group"]
	target_center_xy = _group_center_xy(target_group)

	stem = Path(glb_name).stem
	sample_dir = output_dir / stem
	sample_dir.mkdir(parents=True, exist_ok=True)
	TMP_ROOT_DIR.mkdir(parents=True, exist_ok=True)
	temp_ctx = None
	if KEEP_RAW_VIEWS:
		raw_dir = sample_dir / "raw_views"
		raw_dir.mkdir(parents=True, exist_ok=True)
	elif KEEP_SCENE_VIEWS:
		raw_dir = sample_dir / "scene_views"
		raw_dir.mkdir(parents=True, exist_ok=True)
	else:
		temp_ctx = tempfile.TemporaryDirectory(prefix=f"{stem}_scene_views_", dir=str(TMP_ROOT_DIR))
		raw_dir = Path(temp_ctx.name)
	ref_dir = sample_dir / "ref_rotations"
	top_mcq_dir = sample_dir / "top_mcq_images"
	iso_mcq_dir = sample_dir / "isometric_mcq_images"
	qa4_top_dir = sample_dir / "top_scene_building_rotate_images"
	qa4_iso_dir = sample_dir / "iso_scene_building_rotate_images"
	for d in (ref_dir, top_mcq_dir, iso_mcq_dir, qa4_top_dir, qa4_iso_dir):
		d.mkdir(parents=True, exist_ok=True)

	def render_labeled(view_mode: str, qa_label_map: dict[str, int], rel_out_path: str, rel_raw_path: str | None = None):
		raw_rel = rel_raw_path or str(Path("raw") / Path(rel_out_path).name)
		raw_abs = raw_dir / raw_rel
		raw_abs.parent.mkdir(parents=True, exist_ok=True)
		dc_utils.apply_camera_state(cam_obj, cam_data, camera_states[view_mode])
		wall_alpha = TOP_WALL_ALPHA if str(view_mode) == "top" else ISO_WALL_ALPHA
		dc_utils._render_view_with_wall_alpha(cam_obj, str(raw_abs), wall_targets, wall_alpha)
		rw = int(bpy.context.scene.render.resolution_x)
		rh = int(bpy.context.scene.render.resolution_y)
		centers = dc_utils.project_group_centers(id_to_entry, list(qa_label_map.keys()), cam_obj, rw, rh)
		north_vec = dc_utils.north_screen_vector(cam_obj, north_world_dir=north_world_dir)
		px_per_unit = dc_utils.scale_marker_px_per_unit(cam_obj, rw)
		out_abs = sample_dir / rel_out_path
		_annotate_numbers(raw_abs, out_abs, qa_label_map, centers, north_vec, px_per_unit)
		return str(out_abs.resolve())

	# Base images
	# No scene-level base images; keep only QA-used images.

	# QA1
	dir1 = random.choice(ROT_DIR_POOL)
	angle1 = random.choice(ROT_ANGLE_POOL)
	signed1 = _rotate_dir_to_signed_deg(dir1, angle1)
	rel1 = _relation_after_rotation(target_group, other_group, target_center_xy, signed1, COLLISION_PAD)
	overlap1 = rel1 == "COLLIDE"

	qa1_map = assign_independent_labels([{"id": target_id}, {"id": other_id}])
	qa1_top_rel = "qa1_top.png"
	render_labeled("top", qa1_map, qa1_top_rel, rel_raw_path="qa1/top_raw.png")
	qa1 = qa_task1_overlap_yesno(sample_dir, qa1_map[target_id], qa1_map[other_id], dir1, angle1, top_image_rel=qa1_top_rel)
	qa1["answer"] = "Yes" if overlap1 else "No"

	# QA1 reference image
	orig = _rotate_group_around_point(target_group, target_center_xy, signed1)
	render_labeled("top", qa1_map, str(Path("ref_rotations") / f"qa1_ref_{dir1}_{angle1:03d}.png"), rel_raw_path=str(Path("qa1") / f"ref_{dir1}_{angle1:03d}_raw.png"))
	_restore_group_matrices(target_group, orig)

	# QA2
	task2_result = _pick_task2_direction_and_options(target_group, other_group, target_center_xy)
	qa2 = None
	if task2_result is not None:
		qa2_map = assign_independent_labels([{"id": target_id}, {"id": other_id}])
		qa2_top_rel = "qa2_top.png"
		render_labeled("top", qa2_map, qa2_top_rel, rel_raw_path="qa2/top_raw.png")
		dir2, options_deg, correct_deg = task2_result
		qa2 = qa_task2_avoid_collision_angle_mcq(sample_dir, qa2_map[target_id], qa2_map[other_id], dir2, options_deg, correct_deg, top_image_rel=qa2_top_rel)
		for deg in options_deg:
			signed = _rotate_dir_to_signed_deg(dir2, deg)
			orig = _rotate_group_around_point(target_group, target_center_xy, signed)
			render_labeled("top", qa2_map, str(Path("ref_rotations") / f"qa2_ref_{dir2}_{deg:03d}.png"), rel_raw_path=str(Path("qa2") / f"ref_{dir2}_{deg:03d}_raw.png"))
			_restore_group_matrices(target_group, orig)

	# QA3 (top)
	dir3 = random.choice(ROT_DIR_POOL)
	option_degs = random.sample(ROT_ANGLE_POOL, ROT_IMG_MCQ_K)
	query_deg = random.choice(option_degs)
	qa3_map = assign_independent_labels([{"id": target_id}])
	qa3_top_origin_rel = "qa3_top_origin.png"
	render_labeled("top", qa3_map, qa3_top_origin_rel, rel_raw_path="qa3/top_origin_raw.png")

	option_files_top = []
	correct_file_top = None
	for deg in option_degs:
		signed = _rotate_dir_to_signed_deg(dir3, deg)
		orig = _rotate_group_around_point(target_group, target_center_xy, signed)
		rel_fp = str(Path("top_mcq_images") / f"rot_{dir3}_{deg:03d}.png")
		render_labeled("top", qa3_map, rel_fp, rel_raw_path=str(Path("qa3") / f"top_rot_{dir3}_{deg:03d}_raw.png"))
		_restore_group_matrices(target_group, orig)
		option_files_top.append(rel_fp)
		if deg == query_deg:
			correct_file_top = rel_fp

	qa3 = None
	if correct_file_top:
		qa3 = qa_task3_rotation_image_mcq(sample_dir, qa3_map[target_id], dir3, query_deg, option_files_top, correct_file_top, view="top", origin_image_rel=qa3_top_origin_rel)

	# QA3 (iso) with retry over candidate objects.
	qa_iso = None
	qa3_iso_origin_rel = "qa3_isometric_origin.png"
	dc_utils.apply_camera_state(cam_obj, cam_data, camera_states["isometric"])
	rw_iso = int(bpy.context.scene.render.resolution_x)
	rh_iso = int(bpy.context.scene.render.resolution_y)

	qa3_iso_candidates = non_round_ids[:]
	random.shuffle(qa3_iso_candidates)
	qa3_iso_candidates = qa3_iso_candidates[: max(1, QA3_ISO_RETRY_MAX_OBJECTS)]

	for iso_target_id in qa3_iso_candidates:
		iso_entry = id_to_entry[iso_target_id]
		iso_group = iso_entry["group"]
		iso_center_xy = _group_center_xy(iso_group)

		# Try different query angle choices for this candidate.
		query_try = option_degs[:]
		random.shuffle(query_try)
		picked_query_deg = None
		for qdeg in query_try:
			vis_ok = _isometric_group_visibility_ok(iso_group, cam_obj, rw_iso, rh_iso)
			if not vis_ok:
				continue
			signed_q = _rotate_dir_to_signed_deg(dir3, qdeg)
			orig_vis = _rotate_group_around_point(iso_group, iso_center_xy, signed_q)
			dc_utils.apply_camera_state(cam_obj, cam_data, camera_states["isometric"])
			vis_after = _isometric_group_visibility_ok(iso_group, cam_obj, rw_iso, rh_iso)
			_restore_group_matrices(iso_group, orig_vis)
			if vis_after:
				picked_query_deg = qdeg
				break

		if picked_query_deg is None:
			continue

		qa3_iso_map = assign_independent_labels([{"id": iso_target_id}])
		render_labeled("isometric", qa3_iso_map, qa3_iso_origin_rel, rel_raw_path="qa3/isometric_origin_raw.png")
		option_files_iso = []
		correct_file_iso = None
		for deg in option_degs:
			signed = _rotate_dir_to_signed_deg(dir3, deg)
			orig = _rotate_group_around_point(iso_group, iso_center_xy, signed)
			rel_fp = str(Path("isometric_mcq_images") / f"iso_rot_{dir3}_{deg:03d}.png")
			render_labeled("isometric", qa3_iso_map, rel_fp, rel_raw_path=str(Path("qa3") / f"iso_rot_{dir3}_{deg:03d}_raw.png"))
			_restore_group_matrices(iso_group, orig)
			option_files_iso.append(rel_fp)
			if deg == picked_query_deg:
				correct_file_iso = rel_fp

		if correct_file_iso:
			qa_iso = qa_task3_rotation_image_mcq(
				sample_dir,
				qa3_iso_map[iso_target_id],
				dir3,
				picked_query_deg,
				option_files_iso,
				correct_file_iso,
				view="isometric",
				origin_image_rel=qa3_iso_origin_rel,
			)
			break

	# QA4
	qa4 = None
	qa4_iso = None
	qa4_target_label = None
	qa4_candidates = [x for x in non_round_ids if x != target_id]
	if qa4_candidates:
		qa4_target_id = random.choice(qa4_candidates)
		qa4_map = assign_independent_labels([{"id": qa4_target_id}])
		qa4_target_label = qa4_map[qa4_target_id]
		qa4_entry = id_to_entry[qa4_target_id]
		qa4_group = qa4_entry["group"]
		qa4_center_xy = _group_center_xy(qa4_group)
		dir4 = random.choice(ROT_DIR_POOL)
		option_degs4 = _pick_unique_angles_for_qa4(dir4, qa4_center_xy, scene_center)
		if option_degs4:
			query_deg4 = random.choice(option_degs4)
			qa4_top_origin_rel = "qa4_top_origin.png"
			render_labeled("top", qa4_map, qa4_top_origin_rel, rel_raw_path="qa4/top_origin_raw.png")
			option_files_qa4_top = []
			correct_file_qa4_top = None
			for deg in option_degs4:
				signed = _rotate_dir_to_signed_deg(dir4, deg)
				orig = _rotate_group_around_point(qa4_group, scene_center, signed)
				rel_fp = str(Path("top_scene_building_rotate_images") / f"rot_center_{dir4}_{deg:03d}.png")
				render_labeled("top", qa4_map, rel_fp, rel_raw_path=str(Path("qa4") / f"top_center_{dir4}_{deg:03d}_raw.png"))
				_restore_group_matrices(qa4_group, orig)
				option_files_qa4_top.append(rel_fp)
				if deg == query_deg4:
					correct_file_qa4_top = rel_fp

			# Visibility filter for isometric QA4 (original + query rotation).
			dc_utils.apply_camera_state(cam_obj, cam_data, camera_states["isometric"])
			rw_iso = int(bpy.context.scene.render.resolution_x)
			rh_iso = int(bpy.context.scene.render.resolution_y)
			qa4_iso_ok = _isometric_group_visibility_ok(qa4_group, cam_obj, rw_iso, rh_iso)
			if qa4_iso_ok:
				signed_q4 = _rotate_dir_to_signed_deg(dir4, query_deg4)
				orig_vis4 = _rotate_group_around_point(qa4_group, scene_center, signed_q4)
				dc_utils.apply_camera_state(cam_obj, cam_data, camera_states["isometric"])
				qa4_iso_ok = _isometric_group_visibility_ok(qa4_group, cam_obj, rw_iso, rh_iso)
				_restore_group_matrices(qa4_group, orig_vis4)

			option_files_qa4_iso = []
			correct_file_qa4_iso = None
			qa4_iso_origin_rel = "qa4_isometric_origin.png"
			if qa4_iso_ok:
				render_labeled("isometric", qa4_map, qa4_iso_origin_rel, rel_raw_path="qa4/isometric_origin_raw.png")
				for deg in option_degs4:
					signed = _rotate_dir_to_signed_deg(dir4, deg)
					orig = _rotate_group_around_point(qa4_group, scene_center, signed)
					rel_fp = str(Path("iso_scene_building_rotate_images") / f"iso_rot_center_{dir4}_{deg:03d}.png")
					render_labeled("isometric", qa4_map, rel_fp, rel_raw_path=str(Path("qa4") / f"iso_center_{dir4}_{deg:03d}_raw.png"))
					_restore_group_matrices(qa4_group, orig)
					option_files_qa4_iso.append(rel_fp)
					if deg == query_deg4:
						correct_file_qa4_iso = rel_fp

			if correct_file_qa4_top:
				qa4 = qa_task4_rotation_position_mcq(sample_dir, qa4_target_label, dir4, query_deg4, option_files_qa4_top, correct_file_qa4_top, view="top", origin_image_rel=qa4_top_origin_rel)
			if correct_file_qa4_iso:
				qa4_iso = qa_task4_rotation_position_mcq(sample_dir, qa4_target_label, dir4, query_deg4, option_files_qa4_iso, correct_file_qa4_iso, view="isometric", origin_image_rel=qa4_iso_origin_rel)

	if temp_ctx is not None:
		temp_ctx.cleanup()

	qa_top = [qa1]
	if qa2:
		qa_top.append(qa2)
	if qa3:
		qa_top.append(qa3)
	if qa4:
		qa_top.append(qa4)

	qa_iso_list = []
	if qa_iso:
		qa_iso_list.append(qa_iso)
	if qa4_iso:
		qa_iso_list.append(qa4_iso)

	label_mapping = [
		{"scene_object_id": oid, "label_id": label_map[oid], "instance_index": id_to_entry[oid]["instance_index"]}
		for oid in sorted(valid_ids, key=lambda x: label_map[x])
	]

	res = {
		"glb_name": glb_name,
		"scene_name": scene_name,
		"object_count": len(valid_ids),
		"label_mapping": label_mapping,
		"qa": {
			"top": qa_top,
			"isometric": qa_iso_list,
		},
	"special_refs": {
		"target_id": qa1_map[target_id],
		"other_id": qa1_map[other_id],
		"qa4_target_id": qa4_target_label,
			"scene_center_definition": "center of the bounding rectangle of all objects (x/y min-max midpoint)",
			"scene_center_xy": [bounds["center_x"], bounds["center_y"]],
			"collision_pad": COLLISION_PAD,
		},
	}
	return res


def main():
	args = parse_args()
	random.seed(args.seed)

	glb_dir = args.glb_dir.resolve()
	mapping_json = args.mapping_json.resolve()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not glb_dir.exists():
		raise FileNotFoundError(f"GLB dir not found: {glb_dir}")
	if not mapping_json.exists():
		raise FileNotFoundError(f"Mapping json not found: {mapping_json}")

	with open(mapping_json, "r", encoding="utf-8") as f:
		scene_mapping = json.load(f)

	glb_files = sorted([p.name for p in glb_dir.glob("*.glb") if p.is_file()])
	scene_keys = [k for k in glb_files if k in scene_mapping]
	if args.region:
		region_kw = args.region.strip().lower()
		scene_keys = [
			k
			for k in scene_keys
			if (
				region_kw in k.lower()
				or region_kw in str(scene_mapping.get(k, {}).get("scene_name", "")).lower()
			)
		]
	if args.max_scenes > 0:
		scene_keys = scene_keys[: args.max_scenes]

	print("=" * 80)
	print("Indoor mental rotation QA construct")
	print(f"Total glb files: {len(glb_files)}")
	print(f"Matched in mapping: {len(scene_keys)}")
	if args.region:
		print(f"Region filter: {args.region}")

	all_results = []
	for idx, glb_name in enumerate(scene_keys, start=1):
		try:
			result = process_scene(glb_name, scene_mapping[glb_name], glb_dir, output_dir)
		except Exception as e:
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: exception {e}")
			continue
		if not result:
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: invalid scene")
			continue
		all_results.append(result)
		print(
			f"[{idx}/{len(scene_keys)}] done {glb_name}: "
			f"qa_top={len(result['qa']['top'])} qa_iso={len(result['qa']['isometric'])}"
		)

	out_json = output_dir / "metadata_indoor.json"
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(all_results, f, ensure_ascii=False, indent=2)

	print("=" * 80)
	print(f"Done. valid_scenes={len(all_results)}/{len(scene_keys)}")
	print(f"Output: {out_json}")


if __name__ == "__main__":
	main()
