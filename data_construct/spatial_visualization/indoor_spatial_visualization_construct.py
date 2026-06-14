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
import time
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path.home() / "SpatialAct")))
BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", str(PROJECT_ROOT / "blender-3.2.2-linux-x64/blender")))
INTERNSCENES_ROOT = Path(os.environ.get("INTERNSCENES_ROOT", str(Path.home() / "InternScenes")))
SHARED_UTILS_PATH = PROJECT_ROOT / "benchmark/data_construct/utils.py"

_utils_spec = importlib.util.spec_from_file_location("dc_render_utils_task4", SHARED_UTILS_PATH)
if _utils_spec is None or _utils_spec.loader is None:
	raise RuntimeError(f"Cannot load shared utils from {SHARED_UTILS_PATH}")
dc_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(dc_utils)

try:
	import bpy
	from mathutils import Matrix, Vector
	from mathutils.bvhtree import BVHTree
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
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark/data/spatial_visualization/indoor_scenes_complex-10-15"
DEFAULT_WHITELIST_JSON = INTERNSCENES_ROOT / "scenes/normal_overlap_whitelist.json"
RESOLUTION_X = 1080
RESOLUTION_Y = 720
SAMPLES = 64
DYNAMIC_RESOLUTION = True
BASE_LONG_EDGE = 1280
MIN_SHORT_EDGE = 720

MOVE_DIR_POOL = ["E", "W", "N", "S"]
MOVE_DIST_POOL = [1, 2, 3, 5, 6, 10]
# Full 3D overlap checks are significantly heavier than 2D OBB checks.
# Keep tries moderate to avoid extreme per-scene runtime.
MOVE_TASK_TRIES = 600
SWAP_TASK_TRIES = 600
MOVE_OTHER_TOPK = 3

UNIT_SCALE = 1.0
LEFT_STRIP_UI_SCALE_MULT = 1.3  # Match task0 for isometric view
LEFT_STRIP_UI_SCALE_MULT_TOP = 1.55  # Match task0 for top view
COLLISION_PAD = 0.07
IN_BOUNDS_MARGIN = 0.5
MIN_OBJECTS = 3
# 3D BVH overlap threshold:
# require at least this many overlapping triangle pairs to count as collision.
# Raise this to reduce false positives from tiny grazes / numerical noise.
MIN_OVERLAP_TRI_PAIRS = 3

SCALE_BAR_HEIGHT = 12
SCALE_TICK_HEIGHT = 22
SCALE_MARGIN = 28

LABEL_RADIUS = 13
LABEL_FONT_SIZE = 16
KEEP_RAW_VIEWS = False
TOP_MIN_BBOX_PX = 700.0
TOP_MIN_AREA_RATIO = 0.0012
TOP_WALL_ALPHA = 1.0
ISO_WALL_ALPHA = 0.55

# Performance caches for repeated 3D overlap checks.
_GROUP_VERSION: dict[tuple[str, ...], int] = {}
_GROUP_BOUNDS_CACHE: dict[tuple[str, ...], tuple[int, dict]] = {}
_GROUP_BVH_CACHE: dict[tuple[str, ...], tuple[int, BVHTree | None]] = {}


def parse_args() -> argparse.Namespace:
	argv = sys.argv[1:]
	if "--" in argv:
		argv = argv[argv.index("--") + 1 :]

	parser = argparse.ArgumentParser(description="Construct indoor spatial visualization QA")
	parser.add_argument("--glb-dir", type=Path, default=DEFAULT_GLB_DIR)
	parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--whitelist-json", type=Path, default=DEFAULT_WHITELIST_JSON)
	parser.add_argument("--seed", type=int, default=30)
	parser.add_argument("--max-scenes", type=int, default=0, help="0 means all")
	parser.add_argument(
		"--min-overlap-tri-pairs",
		type=int,
		default=3,
		help="3D collision threshold: minimum overlapping triangle-pair count",
	)
	parser.add_argument(
		"--max-scene-seconds",
		type=float,
		default=180.0,
		help="Per-scene timeout in seconds; <=0 disables timeout",
	)
	parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
	parser.add_argument("--worker-index", type=int, default=-1, help="0-based worker index; -1 means parent mode")
	parser.add_argument(
		"--qa-pairs-per-scene",
		type=int,
		default=1,
		help="Number of (QA1 move + QA2 swap) pairs to generate per scene",
	)
	parser.add_argument(
		"--region",
		type=str,
		default="",
		help="Filter scenes by region keyword (match against scene_name and glb filename, case-insensitive)",
	)
	return parser.parse_args(argv)


def load_whitelist_pairs(path: Path) -> set[tuple[str, str]]:
	if not path.exists():
		return set()
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	pairs = set()
	for a, b in data.get("allowed_overlap_pairs", []):
		ca = str(a).strip().lower()
		cb = str(b).strip().lower()
		pairs.add(tuple(sorted((ca, cb))))
	return pairs


def _category_tokens(cat: str) -> set[str]:
	s = str(cat or "").lower().replace("_", " ").replace("-", " ")
	return {x for x in s.split() if x}


def _is_pair_whitelisted(cat_a: str, cat_b: str, whitelist_pairs: set[tuple[str, str]]) -> bool:
	a = str(cat_a).strip().lower()
	b = str(cat_b).strip().lower()
	if tuple(sorted((a, b))) in whitelist_pairs:
		return True
	ta = _category_tokens(a)
	tb = _category_tokens(b)
	for xa in ta:
		for xb in tb:
			if tuple(sorted((xa, xb))) in whitelist_pairs:
				return True
	return False


def _merge_worker_outputs(output_dir: Path, workers: int) -> tuple[list, list[Path]]:
	parts = [output_dir / f"metadata_indoor.worker_{i}_of_{workers}.json" for i in range(workers)]
	merged = []
	used = []
	for p in parts:
		if not p.exists():
			continue
		try:
			with open(p, "r", encoding="utf-8") as f:
				data = json.load(f)
			if isinstance(data, list):
				merged.extend(data)
			used.append(p)
		except Exception:
			continue
	# stable order by scene/glb for reproducibility
	merged.sort(key=lambda x: str(x.get("glb_name", "")))
	return merged, used


def _auto_merge_if_all_workers_ready(output_dir: Path, workers: int) -> bool:
	parts = [output_dir / f"metadata_indoor.worker_{i}_of_{workers}.json" for i in range(workers)]
	if not parts or any((not p.exists()) for p in parts):
		return False
	merged, used_parts = _merge_worker_outputs(output_dir, workers)
	out_json = output_dir / "metadata_indoor.json"
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(merged, f, ensure_ascii=False, indent=2)
	for p in used_parts:
		try:
			p.unlink()
		except Exception:
			pass
	print("=" * 80)
	print(f"Auto-merged workers -> {out_json}  (scenes={len(merged)})")
	return True


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
	key = tuple(sorted(obj.name for obj in group_objs))
	ver = _GROUP_VERSION.get(key, 0)
	cached = _GROUP_BOUNDS_CACHE.get(key)
	if cached is not None and cached[0] == ver:
		return cached[1]

	pts = []
	for obj in group_objs:
		for c in obj.bound_box:
			pts.append(obj.matrix_world @ Vector(c))
	xs = [p.x for p in pts]
	ys = [p.y for p in pts]
	zs = [p.z for p in pts]
	out = {
		"min_x": float(min(xs)),
		"max_x": float(max(xs)),
		"min_y": float(min(ys)),
		"max_y": float(max(ys)),
		"min_z": float(min(zs)),
		"max_z": float(max(zs)),
		"center_x": float((min(xs) + max(xs)) / 2),
		"center_y": float((min(ys) + max(ys)) / 2),
		"center_z": float((min(zs) + max(zs)) / 2),
		"width": float(max(xs) - min(xs)),
		"depth": float(max(ys) - min(ys)),
		"height": float(max(zs) - min(zs)),
		"half_w": float((max(xs) - min(xs)) / 2),
		"half_d": float((max(ys) - min(ys)) / 2),
	}
	_GROUP_BOUNDS_CACHE[key] = (ver, out)
	return out


def _project_group_bbox_metrics(group_objs: list, cam, width: int, height: int) -> tuple[float, float]:
	scene = bpy.context.scene
	xs, ys = [], []
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
		return False
	group_names = {o.name for o in group_objs}
	if hit_obj.name in group_names:
		return True
	try:
		if hit_obj.original is not None and hit_obj.original.name in group_names:
			return True
	except Exception:
		pass
	return False


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
		"min_x": float(min_x),
		"max_x": float(max_x),
		"min_y": float(min_y),
		"max_y": float(max_y),
		"min_z": float(min_z),
		"max_z": float(max_z),
		"center_x": float((min_x + max_x) / 2),
		"center_y": float((min_y + max_y) / 2),
		"center_z": float((min_z + max_z) / 2),
		"width": float(max_x - min_x),
		"depth": float(max_y - min_y),
		"height": float(max_z - min_z),
	}


def _compute_pca_obb_xy(points_xy: list[tuple[float, float]]) -> dict | None:
	if not points_xy:
		return None
	n = len(points_xy)
	mx = sum(p[0] for p in points_xy) / n
	my = sum(p[1] for p in points_xy) / n
	xx = sum((p[0] - mx) ** 2 for p in points_xy)
	yy = sum((p[1] - my) ** 2 for p in points_xy)
	xy = sum((p[0] - mx) * (p[1] - my) for p in points_xy)

	if abs(xy) < 1e-9:
		angle = 0.0 if xx >= yy else (math.pi / 2.0)
	else:
		angle = 0.5 * math.atan2(2.0 * xy, xx - yy)

	ca = math.cos(angle)
	sa = math.sin(angle)
	ax = (ca, sa)
	ay = (-sa, ca)

	rot = []
	for (x, y) in points_xy:
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


def _build_meta_from_group(label_id: int, scene_obj_id: str, instance_index: int, group_objs: list) -> dict:
	b = _group_bounds(group_objs)
	pts = []
	for obj in group_objs:
		verts = obj.data.vertices
		step = max(1, len(verts) // 600)
		for i in range(0, len(verts), step):
			w = obj.matrix_world @ verts[i].co
			pts.append((float(w.x), float(w.y)))

	obb = _compute_pca_obb_xy(pts)
	if obb is None:
		obb = {
			"angle": 0.0,
			"ax": (1.0, 0.0),
			"ay": (0.0, 1.0),
			"hx": b["half_w"],
			"hy": b["half_d"],
			"center": (b["center_x"], b["center_y"]),
		}

	return {
		"label_id": int(label_id),
		"scene_object_id": str(scene_obj_id),
		"instance_index": int(instance_index),
		"pos": (b["center_x"], b["center_y"]),
		"half_w": b["half_w"],
		"half_d": b["half_d"],
		"height": b["height"],
		"obb": obb,
	}


def _normalize2(v: tuple[float, float]) -> tuple[float, float]:
	l = math.hypot(v[0], v[1])
	if l < 1e-9:
		return (1.0, 0.0)
	return (v[0] / l, v[1] / l)


def _project_obb(center, ax, ay, hx, hy, axis):
	dots = []
	for sx in (-1.0, 1.0):
		for sy in (-1.0, 1.0):
			px = center[0] + ax[0] * (sx * hx) + ay[0] * (sy * hy)
			py = center[1] + ax[1] * (sx * hx) + ay[1] * (sy * hy)
			dots.append(px * axis[0] + py * axis[1])
	return min(dots), max(dots)


def _aabb_overlap_xy(ma: dict, mb: dict, inflate: float = 0.0, pos_a=None, pos_b=None) -> bool:
	ax, ay = pos_a if pos_a is not None else tuple(ma["obb"]["center"][:2])
	bx, by = pos_b if pos_b is not None else tuple(mb["obb"]["center"][:2])
	inf = float(inflate)

	min_ax = ax - ma["half_w"] - inf
	max_ax = ax + ma["half_w"] + inf
	min_ay = ay - ma["half_d"] - inf
	max_ay = ay + ma["half_d"] + inf

	min_bx = bx - mb["half_w"] - inf
	max_bx = bx + mb["half_w"] + inf
	min_by = by - mb["half_d"] - inf
	max_by = by + mb["half_d"] + inf

	if max_ax < min_bx or min_ax > max_bx:
		return False
	if max_ay < min_by or min_ay > max_by:
		return False
	return True


def _obb_overlap(ma: dict, mb: dict, inflate: float = 0.0, pos_a=None, pos_b=None) -> bool:
	oa = ma["obb"]
	ob = mb["obb"]

	ca = tuple(pos_a[:2]) if pos_a is not None else tuple(oa["center"][:2])
	cb = tuple(pos_b[:2]) if pos_b is not None else tuple(ob["center"][:2])

	ax_a, ay_a = oa["ax"], oa["ay"]
	ax_b, ay_b = ob["ax"], ob["ay"]

	hx_a = float(oa["hx"] + inflate)
	hy_a = float(oa["hy"] + inflate)
	hx_b = float(ob["hx"] + inflate)
	hy_b = float(ob["hy"] + inflate)

	axes = [ax_a, ay_a, ax_b, ay_b]
	for axis in axes:
		axis = _normalize2(axis)
		a0, a1 = _project_obb(ca, ax_a, ay_a, hx_a, hy_a, axis)
		b0, b1 = _project_obb(cb, ax_b, ay_b, hx_b, hy_b, axis)
		if a1 < b0 or b1 < a0:
			return False
	return True


def _relation_2d_with_pad(meta_a: dict, meta_b: dict, pad: float, moved_pos=None) -> str:
	pos_a = moved_pos if moved_pos is not None else None
	if not _aabb_overlap_xy(meta_a, meta_b, inflate=pad, pos_a=pos_a, pos_b=None):
		return "SAFE"
	if _obb_overlap(meta_a, meta_b, inflate=0.0, pos_a=pos_a, pos_b=None):
		return "COLLIDE"
	if _obb_overlap(meta_a, meta_b, inflate=pad, pos_a=pos_a, pos_b=None):
		return "NEAR"
	return "SAFE"


def _collision_pairs(labels: list[int], metas_by_label: dict[int, dict], pos_overrides: dict[int, tuple[float, float]] | None = None):
	if pos_overrides is None:
		pos_overrides = {}
	pairs = []
	for a, b in combinations(labels, 2):
		ma = metas_by_label[a]
		mb = metas_by_label[b]
		pa = pos_overrides.get(a)
		pb = pos_overrides.get(b)
		if not _aabb_overlap_xy(ma, mb, inflate=0.0, pos_a=pa, pos_b=pb):
			continue
		if _obb_overlap(ma, mb, inflate=0.0, pos_a=pa, pos_b=pb):
			pairs.append((a, b))
	return pairs


def _relation_with_overrides(ma: dict, mb: dict, pad: float, pos_a=None, pos_b=None) -> str:
	if not _aabb_overlap_xy(ma, mb, inflate=pad, pos_a=pos_a, pos_b=pos_b):
		return "SAFE"
	if _obb_overlap(ma, mb, inflate=0.0, pos_a=pos_a, pos_b=pos_b):
		return "COLLIDE"
	if _obb_overlap(ma, mb, inflate=pad, pos_a=pos_a, pos_b=pos_b):
		return "NEAR"
	return "SAFE"


def _has_related_near(
	labels: list[int],
	metas_by_label: dict[int, dict],
	target_labels: set[int],
	pos_overrides: dict[int, tuple[float, float]] | None = None,
) -> bool:
	if pos_overrides is None:
		pos_overrides = {}
	for a, b in combinations(labels, 2):
		if a not in target_labels and b not in target_labels:
			continue
		ma = metas_by_label[a]
		mb = metas_by_label[b]
		pa = pos_overrides.get(a)
		pb = pos_overrides.get(b)
		rel = _relation_with_overrides(ma, mb, pad=COLLISION_PAD, pos_a=pa, pos_b=pb)
		if rel == "NEAR":
			return True
	return False


def _load_font(size: int):
	if not PIL_AVAILABLE:
		return None
	if "bpy" in sys.modules:
		return ImageFont.load_default()
	try:
		return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
	except Exception:
		return ImageFont.load_default()


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



CAMERA_TOP_DIST_SCALE = 2.9
CAMERA_ISO_DIST_SCALE = 3.2
CAMERA_FIT_MARGIN = 1.10
CAMERA_SAFETY_SCALE = 1.03
FIXED_ISOMETRIC_MODE = "isometric_north_ur"

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
	raw_aspect = span_x / span_y
	aspect = max(raw_aspect, 1.0 / raw_aspect) if render_mode == "top" else max(raw_aspect, 1.0)
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
	elif render_mode == "isometric":
		dc_utils.setup_camera_for_mode(
			cam_obj,
			cam_data,
			render_ctx,
			FIXED_ISOMETRIC_MODE,
			top_dist_scale=CAMERA_TOP_DIST_SCALE,
			iso_dist_scale=CAMERA_ISO_DIST_SCALE,
			fit_margin=CAMERA_FIT_MARGIN,
			fit_safety=CAMERA_SAFETY_SCALE,
		)
		return
	else:
		raise ValueError(f"Unsupported render mode: {render_mode}")

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
	cam_data.clip_start = 0.1
	cam_data.clip_end = max(radius * 20, 1000)


def render_view(cam, output_path: str):
	scene = bpy.context.scene
	scene.camera = cam
	scene.render.filepath = output_path
	bpy.ops.render.render(write_still=True)


def _project_group_centers(id_to_entry: dict[str, dict], object_ids: list[str], cam_obj, width: int, height: int) -> dict[str, tuple[int, int]]:
	scene = bpy.context.scene
	out: dict[str, tuple[int, int]] = {}
	for oid in object_ids:
		entry = id_to_entry.get(oid)
		if not entry:
			continue
		b = _group_bounds(entry["group"])
		world_c = Vector((b["center_x"], b["center_y"], b["center_z"]))
		v = world_to_camera_view(scene, cam_obj, world_c)
		if v.z < 0:
			continue
		px = int(round(v.x * width))
		py = int(round((1.0 - v.y) * height))
		out[oid] = (px, py)
	return out


def _draw_scale_marker(drawer, img_w: int, img_h: int, cam_obj):
	px_per_unit = dc_utils.scale_marker_px_per_unit(cam_obj, img_w)
	bar_len = max(30, int(round(UNIT_SCALE * px_per_unit)))
	x0 = SCALE_MARGIN
	y0 = img_h - SCALE_MARGIN

	drawer.rectangle([x0, y0 - SCALE_BAR_HEIGHT // 2, x0 + bar_len, y0 + SCALE_BAR_HEIGHT // 2], fill=(255, 255, 255))
	drawer.rectangle([x0 - 1, y0 - SCALE_TICK_HEIGHT // 2, x0 + 1, y0 + SCALE_TICK_HEIGHT // 2], fill=(255, 255, 255))
	drawer.rectangle([x0 + bar_len - 1, y0 - SCALE_TICK_HEIGHT // 2, x0 + bar_len + 1, y0 + SCALE_TICK_HEIGHT // 2], fill=(255, 255, 255))
	drawer.text((x0 + bar_len + 12, y0 - 12), "1m", fill=(255, 255, 255), font=_load_font(20))


def _north_screen_vector(cam) -> tuple[float, float]:
	north_world_dir = Vector((0.0, 1.0, 0.0))
	m = cam.matrix_world.to_3x3()
	cam_right = (m @ Vector((1.0, 0.0, 0.0))).normalized()
	cam_up = (m @ Vector((0.0, 1.0, 0.0))).normalized()
	d = north_world_dir.normalized()
	screen_x = float(d.dot(cam_right))
	screen_y = float(-d.dot(cam_up))
	norm = math.sqrt(screen_x * screen_x + screen_y * screen_y)
	if norm > 1e-8:
		return (screen_x / norm, screen_y / norm)
	return (0.0, -1.0)


def _draw_north_indicator(drawer, north_vec: tuple[float, float]):
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
	drawer.text((78, 40), "NORTH", fill=(255, 0, 0), font=_load_font(18))


def _annotate_top(
	base_image: Path,
	out_image: Path,
	label_map: dict[str, int],
	centers: dict[str, tuple[int, int]],
	cam_obj,
	with_scale_marker: bool,
	north_world_dir=None,
):
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	px_per_unit = dc_utils.scale_marker_px_per_unit(cam_obj, img.width)
	north_vec = dc_utils.north_screen_vector(cam_obj, north_world_dir=north_world_dir)

	font = _load_font(LABEL_FONT_SIZE)
	# Only label target objects that appear in label_map
	for oid, label in label_map.items():
		if oid not in centers:
			continue
		cx, cy = centers[oid]
		cx = max(16, min(img.width - 16, cx))
		cy = max(16, min(img.height - 16, cy))
		r = LABEL_RADIUS
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 60), outline=(255, 255, 255), width=2)
		try:
			draw.text((cx, cy), str(label), fill=(255, 255, 255), font=font, anchor="mm")
		except TypeError:
			bbox = draw.textbbox((0, 0), str(label), font=font)
			draw.text((cx - (bbox[2] - bbox[0]) // 2, cy - (bbox[3] - bbox[1]) // 2), str(label), fill=(255, 255, 255), font=font)

	img = _compose_with_left_strip_scaled(img, north_vec, px_per_unit, ui_scale_mult=LEFT_STRIP_UI_SCALE_MULT_TOP)
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


def _move_dir_to_vec(
	d: str,
	north_world_xy: tuple[float, float],
	east_world_xy: tuple[float, float],
) -> tuple[float, float]:
	nx, ny = north_world_xy
	ex, ey = east_world_xy
	return {
		"E": (ex, ey),
		"W": (-ex, -ey),
		"N": (nx, ny),
		"S": (-nx, -ny),
	}[d]


def _top_screen_basis_world(cam_obj) -> tuple[tuple[float, float], tuple[float, float]]:
	"""
	Build a world-XY basis aligned with top-view screen axes:
	- north_world_xy: screen-up direction in world XY
	- east_world_xy: screen-right direction in world XY
	"""
	m = cam_obj.matrix_world.to_3x3()
	cam_right = m @ Vector((1.0, 0.0, 0.0))
	cam_up = m @ Vector((0.0, 1.0, 0.0))

	def _norm_xy(vx: float, vy: float, fallback: tuple[float, float]) -> tuple[float, float]:
		n = math.hypot(vx, vy)
		if n < 1e-8:
			return fallback
		return (vx / n, vy / n)

	# Camera local +Y corresponds to image-up direction.
	north_world_xy = _norm_xy(float(cam_up.x), float(cam_up.y), (0.0, 1.0))
	east_world_xy = _norm_xy(float(cam_right.x), float(cam_right.y), (1.0, 0.0))
	return north_world_xy, east_world_xy


def _move_dir_to_words_en(d: str) -> str:
	return {"E": "East", "W": "West", "N": "North", "S": "South"}[d]


def _abs_images(sample_dir: Path, names: list[str]) -> list[str]:
	return [str((sample_dir / n).resolve()) for n in names]


def _qa_task_move_collision_yesno(
	sample_dir: Path,
	move_id: int,
	other_id: int,
	dir_token: str,
	dist_m: int,
	will_collide: bool,
	image_name: str = "top_move.png",
) -> dict:
	dir_en = _move_dir_to_words_en(dir_token)
	q = (
		"There is one image: a top-view image.\n"
		"Directions in the image: North=up, South=down, East=right, West=left.\n"
		f"If object {move_id} is moved {dir_en} by {dist_m} meters, "
		f"will it overlap (collide) with object {other_id}? Answer with Yes or No.\n\n"
		"Top-view image: <image>\n"
		"Note: the image includes a 1m scale bar.\n"
	)
	return {
		"question": q,
		"answer": "Yes" if will_collide else "No",
		"images": _abs_images(sample_dir, [str(image_name)]),
		"task_type": "top_move_overlap",
		"meta": {
			"move_id": move_id,
			"other_id": other_id,
			"direction": dir_en,
			"dist_m": int(dist_m),
			"mapping": {"North": "up", "South": "down", "East": "right", "West": "left"},
			"adjacent": "nearest_neighbor",
		},
	}


def _qa_task_swap_collision_yesno(
	sample_dir: Path,
	a_id: int,
	b_id: int,
	before_collide: bool,
	after_collide: bool,
	before_pairs: list[list[int]] | None = None,
	after_pairs: list[list[int]] | None = None,
	image_name: str = "top_swap.png",
) -> dict:
	q = (
		"There is one image: a top-view image.\n"
		f"If we swap the positions of object {a_id} and object {b_id}, "
		f"will object {a_id} or object {b_id} overlap (collide) with any object AFTER the swap? "
		"Answer with Yes or No.\n\n"
		"Top-view image: <image>"
	)
	return {
		"question": q,
		"answer": "Yes" if after_collide else "No",
		"images": _abs_images(sample_dir, [str(image_name)]),
		"task_type": "top_swap_overlap",
			"meta": {
				"swap_a": a_id,
				"swap_b": b_id,
				"before_collide": before_collide,
				"after_collide": after_collide,
				"before_related_pairs": before_pairs or [],
				"after_related_pairs": after_pairs or [],
				"case": "collide_to_safe" if (before_collide and not after_collide) else "safe_to_collide",
				"collision_scope": "swap_pair_related",
				"collision_mode": "3D_BVH_final_state_overlap",
		},
	}


def _in_bounds_center(x: float, y: float, bounds: dict, margin: float = 0.5) -> bool:
	return (
		(bounds["min_x"] + margin) <= x <= (bounds["max_x"] - margin)
		and (bounds["min_y"] + margin) <= y <= (bounds["max_y"] - margin)
	)


def _nearest_neighbor_id(labels: list[int], metas_by_label: dict[int, dict], pivot_id: int) -> int:
	px, py = metas_by_label[pivot_id]["obb"]["center"]
	best = None
	best_d = 1e30
	for i in labels:
		if i == pivot_id:
			continue
		qx, qy = metas_by_label[i]["obb"]["center"]
		d = (px - qx) ** 2 + (py - qy) ** 2
		if d < best_d:
			best_d = d
			best = i
	return int(best) if best is not None else int(random.choice([i for i in labels if i != pivot_id]))


def _nearest_k_neighbor_ids(labels: list[int], metas_by_label: dict[int, dict], pivot_id: int, k: int) -> list[int]:
	px, py = metas_by_label[pivot_id]["obb"]["center"]
	cands = []
	for i in labels:
		if i == pivot_id:
			continue
		qx, qy = metas_by_label[i]["obb"]["center"]
		d = (px - qx) ** 2 + (py - qy) ** 2
		cands.append((d, i))
	cands.sort(key=lambda x: x[0])
	return [i for _, i in cands[: max(1, int(k))]]


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
				"category": str(obj.get("category", "unknown")).strip().lower(),
				"group": g,
				"center": (b["center_x"], b["center_y"], b["center_z"]),
			}
		)
	return logical


def _translate_group(group_objs: list, dx: float, dy: float, dz: float = 0.0) -> dict:
	orig = {obj.name: obj.matrix_world.copy() for obj in group_objs}
	t = Matrix.Translation(Vector((dx, dy, dz)))
	for obj in group_objs:
		obj.matrix_world = t @ obj.matrix_world
	bpy.context.view_layer.update()
	key = tuple(sorted(obj.name for obj in group_objs))
	_GROUP_VERSION[key] = _GROUP_VERSION.get(key, 0) + 1
	return orig


def _restore_group(group_objs: list, orig: dict):
	for obj in group_objs:
		if obj.name in orig:
			obj.matrix_world = orig[obj.name]
	bpy.context.view_layer.update()
	key = tuple(sorted(obj.name for obj in group_objs))
	_GROUP_VERSION[key] = _GROUP_VERSION.get(key, 0) + 1


def _build_group_bvh(group: list) -> BVHTree | None:
	key = tuple(sorted(obj.name for obj in group))
	ver = _GROUP_VERSION.get(key, 0)
	cached = _GROUP_BVH_CACHE.get(key)
	if cached is not None and cached[0] == ver:
		return cached[1]

	verts_world = []
	tris = []
	v_ofs = 0
	depsgraph = bpy.context.evaluated_depsgraph_get()
	for obj in group:
		obj_eval = obj.evaluated_get(depsgraph)
		mesh = obj_eval.to_mesh()
		if mesh is None:
			continue
		try:
			mesh.calc_loop_triangles()
			world_matrix = obj_eval.matrix_world
			for v in mesh.vertices:
				verts_world.append(world_matrix @ v.co)
			for tri in mesh.loop_triangles:
				a, b, c = tri.vertices
				tris.append((a + v_ofs, b + v_ofs, c + v_ofs))
			v_ofs += len(mesh.vertices)
		finally:
			obj_eval.to_mesh_clear()
	if not tris or not verts_world:
		_GROUP_BVH_CACHE[key] = (ver, None)
		return None
	try:
		bvh = BVHTree.FromPolygons(verts_world, tris, all_triangles=True, epsilon=1e-6)
		_GROUP_BVH_CACHE[key] = (ver, bvh)
		return bvh
	except Exception:
		_GROUP_BVH_CACHE[key] = (ver, None)
		return None


def _groups_overlap_3d(group_a: list, group_b: list) -> bool:
	# Fast broad-phase reject using 3D AABB.
	ba = _group_bounds(group_a)
	bb = _group_bounds(group_b)
	if ba["max_x"] < bb["min_x"] or ba["min_x"] > bb["max_x"]:
		return False
	if ba["max_y"] < bb["min_y"] or ba["min_y"] > bb["max_y"]:
		return False
	if ba["max_z"] < bb["min_z"] or ba["min_z"] > bb["max_z"]:
		return False

	bvh_a = _build_group_bvh(group_a)
	bvh_b = _build_group_bvh(group_b)
	if bvh_a is None or bvh_b is None:
		return False
	try:
		return len(bvh_a.overlap(bvh_b)) >= MIN_OVERLAP_TRI_PAIRS
	except Exception:
		return False


def process_scene(
	glb_name: str,
	info: dict,
	glb_dir: Path,
	output_dir: Path,
	sample_idx: int,
	whitelist_pairs: set[tuple[str, str]],
	max_scene_seconds: float = 180.0,
	qa_pairs_per_scene: int = 1,
) -> dict | None:
	t_start = time.monotonic()

	def _timed_out() -> bool:
		if max_scene_seconds <= 0:
			return False
		return (time.monotonic() - t_start) > max_scene_seconds

	layout_info = info.get("layout_info", [])
	scene_name = info.get("scene_name", glb_name)
	layout_objects = convert_layout_to_objects(layout_info)
	if len(layout_objects) < MIN_OBJECTS:
		return None
	if _timed_out():
		return None

	glb_path = glb_dir / glb_name
	if not glb_path.exists():
		return None

	bpy.ops.wm.read_factory_settings(use_empty=True)
	bpy.ops.import_scene.gltf(filepath=str(glb_path))
	if _timed_out():
		return None

	groups = collect_instance_groups(max_instance_idx=len(layout_objects))
	logical = _build_logical_objects(layout_objects, groups)
	if len(logical) < MIN_OBJECTS:
		return None
	if _timed_out():
		return None

	dc_utils.set_render_and_world(SAMPLES, RESOLUTION_X, RESOLUTION_Y, transparent_bg=True)
	bounds = _scene_bounds_from_groups(groups)
	scene_center = Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"]))
	scene_radius = max(bounds["width"], bounds["depth"], 1.0)
	dc_utils.setup_lighting(scene_center, scene_radius)
	render_ctx = dc_utils.compute_render_frame(_all_mesh_objects())
	wall_targets = dc_utils._resolve_wall_targets(_all_mesh_objects())
	cam_obj, cam_data = dc_utils.create_preview_camera()
	dyn_w, dyn_h = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)
	bpy.context.scene.render.resolution_x = dyn_w
	bpy.context.scene.render.resolution_y = dyn_h
	dc_utils.setup_camera_for_mode(
		cam_obj,
		cam_data,
		render_ctx,
		"top",
		top_dist_scale=CAMERA_TOP_DIST_SCALE,
		iso_dist_scale=CAMERA_ISO_DIST_SCALE,
		fit_margin=CAMERA_FIT_MARGIN,
		fit_safety=CAMERA_SAFETY_SCALE,
	)
	north_world_xy, east_world_xy = _top_screen_basis_world(cam_obj)

	id_to_entry = {e["id"]: e for e in logical}
	valid_ids = [e["id"] for e in logical]
	rw_top = int(bpy.context.scene.render.resolution_x)
	rh_top = int(bpy.context.scene.render.resolution_y)
	valid_ids = [oid for oid in valid_ids if _top_group_visibility_ok(id_to_entry[oid]["group"], cam_obj, rw_top, rh_top)]
	label_map = assign_independent_labels([{"id": oid} for oid in valid_ids])
	if len(valid_ids) < MIN_OBJECTS:
		return None

	metas_by_label: dict[int, dict] = {}
	for oid in valid_ids:
		entry = id_to_entry[oid]
		lb = label_map[oid]
		metas_by_label[lb] = _build_meta_from_group(lb, oid, entry["instance_index"], entry["group"])
	labels = sorted(metas_by_label.keys())
	label_to_oid = {lb: oid for oid, lb in label_map.items()}
	label_to_category = {
		lb: str(id_to_entry[label_to_oid[lb]].get("category", "unknown")).strip().lower()
		for lb in labels
	}

	def _pair_collide_3d_with_whitelist(lb_a: int, lb_b: int) -> bool:
		ca = label_to_category.get(lb_a, "")
		cb = label_to_category.get(lb_b, "")
		if _is_pair_whitelisted(ca, cb, whitelist_pairs):
			return False
		return _groups_overlap_3d(label_to_group[lb_a], label_to_group[lb_b])

	north_world_dir = dc_utils.canonical_north_world(render_ctx)

	def render_labeled(
		rel_out_path: str,
		rel_raw_path: str,
		with_scale_marker: bool,
		label_map_override=None,
		render_mode: str = "top",
	):
		raw_abs = raw_dir / rel_raw_path
		raw_abs.parent.mkdir(parents=True, exist_ok=True)
		wall_alpha = TOP_WALL_ALPHA if str(render_mode) == "top" else ISO_WALL_ALPHA
		dc_utils._render_view_with_wall_alpha(cam_obj, str(raw_abs), wall_targets, wall_alpha)
		# Only label target object IDs
		lm = label_map_override if label_map_override is not None else label_map
		rw = int(bpy.context.scene.render.resolution_x)
		rh = int(bpy.context.scene.render.resolution_y)
		centers = dc_utils.project_group_centers(id_to_entry, list(lm.keys()), cam_obj, rw, rh)
		out_abs = sample_dir / rel_out_path
		_annotate_top(raw_abs, out_abs, lm, centers, cam_obj, with_scale_marker=with_scale_marker, north_world_dir=north_world_dir)
		return str(out_abs.resolve())

	stem = Path(glb_name).stem
	sample_dir = output_dir / stem
	created_now = not sample_dir.exists()
	sample_dir.mkdir(parents=True, exist_ok=True)

	def _fail_cleanup():
		# Always remove hidden raw intermediates on failure.
		if not KEEP_RAW_VIEWS:
			try:
				if "raw_dir" in locals() and Path(raw_dir).exists():
					shutil.rmtree(raw_dir, ignore_errors=True)
			except Exception:
				pass
		if created_now:
			shutil.rmtree(sample_dir, ignore_errors=True)

	if KEEP_RAW_VIEWS:
		raw_dir = sample_dir / "raw_views"
		raw_dir.mkdir(parents=True, exist_ok=True)
	else:
		raw_dir = sample_dir / ".raw_views"
		raw_dir.mkdir(parents=True, exist_ok=True)
	ref_dir = sample_dir / "ref_images"
	ref_dir.mkdir(parents=True, exist_ok=True)
	for stale_path in sample_dir.glob("top_move*.png"):
		try:
			stale_path.unlink()
		except Exception:
			pass
	for stale_path in sample_dir.glob("top_swap*.png"):
		try:
			stale_path.unlink()
		except Exception:
			pass
	for stale_path in ref_dir.glob("qa1_move_after*.png"):
		try:
			stale_path.unlink()
		except Exception:
			pass
	for stale_path in ref_dir.glob("qa2_swap_after*.png"):
		try:
			stale_path.unlink()
		except Exception:
			pass

	label_to_group = {lb: id_to_entry[label_to_oid[lb]]["group"] for lb in labels}
	used_move_keys: set[tuple[int, int, str, int, bool]] = set()
	used_swap_keys: set[tuple[int, int, bool, bool]] = set()
	qa_pairs_target = max(1, int(qa_pairs_per_scene))
	pair_payloads: list[dict] = []

	def _pair_suffix(pair_idx: int) -> str:
		return "" if pair_idx == 0 else f"_{pair_idx + 1:02d}"

	def _pair_output_names(pair_idx: int) -> dict[str, str]:
		suf = _pair_suffix(pair_idx)
		return {
			"top_move": f"top_move{suf}.png",
			"top_swap": f"top_swap{suf}.png",
			"ref_move_after": f"qa1_move_after{suf}.png",
			"ref_swap_after": f"qa2_swap_after{suf}.png",
		}

	def _build_one_pair(pair_idx: int) -> dict | None:
		desired_move_collide = ((sample_idx + pair_idx) % 2 == 0)
		desired_after_collide = ((sample_idx + pair_idx) % 2 == 1)
		file_names = _pair_output_names(pair_idx)
		all_pairs = list(combinations(labels, 2))
		random.shuffle(all_pairs)

		def _pick_move(mode: str):
			tries = 0
			move_ids = labels[:]
			random.shuffle(move_ids)
			while tries < MOVE_TASK_TRIES:
				if _timed_out():
					return None, None
				random.shuffle(move_ids)
				for move_id in move_ids:
					if _timed_out():
						return None, None
					if tries >= MOVE_TASK_TRIES:
						break
					tries += 1
					other_topk = _nearest_k_neighbor_ids(labels, metas_by_label, move_id, MOVE_OTHER_TOPK)
					if not other_topk:
						continue
					meta_m = metas_by_label[move_id]
					group_m = label_to_group[move_id]
					old_x, old_y = meta_m["obb"]["center"]

					other_trials = other_topk[:]
					random.shuffle(other_trials)
					for other_id in other_trials:
						if _timed_out():
							return None, None
						before_collide = _pair_collide_3d_with_whitelist(move_id, other_id)
						dirs = MOVE_DIR_POOL[:]
						dists = MOVE_DIST_POOL[:]
						random.shuffle(dirs)
						random.shuffle(dists)

						for d in dirs:
							if _timed_out():
								return None, None
							vx, vy = _move_dir_to_vec(d, north_world_xy, east_world_xy)
							for dist in dists:
								if _timed_out():
									return None, None
								actual_dist = dist * UNIT_SCALE
								nx = float(old_x + vx * actual_dist)
								ny = float(old_y + vy * actual_dist)
								if not _in_bounds_center(nx, ny, bounds, margin=IN_BOUNDS_MARGIN):
									continue
								orig_m = _translate_group(group_m, nx - old_x, ny - old_y)
								try:
									after_collide = _pair_collide_3d_with_whitelist(move_id, other_id)
								finally:
									_restore_group(group_m, orig_m)

								if mode == "strict":
									if after_collide != desired_move_collide:
										continue
								elif mode == "flip":
									if after_collide == before_collide:
										continue
								else:
									continue

								move_key = (int(move_id), int(other_id), str(d), int(dist), bool(after_collide))
								if move_key in used_move_keys:
									continue
								info = {
									"move_label": move_id,
									"other_label": other_id,
									"direction": d,
									"dist_m": int(dist),
									"after_collide": after_collide,
									"move_key": move_key,
								}
								return info, (move_id, nx, ny)
			return None, None

		def _pick_swap(mode: str):
			def _pair_3d(a: int, b: int) -> bool:
				return _pair_collide_3d_with_whitelist(a, b)

			def _related_collision_pairs(a: int, b: int) -> list[tuple[int, int]]:
				pairs = []
				if _pair_3d(a, b):
					pairs.append((min(a, b), max(a, b)))
				for t in labels:
					if t == a or t == b:
						continue
					if _pair_3d(a, t):
						pairs.append((min(a, t), max(a, t)))
					if _pair_3d(b, t):
						pairs.append((min(b, t), max(b, t)))
				return sorted(set(pairs))

			for a_id, b_id in all_pairs[:SWAP_TASK_TRIES]:
				if _timed_out():
					return None, None
				ga = label_to_group[a_id]
				gb = label_to_group[b_id]
				ba0 = _group_bounds(ga)
				bb0 = _group_bounds(gb)
				pa = (ba0["center_x"], ba0["center_y"], ba0["center_z"])
				pb = (bb0["center_x"], bb0["center_y"], bb0["center_z"])
				before_pairs = _related_collision_pairs(a_id, b_id)
				before_related = len(before_pairs) > 0
				group_a = ga
				group_b = gb
				orig_a = _translate_group(group_a, pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
				orig_b = _translate_group(group_b, pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2])
				try:
					after_pairs = _related_collision_pairs(a_id, b_id)
					after_related = len(after_pairs) > 0
				finally:
					_restore_group(group_b, orig_b)
					_restore_group(group_a, orig_a)

				if mode == "strict":
					if after_related != desired_after_collide:
						continue
				elif mode == "flip":
					if after_related == before_related:
						continue
				else:
					continue

				swap_key = (min(int(a_id), int(b_id)), max(int(a_id), int(b_id)), bool(before_related), bool(after_related))
				if swap_key in used_swap_keys:
					continue
				info = {
					"swap_a_label": a_id,
					"swap_b_label": b_id,
					"before_related": before_related,
					"after_related": after_related,
					"before_related_pairs": before_pairs,
					"after_related_pairs": after_pairs,
					"swap_key": swap_key,
				}
				return info, (a_id, b_id)
			return None, None

		move_info, move_ref = _pick_move("strict")
		if move_info is None:
			move_info, move_ref = _pick_move("flip")

		swap_info, swap_ref = _pick_swap("strict")
		if swap_info is None:
			swap_info, swap_ref = _pick_swap("flip")

		if _timed_out():
			return None
		if move_info is None and swap_info is None:
			return None

		move_label_map = None
		swap_label_map = None
		move_task = None
		swap_task = None
		if move_info is not None:
			move_label = int(move_info["move_label"])
			other_label = int(move_info["other_label"])
			move_oid = label_to_oid[move_label]
			other_oid = label_to_oid[other_label]
			qa1_map = assign_independent_labels([{"id": move_oid}, {"id": other_oid}])
			move_label_map = {
				move_oid: qa1_map[move_oid],
				other_oid: qa1_map[other_oid],
			}
			move_task = _qa_task_move_collision_yesno(
				sample_dir,
				qa1_map[move_oid],
				qa1_map[other_oid],
				move_info["direction"],
				int(move_info["dist_m"]),
				bool(move_info["after_collide"]),
				image_name=file_names["top_move"],
			)
		if swap_info is not None:
			swap_a_label = int(swap_info["swap_a_label"])
			swap_b_label = int(swap_info["swap_b_label"])
			swap_a_oid = label_to_oid[swap_a_label]
			swap_b_oid = label_to_oid[swap_b_label]
			qa2_map = assign_independent_labels([{"id": swap_a_oid}, {"id": swap_b_oid}])
			swap_label_map = {
				swap_a_oid: qa2_map[swap_a_oid],
				swap_b_oid: qa2_map[swap_b_oid],
			}
			swap_task = _qa_task_swap_collision_yesno(
				sample_dir,
				qa2_map[swap_a_oid],
				qa2_map[swap_b_oid],
				bool(swap_info["before_related"]),
				bool(swap_info["after_related"]),
				[list(p) for p in swap_info.get("before_related_pairs", [])],
				[list(p) for p in swap_info.get("after_related_pairs", [])],
				image_name=file_names["top_swap"],
			)
		if move_label_map is None:
			if swap_label_map is not None:
				fallback_oids = list(swap_label_map.keys())
				local_map = assign_independent_labels([{"id": oid} for oid in fallback_oids])
				move_label_map = {oid: local_map[oid] for oid in fallback_oids}
			else:
				fallback = labels[:2] if len(labels) >= 2 else labels
				fallback_oids = [label_to_oid[lb] for lb in fallback]
				local_map = assign_independent_labels([{"id": oid} for oid in fallback_oids])
				move_label_map = {oid: local_map[oid] for oid in fallback_oids}
		if swap_label_map is None:
			if move_label_map is not None:
				fallback_oids = list(move_label_map.keys())
				local_map = assign_independent_labels([{"id": oid} for oid in fallback_oids])
				swap_label_map = {oid: local_map[oid] for oid in fallback_oids}
			else:
				fallback = labels[:2] if len(labels) >= 2 else labels
				fallback_oids = [label_to_oid[lb] for lb in fallback]
				local_map = assign_independent_labels([{"id": oid} for oid in fallback_oids])
				swap_label_map = {oid: local_map[oid] for oid in fallback_oids}

		top_swap_path = render_labeled(
			file_names["top_swap"],
			f"base/top_swap_raw{_pair_suffix(pair_idx)}.png",
			with_scale_marker=False,
			label_map_override=swap_label_map,
		)
		top_move_path = render_labeled(
			file_names["top_move"],
			f"base/top_move_raw{_pair_suffix(pair_idx)}.png",
			with_scale_marker=True,
			label_map_override=move_label_map,
		)
		if _timed_out():
			return None

		move_ref_abs = None
		swap_ref_abs = None
		if move_ref is not None:
			move_label, nx, ny = move_ref
			move_oid = label_to_oid[move_label]
			entry = id_to_entry[move_oid]
			b = _group_bounds(entry["group"])
			dx = nx - b["center_x"]
			dy = ny - b["center_y"]
			orig = _translate_group(entry["group"], dx, dy)
			try:
				render_labeled(
					str(Path("ref_images") / file_names["ref_move_after"]),
					f"ref/qa1_move_after_raw{_pair_suffix(pair_idx)}.png",
					with_scale_marker=True,
					label_map_override=move_label_map,
				)
			finally:
				_restore_group(entry["group"], orig)
			move_ref_abs = str((sample_dir / "ref_images" / file_names["ref_move_after"]).resolve())

		if swap_ref is not None:
			a_id, b_id = swap_ref
			a_oid = label_to_oid[a_id]
			b_oid = label_to_oid[b_id]
			a_entry = id_to_entry[a_oid]
			b_entry = id_to_entry[b_oid]
			ba = _group_bounds(a_entry["group"])
			bb = _group_bounds(b_entry["group"])
			da = (
				bb["center_x"] - ba["center_x"],
				bb["center_y"] - ba["center_y"],
				bb["center_z"] - ba["center_z"],
			)
			db = (
				ba["center_x"] - bb["center_x"],
				ba["center_y"] - bb["center_y"],
				ba["center_z"] - bb["center_z"],
			)
			orig_a = _translate_group(a_entry["group"], da[0], da[1], da[2])
			orig_b = _translate_group(b_entry["group"], db[0], db[1], db[2])
			try:
				render_labeled(
					str(Path("ref_images") / file_names["ref_swap_after"]),
					f"ref/qa2_swap_after_raw{_pair_suffix(pair_idx)}.png",
					with_scale_marker=False,
					label_map_override=swap_label_map,
				)
			finally:
				_restore_group(a_entry["group"], orig_a)
				_restore_group(b_entry["group"], orig_b)
			swap_ref_abs = str((sample_dir / "ref_images" / file_names["ref_swap_after"]).resolve())

		if move_info is not None:
			used_move_keys.add(move_info["move_key"])
		if swap_info is not None:
			used_swap_keys.add(swap_info["swap_key"])

		pair_images = {
			"top_swap": top_swap_path,
			"top_move": top_move_path,
		}
		if move_ref_abs is not None:
			pair_images["qa1_move_after"] = move_ref_abs
		if swap_ref_abs is not None:
			pair_images["qa2_swap_after"] = swap_ref_abs

		return {
			"qa": [q for q in [move_task, swap_task] if q is not None],
			"images": pair_images,
			"desired_move_collide": bool(desired_move_collide),
			"desired_after_collide": bool(desired_after_collide),
		}

	for pair_idx in range(qa_pairs_target):
		pair_payload = _build_one_pair(pair_idx)
		if pair_payload is None:
			if _timed_out():
				break
			continue
		pair_payloads.append(pair_payload)

	if _timed_out():
		_fail_cleanup()
		return None
	if not pair_payloads:
		_fail_cleanup()
		return None

	if not KEEP_RAW_VIEWS:
		shutil.rmtree(raw_dir, ignore_errors=True)

	qa_top = []
	images = {}
	pair_summaries = []
	for i, payload in enumerate(pair_payloads):
		qa_top.extend(payload["qa"])
		if i == 0:
			images.update(payload["images"])
		else:
			suf = f"_{i + 1:02d}"
			for k, v in payload["images"].items():
				images[f"{k}{suf}"] = v
		pair_summaries.append(
			{
				"pair_index": int(i + 1),
				"qa_count": int(len(payload["qa"])),
				"desired_move_collide": bool(payload["desired_move_collide"]),
				"desired_swap_after_collide": bool(payload["desired_after_collide"]),
				"images": payload["images"],
			}
		)

	label_mapping = [
		{"scene_object_id": oid, "label_id": label_map[oid], "instance_index": id_to_entry[oid]["instance_index"]}
		for oid in sorted(valid_ids, key=lambda x: label_map[x])
	]

	return {
		"glb_name": glb_name,
		"scene_name": scene_name,
		"object_count": len(valid_ids),
		"label_mapping": label_mapping,
		"images": images,
		"qa": {"top": qa_top},
			"special_refs": {
				"scale_marker_m": 1.0,
				"unit_scale": UNIT_SCALE,
				"collision_pad": COLLISION_PAD,
				"min_overlap_tri_pairs": MIN_OVERLAP_TRI_PAIRS,
				"whitelist_exemption": True,
				"qa_pairs_target": int(qa_pairs_target),
				"qa_pairs_generated": int(len(pair_payloads)),
				"qa_pair_details": pair_summaries,
				"qa2_scope": "swap_pair_related",
			},
		}


def main():
	global MIN_OVERLAP_TRI_PAIRS
	args = parse_args()
	random.seed(args.seed)
	MIN_OVERLAP_TRI_PAIRS = max(1, int(args.min_overlap_tri_pairs))

	glb_dir = args.glb_dir.resolve()
	mapping_json = args.mapping_json.resolve()
	output_dir = args.output_dir.resolve()
	whitelist_json = args.whitelist_json.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not glb_dir.exists():
		raise FileNotFoundError(f"GLB dir not found: {glb_dir}")
	if not mapping_json.exists():
		raise FileNotFoundError(f"Mapping json not found: {mapping_json}")

	whitelist_pairs = load_whitelist_pairs(whitelist_json)

	# Parent parallel mode: spawn worker Blender processes and merge outputs.
	if args.workers > 1 and args.worker_index < 0:
		script_path = str(Path(__file__).resolve())
		base_args = []
		if args.max_scenes > 0:
			base_args += ["--max-scenes", str(args.max_scenes)]
		if args.max_scene_seconds > 0:
			base_args += ["--max-scene-seconds", str(args.max_scene_seconds)]
		if args.whitelist_json:
			base_args += ["--whitelist-json", str(args.whitelist_json.resolve())]
		if args.region:
			base_args += ["--region", args.region]
		base_args += [
			"--glb-dir", str(glb_dir),
			"--mapping-json", str(mapping_json),
			"--output-dir", str(output_dir),
			"--seed", str(args.seed),
			"--workers", str(args.workers),
			"--qa-pairs-per-scene", str(args.qa_pairs_per_scene),
		]
		print("=" * 80)
		print(f"Launching {args.workers} parallel workers...")
		procs = []
		for wi in range(args.workers):
			cmd = [
				str(BLENDER_BIN),
				"--background",
				"--python",
				script_path,
				"--",
				*base_args,
				"--worker-index",
				str(wi),
			]
			procs.append((wi, subprocess.Popen(cmd, env=os.environ.copy())))
		exit_codes = []
		for wi, p in procs:
			rc = p.wait()
			exit_codes.append((wi, rc))
		bad = [(wi, rc) for wi, rc in exit_codes if rc != 0]
		if bad:
			raise RuntimeError(f"Some workers failed: {bad}")
		return

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
	if args.workers > 1 and args.worker_index >= 0:
		scene_keys = [k for i, k in enumerate(scene_keys) if (i % args.workers) == args.worker_index]

	print("=" * 80)
	print("Indoor spatial visualization QA construct")
	print(f"Total glb files: {len(glb_files)}")
	print(f"Matched in mapping: {len(scene_keys)}")
	print(f"Whitelist pairs: {len(whitelist_pairs)} from {whitelist_json}")
	if args.region:
		print(f"Region filter: {args.region}")

	all_results = []
	for idx, glb_name in enumerate(scene_keys, start=1):
		scene_t0 = time.monotonic()
		try:
			result = process_scene(
				glb_name,
				scene_mapping[glb_name],
				glb_dir,
				output_dir,
				sample_idx=idx - 1,
				whitelist_pairs=whitelist_pairs,
				max_scene_seconds=float(args.max_scene_seconds),
				qa_pairs_per_scene=int(args.qa_pairs_per_scene),
			)
		except Exception as e:
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: exception {e}")
			continue
		if not result:
			elapsed = time.monotonic() - scene_t0
			if args.max_scene_seconds > 0 and elapsed >= float(args.max_scene_seconds):
				print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: timeout ({elapsed:.1f}s >= {args.max_scene_seconds:.1f}s)")
			else:
				print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: invalid scene")
			continue
		all_results.append(result)
		print(f"[{idx}/{len(scene_keys)}] done {glb_name}: qa_top={len(result['qa']['top'])}")

	if args.workers > 1 and args.worker_index >= 0:
		out_json = output_dir / f"metadata_indoor.worker_{args.worker_index}_of_{args.workers}.json"
	else:
		out_json = output_dir / "metadata_indoor.json"
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(all_results, f, ensure_ascii=False, indent=2)

	# If workers are run manually (or parent process is skipped), auto-finalize
	# when all worker parts are present.
	if args.workers > 1 and args.worker_index >= 0:
		_auto_merge_if_all_workers_ready(output_dir, args.workers)

	print("=" * 80)
	print(f"Done. valid_scenes={len(all_results)}/{len(scene_keys)}")
	print(f"Output: {out_json}")


if __name__ == "__main__":
	main()
