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
from pathlib import Path

try:
	from shapely.geometry import LineString, Point, Polygon
	from shapely.ops import nearest_points, unary_union
	SHAPELY_AVAILABLE = True
except Exception:
	SHAPELY_AVAILABLE = False


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path.home() / "SpatialAct")))
BLENDER_BIN = Path(os.environ.get("BLENDER_BIN", str(PROJECT_ROOT / "blender-3.2.2-linux-x64/blender")))
INTERNSCENES_ROOT = Path(os.environ.get("INTERNSCENES_ROOT", str(Path.home() / "InternScenes")))
SHARED_UTILS_PATH = PROJECT_ROOT / "benchmark/data_construct/utils.py"

_utils_spec = importlib.util.spec_from_file_location("dc_render_utils_task567_indoor", SHARED_UTILS_PATH)
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
	from PIL import Image, ImageDraw, ImageFont, ImageChops
	PIL_AVAILABLE = True
except ImportError:
	PIL_AVAILABLE = False


DEFAULT_GLB_DIR = INTERNSCENES_ROOT / "scenes/glb_files_wall_complex-10-15_clean_keep"
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_LAYOUT_ROOT = INTERNSCENES_ROOT / "data/Layout_info"
DEFAULT_WHITELIST_JSON = INTERNSCENES_ROOT / "scenes/normal_overlap_whitelist.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark/data/error_mode_clean/indoor_scenes_complex-10-15"

RESOLUTION_X = 1080
RESOLUTION_Y = 720
SAMPLES = 64
BASE_LONG_EDGE = 1280
MIN_SHORT_EDGE = 720
LABEL_RADIUS = 14
LABEL_FONT_SIZE = 18

TOP_MIN_BBOX_PX = 220.0
TOP_MIN_AREA_RATIO = 0.00035
ENABLE_TOP_VISIBILITY_PREFILTER = False
OVERLAP_TOP_MIN_INTERSECTION_PX = 260.0
# Additional overlap visibility gate: true top projected-mask overlap area (px).
OVERLAP_TOP_MASK_MIN_INTERSECTION_PX = 80.0
# Raster cap for mask overlap computation speed.
OVERLAP_MASK_RASTER_MAX_SIDE = 960
# 3D BVH overlap strength threshold: minimum overlapped triangle-pair count.
BVH_OVERLAP_MIN_HITS = 5
# Minimum XY penetration depth (meters) for overlap acceptance.
OVERLAP_MIN_PENETRATION_DEPTH_M = 0.08
DEBUG_BVH_OVERLAP_HITS = False

MIN_OBJECTS = 3
MAX_LABELED_OBJECTS = 5
UNIT_SCALE = 1.0

MOVE_DIRS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
MOVE_DISTS_M = [0.5, 1.0, 1.5, 2.0, 2.5]
ROTATE_DEGS = [30, 35, 40, 45, 50, 55, 60, 120, 135, 150]
SCALE_PCTS = [10, 15, 20, 25, 30]
WALL_CONFLICT_EXCLUDE_TARGET_KEYWORDS = {
	"window", "window frame", "door", "door frame", "curtain", "blinds",
	"painting", "picture", "picture frame", "photo", "poster", "wall art", "artwork", "mirror",
}
DEFAULT_REGION_BOUNDARY_RELAX_CATEGORIES = (
	"window,window frame,door,door frame,curtain,blinds,shelf,cabinet,toilet,sink,counter,bathtub,shower"
)
REGION_BOUNDARY_RELAX_KEYWORDS: set[str] = {
	x.strip().lower() for x in DEFAULT_REGION_BOUNDARY_RELAX_CATEGORIES.split(",") if x.strip()
}
REGION_BOUNDARY_RATIO_DEFAULT = 0.98
REGION_BOUNDARY_RATIO_RELAXED = 0.70

ISSUE_OVERLAP = "overlap"
ISSUE_WALL_DOOR = "wall_door_conflict"
ISSUE_ANGLE = "orientation"
ISSUE_WALL = "wall_conflict"
ISSUE_DOOR = "door_conflict"
ISSUE_PATH = "path_conflict"

CAMERA_TOP_DIST_SCALE = 2.9
# Keep camera behavior consistent with task4 spatial visualization.
CAMERA_ISO_DIST_SCALE = 3.2
CAMERA_FIT_MARGIN = 1.10
CAMERA_SAFETY_SCALE = 1.03
FIXED_ISOMETRIC_MODE = "isometric_north_ur"
# Wall opacity amount: 0.0 fully transparent, 1.0 fully opaque.
TOP_WALL_ALPHA = 1.0
ISO_WALL_ALPHA = 0.55
# Compensate stacked wall shells: per-surface opacity = alpha ** gamma.
# Keep semantic endpoints unchanged: 0 -> 0 (transparent), 1 -> 1 (opaque).
WALL_ALPHA_RESPONSE_GAMMA = 8.0
ISO_MIN_BBOX_PX = 360.0
ISO_MIN_AREA_RATIO = 0.00075
TOP_FRAME_SAFE_MARGIN_PX = 8.0
# Render-time wall cap: keep wall top above objects (reference style), rely on camera
# parameters (not low wall height) to expose interior in isometric view.
WALL_TOP_MARGIN = 0.04
WALL_TOP_MAX_ABOVE_FLOOR = 2.90
WALL_TOP_OBJECT_PERCENTILE = 1.00
USE_WALL_PROXY_RENDER = False
# Whether wall target resolver should also include door/window frames as walls.
WALL_INCLUDE_OPENING_FRAMES = False
# Wall conflict criterion: overlap_depth / wall_depth (per axis).
# Conflict when max ratio across 4 directions exceeds this threshold.
WALL_CONFLICT_DEPTH_RATIO_THRESHOLD = 0.14
# Additional gate for generated wall_conflict cases: require clear wall penetration.
WALL_CONFLICT_VISIBLE_MIN_OVERFLOW_M = 0.16
WALL_CONFLICT_VISIBLE_MIN_OVERFLOW_RATIO = 0.03
WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO = 0.20

WALL_CONTACT_BLOCK_MIN_PENETRATION_RATIO = 0.01
WALL_CONTACT_BLOCK_MAX_DISTANCE_M = 0.02
# Per-scene object budget (each anomaly main type) for synthetic construction.
# For one scene, each main type tries at most N candidate objects; per-object action
# enumeration remains exhaustive until a valid case is found.
OVERLAP_OBJECT_BUDGET_PER_SCENE = 8
WALL_DOOR_OBJECT_BUDGET_PER_SCENE = 8
ORIENTATION_OBJECT_BUDGET_PER_SCENE = 12
# Orientation target filters (shape/size/near-wall).
ORIENTATION_MIN_VOLUME_THRESHOLD = 0.06
ORIENTATION_MAX_ASPECT_RATIO = 3.40
ORIENTATION_WALL_NEAR_DIST_M = 1.90
# Keep angle threshold aligned with preprocess-clean orientation rule.
ORIENTATION_WALL_AXIS_ALIGN_TOL_DEG = 12.0
ORIENTATION_MIN_ANGLE_DELTA_DEG = ORIENTATION_WALL_AXIS_ALIGN_TOL_DEG
# Disallow rotation injection on near-circular objects.
ROTATE_NEAR_CIRCULAR_MAX_ASPECT = 1.15
ROTATE_NEAR_CIRCULAR_MAX_RECTANGULARITY = 0.90
ROTATE_NEAR_CIRCULAR_STRICT_ASPECT = 1.05
# Heavy post-injection consistency check.
ENABLE_STRICT_INJECTED_ISSUE_CHECK = True
POST_PROCESS_RETRY_MAX_ATTEMPTS_PER_CASE = 8


def parse_args() -> argparse.Namespace:
	argv = sys.argv[1:]
	if "--" in argv:
		argv = argv[argv.index("--") + 1 :]

	parser = argparse.ArgumentParser(description="Construct indoor error-mode QA")
	parser.add_argument("--glb-dir", type=Path, default=DEFAULT_GLB_DIR)
	parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
	parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
	parser.add_argument("--whitelist-json", type=Path, default=DEFAULT_WHITELIST_JSON)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--seed", type=int, default=30)
	parser.add_argument("--max-scenes", type=int, default=0, help="0 means all")
	parser.add_argument("--region", type=str, default="", help="keyword filter")
	parser.add_argument(
		"--case-ids",
		type=str,
		default="",
		help="Comma-separated 1-based case ids to run (e.g. 1,2,3). Empty means default behavior.",
	)
	parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
	parser.add_argument("--worker-index", type=int, default=-1, help="0-based worker index; -1 means parent mode")
	if hasattr(argparse, "BooleanOptionalAction"):
		parser.add_argument(
			"--all-cases",
			"--all_cases",
			dest="all_cases",
			action=argparse.BooleanOptionalAction,
			default=True,
			help="Whether to try all 7 case types per scene (default: true). Use --no-all-cases to run single-case mode.",
		)
	else:
		group = parser.add_mutually_exclusive_group()
		group.add_argument("--all-cases", "--all_cases", dest="all_cases", action="store_true")
		group.add_argument("--no-all-cases", dest="all_cases", action="store_false")
		parser.set_defaults(all_cases=True)
	return parser.parse_args(argv)


CASE_CYCLE = [
	(ISSUE_OVERLAP, "move"),
	(ISSUE_OVERLAP, "rotate"),
	(ISSUE_OVERLAP, "scale"),
	(ISSUE_WALL, "move"),
	(ISSUE_WALL, "rotate"),
	(ISSUE_WALL, "scale"),
	(ISSUE_ANGLE, "rotate"),
]


def _target_case(sample_idx: int) -> tuple[str, str]:
	return CASE_CYCLE[sample_idx % len(CASE_CYCLE)]


def _target_case_by_id(case_id: int) -> tuple[str, str]:
	return CASE_CYCLE[int(case_id) % len(CASE_CYCLE)]


def _parse_case_ids_arg(case_ids_text: str) -> list[int] | None:
	txt = str(case_ids_text or "").strip()
	if not txt:
		return None
	out: list[int] = []
	for tok in txt.split(","):
		s = tok.strip()
		if not s:
			continue
		if not s.isdigit():
			raise ValueError(f"Invalid --case-ids token: {s!r}")
		case_no = int(s)
		if case_no < 1 or case_no > len(CASE_CYCLE):
			raise ValueError(f"--case-ids value out of range [1,{len(CASE_CYCLE)}]: {case_no}")
		case_id = case_no - 1
		if case_id not in out:
			out.append(case_id)
	return out if out else None


def _issue_main_type(issue: dict) -> str:
	main_t = str(issue.get("main_type", "")).strip()
	if main_t in {ISSUE_WALL, ISSUE_WALL_DOOR, ISSUE_DOOR, ISSUE_PATH}:
		return ISSUE_WALL
	if main_t:
		return main_t
	t = str(issue.get("type", "")).strip()
	if " by " in t:
		t = t.split(" by ", 1)[0].strip()
	if t in {ISSUE_WALL, ISSUE_WALL_DOOR, ISSUE_DOOR, ISSUE_PATH}:
		return ISSUE_WALL
	return t


def _issue_type_from_wall_subtype(subtype: str | None) -> str:
	_ = subtype
	return ISSUE_WALL


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


def augment_overlap_whitelist_with_surface_support(whitelist: set[tuple[str, str]]) -> set[tuple[str, str]]:
	
	host_cats = ["desk", "table", "counter", "shelf", "cabinet", "wardrobe", "dresser", "stand"]
	top_items = [
		"book", "cup", "bottle", "vase", "plant", "laptop", "keyboard", "mouse", "monitor",
		"computer", "speaker", "telephone", "file", "paper", "paper cutter", "bowl", "plate",
		"fruit", "food", "teapot", "kettle", "kitchenware", "decoration", "object", "container",
		"box", "basket", "jar", "clock", "toy", "tissue box", "toiletry", "soap dispenser",
		"soap dish", "bag", "backpack", "shoe", "towel", "clothes", "jacket", "lamp", "light",
	]
	out = set(whitelist)
	for h in host_cats:
		for t in top_items:
			out.add(tuple(sorted((h, t))))
	for a, b in [("sink", "cabinet"), ("sink", "counter"), ("basin", "cabinet"), ("basin", "counter")]:
		out.add(tuple(sorted((a, b))))
	for a, b in [
		("chair", "table"), ("chair", "desk"), ("chair", "counter"),
		("chair", "carpet"), ("chair", "rug"),
		("stool", "table"), ("stool", "desk"), ("stool", "counter"),
		("bench", "table"), ("bench", "desk"), ("bench", "counter"),
	]:
		out.add(tuple(sorted((a, b))))
	return out


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


def build_wall_path(glb_name: str, layout_root: Path) -> Path | None:
	stem = Path(glb_name).stem
	
	if stem.endswith("_clean"):
		stem = stem[:-6]
	if stem.startswith("3rscan__"):
		scene_id = stem.split("__", 1)[1]
		cands = [
			layout_root / "3rscan" / scene_id / "StructureMesh/wall.glb",
			layout_root / "3rscan" / scene_id / "wall.glb",
		]
		for p in cands:
			if p.exists():
				return p
		return None
	if stem.startswith("arkitscenes__"):
		parts = stem.split("__")
		if len(parts) >= 3:
			cands = [
				layout_root / "arkitscenes" / parts[1] / parts[2] / "StructureMesh/wall.glb",
				layout_root / "arkitscenes" / parts[1] / parts[2] / "wall.glb",
			]
			for p in cands:
				if p.exists():
					return p

		if len(parts) >= 3:
			scene_id = parts[2]
			for p in layout_root.glob(f"arkitscenes/*/{scene_id}/StructureMesh/wall.glb"):
				if p.exists():
					return p
	if stem.startswith("scannet__"):
		scene_id = stem.split("__", 1)[1]
		cands = [
			layout_root / "scannet" / scene_id / "StructureMesh/wall.glb",
			layout_root / "scannet" / scene_id / "wall.glb",
		]
		for p in cands:
			if p.exists():
				return p
		return None
	if stem.startswith("matterport3d__"):
		parts = stem.split("__")
		if len(parts) >= 3 and parts[2].startswith("region"):
			house_id = parts[1]
			region_id = parts[2]
			cands = [
				layout_root / "matterport3d" / house_id / region_id / "StructureMesh/wall.glb",
				layout_root / "matterport3d" / house_id / region_id / "wall.glb",
			]
			for p in cands:
				if p.exists():
					return p
		scene_id = stem.split("__", 1)[1]
		cands = [
			layout_root / "matterport3d" / scene_id / "StructureMesh/wall.glb",
			layout_root / "matterport3d" / scene_id / "wall.glb",
		]
		for p in cands:
			if p.exists():
				return p
		return None
	return None


def _import_wall_bounds(wall_glb: Path | None) -> dict | None:
	if wall_glb is None or not wall_glb.exists():
		return None
	before = {o.name for o in bpy.data.objects}
	try:
		bpy.ops.import_scene.gltf(filepath=str(wall_glb))
	except Exception:
		return None
	new_mesh_objs = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
	if not new_mesh_objs:
		return None
	xs, ys, zs = [], [], []
	for obj in new_mesh_objs:
		for c in obj.bound_box:
			w = obj.matrix_world @ Vector(c)
			xs.append(float(w.x))
			ys.append(float(w.y))
			zs.append(float(w.z))
	try:
		if not xs:
			return None
		return {
			"min_x": min(xs), "max_x": max(xs),
			"min_y": min(ys), "max_y": max(ys),
			"min_z": min(zs), "max_z": max(zs),
		}
	finally:
	
		for obj in new_mesh_objs:
			try:
				bpy.data.objects.remove(obj, do_unlink=True)
			except Exception:
				pass


def _import_wall_components(wall_glb: Path | None) -> list:
	"""Import wall.glb and extract merged 2D wall-band polygons (horizontal faces)."""
	if not SHAPELY_AVAILABLE or wall_glb is None or not wall_glb.exists():
		return []
	before = {o.name for o in bpy.data.objects}
	try:
		bpy.ops.import_scene.gltf(filepath=str(wall_glb))
	except Exception:
		return []
	new_mesh_objs = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
	polys = []
	try:
		for obj in new_mesh_objs:
			mesh = obj.data
			mw = obj.matrix_world
			nm = mw.to_3x3()
			for poly in mesh.polygons:
				try:
					wn = nm @ poly.normal
				except Exception:
					continue
				# Keep top/bottom-ish faces to get wall thickness bands in XY.
				if abs(float(wn.z)) < 0.7:
					continue
				pts = []
				for vid in poly.vertices:
					w = mw @ mesh.vertices[vid].co
					pts.append((float(w.x), float(w.y)))
				try:
					p = Polygon(pts)
				except Exception:
					continue
				if not p.is_valid:
					p = p.buffer(0)
				if p.is_empty or float(p.area) <= 1e-6:
					continue
				polys.append(p)
		if not polys:
			return []
		try:
			merged = unary_union(polys)
		except Exception:
			return []
		comps = []
		geoms = list(getattr(merged, "geoms", [merged]))
		for g in geoms:
			if g is None or g.is_empty:
				continue
			if not g.is_valid:
				g = g.buffer(0)
			if g.is_empty:
				continue
			if g.geom_type == "Polygon":
				if float(g.area) > 1e-5:
					comps.append(g)
			elif g.geom_type == "MultiPolygon":
				for gg in g.geoms:
					if gg is not None and (not gg.is_empty) and float(gg.area) > 1e-5:
						comps.append(gg)
		return comps
	finally:
		for obj in new_mesh_objs:
			try:
				bpy.data.objects.remove(obj, do_unlink=True)
			except Exception:
				pass


def _import_wall_proxy(wall_glb: Path | None) -> list:
	"""Import external wall.glb as render proxy walls."""
	if wall_glb is None or not wall_glb.exists():
		return []
	before = {o.name for o in bpy.data.objects}
	try:
		bpy.ops.import_scene.gltf(filepath=str(wall_glb))
	except Exception:
		return []
	new_mesh_objs = [o for o in bpy.data.objects if o.name not in before and o.type == "MESH"]
	for obj in new_mesh_objs:
		try:
			obj["_dc_wall_proxy"] = True
			if not str(obj.name).startswith("__wall_proxy__"):
				obj.name = f"__wall_proxy__{obj.name}"
		except Exception:
			pass
	return new_mesh_objs


def _hide_original_wall_structures(wall_bounds: dict | None):
	"""Hide wall-like meshes from composed scene when proxy walls are available."""
	if wall_bounds is None:
		return
	wminx, wmaxx = float(wall_bounds["min_x"]), float(wall_bounds["max_x"])
	wminy, wmaxy = float(wall_bounds["min_y"]), float(wall_bounds["max_y"])
	wminz, wmaxz = float(wall_bounds["min_z"]), float(wall_bounds["max_z"])
	for obj in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
		try:
			if bool(obj.get("_dc_wall_proxy", False)):
				continue
		except Exception:
			pass
		n = str(obj.name).lower()
		# Keep scene instances; only consider structure-like meshes.
		if re.match(r"^\d+_", str(obj.name)) is not None:
			continue
		if "floor" in n or "ceiling" in n:
			continue
		b = _group_bounds([obj])
		ominx, omaxx = float(b["min_x"]), float(b["max_x"])
		ominy, omaxy = float(b["min_y"]), float(b["max_y"])
		ominz, omaxz = float(b["min_z"]), float(b["max_z"])
		ix = max(0.0, min(wmaxx, omaxx) - max(wminx, ominx))
		iy = max(0.0, min(wmaxy, omaxy) - max(wminy, ominy))
		iz = max(0.0, min(wmaxz, omaxz) - max(wminz, ominz))
		if ix <= 0.0 or iy <= 0.0 or iz <= 0.0:
			continue
		ow = max(1e-6, omaxx - ominx)
		od = max(1e-6, omaxy - ominy)
		oh = max(1e-6, omaxz - ominz)
		if oh < 0.30:
			continue
		try:
			obj.hide_render = True
		except Exception:
			pass


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


def _build_group_bvh(group: list) -> BVHTree | None:
	if not group:
		return None
	depsgraph = bpy.context.evaluated_depsgraph_get()
	verts_world: list[Vector] = []
	tris: list[tuple[int, int, int]] = []
	v_ofs = 0
	for obj in group:
		if obj.type != "MESH":
			continue
		obj_eval = obj.evaluated_get(depsgraph)
		mesh = obj_eval.to_mesh()
		if mesh is None:
			continue
		try:
			mesh.calc_loop_triangles()
			mw = obj_eval.matrix_world
			for v in mesh.vertices:
				verts_world.append(mw @ v.co)
			for tri in mesh.loop_triangles:
				i0, i1, i2 = tri.vertices
				tris.append((v_ofs + i0, v_ofs + i1, v_ofs + i2))
			v_ofs += len(mesh.vertices)
		finally:
			obj_eval.to_mesh_clear()
	if len(verts_world) < 3 or len(tris) == 0:
		return None
	try:
		return BVHTree.FromPolygons(verts_world, tris, all_triangles=True, epsilon=1e-6)
	except Exception:
		return None


def _bvh_overlap_hit_count(bvh_a: BVHTree | None, bvh_b: BVHTree | None) -> int:
	if bvh_a is None or bvh_b is None:
		return 0
	try:
		return int(len(bvh_a.overlap(bvh_b)))
	except Exception:
		return 0


def _bvh_overlap_strong(bvh_a: BVHTree | None, bvh_b: BVHTree | None, min_hits: int = BVH_OVERLAP_MIN_HITS) -> bool:
	return _bvh_overlap_hit_count(bvh_a, bvh_b) >= int(max(1, min_hits))


def _group_bounds(group_objs: list) -> dict:
	pts = []
	for obj in group_objs:
		for c in obj.bound_box:
			pts.append(obj.matrix_world @ Vector(c))
	xs = [p.x for p in pts]
	ys = [p.y for p in pts]
	zs = [p.z for p in pts]
	return {
		"min_x": float(min(xs)),
		"max_x": float(max(xs)),
		"min_y": float(min(ys)),
		"max_y": float(max(ys)),
		"min_z": float(min(zs)),
		"max_z": float(max(zs)),
		"center_x": float((min(xs) + max(xs)) / 2),
		"center_y": float((min(ys) + max(ys)) / 2),
		"center_z": float((min(zs) + max(zs)) / 2),
		"half_w": float((max(xs) - min(xs)) / 2),
		"half_d": float((max(ys) - min(ys)) / 2),
		"width": float(max(xs) - min(xs)),
		"depth": float(max(ys) - min(ys)),
		"height": float(max(zs) - min(zs)),
	}


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
		"ax": (float(ax[0]), float(ax[1])),
		"ay": (float(ay[0]), float(ay[1])),
		"hx": float(hx),
		"hy": float(hy),
		"center": (float(cx), float(cy)),
	}


def _convex_hull_xy(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
	if len(points) <= 1:
		return points[:]
	pts = sorted(set(points))
	if len(pts) <= 1:
		return pts

	def cross(o, a, b):
		return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

	lower: list[tuple[float, float]] = []
	for p in pts:
		while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
			lower.pop()
		lower.append(p)
	upper: list[tuple[float, float]] = []
	for p in reversed(pts):
		while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
			upper.pop()
		upper.append(p)
	return lower[:-1] + upper[:-1]


def _polygon_area_xy(poly: list[tuple[float, float]]) -> float:
	if len(poly) < 3:
		return 0.0
	s = 0.0
	for i in range(len(poly)):
		x1, y1 = poly[i]
		x2, y2 = poly[(i + 1) % len(poly)]
		s += x1 * y2 - x2 * y1
	return abs(s) * 0.5


def _round_like_category(cat: str) -> bool:
	c = str(cat).lower()
	round_tokens = {"round", "circle", "circular", "cylinder", "cylindrical", "disc", "disk", "orb"}
	return any(t in c for t in round_tokens)


def _orientation_rect_like(meta: dict, cat: str) -> bool:
	if _round_like_category(cat):
		return False
	rectangularity = float(meta.get("rectangularity", 0.0))
	if rectangularity < 0.86:
		return False
	return True


def _is_near_circular_for_rotate(meta: dict, cat: str | None) -> bool:
	"""Global rotate-target guard: avoid round/near-round objects."""
	c = str(cat or "").strip().lower()
	if _round_like_category(c):
		return True
	try:
		hx = float(meta["obb"]["hx"])
		hy = float(meta["obb"]["hy"])
	except Exception:
		return False
	aspect = max(hx, hy) / max(min(hx, hy), 1e-6)
	rectangularity = float(meta.get("rectangularity", 1.0))
	if aspect <= float(ROTATE_NEAR_CIRCULAR_MAX_ASPECT) and rectangularity <= float(ROTATE_NEAR_CIRCULAR_MAX_RECTANGULARITY):
		return True
	if aspect <= float(ROTATE_NEAR_CIRCULAR_STRICT_ASPECT):
		return True
	return False


def _group_footprint(group_objs: list):
	"""Extract group footprint using near-horizontal mesh faces (region-style)."""
	if not SHAPELY_AVAILABLE or not group_objs:
		return None
	depsgraph = bpy.context.evaluated_depsgraph_get()
	valid_polys = []
	for obj in group_objs:
		if obj is None or obj.type != "MESH":
			continue
		try:
			obj_eval = obj.evaluated_get(depsgraph)
			mesh = obj_eval.to_mesh()
		except Exception:
			mesh = None
		if mesh is None:
			continue
		try:
			world_matrix = obj.matrix_world
			transform = world_matrix.to_3x3()
			for poly in mesh.polygons:
				world_coords = [world_matrix @ mesh.vertices[idx].co for idx in poly.vertices]
				if len(world_coords) < 3:
					continue
				world_normal = transform @ poly.normal
				# Keep top/bottom-ish faces; ignore near-vertical faces.
				if abs(world_normal.z) < 0.7:
					continue
				coords_2d = [(float(v.x), float(v.y)) for v in world_coords]
				face_poly = Polygon(coords_2d)
				if not face_poly.is_valid:
					face_poly = face_poly.buffer(0)
				if not face_poly.is_empty and float(face_poly.area) > 0.01:
					valid_polys.append(face_poly)
		finally:
			obj_eval.to_mesh_clear()
	if not valid_polys:
		return None
	try:
		footprint = unary_union(valid_polys)
	except Exception:
		return None
	if footprint is None or footprint.is_empty:
		return None
	return footprint


def _angle_diff_mod_180(a: float, b: float) -> float:
	d = abs((a - b) % 180.0)
	if d > 90.0:
		d = 180.0 - d
	return d


def _is_orientation_candidate_shape_group(group_objs: list, cat: str) -> bool:
	"""Region-style orientation candidate shape check (rectangle / near-rectangle / L-shape)."""
	if not SHAPELY_AVAILABLE:
		return False
	footprint = _group_footprint(group_objs)
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
	return bool(is_rectangle or is_l_shape)


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
			"ax": (1.0, 0.0),
			"ay": (0.0, 1.0),
			"hx": b["half_w"],
			"hy": b["half_d"],
			"center": (b["center_x"], b["center_y"]),
		}
	hull = _convex_hull_xy(pts)
	hull_area = _polygon_area_xy(hull)
	obb_area = max(1e-6, 4.0 * float(obb["hx"]) * float(obb["hy"]))
	rectangularity = min(1.0, hull_area / obb_area)
	aspect_ratio = max(float(obb["hx"]), float(obb["hy"])) / max(min(float(obb["hx"]), float(obb["hy"])), 1e-6)

	return {
		"label_id": int(label_id),
		"scene_object_id": str(scene_obj_id),
		"instance_index": int(instance_index),
		"half_w": b["half_w"],
		"half_d": b["half_d"],
		"height": b["height"],
		"obb": obb,
		"rectangularity": float(rectangularity),
		"aspect_ratio": float(aspect_ratio),
	}


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


def _project_group_mask_overlap_area_px(
	group_a: list,
	group_b: list,
	cam,
	width: int,
	height: int,
) -> float:
	"""Approximate true top projected overlap by rasterizing mesh triangles into two masks."""
	if (not PIL_AVAILABLE) or cam is None or width <= 1 or height <= 1:
		return 0.0
	scene = bpy.context.scene
	depsgraph = bpy.context.evaluated_depsgraph_get()
	max_side = max(int(width), int(height))
	scale = 1.0
	if max_side > int(OVERLAP_MASK_RASTER_MAX_SIDE):
		scale = float(OVERLAP_MASK_RASTER_MAX_SIDE) / float(max_side)
	rw = max(1, int(round(float(width) * scale)))
	rh = max(1, int(round(float(height) * scale)))

	def _raster_group_mask(group_objs: list):
		mask = Image.new("1", (rw, rh), 0)
		draw = ImageDraw.Draw(mask)
		for obj in group_objs:
			if obj is None or obj.type != "MESH":
				continue
			obj_eval = obj.evaluated_get(depsgraph)
			mesh = obj_eval.to_mesh()
			if mesh is None:
				continue
			try:
				mesh.calc_loop_triangles()
				mw = obj_eval.matrix_world
				for tri in mesh.loop_triangles:
					pts = []
					valid = True
					for vid in tri.vertices:
						wc = mw @ mesh.vertices[vid].co
						v = world_to_camera_view(scene, cam, wc)
						if v.z <= 1e-6:
							valid = False
							break
						sx = float(v.x) * float(width) * scale
						sy = (1.0 - float(v.y)) * float(height) * scale
						pts.append((sx, sy))
					if not valid:
						continue
					draw.polygon(pts, fill=1)
			finally:
				obj_eval.to_mesh_clear()
		return mask

	try:
		m1 = _raster_group_mask(group_a)
		m2 = _raster_group_mask(group_b)
		inter = ImageChops.logical_and(m1, m2)
		hist = inter.histogram()
		inter_px_scaled = float(sum(hist[1:])) if len(hist) >= 2 else 0.0
		if scale <= 1e-9:
			return 0.0
		return inter_px_scaled / (scale * scale)
	except Exception:
		return 0.0


def _project_group_bbox_rect(group_objs: list, cam, width: int, height: int) -> tuple[float, float, float, float] | None:
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
		return None
	min_x, max_x = max(0.0, min(xs)), min(float(width), max(xs))
	min_y, max_y = max(0.0, min(ys)), min(float(height), max(ys))
	if max_x <= min_x or max_y <= min_y:
		return None
	return (min_x, min_y, max_x, max_y)


def _project_group_bbox_rect_raw(group_objs: list, cam, width: int, height: int) -> tuple[float, float, float, float] | None:
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
		return None
	min_x, max_x = min(xs), max(xs)
	min_y, max_y = min(ys), max(ys)
	if max_x <= min_x or max_y <= min_y:
		return None
	return (min_x, min_y, max_x, max_y)


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
	# Looser pre-filter: allow either absolute pixel size or relative area to pass.
	if area_px < TOP_MIN_BBOX_PX and area_ratio < TOP_MIN_AREA_RATIO:
		return False
	# Accept center-visible OR any in-frame footprint for pre-filter counting.
	if _is_group_center_visible(group_objs, cam):
		return True
	return _group_any_part_in_frame(group_objs, cam, width, height)


def _group_any_part_in_frame(group_objs: list, cam, width: int, height: int) -> bool:
	"""Check if any part of the group is visible in the camera frame."""
	if not group_objs:
		return False
	# Check multiple points: center and corners
	b = _group_bounds(group_objs)
	points = [
		Vector((b["center_x"], b["center_y"], b["center_z"])),
		Vector((b["min_x"], b["min_y"], b["center_z"])),
		Vector((b["max_x"], b["min_y"], b["center_z"])),
		Vector((b["min_x"], b["max_y"], b["center_z"])),
		Vector((b["max_x"], b["max_y"], b["center_z"])),
	]
	for pt in points:
		if _project_point_in_frame(pt, cam, width, height):
			return True
	return False


def _group_fully_in_frame(group_objs: list, cam, width: int, height: int, margin_px: float = TOP_FRAME_SAFE_MARGIN_PX) -> bool:
	"""Require projected bbox to stay fully inside frame with a small safety margin."""
	rect = _project_group_bbox_rect_raw(group_objs, cam, width, height)
	if rect is None:
		return False
	min_x, min_y, max_x, max_y = rect
	margin = float(max(0.0, margin_px))
	if min_x < margin or min_y < margin:
		return False
	if max_x > (float(width) - margin) or max_y > (float(height) - margin):
		return False
	return True


def _is_object_visible_after_action(group_objs: list, cam, width: int, height: int) -> bool:
	"""Check if object is visible after action - allow smaller area but must be in frame."""
	area_px, area_ratio = _project_group_bbox_metrics(group_objs, cam, width, height)
	if area_px < 100.0 or area_ratio < 0.0003:
		return False
	if _is_group_center_visible(group_objs, cam):
		return True
	return _group_any_part_in_frame(group_objs, cam, width, height)


def _iso_group_visibility_ok(group_objs: list, cam, width: int, height: int) -> bool:
	area_px, area_ratio = _project_group_bbox_metrics(group_objs, cam, width, height)
	if area_px < ISO_MIN_BBOX_PX or area_ratio < ISO_MIN_AREA_RATIO:
		return False
	return _is_group_center_visible(group_objs, cam)


def _project_point_in_frame(world_pt: Vector, cam, width: int, height: int) -> bool:
	scene = bpy.context.scene
	v = world_to_camera_view(scene, cam, world_pt)
	if v.z <= 1e-6:
		return False
	sx = float(v.x) * float(width)
	sy = (1.0 - float(v.y)) * float(height)
	return (0.0 <= sx <= float(width)) and (0.0 <= sy <= float(height))


def _point_visible_from_cam(world_pt: Vector, cam, allowed_group_objs: list) -> bool:
	scene = bpy.context.scene
	depsgraph = bpy.context.evaluated_depsgraph_get()
	origin = cam.matrix_world.translation
	vec = world_pt - origin
	dist = vec.length
	if dist < 1e-6:
		return True
	direction = vec.normalized()
	hit, _loc, _normal, _face_idx, hit_obj, _hit_m = scene.ray_cast(depsgraph, origin, direction, distance=dist + 1e-4)
	if not hit or hit_obj is None:
		return False
	group_names = {o.name for o in allowed_group_objs}
	if hit_obj.name in group_names:
		return True
	try:
		if hit_obj.original is not None and hit_obj.original.name in group_names:
			return True
	except Exception:
		pass
	return False


def _wall_conflict_part_visible_iso(
	label_id: int,
	cam,
	width: int,
	height: int,
	label_to_oid: dict[int, str],
	id_to_entry: dict[str, dict],
	cur_metas: dict[int, dict],
	wall_bounds: dict | None,
	wall_components: list | None = None,
) -> bool:
	"""For wall_conflict, require the conflicting side/part to be visible in isometric."""
	if wall_bounds is None:
		return False
	oid = label_to_oid.get(label_id)
	if oid is None or oid not in id_to_entry:
		return False
	m = cur_metas.get(label_id)
	if m is None:
		return False
	cx, cy = m["obb"]["center"]
	hw = float(m["half_w"])
	hd = float(m["half_d"])
	cz = float(_group_bounds(id_to_entry[oid]["group"])["center_z"])

	# Probe points on the object boundary toward each violated wall side.
	probes: list[Vector] = []
	ratios = _wall_overlap_depth_ratios(cx, cy, hw, hd, wall_bounds)
	if float(ratios["left"]) > float(WALL_CONFLICT_DEPTH_RATIO_THRESHOLD):
		probes.append(Vector((cx - hw, cy, cz)))
	if float(ratios["right"]) > float(WALL_CONFLICT_DEPTH_RATIO_THRESHOLD):
		probes.append(Vector((cx + hw, cy, cz)))
	if float(ratios["bottom"]) > float(WALL_CONFLICT_DEPTH_RATIO_THRESHOLD):
		probes.append(Vector((cx, cy - hd, cz)))
	if float(ratios["top"]) > float(WALL_CONFLICT_DEPTH_RATIO_THRESHOLD):
		probes.append(Vector((cx, cy + hd, cz)))
	if not probes:
		if SHAPELY_AVAILABLE and wall_components:
			near = _nearest_wall_overlap_ratio_for_object(
				m,
				wall_components,
				group_objs=id_to_entry[oid]["group"],
			)
			if float(near.get("ratio", 0.0)) >= float(WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO):
				wp = near.get("wall_point")
				if isinstance(wp, tuple) and len(wp) == 2:
					probes.append(Vector((float(wp[0]), float(wp[1]), cz)))
	if not probes:
		return False
	group_objs = id_to_entry[oid]["group"]
	for p in probes:
		if not _project_point_in_frame(p, cam, width, height):
			continue
		if _point_visible_from_cam(p, cam, group_objs):
			return True
	return False


def _rebuild_metas_by_label(labels: list[int], label_to_oid: dict[int, str], id_to_entry: dict[str, dict]) -> dict[int, dict]:
	metas = {}
	for lb in labels:
		oid = label_to_oid[lb]
		e = id_to_entry[oid]
		metas[lb] = _build_meta_from_group(lb, oid, e["instance_index"], e["group"])
	return metas


def _wall_overlap_depth_ratios(cx: float, cy: float, hw: float, hd: float, wall_bounds: dict | None) -> dict[str, float]:
	"""Compute normalized overlap depth (overlap_depth / wall_depth) for 4 wall directions."""
	if wall_bounds is None:
		return {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0, "max_ratio": 0.0}

	wminx = float(wall_bounds["min_x"])
	wmaxx = float(wall_bounds["max_x"])
	wminy = float(wall_bounds["min_y"])
	wmaxy = float(wall_bounds["max_y"])
	wdepth_x = max(1e-6, wmaxx - wminx)
	wdepth_y = max(1e-6, wmaxy - wminy)

	overlap_left = max(0.0, wminx - (float(cx) - float(hw)))
	overlap_right = max(0.0, (float(cx) + float(hw)) - wmaxx)
	overlap_bottom = max(0.0, wminy - (float(cy) - float(hd)))
	overlap_top = max(0.0, (float(cy) + float(hd)) - wmaxy)

	r_left = overlap_left / wdepth_x
	r_right = overlap_right / wdepth_x
	r_bottom = overlap_bottom / wdepth_y
	r_top = overlap_top / wdepth_y
	mx = max(r_left, r_right, r_bottom, r_top)
	return {
		"left": float(r_left),
		"right": float(r_right),
		"bottom": float(r_bottom),
		"top": float(r_top),
		"max_ratio": float(mx),
	}


def _is_region_boundary_relaxed_category(cat: str | None) -> bool:
	if not cat:
		return False
	c = str(cat).strip().lower()
	if c in REGION_BOUNDARY_RELAX_KEYWORDS:
		return True
	return any(k in c for k in REGION_BOUNDARY_RELAX_KEYWORDS)


def _inside_wall_region_by_bbox(
	cx: float,
	cy: float,
	hw: float,
	hd: float,
	wall_bounds: dict | None,
	inside_ratio_thr: float = REGION_BOUNDARY_RATIO_DEFAULT,
) -> bool:
	if wall_bounds is None:
		return True
	obj_min_x = float(cx) - float(hw)
	obj_max_x = float(cx) + float(hw)
	obj_min_y = float(cy) - float(hd)
	obj_max_y = float(cy) + float(hd)
	obj_area = max(1e-9, (obj_max_x - obj_min_x) * (obj_max_y - obj_min_y))

	wminx = float(wall_bounds["min_x"])
	wmaxx = float(wall_bounds["max_x"])
	wminy = float(wall_bounds["min_y"])
	wmaxy = float(wall_bounds["max_y"])

	# Match preprocess rule: centroid must be in wall region.
	if not (wminx <= float(cx) <= wmaxx and wminy <= float(cy) <= wmaxy):
		return False
	ix = max(0.0, min(wmaxx, obj_max_x) - max(wminx, obj_min_x))
	iy = max(0.0, min(wmaxy, obj_max_y) - max(wminy, obj_min_y))
	inter_area = ix * iy
	ratio = inter_area / obj_area
	return bool(ratio >= float(inside_ratio_thr))


def _inside_wall_region_by_category_bbox(
	cx: float,
	cy: float,
	hw: float,
	hd: float,
	cat: str | None,
	wall_bounds: dict | None,
	ratio_default: float = REGION_BOUNDARY_RATIO_DEFAULT,
	ratio_relaxed: float = REGION_BOUNDARY_RATIO_RELAXED,
) -> bool:
	thr = float(ratio_relaxed) if _is_region_boundary_relaxed_category(cat) else float(ratio_default)
	return _inside_wall_region_by_bbox(cx, cy, hw, hd, wall_bounds, inside_ratio_thr=thr)


def _meta_obb_polygon(meta: dict):
	"""Build shapely polygon from OBB meta."""
	if not SHAPELY_AVAILABLE or not isinstance(meta, dict):
		return None
	try:
		obb = meta.get("obb", {})
		cx, cy = obb["center"]
		ax0, ax1 = obb["ax"]
		ay0, ay1 = obb["ay"]
		hx = float(obb["hx"])
		hy = float(obb["hy"])
	except Exception:
		return None
	p1 = (cx - ax0 * hx - ay0 * hy, cy - ax1 * hx - ay1 * hy)
	p2 = (cx + ax0 * hx - ay0 * hy, cy + ax1 * hx - ay1 * hy)
	p3 = (cx + ax0 * hx + ay0 * hy, cy + ax1 * hx + ay1 * hy)
	p4 = (cx - ax0 * hx + ay0 * hy, cy - ax1 * hx + ay1 * hy)
	try:
		poly = Polygon([p1, p2, p3, p4])
	except Exception:
		return None
	if not poly.is_valid:
		poly = poly.buffer(0)
	if poly.is_empty or float(poly.area) <= 1e-9:
		return None
	return poly


def _wall_probe_polygon_from_group_or_meta(meta: dict, group_objs: list | None = None):
	if not SHAPELY_AVAILABLE:
		return None
	if group_objs:
		try:
			footprint = _group_footprint(group_objs)
		except Exception:
			footprint = None
		if footprint is not None and (not footprint.is_empty):
			try:
				area = float(getattr(footprint, "area", 0.0) or 0.0)
			except Exception:
				area = 0.0
			if area > 1e-9:
				return footprint
	return _meta_obb_polygon(meta)


def _nearest_wall_overlap_ratio_for_object(
	meta: dict,
	wall_components: list | None,
	group_objs: list | None = None,
) -> dict:
	"""Wall overlap ratio using footprint-first geometry, with OBB fallback."""
	if (not SHAPELY_AVAILABLE) or (not wall_components):
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float("inf")}
	obj_poly = _wall_probe_polygon_from_group_or_meta(meta, group_objs=group_objs)
	if obj_poly is None or obj_poly.is_empty:
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float("inf")}
	return _nearest_wall_overlap_ratio(obj_poly, wall_components)


def _is_wall_conflict_with_ctx(
	cur_metas: dict[int, dict],
	a: int,
	label_to_oid: dict[int, str],
	id_to_entry: dict[str, dict],
	wall_bounds: dict | None,
	wall_components: list | None,
	desired_subtype: str | None = None,
	min_penetration_ratio: float | None = None,
	include_touch: bool = False,
) -> tuple[bool, int | None, str | None]:
	oa = label_to_oid.get(int(a))
	ca = str(id_to_entry.get(oa, {}).get("category", "")).lower() if oa is not None else ""
	if any(k in ca for k in WALL_CONFLICT_EXCLUDE_TARGET_KEYWORDS):
		return False, None, None
	m = cur_metas.get(int(a))
	if m is None:
		return False, None, None
	cx, cy = m["obb"]["center"]
	hw = m["half_w"]
	hd = m["half_d"]
	if min_penetration_ratio is None:
		min_penetration_ratio = float(WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO)
	penetrate_wall = False
	touch_wall = False
	if SHAPELY_AVAILABLE and wall_components:
		group_objs = id_to_entry.get(oa, {}).get("group") if oa is not None else None
		near = _nearest_wall_overlap_ratio_for_object(m, wall_components, group_objs=group_objs)
		ratio = float(near.get("ratio", 0.0))
		dist = float(near.get("distance", float("inf")))
		penetrate_wall = ratio >= float(min_penetration_ratio)
		if include_touch:
			touch_wall = (
				ratio >= float(WALL_CONTACT_BLOCK_MIN_PENETRATION_RATIO)
				or dist <= float(WALL_CONTACT_BLOCK_MAX_DISTANCE_M)
			)
	elif include_touch and wall_bounds is not None:
		ratios = _wall_overlap_depth_ratios(cx, cy, hw, hd, wall_bounds)
		touch_wall = float(ratios.get("max_ratio", 0.0)) >= float(WALL_CONTACT_BLOCK_MIN_PENETRATION_RATIO)
	out = bool(penetrate_wall or touch_wall)
	if out and (desired_subtype is None or desired_subtype == "furniture_wall_intersection"):
		return True, None, "furniture_wall_intersection"
	return False, None, None


def _is_wall_touch_or_conflict_with_ctx(
	cur_metas: dict[int, dict],
	a: int,
	label_to_oid: dict[int, str],
	id_to_entry: dict[str, dict],
	wall_bounds: dict | None,
	wall_components: list | None,
	desired_subtype: str | None = None,
) -> tuple[bool, int | None, str | None]:
	return _is_wall_conflict_with_ctx(
		cur_metas,
		a,
		label_to_oid=label_to_oid,
		id_to_entry=id_to_entry,
		wall_bounds=wall_bounds,
		wall_components=wall_components,
		desired_subtype=desired_subtype,
		min_penetration_ratio=float(WALL_CONTACT_BLOCK_MIN_PENETRATION_RATIO),
		include_touch=True,
	)


def _closest_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
	vx, vy = (bx - ax), (by - ay)
	den = vx * vx + vy * vy
	if den <= 1e-12:
		return (ax, ay)
	t = ((px - ax) * vx + (py - ay) * vy) / den
	t = max(0.0, min(1.0, t))
	return (ax + t * vx, ay + t * vy)


def _extract_linestrings(geom) -> list:
	if geom is None or geom.is_empty:
		return []
	gt = geom.geom_type
	if gt == "LineString":
		return [geom]
	if gt == "MultiLineString":
		return [g for g in geom.geoms if g is not None and (not g.is_empty)]
	if gt == "GeometryCollection":
		out = []
		for g in geom.geoms:
			out.extend(_extract_linestrings(g))
		return out
	return []


def _collect_geom_coords_xy(geom) -> list[tuple[float, float]]:
	if geom is None or geom.is_empty:
		return []
	gt = geom.geom_type
	if gt == "Polygon":
		return [(float(x), float(y)) for x, y in geom.exterior.coords]
	if gt == "MultiPolygon":
		out = []
		for g in geom.geoms:
			out.extend(_collect_geom_coords_xy(g))
		return out
	if gt == "LineString":
		return [(float(x), float(y)) for x, y in geom.coords]
	if gt == "MultiLineString":
		out = []
		for g in geom.geoms:
			out.extend(_collect_geom_coords_xy(g))
		return out
	if gt == "Point":
		return [(float(geom.x), float(geom.y))]
	if gt == "MultiPoint":
		return [(float(g.x), float(g.y)) for g in geom.geoms]
	if gt == "GeometryCollection":
		out = []
		for g in geom.geoms:
			out.extend(_collect_geom_coords_xy(g))
		return out
	return []


def _nearest_wall_overlap_ratio(obj_poly, wall_components: list) -> dict:
	"""
	Compute ratio = overlap_depth / wall_thickness against nearest wall component.
	Depth and thickness are measured along the local wall normal.
	"""
	if (not SHAPELY_AVAILABLE) or obj_poly is None or obj_poly.is_empty or (not wall_components):
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float("inf")}
	best = None
	best_dist = 1e30
	for wp in wall_components:
		if wp is None or wp.is_empty:
			continue
		try:
			d = float(obj_poly.distance(wp))
		except Exception:
			continue
		if d < best_dist:
			best_dist = d
			best = wp
	if best is None:
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float("inf")}

	try:
		_p_obj, p_wall = nearest_points(obj_poly, best.boundary)
	except Exception:
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float(best_dist)}
	px, py = float(p_wall.x), float(p_wall.y)

	coords = list(best.exterior.coords) if hasattr(best, "exterior") else []
	if len(coords) < 2:
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float(best_dist)}
	best_seg = None
	best_seg_d2 = 1e30
	best_cp = (px, py)
	for i in range(len(coords) - 1):
		ax, ay = float(coords[i][0]), float(coords[i][1])
		bx, by = float(coords[i + 1][0]), float(coords[i + 1][1])
		cx, cy = _closest_point_on_segment(px, py, ax, ay, bx, by)
		d2 = (cx - px) ** 2 + (cy - py) ** 2
		if d2 < best_seg_d2:
			best_seg_d2 = d2
			best_seg = (ax, ay, bx, by)
			best_cp = (cx, cy)
	if best_seg is None:
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float(best_dist)}

	ax, ay, bx, by = best_seg
	tx, ty = (bx - ax), (by - ay)
	tl = math.hypot(tx, ty)
	if tl < 1e-9:
		return {"ratio": 0.0, "overlap_depth": 0.0, "wall_thickness": 1.0, "distance": float(best_dist)}
	tx, ty = tx / tl, ty / tl
	n1 = (-ty, tx)
	n2 = (ty, -tx)
	cpx, cpy = best_cp
	eps = 1e-3
	try:
		in1 = bool(best.buffer(1e-9).contains(Point(cpx + n1[0] * eps, cpy + n1[1] * eps)))
	except Exception:
		in1 = False
	nx, ny = n1 if in1 else n2

	b = best.bounds
	line_half_len = max(float(b[2] - b[0]), float(b[3] - b[1]), 1.0) * 2.5 + 1.0
	try:
		cross_line = LineString([(cpx - nx * line_half_len, cpy - ny * line_half_len), (cpx + nx * line_half_len, cpy + ny * line_half_len)])
		inter_line = best.intersection(cross_line)
		line_parts = _extract_linestrings(inter_line)
	except Exception:
		line_parts = []
	wall_thickness = 0.0
	if line_parts:
		p_ref = Point(cpx, cpy)
		line_parts.sort(key=lambda ls: float(ls.distance(p_ref)))
		wall_thickness = float(line_parts[0].length)
	if wall_thickness <= 1e-9:
		wall_thickness = 1e-9

	try:
		ov = obj_poly.intersection(best)
	except Exception:
		ov = None
	if ov is None or ov.is_empty:
		return {
			"ratio": 0.0,
			"overlap_depth": 0.0,
			"wall_thickness": float(wall_thickness),
			"distance": float(best_dist),
			"normal": (float(nx), float(ny)),
			"wall_point": (float(cpx), float(cpy)),
		}
	coords_ov = _collect_geom_coords_xy(ov)
	if not coords_ov:
		return {
			"ratio": 0.0,
			"overlap_depth": 0.0,
			"wall_thickness": float(wall_thickness),
			"distance": float(best_dist),
			"normal": (float(nx), float(ny)),
			"wall_point": (float(cpx), float(cpy)),
		}
	projs = [float(x) * float(nx) + float(y) * float(ny) for x, y in coords_ov]
	overlap_depth = max(0.0, max(projs) - min(projs))
	ratio = overlap_depth / max(wall_thickness, 1e-9)
	return {
		"ratio": float(ratio),
		"overlap_depth": float(overlap_depth),
		"wall_thickness": float(wall_thickness),
		"distance": float(best_dist),
		"normal": (float(nx), float(ny)),
		"wall_point": (float(cpx), float(cpy)),
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


def _aabb_overlap_xy(ma: dict, mb: dict, pos_a=None, pos_b=None) -> bool:
	ax, ay = pos_a if pos_a is not None else tuple(ma["obb"]["center"][:2])
	bx, by = pos_b if pos_b is not None else tuple(mb["obb"]["center"][:2])
	min_ax = ax - ma["half_w"]
	max_ax = ax + ma["half_w"]
	min_ay = ay - ma["half_d"]
	max_ay = ay + ma["half_d"]
	min_bx = bx - mb["half_w"]
	max_bx = bx + mb["half_w"]
	min_by = by - mb["half_d"]
	max_by = by + mb["half_d"]
	if max_ax < min_bx or min_ax > max_bx:
		return False
	if max_ay < min_by or min_ay > max_by:
		return False
	return True


def _obb_overlap(ma: dict, mb: dict, pos_a=None, pos_b=None) -> bool:
	oa = ma["obb"]
	ob = mb["obb"]
	ca = tuple(pos_a[:2]) if pos_a is not None else tuple(oa["center"][:2])
	cb = tuple(pos_b[:2]) if pos_b is not None else tuple(ob["center"][:2])
	ax_a, ay_a = oa["ax"], oa["ay"]
	ax_b, ay_b = ob["ax"], ob["ay"]
	hx_a = float(oa["hx"])
	hy_a = float(oa["hy"])
	hx_b = float(ob["hx"])
	hy_b = float(ob["hy"])

	for axis in [ax_a, ay_a, ax_b, ay_b]:
		axis = _normalize2(axis)
		a0, a1 = _project_obb(ca, ax_a, ay_a, hx_a, hy_a, axis)
		b0, b1 = _project_obb(cb, ax_b, ay_b, hx_b, hy_b, axis)
		if a1 < b0 or b1 < a0:
			return False
	return True


def _obb_penetration_depth_xy(ma: dict, mb: dict, pos_a=None, pos_b=None) -> float:
	"""Return minimum positive overlap depth on SAT axes (meters) in XY; 0 if separated."""
	oa = ma["obb"]
	ob = mb["obb"]
	ca = tuple(pos_a[:2]) if pos_a is not None else tuple(oa["center"][:2])
	cb = tuple(pos_b[:2]) if pos_b is not None else tuple(ob["center"][:2])
	ax_a, ay_a = oa["ax"], oa["ay"]
	ax_b, ay_b = ob["ax"], ob["ay"]
	hx_a = float(oa["hx"])
	hy_a = float(oa["hy"])
	hx_b = float(ob["hx"])
	hy_b = float(ob["hy"])

	min_overlap = 1e30
	for axis in [ax_a, ay_a, ax_b, ay_b]:
		axis = _normalize2(axis)
		a0, a1 = _project_obb(ca, ax_a, ay_a, hx_a, hy_a, axis)
		b0, b1 = _project_obb(cb, ax_b, ay_b, hx_b, hy_b, axis)
		overlap = min(a1, b1) - max(a0, b0)
		if overlap <= 0.0:
			return 0.0
		if overlap < min_overlap:
			min_overlap = overlap
	if min_overlap >= 1e29:
		return 0.0
	return float(min_overlap)


def _collision_pairs(
	labels: list[int],
	metas_by_label: dict[int, dict],
	pos_overrides: dict[int, tuple[float, float]] | None = None,
	label_to_oid: dict[int, str] | None = None,
	id_to_entry: dict[str, dict] | None = None,
):
	if pos_overrides is None:
		pos_overrides = {}
	pairs = []
	use_3d = label_to_oid is not None and id_to_entry is not None and len(pos_overrides) == 0
	for i in range(len(labels)):
		for j in range(i + 1, len(labels)):
			a = labels[i]
			b = labels[j]
			ma = metas_by_label[a]
			mb = metas_by_label[b]
			pa = pos_overrides.get(a)
			pb = pos_overrides.get(b)
			if not _aabb_overlap_xy(ma, mb, pos_a=pa, pos_b=pb):
				continue
			if use_3d:
				try:
					ga = id_to_entry[label_to_oid[a]]["group"]
					gb = id_to_entry[label_to_oid[b]]["group"]
				except Exception:
					ga = None
					gb = None
				if ga is not None and gb is not None:
					bvh_a = _build_group_bvh(ga)
					bvh_b = _build_group_bvh(gb)
					if bvh_a is None or bvh_b is None:
						continue
					if _bvh_overlap_strong(bvh_a, bvh_b):
						pairs.append((a, b))
				continue

			if _obb_overlap(ma, mb, pos_a=pa, pos_b=pb):
				pairs.append((a, b))
	return pairs


_MOVE_NORTH_WORLD_XY: tuple[float, float] = (0.0, 1.0)
_MOVE_EAST_WORLD_XY: tuple[float, float] = (1.0, 0.0)


def _set_move_basis_from_north(north_world_dir):
	"""Align move direction tokens (N/E/S/W) with the rendered North indicator."""
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
	# Right-hand 90° rotation from north gives east in XY.
	ex, ey = ny, -nx
	en = math.hypot(ex, ey)
	if en < 1e-9:
		ex, ey = 1.0, 0.0
	else:
		ex, ey = ex / en, ey / en
	_MOVE_NORTH_WORLD_XY = (nx, ny)
	_MOVE_EAST_WORLD_XY = (ex, ey)


def _move_vec(d: str) -> tuple[float, float]:
	nx, ny = _MOVE_NORTH_WORLD_XY
	ex, ey = _MOVE_EAST_WORLD_XY
	if d == "N":
		return (nx, ny)
	if d == "S":
		return (-nx, -ny)
	if d == "E":
		return (ex, ey)
	if d == "W":
		return (-ex, -ey)
	if d == "NE":
		return ((nx + ex) / math.sqrt(2.0), (ny + ey) / math.sqrt(2.0))
	if d == "NW":
		return ((nx - ex) / math.sqrt(2.0), (ny - ey) / math.sqrt(2.0))
	if d == "SE":
		return ((-nx + ex) / math.sqrt(2.0), (-ny + ey) / math.sqrt(2.0))
	if d == "SW":
		return ((-nx - ex) / math.sqrt(2.0), (-ny - ey) / math.sqrt(2.0))
	raise KeyError(f"Unknown move direction token: {d}")


def _dir_word(d: str) -> str:
	return {
		"N": "North", "S": "South", "E": "East", "W": "West",
		"NE": "Northeast", "NW": "Northwest", "SE": "Southeast", "SW": "Southwest",
	}[d]


def _opposite_dir(d: str) -> str:
	return {"N": "S", "S": "N", "E": "W", "W": "E", "NE": "SW", "NW": "SE", "SE": "NW", "SW": "NE"}[d]


def _nearest_neighbor_label(labels: list[int], metas_by_label: dict[int, dict], pivot_label: int) -> int:
	px, py = metas_by_label[pivot_label]["obb"]["center"]
	best = None
	best_d = 1e30
	for i in labels:
		if i == pivot_label:
			continue
		qx, qy = metas_by_label[i]["obb"]["center"]
		d = (px - qx) ** 2 + (py - qy) ** 2
		if d < best_d:
			best_d = d
			best = i
	return int(best) if best is not None else int(random.choice([i for i in labels if i != pivot_label]))


def _select_operable_labels(labels: list[int], metas_by_label: dict[int, dict], max_count: int = MAX_LABELED_OBJECTS) -> list[int]:
	"""Select up to max_count labels with spatial separation for clearer visualization."""
	if len(labels) <= max_count:
		return sorted(labels)

	centers = {lb: metas_by_label[lb]["obb"]["center"] for lb in labels}
	xs = [centers[lb][0] for lb in labels]
	ys = [centers[lb][1] for lb in labels]
	diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
	min_sep = max(0.8, 0.15 * diag)

	pool = labels[:]
	random.shuffle(pool)
	selected = [pool.pop(0)]

	def nearest_dist(lb: int, chosen: list[int]) -> float:
		cx, cy = centers[lb]
		ds = [math.hypot(cx - centers[s][0], cy - centers[s][1]) for s in chosen]
		return min(ds) if ds else 1e9

	while len(selected) < max_count and pool:
		cands = sorted(pool, key=lambda lb: nearest_dist(lb, selected), reverse=True)
		pick = None
		for lb in cands:
			if nearest_dist(lb, selected) >= min_sep:
				pick = lb
				break
		if pick is None:
			pick = cands[0]
		selected.append(pick)
		pool.remove(pick)

	return sorted(selected)


def _select_operable_labels_orientation_friendly(
	labels: list[int],
	metas_by_label: dict[int, dict],
	label_to_oid: dict[int, str],
	id_to_entry: dict[str, dict],
	wall_bounds: dict | None,
	max_count: int = MAX_LABELED_OBJECTS,
) -> list[int]:
	"""For orientation case, prefer one strong near-wall/corner target in labeled subset."""
	if len(labels) <= max_count:
		return sorted(labels)

	def _cat(lb: int) -> str:
		oid = label_to_oid.get(lb)
		return str(id_to_entry.get(oid, {}).get("category", "")).lower() if oid is not None else ""

	def _center(lb: int) -> tuple[float, float]:
		return metas_by_label[lb]["obb"]["center"]

	corner_target_cats = {
		"chair", "stool", "bench", "table", "desk", "cabinet", "shelf", "nightstand",
		"dresser", "wardrobe", "sofa", "couch", "bed", "tv stand", "stand",
	}

	target_label: int | None = None
	if wall_bounds is not None:
		# Wall-corner mode: target must be the nearest-to-wall object first.
		# Tie-breakers use corner/square preference only among nearest-wall candidates.
		cands: list[tuple[float, float, int]] = []  # (nearest_wall_dist, tie_score, label)
		for lb in labels:
			cat = _cat(lb)
			if not any(k in cat for k in corner_target_cats):
				continue
			meta = metas_by_label[lb]
			oid_lb = label_to_oid.get(lb)
			if oid_lb is None:
				continue
			if not _is_orientation_candidate_shape_group(id_to_entry[oid_lb]["group"], cat):
				continue
			hx = float(meta["obb"]["hx"])
			hy = float(meta["obb"]["hy"])
			if min(hx, hy) < 1e-4:
				continue
			ratio = max(hx, hy) / max(min(hx, hy), 1e-6)
			# Prefer square-ish objects near wall corner.
			square_score = max(0.0, 1.25 - ratio)
			cx, cy = meta["obb"]["center"]
			d_l = cx - wall_bounds["min_x"]
			d_r = wall_bounds["max_x"] - cx
			d_b = cy - wall_bounds["min_y"]
			d_t = wall_bounds["max_y"] - cy
			ds = sorted([d_l, d_r, d_b, d_t])
			nearest_wall_dist = float(ds[0])
			corner_score = max(0.0, 1.2 - ds[0]) + max(0.0, 1.2 - ds[1])
			tie_score = 2.0 * corner_score + square_score
			cands.append((nearest_wall_dist, tie_score, lb))
		if cands:
			min_d = min(x[0] for x in cands)
			eps = 1e-6
			nearest = [x for x in cands if abs(x[0] - min_d) <= eps]
			nearest.sort(key=lambda x: x[1], reverse=True)
			target_label = nearest[0][2]

	if target_label is None:
		return _select_operable_labels(labels, metas_by_label, max_count=max_count)

	centers = {lb: metas_by_label[lb]["obb"]["center"] for lb in labels}
	selected = [target_label]
	pool = [lb for lb in labels if lb not in set(selected)]

	def nearest_dist(lb: int, chosen: list[int]) -> float:
		cx, cy = centers[lb]
		ds = [math.hypot(cx - centers[s][0], cy - centers[s][1]) for s in chosen]
		return min(ds) if ds else 1e9

	while len(selected) < max_count and pool:
		# Keep remaining labels separated for readability.
		pick = max(pool, key=lambda lb: nearest_dist(lb, selected))
		selected.append(pick)
		pool.remove(pick)

	return sorted(selected)


def _select_operable_labels_with_forced(
	labels: list[int],
	metas_by_label: dict[int, dict],
	forced_labels: list[int],
	max_count: int = MAX_LABELED_OBJECTS,
) -> list[int]:
	"""Select up to max_count labels while forcing specific labels to be included."""
	all_labels = sorted(labels)
	if len(all_labels) <= max_count:
		return all_labels

	forced_set = {lb for lb in forced_labels if lb in set(all_labels)}
	if len(forced_set) >= max_count:
		return sorted(list(forced_set))[:max_count]

	if not forced_set:
		return _select_operable_labels(all_labels, metas_by_label, max_count=max_count)

	centers = {lb: metas_by_label[lb]["obb"]["center"] for lb in all_labels}
	selected = list(sorted(forced_set))
	pool = [lb for lb in all_labels if lb not in forced_set]

	def nearest_dist(lb: int, chosen: list[int]) -> float:
		cx, cy = centers[lb]
		ds = [math.hypot(cx - centers[s][0], cy - centers[s][1]) for s in chosen]
		return min(ds) if ds else 1e9

	while len(selected) < max_count and pool:
		pick = max(pool, key=lambda lb: nearest_dist(lb, selected))
		selected.append(pick)
		pool.remove(pick)

	return sorted(selected)


def _remap_issue_and_action_labels(
	issue_meta: dict,
	inject_action: dict,
	old_label_to_oid: dict[int, str],
	new_oid_to_label: dict[str, int],
) -> tuple[dict, dict] | tuple[None, None]:
	"""Remap anomaly labels from candidate-label space to display-label space."""
	def _desc_from_issue_item(it: dict) -> str:
		lbs = []
		for x in it.get("object_labels", []):
			sx = str(x).strip()
			if sx.isdigit():
				lbs.append(int(sx))
		main_t = str(it.get("main_type", "")).strip() or _issue_main_type(it)
		subtype = str(it.get("subtype", "")).strip()
		if main_t == ISSUE_OVERLAP:
			if len(lbs) >= 2:
				return f"Object labels [{lbs[0]}, {lbs[1]}] have a physically implausible overlap."
			if len(lbs) == 1:
				return f"Object label [{lbs[0]}] has a physically implausible overlap."
		if main_t in {ISSUE_WALL_DOOR, ISSUE_WALL, ISSUE_DOOR, ISSUE_PATH}:
			if len(lbs) >= 1:
				if subtype:
					return f"Object label [{lbs[0]}] causes {subtype.replace('_', ' ')}."
				return f"Object label [{lbs[0]}] conflicts with wall."
		if main_t == ISSUE_ANGLE and len(lbs) >= 1:
			return f"Object label [{lbs[0]}] has an abnormal orientation."
		return str(it.get("description", ""))

	try:
		issue_new = json.loads(json.dumps(issue_meta, ensure_ascii=False))
	except Exception:
		issue_new = {"issues": []}

	for it in issue_new.get("issues", []):
		for key in ("object_labels", "building_labels"):
			raw = it.get(key, [])
			new_labels: list[str] = []
			for x in raw:
				sx = str(x).strip()
				if not sx.isdigit():
					continue
				old_lb = int(sx)
				oid = old_label_to_oid.get(old_lb)
				if oid is None:
					continue
				new_lb = new_oid_to_label.get(oid)
				if new_lb is None:
					continue
				new_labels.append(str(int(new_lb)))
			if key == "object_labels":
				it[key] = new_labels
			elif key in it:
				it.pop(key, None)
		it["description"] = _desc_from_issue_item(it)

	try:
		old_act_lb = int(inject_action["id"])
	except Exception:
		return None, None
	oid = old_label_to_oid.get(old_act_lb)
	if oid is None or oid not in new_oid_to_label:
		return None, None
	action_new = dict(inject_action)
	action_new["id"] = int(new_oid_to_label[oid])
	return issue_new, action_new


def set_render_and_world():
	dc_utils.set_render_and_world(SAMPLES, RESOLUTION_X, RESOLUTION_Y, transparent_bg=True)
	# In dense indoor scenes with layered wall shells, low transparent-bounce budgets
	# make semi-transparent walls look dark/opaque. Raise these limits for stable see-through.
	try:
		cyc = bpy.context.scene.cycles
		cyc.transparent_max_bounces = max(int(getattr(cyc, "transparent_max_bounces", 0)), 128)
		cyc.transparent_min_bounces = max(int(getattr(cyc, "transparent_min_bounces", 0)), 32)
		cyc.max_bounces = max(int(getattr(cyc, "max_bounces", 0)), 128)
	except Exception:
		pass


def setup_lighting(scene_center: Vector, scene_radius: float):
	for obj in list(bpy.data.objects):
		if obj.type == "LIGHT":
			bpy.data.objects.remove(obj, do_unlink=True)
	sun = bpy.data.objects.new("Sun", bpy.data.lights.new("SunData", type="SUN"))
	bpy.context.collection.objects.link(sun)
	sun.location = scene_center + Vector((scene_radius * 0.6, -scene_radius * 0.5, scene_radius * 3.2))
	sun.rotation_euler = (math.radians(65), 0.0, math.radians(35))
	sun.data.energy = 4.2
	if hasattr(sun.data, "angle"):
		sun.data.angle = math.radians(4.5)
	if hasattr(sun.data, "use_shadow"):
		sun.data.use_shadow = True
	fill = bpy.data.objects.new("Fill", bpy.data.lights.new("FillData", type="AREA"))
	bpy.context.collection.objects.link(fill)
	fill.location = scene_center + Vector((0.0, 0.0, scene_radius * 2.2))
	fill.rotation_euler = (math.radians(90), 0.0, 0.0)
	fill.data.energy = 1400.0
	fill.data.size = max(scene_radius * 1.6, 3.5)
	if hasattr(fill.data, "use_shadow"):
		fill.data.use_shadow = False
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
		if hasattr(data, "use_shadow"):
			data.use_shadow = False

	# Keep object shadows but suppress structure/wall-like shadows to avoid heavy occlusion.
	structure_shadow_tokens = {
		"wall", "floor", "ceiling", "window", "door", "frame",
		"curtain", "blinds", "geometry_",
	}
	for obj in bpy.context.scene.objects:
		if obj.type != "MESH":
			continue
		n = str(obj.name).lower()
		is_structure_like = any(t in n for t in structure_shadow_tokens)
		if hasattr(obj, "visible_shadow"):
			obj.visible_shadow = (not is_structure_like)
		if hasattr(obj, "cycles_visibility"):
			try:
				obj.cycles_visibility.shadow = (not is_structure_like)
			except Exception:
				pass


def _resolve_wall_targets(meshes: list, wall_bounds: dict | None) -> list:
	proxy_targets = [o for o in meshes if bool(getattr(o, "get", lambda *_: False)("_dc_wall_proxy", False))]
	if proxy_targets:
		return proxy_targets

	def _is_scene_instance_mesh(obj_name: str) -> bool:
		# Layout instances are usually "<idx>_..."; structure meshes typically are not.
		return re.match(r"^\d+_", str(obj_name)) is not None

	def _wall_like_name(obj_name: str) -> bool:
		n = str(obj_name).lower()
		if ("floor" in n) or ("ceiling" in n):
			return False
		# Avoid decorative frames that are not room structures.
		if any(t in n for t in ("picture frame", "photo frame", "poster frame", "art frame")):
			return False
		if "wall" in n:
			return True
		if not WALL_INCLUDE_OPENING_FRAMES:
			return False
		return (
			("doorframe" in n)
			or ("door frame" in n)
			or ("windowframe" in n)
			or ("window frame" in n)
		)

	named_targets = [o for o in meshes if _wall_like_name(o.name)]
	if wall_bounds is None:
		return named_targets

	# Geometry fallback: match meshes by overlap with external wall bounds, independent of name.
	wminx, wmaxx = float(wall_bounds["min_x"]), float(wall_bounds["max_x"])
	wminy, wmaxy = float(wall_bounds["min_y"]), float(wall_bounds["max_y"])
	wminz, wmaxz = float(wall_bounds["min_z"]), float(wall_bounds["max_z"])
	w_area = max(1e-6, (wmaxx - wminx) * (wmaxy - wminy))
	w_h = max(1e-6, (wmaxz - wminz))
	best_obj = None
	best_score = -1.0
	scored: list[tuple[float, object, dict]] = []
	wcx = 0.5 * (wminx + wmaxx)
	wcy = 0.5 * (wminy + wmaxy)
	wdiag = max(1e-6, math.hypot(wmaxx - wminx, wmaxy - wminy))
	for obj in meshes:
		n = str(obj.name).lower()
		# Skip likely furniture/instances and obvious non-wall structures.
		if _is_scene_instance_mesh(obj.name):
			continue
		if any(tok in n for tok in ("floor", "ceiling", "ground", "rug", "carpet")):
			continue
		b = _group_bounds([obj])
		ominx, omaxx = float(b["min_x"]), float(b["max_x"])
		ominy, omaxy = float(b["min_y"]), float(b["max_y"])
		ominz, omaxz = float(b["min_z"]), float(b["max_z"])
		ow = max(1e-6, omaxx - ominx)
		od = max(1e-6, omaxy - ominy)
		oh = max(1e-6, omaxz - ominz)
	
		if oh < 0.30:
			continue
		if oh / max(ow, od) < 0.06:
			continue
		ix = max(0.0, min(wmaxx, omaxx) - max(wminx, ominx))
		iy = max(0.0, min(wmaxy, omaxy) - max(wminy, ominy))
		iz = max(0.0, min(wmaxz, omaxz) - max(wminz, ominz))
		inter = ix * iy
		o_area = max(1e-6, (omaxx - ominx) * (omaxy - ominy))
		# Prefer wall-footprint overlap + vertical overlap + sizeable wall-like extent.
		xy_cover_wall = inter / w_area
		xy_cover_obj = inter / o_area
		z_overlap = iz / w_h
		height_ratio = min(1.0, oh / w_h)
		ocx = 0.5 * (ominx + omaxx)
		ocy = 0.5 * (ominy + omaxy)
		center_dist = math.hypot(ocx - wcx, ocy - wcy)
		proximity = max(0.0, 1.0 - center_dist / (1.8 * wdiag))
		size_ratio = min((ow * od) / w_area, w_area / max(1e-6, ow * od))
		score = (
			0.38 * xy_cover_wall
			+ 0.22 * xy_cover_obj
			+ 0.18 * z_overlap
			+ 0.10 * height_ratio
			+ 0.08 * proximity
			+ 0.04 * size_ratio
		)
		if score > best_score:
			best_score = score
			best_obj = obj
		scored.append((score, obj, {"xyw": xy_cover_wall, "xyo": xy_cover_obj, "z": z_overlap, "prox": proximity}))
	bounds_targets: list = []
	if best_obj is not None and best_score > 0.03:
		# Keep top candidates close to best score to support split wall geometry.
		scored.sort(key=lambda x: x[0], reverse=True)
		cands = [o for s, o, _ in scored if s >= max(0.03, best_score * 0.55)]
		try:
			debug_top = ", ".join(f"{o.name}:{s:.3f}" for s, o, _ in scored[:8])
			print(f"[WALL] resolve by bounds best={best_score:.3f}, top={debug_top}")
		except Exception:
			pass
		bounds_targets = cands if cands else [best_obj]

	elif scored:
		scored.sort(key=lambda x: x[0], reverse=True)
		keep_n = min(4, len(scored))
		bounds_targets = [o for _, o, _ in scored[:keep_n]]
		try:
			debug_top = ", ".join(f"{o.name}:{s:.3f}" for s, o, _ in scored[:8])
			print(f"[WALL] resolve fallback top={debug_top}")
		except Exception:
			pass

	# Use union: explicit name match + bounds match.
	# Some scenes have both named structural frames and unnamed geometry_* wall surfaces.
	union_by_name: dict[str, object] = {}
	for o in named_targets:
		union_by_name[o.name] = o
	for o in bounds_targets:
		union_by_name[o.name] = o
	return list(union_by_name.values())


def _cap_wall_height(wall_bounds: dict | None, top_margin: float = WALL_TOP_MARGIN):
	"""For rendering, cap all walls to one unified absolute top height."""
	meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
	targets = _resolve_wall_targets(meshes, wall_bounds)
	if not targets:
		return
	target_names = {o.name for o in targets}
	non_wall = [o for o in meshes if o.name not in target_names]
	if not non_wall:
		return

	obj_tops = sorted(float(_group_bounds([o])["max_z"]) for o in non_wall)
	if not obj_tops:
		return
	q = max(0.0, min(1.0, float(WALL_TOP_OBJECT_PERCENTILE)))
	q_idx = int(round((len(obj_tops) - 1) * q))
	robust_top = obj_tops[q_idx]
	wall_min = min(float(_group_bounds([o])["min_z"]) for o in targets)
	wall_max = max(float(_group_bounds([o])["max_z"]) for o in targets)
	desired_top = robust_top + float(top_margin)
	desired_top = min(desired_top, wall_min + float(WALL_TOP_MAX_ABOVE_FLOOR))
	desired_top = max(desired_top, wall_min + 0.6)
	if wall_max <= desired_top + 1e-4:
		return
	for obj in targets:
		ob = _group_bounds([obj])
		ob_min = float(ob["min_z"])
		ob_max = float(ob["max_z"])
		den = max(1e-6, ob_max - ob_min)
		# Per-wall scaling so every wall reaches the same absolute top.
		f = (desired_top - ob_min) / den
		f = max(0.05, min(1.0, float(f)))
		T1 = Matrix.Translation(Vector((0.0, 0.0, ob_min)))
		T2 = Matrix.Translation(Vector((0.0, 0.0, -ob_min)))
		S = Matrix((
			(1.0, 0.0, 0.0, 0.0),
			(0.0, 1.0, 0.0, 0.0),
			(0.0, 0.0, f,   0.0),
			(0.0, 0.0, 0.0, 1.0),
		))
		obj.matrix_world = (T1 @ S @ T2) @ obj.matrix_world
	bpy.context.view_layer.update()


def _set_structure_transparency(alpha: float, wall_bounds: dict | None = None):
	"""Make wall meshes semi-transparent to reduce occlusion in indoor views."""
	alpha = max(0.0, min(1.0, float(alpha)))
	meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
	targets = _resolve_wall_targets(meshes, wall_bounds)
	if not targets:
		print(f"[WALL] transparency alpha={alpha:.2f}, targets=0")
		return
	target_preview = ", ".join(o.name for o in targets[:6])
	if len(targets) > 6:
		target_preview += ", ..."
	print(f"[WALL] transparency alpha={alpha:.2f}, targets={len(targets)}: {target_preview}")

	def _ensure_cycles_opacity(mat, opacity: float):
		"""Force wall opacity in Cycles by mixing original surface with Transparent BSDF."""
		if mat is None:
			return
		try:
			mat.use_nodes = True
		except Exception:
			return
		nt = getattr(mat, "node_tree", None)
		if nt is None:
			return
		nodes = nt.nodes
		links = nt.links
		out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and getattr(n, "is_active_output", False)), None)
		if out is None:
			out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
		if out is None or "Surface" not in out.inputs:
			return
		surface_in = out.inputs["Surface"]
		volume_in = out.inputs.get("Volume")

		mix_node = nodes.get("_DC_WALL_ALPHA_MIX")
		if mix_node is None:
			mix_node = nodes.new("ShaderNodeMixShader")
			mix_node.name = "_DC_WALL_ALPHA_MIX"
			mix_node.label = "_DC_WALL_ALPHA_MIX"
			mix_node.location = (out.location.x - 220.0, out.location.y)

		transp_node = nodes.get("_DC_WALL_ALPHA_TRANSPARENT")
		if transp_node is None:
			transp_node = nodes.new("ShaderNodeBsdfTransparent")
			transp_node.name = "_DC_WALL_ALPHA_TRANSPARENT"
			transp_node.label = "_DC_WALL_ALPHA_TRANSPARENT"
			transp_node.location = (mix_node.location.x - 220.0, mix_node.location.y - 120.0)
		try:
			# Keep transparent lobe color neutral to avoid gray tint accumulation.
			transp_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
		except Exception:
			pass

		opaque_node = nodes.get("_DC_WALL_ALPHA_OPAQUE")
		if opaque_node is None:
			opaque_node = nodes.new("ShaderNodeEmission")
			opaque_node.name = "_DC_WALL_ALPHA_OPAQUE"
			opaque_node.label = "_DC_WALL_ALPHA_OPAQUE"
			opaque_node.location = (mix_node.location.x - 220.0, mix_node.location.y + 120.0)
		try:
			if "Color" in opaque_node.inputs:
				opaque_node.inputs["Color"].default_value = (0.90, 0.90, 0.90, 1.0)
			if "Strength" in opaque_node.inputs:
				opaque_node.inputs["Strength"].default_value = 1.0
		except Exception:
			pass

		try:
			# Mix factor controls transparency amount.
			mix_node.inputs["Fac"].default_value = max(0.0, min(1.0, 1.0 - float(opacity)))
		except Exception:
			pass

		try:
			for lk in list(mix_node.inputs[1].links):
				links.remove(lk)
			links.new(opaque_node.outputs["Emission"], mix_node.inputs[1])
		except Exception:
			pass

		# Transparent shader is shader 2.
		try:
			for lk in list(mix_node.inputs[2].links):
				links.remove(lk)
			links.new(transp_node.outputs["BSDF"], mix_node.inputs[2])
		except Exception:
			pass

		# Ensure Material Output Surface always receives mixed shader output.
		# Same alpha semantics for top/isometric: fac = 1 - opacity.
		try:
			for lk in list(surface_in.links):
				links.remove(lk)
			links.new(mix_node.outputs["Shader"], surface_in)
		except Exception:
			pass

		# Volume can keep object visually opaque in Cycles; disable for wall transparency.
		if volume_in is not None:
			try:
				for lk in list(volume_in.links):
					links.remove(lk)
			except Exception:
				pass

	def _ensure_object_material_slot(obj):
		if len(getattr(obj, "material_slots", [])) > 0:
			return
		mat_name = "_DC_WALL_ALPHA_AUTOGEN"
		mat = bpy.data.materials.get(mat_name)
		if mat is None:
			mat = bpy.data.materials.new(mat_name)
			mat.use_nodes = True
		try:
			obj.data.materials.append(mat)
		except Exception:
			pass

	# Apply to object level
	for obj in targets:
		# Keep pure alpha semantics (0 transparent, 1 opaque), no threshold-based hiding.
		try:
			obj.hide_render = False
		except Exception:
			pass

		# Set object-level transparency
		try:
			if hasattr(obj, 'display_type'):
				obj.display_type = 'TRANSPARENT'
			if hasattr(obj, 'show_transparent'):
				obj.show_transparent = True
		except Exception:
			pass
		_ensure_object_material_slot(obj)

		# Also try material-level transparency
		for slot in getattr(obj, "material_slots", []):
			mat = slot.material
			if mat is None:
				continue

			# Copy material if shared with other objects
			try:
				if getattr(mat, "users", 0) > 1:
					mat = mat.copy()
					slot.material = mat
			except Exception:
				pass

			# Set material transparency mode
			try:
				mat.blend_method = "BLEND"
				mat.shadow_method = "NONE"
				mat.use_screen_refraction = False
				mat.use_backface_culling = False
			except Exception:
				pass
			# In dense indoor meshes, multiple wall layers accumulate opacity quickly.
			# Apply a monotonic gamma response while preserving alpha endpoints.
			effective_opacity = max(0.0, min(1.0, float(alpha) ** float(WALL_ALPHA_RESPONSE_GAMMA)))
			_ensure_cycles_opacity(mat, effective_opacity)


def create_preview_camera(name: str):
	cam_data = bpy.data.cameras.new(name)
	cam_obj = bpy.data.objects.new(name, cam_data)
	bpy.context.collection.objects.link(cam_obj)
	return cam_obj, cam_data


def setup_camera_for_mode(cam_obj, cam_data, render_ctx: dict, mode: str):
	dc_utils.setup_camera_for_mode(
		cam_obj,
		cam_data,
		render_ctx,
		mode,
		top_dist_scale=CAMERA_TOP_DIST_SCALE,
		iso_dist_scale=CAMERA_ISO_DIST_SCALE,
		fit_margin=CAMERA_FIT_MARGIN,
		fit_safety=CAMERA_SAFETY_SCALE,
	)


def render_view(cam, output_path: str):
	dc_utils.render_view(cam, output_path)


def _load_font(size: int):
	if not PIL_AVAILABLE:
		return None
	if "bpy" in sys.modules:
		return ImageFont.load_default()
	try:
		return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
	except Exception:
		return ImageFont.load_default()


def _project_group_centers(id_to_entry: dict[str, dict], object_ids: list[str], cam, width: int, height: int) -> dict[str, tuple[int, int]]:
	return dc_utils.project_group_centers(id_to_entry, object_ids, cam, width, height)


def _projected_pixels_per_meter(
	cam_obj,
	width: int,
	height: int,
	render_ctx: dict | None,
	north_world_dir=None,
) -> float:
	"""
	Compute 1m pixel length from actual scene projection:
	project two world points 1m apart around scene center.
	"""
	if render_ctx is None:
		return dc_utils.scale_marker_px_per_unit(cam_obj, width)
	try:
		center = render_ctx.get("center", None)
		if center is None:
			return dc_utils.scale_marker_px_per_unit(cam_obj, width)
		nx = float(getattr(north_world_dir, "x", 0.0)) if north_world_dir is not None else 0.0
		ny = float(getattr(north_world_dir, "y", 1.0)) if north_world_dir is not None else 1.0
		n = math.hypot(nx, ny)
		if n < 1e-9:
			nx, ny = 0.0, 1.0
		else:
			nx, ny = nx / n, ny / n
		# East is 90° rotation of north in XY.
		ex, ey = ny, -nx
		p0 = Vector((float(center.x), float(center.y), float(center.z)))
		p1 = Vector((float(center.x + ex), float(center.y + ey), float(center.z)))
		scene = bpy.context.scene
		v0 = world_to_camera_view(scene, cam_obj, p0)
		v1 = world_to_camera_view(scene, cam_obj, p1)
		if v0.z <= 1e-6 or v1.z <= 1e-6:
			return dc_utils.scale_marker_px_per_unit(cam_obj, width)
		dx = (float(v1.x) - float(v0.x)) * float(width)
		dy = (float(v1.y) - float(v0.y)) * float(height)
		ppm = math.hypot(dx, dy)
		if ppm > 1e-6 and math.isfinite(ppm):
			return float(ppm)
	except Exception:
		pass
	return dc_utils.scale_marker_px_per_unit(cam_obj, width)


def _compose_with_left_strip_scaled(
	img,
	north_vec: tuple[float, float],
	px_per_unit: float,
	ui_scale_mult: float = 1.0,
	model_pad: int = 6,
	gap: int = 6,
):
	"""Same compact left-strip layout, with controllable UI scale multiplier."""
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

	base_ui = max(0.58, min(1.0, ch / 460.0))
	ui_scale = max(0.58, min(1.8, base_ui * float(ui_scale_mult)))
	# Keep 1m physical length truthful: 1m == px_per_unit.
	bar_px = int(round(max(10.0, (px_per_unit if px_per_unit > 0 else 70.0))))
	# Make strip as tight as possible:
	# only a little wider than needed for 1m bar, while still fitting north panel.
	min_for_scale_panel = bar_px + int(round(40 * ui_scale))
	min_for_north_panel = int(round(130 * ui_scale))
	strip_w = max(min_for_scale_panel, min_for_north_panel) + 2
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


def _annotate_numbers(
	base_image: Path,
	out_image: Path,
	label_map: dict[str, int],
	centers: dict[str, tuple[int, int]],
	cam_obj,
	render_ctx: dict | None = None,
	render_mode: str = "top",
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
	px_per_unit = _projected_pixels_per_meter(
		cam_obj,
		img.width,
		img.height,
		render_ctx=render_ctx,
		north_world_dir=north_world_dir,
	)
	north_vec = dc_utils.north_screen_vector(cam_obj, north_world_dir=north_world_dir)
	font = _load_font(LABEL_FONT_SIZE)
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
	
	strip_scale = max(14.0, float(px_per_unit))
	img = dc_utils.compose_compact_with_left_strip(img, north_vec, strip_scale)
	img.save(out_image)
	return str(out_image.resolve())


def _capture_labeled(
	cam_obj,
	render_ctx: dict,
	render_mode: str,
	id_to_entry: dict[str, dict],
	label_map: dict[str, int],
	raw_path: Path,
	out_path: Path,
	wall_bounds: dict | None = None,
	wall_alpha: float = 1.0,
	north_world_dir=None,
) -> str:
	def _render_with_wall_alpha(cam, out_path_local: Path, alpha: float):
		meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
		targets = _resolve_wall_targets(meshes, wall_bounds)
		alpha = max(0.0, min(1.0, float(alpha)))
		if not targets:
			render_view(cam, str(out_path_local))
			return

		orig_hide = {o.name: bool(getattr(o, "hide_render", False)) for o in targets}

		def _set_targets_hidden(hidden: bool):
			for obj in targets:
				try:
					obj.hide_render = bool(hidden)
				except Exception:
					pass

		if alpha >= 1.0 - 1e-6:
			try:
				_set_targets_hidden(False)
				render_view(cam, str(out_path_local))
			finally:
				for obj in targets:
					try:
						obj.hide_render = orig_hide.get(obj.name, False)
					except Exception:
						pass
			return
		if alpha <= 1e-6:
			try:
				_set_targets_hidden(True)
				render_view(cam, str(out_path_local))
			finally:
				for obj in targets:
					try:
						obj.hide_render = orig_hide.get(obj.name, False)
					except Exception:
						pass
			return

		tmp_on = out_path_local.with_name(f".{out_path_local.stem}_wall_on.png")
		tmp_off = out_path_local.with_name(f".{out_path_local.stem}_wall_off.png")
		try:
			_set_targets_hidden(False)
			render_view(cam, str(tmp_on))
			_set_targets_hidden(True)
			render_view(cam, str(tmp_off))

			if PIL_AVAILABLE and tmp_on.exists() and tmp_off.exists():
				img_on = Image.open(tmp_on).convert("RGBA")
				img_off = Image.open(tmp_off).convert("RGBA")
				if img_on.size != img_off.size:
					img_off = img_off.resize(img_on.size, Image.BILINEAR)
				# alpha semantics unified: 0 -> no-wall pass, 1 -> wall pass.
				Image.blend(img_off, img_on, alpha=alpha).save(out_path_local)
			else:
				shutil.copy2(tmp_on if alpha >= 0.5 else tmp_off, out_path_local)
		finally:
			for obj in targets:
				try:
					obj.hide_render = orig_hide.get(obj.name, False)
				except Exception:
					pass
			for p in [tmp_on, tmp_off]:
				try:
					if p.exists():
						p.unlink()
				except Exception:
					pass

	dyn_w, dyn_h = dc_utils.dynamic_resolution_for_mode(render_ctx, render_mode, BASE_LONG_EDGE, MIN_SHORT_EDGE)
	bpy.context.scene.render.resolution_x = dyn_w
	bpy.context.scene.render.resolution_y = dyn_h
	setup_camera_for_mode(cam_obj, cam_obj.data, render_ctx, render_mode)
	_render_with_wall_alpha(cam_obj, raw_path, wall_alpha)
	centers = _project_group_centers(id_to_entry, list(label_map.keys()), cam_obj, dyn_w, dyn_h)
	return _annotate_numbers(
		raw_path,
		out_path,
		label_map,
		centers,
		cam_obj,
		render_ctx=render_ctx,
		render_mode=render_mode,
		north_world_dir=north_world_dir,
	)


def _translate_group(group_objs: list, dx: float, dy: float) -> dict:
	orig = {obj.name: obj.matrix_world.copy() for obj in group_objs}
	targets = _group_transform_targets(group_objs)
	t = Matrix.Translation(Vector((dx, dy, 0.0)))
	for obj in targets:
		obj.matrix_world = t @ obj.matrix_world
	bpy.context.view_layer.update()
	return orig


def _group_transform_targets(group_objs: list) -> list:
	"""Apply rigid transform on roots only to avoid double-transform in parent-child mesh hierarchies."""
	if not group_objs:
		return []
	in_group = set(group_objs)
	targets = []
	for obj in group_objs:
		p = obj.parent
		has_ancestor_in_group = False
		while p is not None:
			if p in in_group:
				has_ancestor_in_group = True
				break
			p = p.parent
		if not has_ancestor_in_group:
			targets.append(obj)
	return targets if targets else list(group_objs)


def _group_centroid_world(group_objs: list) -> Vector:
	"""Use group OBB center in XY (with group Z center) as rotate/scale pivot."""
	pts_xy: list[tuple[float, float]] = []
	for obj in group_objs:
		if obj.type != "MESH" or obj.data is None:
			continue
		verts = obj.data.vertices
		if len(verts) == 0:
			continue
		step = max(1, len(verts) // 600)
		for i in range(0, len(verts), step):
			w = obj.matrix_world @ verts[i].co
			pts_xy.append((float(w.x), float(w.y)))
	b = _group_bounds(group_objs)
	obb = _compute_pca_obb_xy(pts_xy) if pts_xy else None
	if obb is not None:
		cx, cy = obb["center"]
		return Vector((float(cx), float(cy), float(b["center_z"])))
	return Vector((b["center_x"], b["center_y"], b["center_z"]))


def _rotate_group(group_objs: list, deg: float, clockwise: bool, pivot: Vector | None = None) -> dict:
	orig = {obj.name: obj.matrix_world.copy() for obj in group_objs}
	targets = _group_transform_targets(group_objs)
	sign = -1.0 if clockwise else 1.0
	c_geo = pivot if pivot is not None else _group_centroid_world(group_objs)
	# Rotate around object's center vertical axis (multi-mesh stable pivot).
	c = Vector((float(c_geo.x), float(c_geo.y), float(c_geo.z)))
	T1 = Matrix.Translation(c)
	T2 = Matrix.Translation(-c)
	R = Matrix.Rotation(math.radians(sign * deg), 4, "Z")
	M = T1 @ R @ T2
	for obj in targets:
		obj.matrix_world = M @ obj.matrix_world
	bpy.context.view_layer.update()
	return orig


def _scale_group(group_objs: list, percent: float, scale_up: bool, pivot: Vector | None = None) -> dict:
	orig = {obj.name: obj.matrix_world.copy() for obj in group_objs}
	targets = _group_transform_targets(group_objs)
	f = 1.0 + (percent / 100.0) if scale_up else 1.0 / (1.0 + (percent / 100.0))
	c = pivot if pivot is not None else _group_centroid_world(group_objs)
	T1 = Matrix.Translation(c)
	T2 = Matrix.Translation(-c)
	S = Matrix.Diagonal((f, f, f, 1.0))  # Scale X, Y, Z uniformly
	M = T1 @ S @ T2
	for obj in targets:
		obj.matrix_world = M @ obj.matrix_world
	bpy.context.view_layer.update()
	return orig


def _restore_group(group_objs: list, orig: dict):
	for obj in group_objs:
		if obj.name in orig:
			obj.matrix_world = orig[obj.name]
	bpy.context.view_layer.update()


def _apply_action(
	action: dict,
	label_to_oid: dict[int, str],
	id_to_entry: dict[str, dict],
	pivot_by_label: dict[int, Vector] | None = None,
) -> tuple[list, dict]:
	label = int(action["id"])
	oid = label_to_oid[label]
	group = id_to_entry[oid]["group"]
	pivot = None if pivot_by_label is None else pivot_by_label.get(label)
	if action["op"] == "move":
		dx, dy = _move_vec(action["dir"])
		orig = _translate_group(group, dx * action["dist"] * UNIT_SCALE, dy * action["dist"] * UNIT_SCALE)
		return group, orig
	if action["op"] == "rotate":
		orig = _rotate_group(group, float(action["deg"]), bool(action["clockwise"]), pivot=pivot)
		return group, orig
	if action["op"] == "scale":
		orig = _scale_group(group, float(action["percent"]), bool(action["up"]), pivot=pivot)
		return group, orig
	raise ValueError(f"Unsupported action: {action}")


def _action_text(action: dict) -> str:
	def _fmt_m(v: float) -> str:
		return f"{float(v):.1f}"
	bid = int(action["id"])
	if action["op"] == "move":
		return f"Move object {bid} {_dir_word(action['dir'])} by {_fmt_m(action['dist'])}m"
	if action["op"] == "rotate":
		dir_word = "clockwise" if action["clockwise"] else "counter-clockwise"
		return f"Rotate object {bid} {dir_word} by {int(action['deg'])}° around its center"
	if action["op"] == "scale":
		return f"Scale {'up' if action['up'] else 'down'} object {bid} by {int(action['percent'])}%"
	return "Do nothing"


def _reverse_action(action: dict) -> dict:
	if action["op"] == "move":
		return {"op": "move", "id": action["id"], "dir": _opposite_dir(action["dir"]), "dist": action["dist"]}
	if action["op"] == "rotate":
		return {"op": "rotate", "id": action["id"], "clockwise": (not action["clockwise"]), "deg": action["deg"]}
	if action["op"] == "scale":
		return {"op": "scale", "id": action["id"], "up": (not action["up"]), "percent": action["percent"]}
	raise ValueError(f"Unsupported action: {action}")


def _build_issue_meta(
	issue_type: str,
	labels: list[int],
	description: str,
	subtype: str | None = None,
	action_kind: str | None = None,
) -> dict:
	def _desc_from_labels() -> str:
		lbs = [int(x) for x in labels]
		if issue_type == ISSUE_OVERLAP:
			if len(lbs) >= 2:
				return f"Object labels [{lbs[0]}, {lbs[1]}] have a physically implausible overlap."
			if len(lbs) == 1:
				return f"Object label [{lbs[0]}] has a physically implausible overlap."
		if issue_type in {ISSUE_WALL_DOOR, ISSUE_WALL, ISSUE_DOOR, ISSUE_PATH}:
			if len(lbs) >= 1:
				if subtype:
					return f"Object label [{lbs[0]}] causes {str(subtype).replace('_', ' ')}."
				return f"Object label [{lbs[0]}] conflicts with wall."
		if issue_type == ISSUE_ANGLE and len(lbs) >= 1:
			return f"Object label [{lbs[0]}] has an abnormal orientation."
		return description

	type_text = f"{issue_type} by {action_kind}" if action_kind else issue_type
	issue = {
		"type": type_text,
		"main_type": issue_type,
		"object_labels": [str(x) for x in labels],
		"description": _desc_from_labels(),
		"severity": "medium",
	}
	if subtype:
		issue["subtype"] = subtype
	return {
		"issues": [
			issue
		]
	}


def _anomaly_mapping():
	return {
		ISSUE_OVERLAP: "A",
		ISSUE_WALL_DOOR: "B",
		ISSUE_WALL: "B",
		ISSUE_DOOR: "B",
		ISSUE_PATH: "B",
		ISSUE_ANGLE: "C",
	}


def _fmt_obj_desc(label_id: int | str, label_category_map: dict[int, str] | None) -> str:
	try:
		lb = int(label_id)
	except Exception:
		return f"object {label_id}"
	if not label_category_map:
		return f"object {lb}"
	cat = str(label_category_map.get(lb, "")).strip()
	return f"object {lb} ({cat})" if cat else f"object {lb}"


def _label_legend_line(label_category_map: dict[int, str] | None) -> str:
	if not label_category_map:
		return ""
	parts = []
	for lb in sorted(label_category_map.keys()):
		parts.append(_fmt_obj_desc(lb, label_category_map))
	return "Labeled objects: " + ", ".join(parts)


def _focus_object_line(issue_meta: dict, label_category_map: dict[int, str] | None) -> str:
	issue = (issue_meta.get("issues") or [{}])[0]
	labels: list[int] = []
	for x in issue.get("object_labels", []):
		sx = str(x).strip()
		if sx.isdigit():
			labels.append(int(sx))
	if not labels:
		return ""
	t = _issue_main_type(issue)
	if t == ISSUE_OVERLAP and len(labels) >= 2:
		return (
			"Involved objects: "
			+ _fmt_obj_desc(labels[0], label_category_map)
			+ " and "
			+ _fmt_obj_desc(labels[1], label_category_map)
		)
	if t in {ISSUE_WALL_DOOR, ISSUE_WALL, ISSUE_DOOR, ISSUE_PATH, ISSUE_ANGLE}:
		return "Involved object: " + _fmt_obj_desc(labels[0], label_category_map)
	return "Involved objects: " + ", ".join(_fmt_obj_desc(lb, label_category_map) for lb in labels)


def _description_with_semantic_objects(description: str, label_category_map: dict[int, str] | None) -> str:
	text = str(description or "").strip()
	if not text:
		return ""
	text = re.sub(
		r"Object labels \[(\d+)\], \[(\d+)\]",
		lambda m: f"{_fmt_obj_desc(m.group(1), label_category_map)} and {_fmt_obj_desc(m.group(2), label_category_map)}",
		text,
	)
	text = re.sub(
		r"Object label \[(\d+)\]",
		lambda m: _fmt_obj_desc(m.group(1), label_category_map),
		text,
	)
	return text


def _qa3_problem_summary(issue_meta: dict, label_category_map: dict[int, str] | None) -> str:
	issue = (issue_meta.get("issues") or [{}])[0]
	t = _issue_main_type(issue)
	labels: list[int] = []
	for x in issue.get("object_labels", []):
		sx = str(x).strip()
		if sx.isdigit():
			labels.append(int(sx))
	if t == ISSUE_OVERLAP and len(labels) >= 2:
		a = _fmt_obj_desc(labels[0], label_category_map)
		b = _fmt_obj_desc(labels[1], label_category_map)
		return f"A problem is detected in the image: {a} and {b} have a physically implausible overlap."
	if t in {ISSUE_WALL_DOOR, ISSUE_WALL, ISSUE_DOOR, ISSUE_PATH} and labels:
		a = _fmt_obj_desc(labels[0], label_category_map)
		return f"A problem is detected in the image: {a} conflicts with wall."
	if t == ISSUE_ANGLE and labels:
		a = _fmt_obj_desc(labels[0], label_category_map)
		return f"A problem is detected in the image: {a} has an abnormal orientation."
	desc_line = _description_with_semantic_objects(str(issue.get("description", "")), label_category_map)
	if desc_line:
		return f"A problem is detected in the image: {desc_line}"
	return "A problem is detected in the image."


def qa1_mcq_what_problem(issue_meta: dict, images: list[str], label_category_map: dict[int, str] | None = None) -> dict:
	t = _issue_main_type(issue_meta["issues"][0]) if issue_meta.get("issues") else None
	mapping = _anomaly_mapping()
	choices = [
		"A. One object has a physically implausible overlap/interpenetration with another object",
		"B. One object conflicts with wall",
		"C. One furniture object has an abnormal orientation",
	]
	q = (
		"You are viewing a top-down view of a 3D scene containing multiple labeled objects.\n"
		"If there is a problem, it involves the labeled objects shown in the image.\n"
		"Question: Examine the scene carefully and identify what problem exists. Choose one option.\n"
		"Top-view image: <image>"
	)
	legend = _label_legend_line(label_category_map)
	if legend:
		q = q + "\n" + legend
	return {"question": q + "\n" + "\n".join(choices), "answer": mapping.get(t, "A"), "task_type": "top_error_identify", "meta": issue_meta, "images": images}


def qa2_mcq_what_problem(issue_meta: dict, images: list[str], label_category_map: dict[int, str] | None = None) -> dict:
	t = _issue_main_type(issue_meta["issues"][0]) if issue_meta.get("issues") else None
	mapping = _anomaly_mapping()
	choices = [
		"A. One object has a physically implausible overlap/interpenetration with another object",
		"B. One object conflicts with wall",
		"C. One furniture object has an abnormal orientation",
	]
	q = (
		"You are viewing two images of a 3D scene: a top-down view and an isometric view.\n"
		"If there is a problem, it involves the labeled objects shown in the images.\n"
		"Question: Examine both views carefully and identify what problem exists. Choose one option.\n"
		"Top-view image: <image>\n"
		"Isometric-view image: <image>"
	)
	legend = _label_legend_line(label_category_map)
	if legend:
		q = q + "\n" + legend
	return {"question": q + "\n" + "\n".join(choices), "answer": mapping.get(t, "A"), "task_type": "top_isometric_error_identify", "meta": issue_meta, "images": images}


def qa3_fix(
	issue_meta: dict,
	inject_action: dict,
	images: list[str],
	option_images: list[str],
	option_texts: list[str],
	label_category_map: dict[int, str] | None = None,
) -> dict:
	correct_action = _action_text(_reverse_action(inject_action))
	letters = ["A", "B", "C", "D"]
	choices = [f"{letters[i]}. {option_texts[i]}" for i in range(4)]
	ans_letter = letters[option_texts.index(correct_action)]
	known = _qa3_problem_summary(issue_meta, label_category_map)
	legend = _label_legend_line(label_category_map)
	q = (
		f"{known}\n"
		"Choose ONE action to fix it in ONE step.\n"
		"Top-view image: <image>"
	)
	if any(str(t).startswith("Rotate object") for t in option_texts):
		q = q + "\nFor rotation actions, rotate around its center."
	if legend:
		q = q + "\n" + legend
	return {
		"question": q + "\n\n" + "\n".join(choices),
		"answer": ans_letter,
		"task_type": "top_error_modify",
		"meta": {"issue_meta": issue_meta, "inject_action": inject_action},
		"images": images,
		"reference_images": option_images,
	}


def _make_synthetic_anomaly(
	glb_name: str,
	labels: list[int],
	metas_by_label: dict[int, dict],
	label_to_oid: dict[int, str],
	id_to_entry: dict[str, dict],
	sample_idx: int,
	whitelist_pairs: set[tuple[str, str]],
	wall_bounds: dict | None,
	wall_components: list | None,
	cam_top=None,
	cam_iso=None,
	render_ctx: dict | None = None,
	iso_vis_size: tuple[int, int] | None = None,
	pivot_by_label: dict[int, Vector] | None = None,
	fail_context: dict | None = None,
	excluded_inject_labels: set[int] | None = None,
	excluded_inject_action_texts: set[str] | None = None,
):
	def _set_fail(reason: str, stage: str = "make_synthetic_anomaly", message: str = "", extra: dict | None = None):
		if fail_context is None:
			return
		fail_context.clear()
		fail_context["stage"] = str(stage)
		fail_context["reason"] = str(reason)
		if message:
			fail_context["message"] = str(message)
		if isinstance(extra, dict) and extra:
			fail_context["extra"] = extra

	if pivot_by_label is None:
		pivot_by_label = {}
		for lb in labels:
			oid = label_to_oid.get(lb)
			if oid is None or oid not in id_to_entry:
				continue
			pivot_by_label[lb] = _group_centroid_world(id_to_entry[oid]["group"])

	target_issue, action_kind = _target_case(sample_idx)
	excluded_inject_labels_set = set(int(x) for x in (excluded_inject_labels or set()))
	excluded_inject_action_texts_set = set(str(x) for x in (excluded_inject_action_texts or set()))
	desired_wall_subtype = "furniture_wall_intersection" if target_issue == ISSUE_WALL else None
	# BVH cache by label-id to avoid rebuilding unchanged objects repeatedly.
	_label_bvh_cache: dict[int, BVHTree | None] = {}

	def _invalidate_bvh_for_label(lb: int):
		_label_bvh_cache.pop(int(lb), None)

	def _get_bvh_for_label(lb: int) -> BVHTree | None:
		lb = int(lb)
		if lb in _label_bvh_cache:
			return _label_bvh_cache[lb]
		oid = label_to_oid.get(lb)
		if oid is None:
			_label_bvh_cache[lb] = None
			return None
		group = id_to_entry.get(oid, {}).get("group")
		if not group:
			_label_bvh_cache[lb] = None
			return None
		bvh = _build_group_bvh(group)
		_label_bvh_cache[lb] = bvh
		return bvh

	def _pair_is_colliding_nonwhitelist_ab(cur_metas: dict[int, dict], a: int, b: int) -> bool:
		a = int(a)
		b = int(b)
		if a == b:
			return False
		oa = label_to_oid.get(a)
		ob = label_to_oid.get(b)
		if oa is None or ob is None:
			return False
		ca = str(id_to_entry[oa].get("category", "")).lower()
		cb = str(id_to_entry[ob].get("category", "")).lower()
		if _is_pair_whitelisted(ca, cb, whitelist_pairs):
			return False
		bvh_a = _get_bvh_for_label(a)
		bvh_b = _get_bvh_for_label(b)
		if bvh_a is None or bvh_b is None:
			return False
		hits = _bvh_overlap_hit_count(bvh_a, bvh_b)
		if hits < int(BVH_OVERLAP_MIN_HITS):
			return False
		pen_depth = _obb_penetration_depth_xy(cur_metas[a], cur_metas[b])
		if pen_depth < float(OVERLAP_MIN_PENETRATION_DEPTH_M):
			return False
		if DEBUG_BVH_OVERLAP_HITS:
			print(
				f"[DEBUG] {glb_name}: BVH+pen accepted a={a}, b={b}, "
				f"hits={hits}, hits_thr={int(BVH_OVERLAP_MIN_HITS)}, "
				f"pen={pen_depth:.4f}m, pen_thr={float(OVERLAP_MIN_PENETRATION_DEPTH_M):.4f}m"
			)
		return True

	def _action_blocked(action: dict) -> bool:
		try:
			return _action_text(action) in excluded_inject_action_texts_set
		except Exception:
			return False

	def _pair_is_colliding_nonwhitelist(cur_metas: dict[int, dict], a: int) -> tuple[bool, int | None]:
		for b in labels:
			if int(b) == int(a):
				continue
			if _pair_is_colliding_nonwhitelist_ab(cur_metas, int(a), int(b)):
				return True, int(b)
		return False, None

	def _is_wall_conflict(
			cur_metas: dict[int, dict],
			a: int,
			desired_subtype: str | None = None,
			min_penetration_ratio: float | None = None,
			include_touch: bool = False,
		) -> tuple[bool, int | None, str | None]:
		return _is_wall_conflict_with_ctx(
			cur_metas,
			a,
			label_to_oid=label_to_oid,
			id_to_entry=id_to_entry,
			wall_bounds=wall_bounds,
			wall_components=wall_components,
			desired_subtype=desired_subtype,
			min_penetration_ratio=min_penetration_ratio,
			include_touch=include_touch,
		)

	def _is_wall_touch_or_conflict(
			cur_metas: dict[int, dict],
			a: int,
			desired_subtype: str | None = None,
		) -> tuple[bool, int | None, str | None]:
		return _is_wall_touch_or_conflict_with_ctx(
			cur_metas,
			a,
			label_to_oid=label_to_oid,
			id_to_entry=id_to_entry,
			wall_bounds=wall_bounds,
			wall_components=wall_components,
			desired_subtype=desired_subtype,
		)

	def _wall_conflict_visibly_strong(cur_metas: dict[int, dict], a: int) -> bool:
		hit, _partner, _sub = _is_wall_conflict(cur_metas, a, desired_subtype="furniture_wall_intersection")
		if not hit:
			return False
		m = cur_metas.get(int(a))
		if m is None:
			return False

		max_overflow_m = 0.0
		max_overflow_ratio = 0.0
		if wall_bounds is not None:
			cx, cy = m["obb"]["center"]
			hw = float(m["half_w"])
			hd = float(m["half_d"])
			wminx = float(wall_bounds["min_x"])
			wmaxx = float(wall_bounds["max_x"])
			wminy = float(wall_bounds["min_y"])
			wmaxy = float(wall_bounds["max_y"])
			over_left = max(0.0, wminx - (float(cx) - hw))
			over_right = max(0.0, (float(cx) + hw) - wmaxx)
			over_bottom = max(0.0, wminy - (float(cy) - hd))
			over_top = max(0.0, (float(cy) + hd) - wmaxy)
			max_overflow_m = max(over_left, over_right, over_bottom, over_top)
			wdepth_x = max(1e-6, wmaxx - wminx)
			wdepth_y = max(1e-6, wmaxy - wminy)
			max_overflow_ratio = max(
				over_left / wdepth_x,
				over_right / wdepth_x,
				over_bottom / wdepth_y,
				over_top / wdepth_y,
			)
		# Overflow is an auxiliary visibility cue for outer-boundary conflicts.
		# Do not hard-require it here, because many valid indoor wall conflicts
		# happen on interior wall components without obvious region-boundary overflow.

		penetration_ratio = 0.0
		if SHAPELY_AVAILABLE and wall_components:
			oa = label_to_oid.get(int(a))
			group_objs = id_to_entry.get(oa, {}).get("group") if oa is not None else None
			near = _nearest_wall_overlap_ratio_for_object(m, wall_components, group_objs=group_objs)
			penetration_ratio = float(near.get("ratio", 0.0))

			strong_pen = penetration_ratio >= float(WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO)
			if not bool(strong_pen):
				return False
		if cam_iso is not None and iso_vis_size is not None:
			iw, ih = int(iso_vis_size[0]), int(iso_vis_size[1])
			if not _wall_conflict_part_visible_iso(
				int(a),
				cam_iso,
				iw,
				ih,
				label_to_oid,
				id_to_entry,
				cur_metas,
				wall_bounds,
				wall_components=wall_components,
			):
				return False
		return True

	def _overlap_visibly_strong(cur_metas: dict[int, dict], a: int, b: int, top_size: tuple[int, int] | None) -> bool:
		_ = cur_metas
		if top_size is None or cam_top is None:
			return True
		top_bbox_overlap = _top_overlap_area_px(int(a), int(b), top_size)
		if top_bbox_overlap < float(OVERLAP_TOP_MIN_INTERSECTION_PX):
			return False
		top_mask_overlap = _top_mask_overlap_area_px(int(a), int(b), top_size)
		if top_mask_overlap < float(OVERLAP_TOP_MASK_MIN_INTERSECTION_PX):
			return False
		return True

	def _scene_has_nonwhitelist_collision(cur_metas: dict[int, dict]) -> bool:
		for lb in labels:
			hit, _ = _pair_is_colliding_nonwhitelist(cur_metas, lb)
			if hit:
				return True
		return False

	def _scene_nonwhitelist_collision_pairs(cur_metas: dict[int, dict]) -> set[tuple[int, int]]:
		pairs: set[tuple[int, int]] = set()
		for i in range(len(labels)):
			a = int(labels[i])
			for j in range(i + 1, len(labels)):
				b = int(labels[j])
				if _pair_is_colliding_nonwhitelist_ab(cur_metas, a, b):
					pairs.add((a, b) if a < b else (b, a))
		return pairs

	def _label_has_nonwhitelist_collision(cur_metas: dict[int, dict], lb: int) -> bool:
		hit, _ = _pair_is_colliding_nonwhitelist(cur_metas, int(lb))
		return bool(hit)

	def _label_has_nonwhitelist_collision_except(
		cur_metas: dict[int, dict],
		lb: int,
		allowed_partners: set[int] | None = None,
	) -> bool:
		if allowed_partners is None:
			allowed_partners = set()
		lb = int(lb)
		for other in labels:
			if other == lb or int(other) in allowed_partners:
				continue
			if _pair_is_colliding_nonwhitelist_ab(cur_metas, lb, int(other)):
				return True
		return False

	def _scene_has_wall_door_path_conflict(cur_metas: dict[int, dict]) -> bool:
		for lb in labels:
			hit, _, _ = _is_wall_touch_or_conflict(cur_metas, lb, desired_subtype=None)
			if hit:
				return True
		return False

	def _scene_wall_conflict_labels(cur_metas: dict[int, dict]) -> set[int]:
		out: set[int] = set()
		for lb in labels:
			if _label_has_wall_door_path_conflict(cur_metas, int(lb)):
				out.add(int(lb))
		return out

	def _label_has_wall_door_path_conflict(cur_metas: dict[int, dict], lb: int) -> bool:
		hit, _, _ = _is_wall_touch_or_conflict(cur_metas, lb, desired_subtype=None)
		return bool(hit)

	def _label_has_wall_conflict(cur_metas: dict[int, dict], lb: int) -> bool:
		# Single-mode guard: even light wall contact should invalidate
		# non-wall cases. Actual wall_conflict construction still calls
		# _is_wall_conflict directly and requires the heavy threshold.
		hit, _, _ = _is_wall_touch_or_conflict(cur_metas, lb, desired_subtype="furniture_wall_intersection")
		return bool(hit)

	def _label_out_of_region_by_bbox(cur_metas: dict[int, dict], lb: int) -> bool:
		if wall_bounds is None:
			return False
		lb = int(lb)
		m = cur_metas.get(lb)
		if m is None:
			return False
		cat = _label_category(lb)
		cx, cy = m["obb"]["center"]
		hw = m["half_w"]
		hd = m["half_d"]
		inside = _inside_wall_region_by_category_bbox(
			cx,
			cy,
			hw,
			hd,
			cat,
			wall_bounds,
			ratio_default=REGION_BOUNDARY_RATIO_DEFAULT,
			ratio_relaxed=REGION_BOUNDARY_RATIO_RELAXED,
		)
		return (not inside)

	def _label_category(lb: int) -> str:
		oid = label_to_oid.get(lb)
		if oid is None:
			return ""
		return str(id_to_entry.get(oid, {}).get("category", "")).lower()

	def _wall_side_distances_from_meta(meta: dict) -> dict[str, float] | None:
		if wall_bounds is None:
			return None
		cx, cy = meta["obb"]["center"]
		return {
			"W": float(cx - float(wall_bounds["min_x"])),
			"E": float(float(wall_bounds["max_x"]) - float(cx)),
			"S": float(cy - float(wall_bounds["min_y"])),
			"N": float(float(wall_bounds["max_y"]) - float(cy)),
		}

	def _nearest_wall_side_from_meta(meta: dict) -> tuple[str, float]:
		ds = _wall_side_distances_from_meta(meta)
		if not ds:
			return "W", 1e9
		side = min(ds.keys(), key=lambda k: ds[k])
		return side, float(ds[side])

	def _nearest_wall_dist_from_meta(meta: dict) -> float:
		return float(_nearest_wall_side_from_meta(meta)[1])

	def _normalize_axis_vec(v) -> tuple[float, float]:
		try:
			x = float(v[0])
			y = float(v[1])
		except Exception:
			return (1.0, 0.0)
		n = math.hypot(x, y)
		if n <= 1e-9:
			return (1.0, 0.0)
		return (x / n, y / n)

	def _angle_between_axes_deg(a, b) -> float:
		ax, ay = _normalize_axis_vec(a)
		bx, by = _normalize_axis_vec(b)
		dot = max(-1.0, min(1.0, abs(ax * bx + ay * by)))
		return math.degrees(math.acos(dot))

	def _axis_error_to_wall_side_deg(obb: dict, wall_side: str) -> tuple[float, str]:
		ax = _normalize_axis_vec(obb.get("ax", (1.0, 0.0)))
		ay = _normalize_axis_vec(obb.get("ay", (0.0, 1.0)))
		target = (0.0, 1.0) if wall_side in ("W", "E") else (1.0, 0.0)
		err_ax = _angle_between_axes_deg(ax, target)
		err_ay = _angle_between_axes_deg(ay, target)
		if err_ax <= err_ay:
			return err_ax, "ax"
		return err_ay, "ay"

	def _is_orientation_target_candidate(lb: int, meta: dict) -> bool:
		hx = float(meta["obb"]["hx"])
		hy = float(meta["obb"]["hy"])
		if min(hx, hy) < 1e-4:
			return False
		cat = _label_category(lb)
		if _round_like_category(cat):
			return False
		oid = label_to_oid.get(lb)
		group = id_to_entry.get(oid, {}).get("group", []) if oid is not None else []
		if not group:
			return False
		shape_group_ok = _is_orientation_candidate_shape_group(group, cat)
		ratio = max(hx, hy) / max(min(hx, hy), 1e-6)
		if (not shape_group_ok) and ratio > float(ORIENTATION_MAX_ASPECT_RATIO):
			return False
		hz = float(meta["obb"].get("hz", meta.get("height", hx)))
		volume = hx * hy * hz
		if volume < float(ORIENTATION_MIN_VOLUME_THRESHOLD):
			return False
		if wall_bounds is not None:
			_side, near_dist = _nearest_wall_side_from_meta(meta)
			if near_dist > float(ORIENTATION_WALL_NEAR_DIST_M):
				return False
		return True

	def _orientation_nearest_side_labels(cur_metas: dict[int, dict], ignore_labels: set[int] | None = None) -> set[int]:
		if wall_bounds is None:
			return set()
		ignore = set(ignore_labels or set())
		candidates: list[int] = []
		for lb in labels:
			lb_i = int(lb)
			if lb_i in ignore:
				continue
			m = cur_metas.get(lb_i)
			if m is None:
				continue
			if not _is_orientation_target_candidate(lb_i, m):
				continue
			ds = _wall_side_distances_from_meta(m)
			if not ds:
				continue
			candidates.append(lb_i)
		if not candidates:
			return set()
		# Orientation anomalies should be possible for any clean, wall-near,
		# rectangle-like object. Restricting to only the single closest object
		# per wall side made case_7 effectively unreachable on these scenes.
		return set(candidates)

	def _label_has_orientation_conflict(
		cur_metas: dict[int, dict],
		lb: int,
		eligible_labels: set[int] | None = None,
	) -> bool:
		lb = int(lb)
		cur_m = cur_metas.get(lb)
		if cur_m is None:
			return False
		if not _is_orientation_target_candidate(lb, cur_m):
			return False
		eligible = eligible_labels if eligible_labels is not None else _orientation_nearest_side_labels(cur_metas)
		if lb not in eligible:
			return False
		nearest_side, _near_dist = _nearest_wall_side_from_meta(cur_m)
		d, _edge_axis = _axis_error_to_wall_side_deg(cur_m.get("obb") or {}, nearest_side)
		return bool(d >= float(ORIENTATION_MIN_ANGLE_DELTA_DEG))

	def _scene_has_orientation_conflict(cur_metas: dict[int, dict], ignore_labels: set[int] | None = None) -> bool:
		ignore = set(ignore_labels or set())
		for lb in labels:
			if int(lb) in ignore:
				continue
			if _label_has_orientation_conflict(cur_metas, int(lb)):
				return True
		return False

	def _scene_orientation_conflict_labels(cur_metas: dict[int, dict], ignore_labels: set[int] | None = None) -> set[int]:
		eligible = _orientation_nearest_side_labels(cur_metas, ignore_labels=ignore_labels)
		out: set[int] = set()
		for lb_i in eligible:
			if _label_has_orientation_conflict(cur_metas, lb_i, eligible_labels=eligible):
				out.add(int(lb_i))
		return out

	# Baseline issue sets before any injected action.
	# Exclusivity checks reject only newly introduced side-issues by set difference.
	_baseline_overlap_pairs = _scene_nonwhitelist_collision_pairs(metas_by_label)
	_baseline_wall_labels = _scene_wall_conflict_labels(metas_by_label)
	_baseline_orientation_labels = _scene_orientation_conflict_labels(metas_by_label)
	_baseline_overlap_labels: set[int] = set()
	for pa, pb in _baseline_overlap_pairs:
		_baseline_overlap_labels.add(int(pa))
		_baseline_overlap_labels.add(int(pb))

	def _exclusive_main_issue_ok(cur_metas: dict[int, dict], main_issue_type: str, focus_label: int | None = None) -> bool:
		overlap_pairs = _scene_nonwhitelist_collision_pairs(cur_metas)
		wall_labels = _scene_wall_conflict_labels(cur_metas)
		orientation_labels = _scene_orientation_conflict_labels(cur_metas)
		new_overlap_pairs = overlap_pairs - _baseline_overlap_pairs
		new_wall_labels = wall_labels - _baseline_wall_labels
		new_orientation_labels = orientation_labels - _baseline_orientation_labels
		has_overlap = len(overlap_pairs) > 0
		if main_issue_type == ISSUE_ANGLE:
			if focus_label is None:
				return False
			lb = int(focus_label)
			# Require orientation anomaly to be newly injected on the target label.
			if lb in _baseline_orientation_labels:
				return False
			if not _label_has_orientation_conflict(cur_metas, lb):
				return False
			# Strict single-issue rule on target object: no overlap / wall conflict together.
			if _label_has_nonwhitelist_collision(cur_metas, lb):
				return False
			if _label_has_wall_conflict(cur_metas, lb):
				return False
			new_other_orientation_labels = {x for x in new_orientation_labels if int(x) != lb}
			return (len(new_overlap_pairs) == 0) and (len(new_wall_labels) == 0) and (len(new_other_orientation_labels) == 0)
		if main_issue_type == ISSUE_OVERLAP:
			if focus_label is None:
				return False
			lb = int(focus_label)
			# Require overlap anomaly to be newly injected on the target label.
			if lb in _baseline_overlap_labels:
				return False
			if not _label_has_nonwhitelist_collision(cur_metas, lb):
				return False
			# Strict single-issue rule on target object: no wall/orientation together.
			if _label_has_wall_conflict(cur_metas, lb):
				return False
			if _label_has_orientation_conflict(cur_metas, lb):
				return False
			new_overlap_on_lb = {p for p in new_overlap_pairs if lb in p}
			if len(new_overlap_on_lb) == 0:
				return False
			# Reject collateral new overlaps that do not involve the target object.
			if any(lb not in p for p in new_overlap_pairs):
				return False
			return (len(new_wall_labels) == 0) and (len(new_orientation_labels) == 0)
		if main_issue_type == ISSUE_WALL:
			if focus_label is None:
				return False
			lb = int(focus_label)
			# Require wall conflict to be newly injected on the target label.
			if lb in _baseline_wall_labels:
				return False
			if not _label_has_wall_conflict(cur_metas, lb):
				return False
			# Strict single-issue rule on target object: no overlap / orientation together.
			if _label_has_nonwhitelist_collision(cur_metas, lb):
				return False
			if _label_has_orientation_conflict(cur_metas, lb):
				return False
			new_other_wall_labels = {x for x in new_wall_labels if int(x) != lb}
			return (len(new_overlap_pairs) == 0) and (len(new_orientation_labels) == 0) and (len(new_other_wall_labels) == 0)
		return False

	def _top_overlap_area_px(lb_a: int, lb_b: int, top_size: tuple[int, int] | None) -> float:
		if top_size is None or cam_top is None:
			return 0.0
		oid_a = label_to_oid.get(lb_a)
		oid_b = label_to_oid.get(lb_b)
		if oid_a is None or oid_b is None:
			return 0.0
		w, h = int(top_size[0]), int(top_size[1])
		r1 = _project_group_bbox_rect(id_to_entry[oid_a]["group"], cam_top, w, h)
		r2 = _project_group_bbox_rect(id_to_entry[oid_b]["group"], cam_top, w, h)
		if r1 is None or r2 is None:
			return 0.0
		ix = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
		iy = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
		return float(ix * iy)

	def _top_mask_overlap_area_px(lb_a: int, lb_b: int, top_size: tuple[int, int] | None) -> float:
		if top_size is None or cam_top is None:
			return 0.0
		oid_a = label_to_oid.get(lb_a)
		oid_b = label_to_oid.get(lb_b)
		if oid_a is None or oid_b is None:
			return 0.0
		w, h = int(top_size[0]), int(top_size[1])
		return _project_group_mask_overlap_area_px(
			id_to_entry[oid_a]["group"],
			id_to_entry[oid_b]["group"],
			cam_top,
			w,
			h,
		)

	def _try_overlap_case(action_kind_local: str):
		cands = [lb for lb in labels if int(lb) not in excluded_inject_labels_set]

		def _nearest_sq_dist(lb: int) -> float:
			cx, cy = metas_by_label[lb]["obb"]["center"]
			best = 1e30
			for ob in labels:
				if ob == lb:
					continue
				ox, oy = metas_by_label[ob]["obb"]["center"]
				d2 = (cx - ox) ** 2 + (cy - oy) ** 2
				if d2 < best:
					best = d2
			return best

		if action_kind_local in {"rotate", "scale"}:
			cands = sorted(
				cands,
				key=lambda lb: (
					_nearest_sq_dist(lb),
					-(float(metas_by_label[lb]["obb"]["hx"]) * float(metas_by_label[lb]["obb"]["hy"])),
				),
			)
		else:
			random.shuffle(cands)

		trial_attempts = 0
		objects_checked = 0
		reason_counters = {
			"overlap_not_visually_strong": 0,
			"top_not_fully_in_frame": 0,
		}
		object_budget_base = max(1, int(OVERLAP_OBJECT_BUDGET_PER_SCENE))
		if action_kind_local in {"rotate", "scale"}:
			object_budget = len(cands)
		else:
			object_budget = object_budget_base
		top_vis_size = None
		if cam_top is not None and render_ctx is not None:
			top_vis_size = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)

		for a in cands:
			if objects_checked >= object_budget:
				break
			objects_checked += 1
			if action_kind_local == "rotate" and _is_near_circular_for_rotate(metas_by_label[a], _label_category(a)):
				continue

			oa = label_to_oid[a]
			group = id_to_entry[oa]["group"]
			orig = _translate_group(group, 0.0, 0.0)
			try:
				if action_kind_local == "move":
					for b in labels:
						if b == a:
							continue
						ca = metas_by_label[a]["obb"]["center"]
						cb = metas_by_label[b]["obb"]["center"]
						dx, dy = cb[0] - ca[0], cb[1] - ca[1]
						n = math.hypot(dx, dy)
						if n < 1e-6:
							continue
						vx, vy = dx / n, dy / n
						for dist in [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 2.8, 3.0, 3.5, 4.0]:
							trial_attempts += 1
							_restore_group(group, orig)
							_translate_group(group, vx * dist * UNIT_SCALE, vy * dist * UNIT_SCALE)
							_invalidate_bvh_for_label(a)
							cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
							ok, partner = _pair_is_colliding_nonwhitelist(cur, a)
							if not ok or partner is None:
								continue
							if _label_out_of_region_by_bbox(cur, a):
								continue
							if not _exclusive_main_issue_ok(cur, ISSUE_OVERLAP, focus_label=a):
								continue
							if top_vis_size is not None:
								dyn_w, dyn_h = top_vis_size
								if not _is_object_visible_after_action(group, cam_top, dyn_w, dyn_h):
									reason_counters["top_not_fully_in_frame"] += 1
									continue
							if not _overlap_visibly_strong(cur, a, int(partner), top_vis_size):
								reason_counters["overlap_not_visually_strong"] += 1
								continue
							dir_code = max(MOVE_DIRS, key=lambda k: _move_vec(k)[0] * vx + _move_vec(k)[1] * vy)
							action = {"op": "move", "id": a, "dir": dir_code, "dist": round(dist, 1)}
							if _action_blocked(action):
								continue
							desc = f"Object {a} has a physically implausible overlap with object {partner}."
							return _build_issue_meta(ISSUE_OVERLAP, [a, partner], desc, action_kind=action_kind_local), action
				elif action_kind_local == "rotate":
					for deg in [10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120, 135, 150]:
						for cw in [True, False]:
							trial_attempts += 1
							_restore_group(group, orig)
							_rotate_group(group, deg, cw, pivot=pivot_by_label.get(a))
							_invalidate_bvh_for_label(a)
							cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
							ok, partner = _pair_is_colliding_nonwhitelist(cur, a)
							if not ok or partner is None:
								continue
							if _label_out_of_region_by_bbox(cur, a):
								continue
							if not _exclusive_main_issue_ok(cur, ISSUE_OVERLAP, focus_label=a):
								continue
							if top_vis_size is not None:
								dyn_w, dyn_h = top_vis_size
								if not _is_object_visible_after_action(group, cam_top, dyn_w, dyn_h):
									reason_counters["top_not_fully_in_frame"] += 1
									continue
							if not _overlap_visibly_strong(cur, a, int(partner), top_vis_size):
								reason_counters["overlap_not_visually_strong"] += 1
								continue
							action = {"op": "rotate", "id": a, "clockwise": cw, "deg": int(deg)}
							if _action_blocked(action):
								continue
							desc = f"Object {a} has a physically implausible overlap with object {partner} after rotation."
							return _build_issue_meta(ISSUE_OVERLAP, [a, partner], desc, action_kind=action_kind_local), action
				else:
					scale_trials = [(True, p) for p in [150, 130, 120, 100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 18, 15, 12, 10, 8, 5]]
					for up, pct in scale_trials:
						trial_attempts += 1
						_restore_group(group, orig)
						_scale_group(group, float(pct), bool(up), pivot=pivot_by_label.get(a))
						_invalidate_bvh_for_label(a)
						cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
						ok, partner = _pair_is_colliding_nonwhitelist(cur, a)
						if not ok or partner is None:
							continue
						if _label_out_of_region_by_bbox(cur, a):
							continue
						if not _exclusive_main_issue_ok(cur, ISSUE_OVERLAP, focus_label=a):
							continue
						if top_vis_size is not None:
							dyn_w, dyn_h = top_vis_size
							if not _is_object_visible_after_action(group, cam_top, dyn_w, dyn_h):
								reason_counters["top_not_fully_in_frame"] += 1
								continue
						if not _overlap_visibly_strong(cur, a, int(partner), top_vis_size):
							reason_counters["overlap_not_visually_strong"] += 1
							continue
						action = {"op": "scale", "id": a, "up": bool(up), "percent": int(pct)}
						if _action_blocked(action):
							continue
						desc = f"Object {a} has a physically implausible overlap with object {partner} after scaling."
						return _build_issue_meta(ISSUE_OVERLAP, [a, partner], desc, action_kind=action_kind_local), action
			finally:
				_restore_group(group, orig)

		print(
			f"[DEBUG] skip {glb_name}: overlap case exhausted objects={objects_checked} "
			f"(object_budget={object_budget}), action_trials={trial_attempts}, no valid collision found"
		)
		_set_fail(
			"overlap_case_exhausted",
			stage=f"overlap_{action_kind_local}",
			message="no valid overlap collision found",
			extra={
				"objects_checked": int(objects_checked),
				"object_budget": int(object_budget),
				"action_trials": int(trial_attempts),
				"reason_counters": {k: int(v) for k, v in reason_counters.items()},
			},
		)
		return None

	def _try_wall_case(action_kind_local: str):
		cands = [lb for lb in labels if int(lb) not in excluded_inject_labels_set]
		if action_kind_local in {"rotate", "scale"}:
			cands = sorted(
				cands,
				key=lambda lb: (
					_nearest_wall_dist_from_meta(metas_by_label[lb]),
					-(float(metas_by_label[lb]["obb"]["hx"]) * float(metas_by_label[lb]["obb"]["hy"])),
				),
			)
		else:
			random.shuffle(cands)
		trial_attempts = 0
		objects_checked = 0
		reason_counters = {
			"hit_but_not_strong": 0,
			"new_overlap_conflict": 0,
			"exclusive_main_issue_fail": 0,
			"top_not_fully_in_frame": 0,
			"iso_conflict_not_visible": 0,
		}
		object_budget_base = max(1, int(WALL_DOOR_OBJECT_BUDGET_PER_SCENE))
		if action_kind_local in {"rotate", "scale"}:
			object_budget = len(cands)
		else:
			object_budget = object_budget_base
		top_vis_size = None
		if cam_top is not None and render_ctx is not None:
			top_vis_size = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)

		subtype_order = [desired_wall_subtype, None] if desired_wall_subtype else [None]
		for a in cands:
			if objects_checked >= object_budget:
				break
			objects_checked += 1
			if action_kind_local == "rotate" and _is_near_circular_for_rotate(metas_by_label[a], _label_category(a)):
				continue

			oa = label_to_oid[a]
			group = id_to_entry[oa]["group"]
			orig = _translate_group(group, 0.0, 0.0)
			try:
				for subtype_try in subtype_order:
					if action_kind_local == "move":
						meta = metas_by_label[a]
						cx, cy = meta["obb"]["center"]
						obj_max_dim = max(meta["half_w"] * 2.0, meta["half_d"] * 2.0)
						overshoot = max(0.5, 0.5 * obj_max_dim + 0.1)
						if wall_bounds is None:
							continue
						dirs = [
							("W", wall_bounds["min_x"] - (cx - meta["half_w"]) + overshoot),
							("E", (cx + meta["half_w"]) - wall_bounds["max_x"] + overshoot),
							("S", wall_bounds["min_y"] - (cy - meta["half_d"]) + overshoot),
							("N", (cy + meta["half_d"]) - wall_bounds["max_y"] + overshoot),
						]
						dirs = sorted(dirs, key=lambda x: abs(x[1]))
						for d, need in dirs:
							base = max(0.0, float(need))
							dist_trials = [base, base + 0.20, base + 0.45, base + 0.70, base + 1.0] if base >= 0.05 else [0.12, 0.25, 0.45, 0.70, 1.0]
							for dist_try in dist_trials:
								trial_attempts += 1
								_restore_group(group, orig)
								vx, vy = _move_vec(d)
								_translate_group(group, vx * dist_try, vy * dist_try)
								_invalidate_bvh_for_label(a)
								cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
								hit, partner, subtype = _is_wall_conflict(cur, a, desired_subtype=subtype_try)
								if not hit:
									continue
								if not _wall_conflict_visibly_strong(cur, a):
									reason_counters["hit_but_not_strong"] += 1
									if cam_iso is not None and iso_vis_size is not None:
										reason_counters["iso_conflict_not_visible"] += 1
									continue
								allowed = {int(partner)} if partner is not None else set()
								if _label_has_nonwhitelist_collision_except(cur, a, allowed_partners=allowed):
									reason_counters["new_overlap_conflict"] += 1
									continue
								if not _exclusive_main_issue_ok(cur, ISSUE_WALL, focus_label=a):
									reason_counters["exclusive_main_issue_fail"] += 1
									continue
								if top_vis_size is not None:
									dyn_w, dyn_h = top_vis_size
									if not _group_fully_in_frame(group, cam_top, dyn_w, dyn_h):
										reason_counters["top_not_fully_in_frame"] += 1
										continue
								action = {"op": "move", "id": a, "dir": d, "dist": round(float(dist_try), 1)}
								if _action_blocked(action):
									continue
								desc = f"Object {a} causes {str(subtype).replace('_', ' ')}."
								ids = [a] + ([partner] if partner is not None else [])
								return _build_issue_meta(_issue_type_from_wall_subtype(subtype), ids, desc, subtype=subtype, action_kind=action_kind_local), action
					elif action_kind_local == "rotate":
						for deg in [12, 15, 20, 25, 30, 40, 45, 60, 70, 80, 90, 100, 110, 120, 135, 150, 165]:
							for cw in [True, False]:
								trial_attempts += 1
								_restore_group(group, orig)
								_rotate_group(group, deg, cw, pivot=pivot_by_label.get(a))
								_invalidate_bvh_for_label(a)
								cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
								hit, partner, subtype = _is_wall_conflict(cur, a, desired_subtype=subtype_try)
								if not hit:
									continue
								if not _wall_conflict_visibly_strong(cur, a):
									reason_counters["hit_but_not_strong"] += 1
									if cam_iso is not None and iso_vis_size is not None:
										reason_counters["iso_conflict_not_visible"] += 1
									continue
								allowed = {int(partner)} if partner is not None else set()
								if _label_has_nonwhitelist_collision_except(cur, a, allowed_partners=allowed):
									reason_counters["new_overlap_conflict"] += 1
									continue
								if not _exclusive_main_issue_ok(cur, ISSUE_WALL, focus_label=a):
									reason_counters["exclusive_main_issue_fail"] += 1
									continue
								if top_vis_size is not None:
									dyn_w, dyn_h = top_vis_size
									if not _group_fully_in_frame(group, cam_top, dyn_w, dyn_h):
										reason_counters["top_not_fully_in_frame"] += 1
										continue
								action = {"op": "rotate", "id": a, "clockwise": cw, "deg": int(deg)}
								if _action_blocked(action):
									continue
								desc = f"Object {a} causes {str(subtype).replace('_', ' ')} after rotation."
								ids = [a] + ([partner] if partner is not None else [])
								return _build_issue_meta(_issue_type_from_wall_subtype(subtype), ids, desc, subtype=subtype, action_kind=action_kind_local), action
					else:
						# Keep small scale attempts for case_6; strength gating is enforced
						# by wall-conflict visibility checks rather than pre-pruning.
						for pct in [100, 90, 80, 70, 60, 50, 40, 35, 30, 25, 20, 15, 10]:
							trial_attempts += 1
							_restore_group(group, orig)
							_scale_group(group, float(pct), True, pivot=pivot_by_label.get(a))
							_invalidate_bvh_for_label(a)
							cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
							hit, partner, subtype = _is_wall_conflict(cur, a, desired_subtype=subtype_try)
							if not hit:
								continue
							if not _wall_conflict_visibly_strong(cur, a):
								reason_counters["hit_but_not_strong"] += 1
								if cam_iso is not None and iso_vis_size is not None:
									reason_counters["iso_conflict_not_visible"] += 1
								continue
							allowed = {int(partner)} if partner is not None else set()
							if _label_has_nonwhitelist_collision_except(cur, a, allowed_partners=allowed):
								reason_counters["new_overlap_conflict"] += 1
								continue
							if not _exclusive_main_issue_ok(cur, ISSUE_WALL, focus_label=a):
								reason_counters["exclusive_main_issue_fail"] += 1
								continue
							if top_vis_size is not None:
								dyn_w, dyn_h = top_vis_size
								if not _group_fully_in_frame(group, cam_top, dyn_w, dyn_h):
									reason_counters["top_not_fully_in_frame"] += 1
									continue
							action = {"op": "scale", "id": a, "up": True, "percent": int(pct)}
							if _action_blocked(action):
								continue
							desc = f"Object {a} causes {str(subtype).replace('_', ' ')} after scaling."
							ids = [a] + ([partner] if partner is not None else [])
							return _build_issue_meta(_issue_type_from_wall_subtype(subtype), ids, desc, subtype=subtype, action_kind=action_kind_local), action
			finally:
				_restore_group(group, orig)

		print(
			f"[DEBUG] skip {glb_name}: wall_case exhausted objects={objects_checked} "
			f"(object_budget={object_budget}), action_trials={trial_attempts}, no valid wall conflict found"
		)
		_set_fail(
			"wall_case_exhausted",
			stage=f"wall_{action_kind_local}",
			message="no valid wall conflict found",
			extra={
				"objects_checked": int(objects_checked),
				"object_budget": int(object_budget),
				"action_trials": int(trial_attempts),
				"reason_counters": {k: int(v) for k, v in reason_counters.items()},
			},
		)
		return None

	def _try_orientation_case():
			trial_attempts = 0
			objects_checked = 0
			object_budget = max(1, int(ORIENTATION_OBJECT_BUDGET_PER_SCENE))
			fail_stats = {
				"total_labels": len(labels),
				"base_not_rect_like": 0,
				"base_round_like": 0,
				"base_small_volume": 0,
				"base_not_near_wall": 0,
				"base_has_overlap": 0,
				"base_has_wall": 0,
				"base_has_orientation": 0,
				"base_pass": 0,
					"nearest_filter_out": 0,
					"trial_overlap_conflict": 0,
					"trial_wall_conflict": 0,
					"trial_orientation_conflict": 0,
					"trial_visibility_fail": 0,
				}
	
			def _orientation_only_ok(action: dict) -> tuple[bool, str]:
				g, o = _apply_action(action, label_to_oid, id_to_entry, pivot_by_label=pivot_by_label)
				try:
					try:
						_invalidate_bvh_for_label(int(action.get("id")))
					except Exception:
						pass
					cur = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
					lb = int(action.get("id"))
					if _label_has_nonwhitelist_collision(cur, lb):
						return False, "overlap_conflict"
					if cam_top is not None and render_ctx is not None:
						dyn_w, dyn_h = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)
						for other in labels:
							other_i = int(other)
							if other_i == lb:
								continue
							oid_a = label_to_oid.get(lb)
							oid_b = label_to_oid.get(other_i)
							if oid_a is None or oid_b is None:
								continue
							ca = str(id_to_entry[oid_a].get("category", "")).lower()
							cb = str(id_to_entry[oid_b].get("category", "")).lower()
							if _is_pair_whitelisted(ca, cb, whitelist_pairs):
								continue
							if _top_mask_overlap_area_px(lb, other_i, (dyn_w, dyn_h)) >= float(OVERLAP_TOP_MIN_INTERSECTION_PX):
								return False, "overlap_conflict"
						if _label_has_wall_conflict(cur, lb):
							return False, "wall_conflict"
					if not _exclusive_main_issue_ok(cur, ISSUE_ANGLE, focus_label=lb):
						if _scene_has_nonwhitelist_collision(cur):
							return False, "overlap_conflict"
						if _scene_has_wall_door_path_conflict(cur):
							return False, "wall_conflict"
						return False, "orientation_conflict"
					return True, "ok"
				finally:
					_restore_group(g, o)
	
			if wall_bounds is None:
				print(f"[DEBUG] skip {glb_name}: orientation no wall_bounds")
				_set_fail("orientation_no_wall_bounds", stage="orientation_rotate")
				return None
	
			base_candidates: list[int] = []
			for a in labels:
				if int(a) in excluded_inject_labels_set:
					continue
				m = metas_by_label[a]
				cat = _label_category(a)
				if _round_like_category(cat):
					fail_stats["base_round_like"] += 1
					continue
					oid = label_to_oid.get(a)
					group = id_to_entry.get(oid, {}).get("group", []) if oid is not None else []
					if not group:
						fail_stats["base_not_rect_like"] += 1
						continue
				hx = float(m["obb"]["hx"])
				hy = float(m["obb"]["hy"])
				ratio = max(hx, hy) / max(min(hx, hy), 1e-6)
				if ratio > float(ORIENTATION_MAX_ASPECT_RATIO):
					fail_stats["base_not_rect_like"] += 1
					continue
				hz = float(m["obb"].get("hz", m.get("height", hx)))
				volume = hx * hy * hz
				if volume < float(ORIENTATION_MIN_VOLUME_THRESHOLD):
					fail_stats["base_small_volume"] += 1
					continue
				if _nearest_wall_dist_from_meta(m) > float(ORIENTATION_WALL_NEAR_DIST_M):
					fail_stats["base_not_near_wall"] += 1
					continue
				base_candidates.append(a)
				fail_stats["base_pass"] += 1
	
			eligible: list[int] = []
			if base_candidates:
				ignore = set(int(x) for x in labels) - set(int(x) for x in base_candidates)
				eligible_set = _orientation_nearest_side_labels(metas_by_label, ignore_labels=ignore)
				for lb in base_candidates:
					if int(lb) in eligible_set:
						eligible.append(lb)
					else:
						fail_stats["nearest_filter_out"] += 1
	
			eligible = sorted(
				eligible,
				key=lambda lb: (
					_nearest_wall_dist_from_meta(metas_by_label[lb]),
					-(float(metas_by_label[lb]["obb"]["hx"]) * float(metas_by_label[lb]["obb"]["hy"])),
				),
			)
	
			for a in eligible[:object_budget]:
				objects_checked += 1
				for deg in [25, 30, 40, 45, 60, 75, 90, 120, 135, 150, 20, 18, 15]:
					for cw in [True, False]:
						trial_attempts += 1
						action = {"op": "rotate", "id": a, "clockwise": cw, "deg": deg}
						if _action_blocked(action):
							continue
						ok, reason = _orientation_only_ok(action)
						if not ok:
							if reason == "overlap_conflict":
								fail_stats["trial_overlap_conflict"] += 1
							elif reason == "wall_conflict":
								fail_stats["trial_wall_conflict"] += 1
							else:
								fail_stats["trial_orientation_conflict"] += 1
							continue
						if cam_top is not None and render_ctx is not None:
							dyn_w, dyn_h = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)
							oid = label_to_oid.get(int(a))
							if oid is None:
								continue
							if not _is_object_visible_after_action(id_to_entry[oid]["group"], cam_top, dyn_w, dyn_h):
								fail_stats["trial_visibility_fail"] += 1
								continue
						if cam_iso is not None and iso_vis_size is not None:
							oid = label_to_oid.get(int(a))
							if oid is None:
								continue
							iw, ih = int(iso_vis_size[0]), int(iso_vis_size[1])
							if not _is_object_visible_after_action(id_to_entry[oid]["group"], cam_iso, iw, ih):
								fail_stats["trial_visibility_fail"] += 1
								continue
						desc = f"Object {a} has an abnormal orientation."
						return _build_issue_meta(ISSUE_ANGLE, [a], desc, action_kind="rotate"), action
	
			print(
				f"[DEBUG] skip {glb_name}: orientation case exhausted "
				f"(base_pass={len(base_candidates)}, eligible={len(eligible)}, checked={objects_checked}, trials={trial_attempts}, "
				f"base_not_rect={fail_stats['base_not_rect_like']}, base_round={fail_stats['base_round_like']}, "
				f"base_small_volume={fail_stats['base_small_volume']}, base_not_near_wall={fail_stats['base_not_near_wall']}, "
				f"base_has_overlap={fail_stats['base_has_overlap']}, base_has_wall={fail_stats['base_has_wall']}, "
				f"base_has_orientation={fail_stats['base_has_orientation']}, "
					f"nearest_filter_out={fail_stats['nearest_filter_out']}, trial_overlap_conflict={fail_stats['trial_overlap_conflict']}, "
					f"trial_wall_conflict={fail_stats['trial_wall_conflict']}, "
					f"trial_orientation_conflict={fail_stats['trial_orientation_conflict']}, "
					f"trial_visibility_fail={fail_stats['trial_visibility_fail']}), "
				f"no valid orientation found"
			)
			_set_fail(
				"orientation_case_exhausted",
				stage="orientation_rotate",
				message="no valid orientation found",
				extra={
					"objects_checked": int(objects_checked),
					"action_trials": int(trial_attempts),
					"base_pass": int(len(base_candidates)),
					"eligible": int(len(eligible)),
					"fail_stats": {k: int(v) for k, v in fail_stats.items()},
				},
			)
			return None

	# Strict-by-main-type policy:
	# If the selected main issue type cannot be constructed, return None (skip this scene)
	if target_issue == ISSUE_OVERLAP:
		return _try_overlap_case(action_kind)
	if target_issue == ISSUE_WALL:
		return _try_wall_case(action_kind)
	if target_issue == ISSUE_ANGLE:
		return _try_orientation_case()
	_set_fail("unsupported_target_issue", stage="dispatch", extra={"target_issue": str(target_issue), "action_kind": str(action_kind)})
	return None


def _build_option_texts(inject_action: dict) -> list[str]:
	correct = _action_text(_reverse_action(inject_action))
	opts = [correct]
	op_pool = ["move", "rotate", "scale"]
	if inject_action.get("op") == "rotate":
		op_pool = ["rotate", "move", "scale"]
	elif inject_action.get("op") == "scale":
		op_pool = ["scale", "move", "rotate"]
	while len(opts) < 4:
		op_type = random.choice(op_pool)
		bid = inject_action["id"]
		if op_type == "move":
			a = {"op": "move", "id": bid, "dir": random.choice(MOVE_DIRS), "dist": random.choice(MOVE_DISTS_M)}
		elif op_type == "rotate":
			a = {"op": "rotate", "id": bid, "clockwise": bool(random.getrandbits(1)), "deg": random.choice(ROTATE_DEGS)}
		else:
			a = {"op": "scale", "id": bid, "up": bool(random.getrandbits(1)), "percent": random.choice(SCALE_PCTS)}
		txt = _action_text(a)
		if txt not in opts:
			opts.append(txt)
	random.shuffle(opts)
	return opts


def _sample_option_action_text(inject_action: dict) -> str:
	"""Sample one candidate option text (may be correct or wrong; caller filters)."""
	op_pool = ["move", "rotate", "scale"]
	if inject_action.get("op") == "rotate":
		op_pool = ["rotate", "move", "scale"]
	elif inject_action.get("op") == "scale":
		op_pool = ["scale", "move", "rotate"]
	op_type = random.choice(op_pool)
	bid = inject_action["id"]
	if op_type == "move":
		a = {"op": "move", "id": bid, "dir": random.choice(MOVE_DIRS), "dist": random.choice(MOVE_DISTS_M)}
	elif op_type == "rotate":
		a = {"op": "rotate", "id": bid, "clockwise": bool(random.getrandbits(1)), "deg": random.choice(ROTATE_DEGS)}
	else:
		a = {"op": "scale", "id": bid, "up": bool(random.getrandbits(1)), "percent": random.choice(SCALE_PCTS)}
	return _action_text(a)


def _parse_action_text(action_text: str) -> dict:
	m = re.match(r"^Move (?:object|building) (\d+) (\w+) by ([0-9.]+)m$", action_text)
	if m:
		dir_word = str(m.group(2)).strip().lower()
		dir_map = {
			"north": "N", "south": "S", "east": "E", "west": "W",
			"northeast": "NE", "northwest": "NW", "southeast": "SE", "southwest": "SW",
		}
		return {"op": "move", "id": int(m.group(1)), "dir": dir_map.get(dir_word, "E"), "dist": float(m.group(3))}
	m = re.match(r"^Rotate (?:object|building) (\d+) (clockwise|counter-clockwise) by (\d+)°(?: around its center)?$", action_text)
	if m:
		return {"op": "rotate", "id": int(m.group(1)), "clockwise": m.group(2) == "clockwise", "deg": int(m.group(3))}
	m = re.match(r"^Scale (up|down) (?:object|building) (\d+) by (\d+)%$", action_text)
	if m:
		return {"op": "scale", "id": int(m.group(2)), "up": m.group(1) == "up", "percent": int(m.group(3))}
	raise ValueError(f"Unsupported option text: {action_text}")


def process_scene(
	glb_name: str,
	info: dict,
	glb_dir: Path,
	output_dir: Path,
	layout_root: Path,
	whitelist_pairs: set[tuple[str, str]],
	sample_idx: int,
	case_id: int | None = None,
	fail_context: dict | None = None,
	_excluded_inject_labels: set[int] | None = None,
	_excluded_inject_action_texts: set[str] | None = None,
	_retry_attempt: int = 0,
) -> dict | None:
	def _mark_fail(reason: str, stage: str = "process_scene", message: str = "", extra: dict | None = None):
		if fail_context is None:
			return
		fail_context.clear()
		fail_context["stage"] = str(stage)
		fail_context["reason"] = str(reason)
		if message:
			fail_context["message"] = str(message)
		if isinstance(extra, dict) and extra:
			fail_context["extra"] = extra

	layout_info = info.get("layout_info", [])
	scene_name = info.get("scene_name", glb_name)
	layout_objects = convert_layout_to_objects(layout_info)
	if len(layout_objects) < MIN_OBJECTS:
		print(f"[DEBUG] skip {glb_name}: layout_objects={len(layout_objects)} < MIN_OBJECTS={MIN_OBJECTS}")
		_mark_fail(
			"layout_objects_too_few",
			stage="precheck_layout",
			extra={"layout_objects": int(len(layout_objects)), "min_required": int(MIN_OBJECTS)},
		)
		return None

	glb_path = glb_dir / glb_name
	if not glb_path.exists():
		print(f"[DEBUG] skip {glb_name}: glb file not exists")
		_mark_fail("glb_not_found", stage="precheck_glb", message=str(glb_path))
		return None

	bpy.ops.wm.read_factory_settings(use_empty=True)
	bpy.ops.import_scene.gltf(filepath=str(glb_path))
	groups = collect_instance_groups(max_instance_idx=len(layout_objects))
	logical = []
	for obj in layout_objects:
		g = groups.get(obj["instance_index"])
		if g:
			logical.append(
				{
					"id": obj["id"],
					"instance_index": obj["instance_index"],
					"category": str(obj.get("category", "unknown")),
					"group": g,
				}
			)
	if len(logical) < MIN_OBJECTS:
		print(f"[DEBUG] skip {glb_name}: logical={len(logical)} < MIN_OBJECTS={MIN_OBJECTS}")
		_mark_fail(
			"logical_objects_too_few",
			stage="precheck_logical",
			extra={"logical_objects": int(len(logical)), "min_required": int(MIN_OBJECTS)},
		)
		return None

	set_render_and_world()
	bounds = _scene_bounds_from_groups(groups)
	scene_center = Vector((bounds["center_x"], bounds["center_y"], bounds["center_z"]))
	scene_radius = max(bounds["width"], bounds["depth"], 1.0)
	setup_lighting(scene_center, scene_radius)
	wall_path = build_wall_path(glb_name, layout_root)
	wall_bounds = _import_wall_bounds(wall_path)
	wall_components = _import_wall_components(wall_path)
	if USE_WALL_PROXY_RENDER:
		proxies = _import_wall_proxy(wall_path)
		if proxies:
			print(f"[WALL] imported proxy walls: {len(proxies)} from {wall_path}")
			_hide_original_wall_structures(wall_bounds)
	_cap_wall_height(wall_bounds, top_margin=WALL_TOP_MARGIN)

	render_ctx = dc_utils.compute_render_frame(_all_mesh_objects())
	cam_top, cam_top_data = create_preview_camera("TopCam")
	cam_iso, cam_iso_data = create_preview_camera("IsoCam")
	north_world_dir = dc_utils.canonical_north_world(render_ctx)
	_set_move_basis_from_north(north_world_dir)
	dyn_w_top, dyn_h_top = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)
	bpy.context.scene.render.resolution_x = dyn_w_top
	bpy.context.scene.render.resolution_y = dyn_h_top
	setup_camera_for_mode(cam_top, cam_top_data, render_ctx, "top")
	setup_camera_for_mode(cam_iso, cam_iso_data, render_ctx, FIXED_ISOMETRIC_MODE)
	dyn_w_iso_pre, dyn_h_iso_pre = dc_utils.dynamic_resolution_for_mode(
		render_ctx, FIXED_ISOMETRIC_MODE, BASE_LONG_EDGE, MIN_SHORT_EDGE
	)
	bpy.context.view_layer.update()

	id_to_entry = {e["id"]: e for e in logical}
	valid_ids = [e["id"] for e in logical]
	rw_top = int(bpy.context.scene.render.resolution_x)
	rh_top = int(bpy.context.scene.render.resolution_y)
	if ENABLE_TOP_VISIBILITY_PREFILTER:
		valid_ids = [oid for oid in valid_ids if _top_group_visibility_ok(id_to_entry[oid]["group"], cam_top, rw_top, rh_top)]
		if len(valid_ids) < MIN_OBJECTS:
			try:
				debug_rows = []
				for oid in [e["id"] for e in logical]:
					g = id_to_entry[oid]["group"]
					top_ok = _top_group_visibility_ok(g, cam_top, rw_top, rh_top)
					area_px, area_ratio = _project_group_bbox_metrics(g, cam_top, rw_top, rh_top)
					center_vis = _is_group_center_visible(g, cam_top)
					fail_flags = []
					if area_px < TOP_MIN_BBOX_PX:
						fail_flags.append("small_bbox")
					if area_ratio < TOP_MIN_AREA_RATIO:
						fail_flags.append("small_area_ratio")
					if not center_vis:
						fail_flags.append("center_occluded")
					debug_rows.append(
						{
							"oid": oid,
							"cat": str(id_to_entry[oid].get("category", "unknown")),
							"area_px": float(area_px),
							"area_ratio": float(area_ratio),
							"center_vis": bool(center_vis),
							"top_ok": bool(top_ok),
							"fail": ",".join(fail_flags) if fail_flags else "pass",
						}
					)
				debug_rows.sort(key=lambda r: r["area_px"], reverse=True)
				manual_valid_ids = [e["id"] for e in logical if _top_group_visibility_ok(e["group"], cam_top, rw_top, rh_top)]
				print(f"[DEBUG][VIS] {glb_name}: top view visibility details (rw={rw_top}, rh={rh_top})")
				print(f"[DEBUG][VIS] valid_ids_after_filter(dict_group)={valid_ids}")
				print(f"[DEBUG][VIS] valid_ids_manual(direct_group)={manual_valid_ids}")
				for r in debug_rows:
					print(
						f"[DEBUG][VIS] oid={r['oid']} cat={r['cat']} "
						f"area_px={r['area_px']:.1f} area_ratio={r['area_ratio']:.5f} "
						f"center_vis={r['center_vis']} top_ok={r['top_ok']} fail={r['fail']}"
					)
			except Exception:
				pass
			print(f"[DEBUG] skip {glb_name}: valid_ids(visibility)={len(valid_ids)} < MIN_OBJECTS={MIN_OBJECTS}")
			_mark_fail(
				"visible_objects_too_few",
				stage="visibility_filter",
				extra={"visible_objects": int(len(valid_ids)), "min_required": int(MIN_OBJECTS)},
			)
			return None
	# Build full metas on all visible objects first.
	full_label_map = assign_independent_labels([{"id": oid} for oid in valid_ids])
	full_label_to_oid = {lb: oid for oid, lb in full_label_map.items()}
	full_metas_by_label = {}
	for oid in valid_ids:
		e = id_to_entry[oid]
		lb = full_label_map[oid]
		full_metas_by_label[lb] = _build_meta_from_group(lb, oid, e["instance_index"], e["group"])
	full_labels = sorted(full_metas_by_label.keys())
	full_pivot_by_label: dict[int, Vector] = {}
	for lb in full_labels:
		oid = full_label_to_oid.get(lb)
		if oid is None:
			continue
		full_pivot_by_label[lb] = _group_centroid_world(id_to_entry[oid]["group"])
	effective_case_id = int(sample_idx % len(CASE_CYCLE)) if case_id is None else int(case_id % len(CASE_CYCLE))
	target_issue, target_action_kind = _target_case_by_id(effective_case_id)
	case_no = int(effective_case_id + 1)

	stem = Path(glb_name).stem
	case_tag = f"case_{case_no}_{target_issue}_{target_action_kind}"
	sample_dir = output_dir / stem / case_tag
	ref_dir = sample_dir / "ref_images"

	anomaly_fail_ctx: dict = {}
	anomaly = _make_synthetic_anomaly(
		glb_name=glb_name,
		labels=full_labels,
		metas_by_label=full_metas_by_label,
		label_to_oid=full_label_to_oid,
		id_to_entry=id_to_entry,
		sample_idx=effective_case_id,
		whitelist_pairs=whitelist_pairs,
		wall_bounds=wall_bounds,
		wall_components=wall_components,
		cam_top=cam_top,
		cam_iso=cam_iso,
		render_ctx=render_ctx,
		iso_vis_size=(int(dyn_w_iso_pre), int(dyn_h_iso_pre)),
		pivot_by_label=full_pivot_by_label,
		fail_context=anomaly_fail_ctx,
		excluded_inject_labels=_excluded_inject_labels,
		excluded_inject_action_texts=_excluded_inject_action_texts,
	)
	if anomaly is None:
		print(f"[DEBUG] skip {glb_name}: _make_synthetic_anomaly returned None (target_issue={target_issue})")
		_mark_fail(
			str(anomaly_fail_ctx.get("reason", "anomaly_generation_failed")),
			stage=str(anomaly_fail_ctx.get("stage", "make_synthetic_anomaly")),
			message=str(anomaly_fail_ctx.get("message", "")),
			extra=anomaly_fail_ctx.get("extra") if isinstance(anomaly_fail_ctx.get("extra"), dict) else None,
		)
		return None

	# Keep final labeled objects <= 5, but force-include anomaly-related objects.
	issue_meta_cand, inject_action_cand = anomaly
	forced_cand_labels: set[int] = set()
	try:
		forced_cand_labels.add(int(inject_action_cand.get("id")))
	except Exception:
		pass
	issue0 = (issue_meta_cand.get("issues") or [{}])[0]
	for x in issue0.get("object_labels", []):
		sx = str(x).strip()
		if sx.isdigit():
			forced_cand_labels.add(int(sx))

	selected_full_labels = _select_operable_labels_with_forced(
		full_labels,
		full_metas_by_label,
		sorted(forced_cand_labels),
		max_count=MAX_LABELED_OBJECTS,
	)
	selected_oids = [oid for oid in valid_ids if full_label_map[oid] in set(selected_full_labels)]

	# Re-index selected objects from 1..K for clearer annotation/QA actions.
	label_map = assign_independent_labels([{"id": oid} for oid in selected_oids])
	label_to_oid = {lb: oid for oid, lb in label_map.items()}
	oid_to_label = {oid: lb for oid, lb in label_map.items()}
	label_category_map = {lb: str(id_to_entry[oid].get("category", "unknown")) for oid, lb in label_map.items()}
	metas_by_label = {}
	for oid in selected_oids:
		e = id_to_entry[oid]
		lb = label_map[oid]
		metas_by_label[lb] = _build_meta_from_group(lb, oid, e["instance_index"], e["group"])
	labels = sorted(metas_by_label.keys())
	pivot_by_label: dict[int, Vector] = {}
	for lb in labels:
		oid = label_to_oid.get(lb)
		if oid is None:
			continue
		pivot_by_label[lb] = _group_centroid_world(id_to_entry[oid]["group"])

	issue_meta, inject_action = _remap_issue_and_action_labels(
		issue_meta_cand,
		inject_action_cand,
		full_label_to_oid,
		oid_to_label,
	)
	if issue_meta is None or inject_action is None:
		print(f"[DEBUG] skip {glb_name}: anomaly remap failed")
		_mark_fail("anomaly_remap_failed", stage="remap_labels")
		return None

	created_now = not sample_dir.exists()
	if created_now:
		sample_dir.mkdir(parents=True, exist_ok=True)
	ref_dir.mkdir(parents=True, exist_ok=True)
	for stale in [sample_dir / "top_error.png", sample_dir / "isometric_error.png"]:
		if stale.exists():
			stale.unlink()
	for fp in ref_dir.glob("qa3_option_*.png"):
		fp.unlink()

	group, orig_inject = _apply_action(inject_action, label_to_oid, id_to_entry, pivot_by_label=pivot_by_label)
	try:
		# IMPORTANT:
		# Option validation/rendering happens on the injected (error) state.
		# Rotate/scale options must use the *current* object centers as pivots,
		# not the pre-injection centers, otherwise options can look like they
		# rotate/scale around a stale point and cause visible translation drift.
		pivot_by_label_post_inject: dict[int, Vector] = {}
		for lb in labels:
			oid = label_to_oid.get(lb)
			if oid is None:
				continue
			pivot_by_label_post_inject[lb] = _group_centroid_world(id_to_entry[oid]["group"])

		# Debug-only round-trip check for case_2_overlap_rotate:
		# from injected state -> reverse -> forward, then compare matrix delta.
		if int(effective_case_id) == 1 and str(inject_action.get("op", "")) == "rotate":
			state_injected = {obj.name: obj.matrix_world.copy() for obj in group}
			try:
				rev_action = _reverse_action(inject_action)
				_apply_action(rev_action, label_to_oid, id_to_entry, pivot_by_label=pivot_by_label)
				_apply_action(inject_action, label_to_oid, id_to_entry, pivot_by_label=pivot_by_label)
				max_m = 0.0
				max_t = 0.0
				for obj in group:
					m_ref = state_injected.get(obj.name)
					if m_ref is None:
						continue
					diff_m = obj.matrix_world - m_ref
					# Matrix max-abs delta
					for r in range(4):
						for c in range(4):
							v = abs(float(diff_m[r][c]))
							if v > max_m:
								max_m = v

					dt = (obj.matrix_world.translation - m_ref.translation).length
					if float(dt) > max_t:
						max_t = float(dt)
				print(
					f"[DEBUG_RT] {glb_name} case=2 rotate roundtrip "
					f"max_matrix_abs_delta={max_m:.10f}, max_translation_delta={max_t:.10f}"
				)
			finally:
				_restore_group(group, state_injected)

		# Injected-issue consistency check (post-action, pre-render).
		# Keep this always-on for orientation cases to avoid mixed-type contamination
		# such as "orientation" samples that already conflict with wall.
		if ENABLE_STRICT_INJECTED_ISSUE_CHECK or target_issue == ISSUE_ANGLE:
			issue0_chk = (issue_meta.get("issues") or [{}])[0]
			main_issue_type_chk = _issue_main_type(issue0_chk)
			main_issue_labels_chk = [int(x) for x in issue0_chk.get("object_labels", []) if str(x).isdigit()]

			def _is_nonwhitelist_collision_pair_chk(lb_a: int, lb_b: int) -> bool:
				oid_a = label_to_oid.get(lb_a)
				oid_b = label_to_oid.get(lb_b)
				if oid_a is None or oid_b is None:
					return False
				ca = str(id_to_entry[oid_a].get("category", "")).lower()
				cb = str(id_to_entry[oid_b].get("category", "")).lower()
				if _is_pair_whitelisted(ca, cb, whitelist_pairs):
					return False
				bvh_a = _build_group_bvh(id_to_entry[oid_a]["group"])
				bvh_b = _build_group_bvh(id_to_entry[oid_b]["group"])
				if bvh_a is None or bvh_b is None:
					return False
				return _bvh_overlap_strong(bvh_a, bvh_b)

			def _top_bbox_overlap_area_chk(lb_a: int, lb_b: int) -> float:
				oid_a = label_to_oid.get(lb_a)
				oid_b = label_to_oid.get(lb_b)
				if oid_a is None or oid_b is None:
					return 0.0
				dyn_w, dyn_h = dc_utils.dynamic_resolution_for_mode(render_ctx, "top", BASE_LONG_EDGE, MIN_SHORT_EDGE)
				r1 = _project_group_bbox_rect(id_to_entry[oid_a]["group"], cam_top, dyn_w, dyn_h)
				r2 = _project_group_bbox_rect(id_to_entry[oid_b]["group"], cam_top, dyn_w, dyn_h)
				if r1 is None or r2 is None:
					return 0.0
				ix = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
				iy = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
				return ix * iy

			def _has_wall_conflict_for_label_chk(lb: int) -> bool:
				cur_metas = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
				hit, _partner, _sub = _is_wall_touch_or_conflict_with_ctx(
					cur_metas,
					lb,
					label_to_oid=label_to_oid,
					id_to_entry=id_to_entry,
					wall_bounds=wall_bounds,
					wall_components=wall_components,
					desired_subtype=None,
				)
				if bool(hit):
					return True

				m = cur_metas.get(int(lb))
				if m is None:
					return False
				oid_lb = label_to_oid.get(int(lb))
				group_objs = id_to_entry.get(oid_lb, {}).get("group") if oid_lb is not None else None
				if SHAPELY_AVAILABLE and wall_components:
					near = _nearest_wall_overlap_ratio_for_object(m, wall_components, group_objs=group_objs)
					if float(near.get("ratio", 0.0)) > 1e-6:
						return True
				if wall_bounds is None:
					return False
				cx, cy = m["obb"]["center"]
				hw = float(m["half_w"])
				hd = float(m["half_d"])
				wminx = float(wall_bounds["min_x"])
				wmaxx = float(wall_bounds["max_x"])
				wminy = float(wall_bounds["min_y"])
				wmaxy = float(wall_bounds["max_y"])
				over_left = max(0.0, wminx - (float(cx) - hw))
				over_right = max(0.0, (float(cx) + hw) - wmaxx)
				over_bottom = max(0.0, wminy - (float(cy) - hd))
				over_top = max(0.0, (float(cy) + hd) - wmaxy)
				return bool(max(over_left, over_right, over_bottom, over_top) > 1e-6)

			def _wall_conflict_visibly_strong_chk(lb: int) -> bool:
				cur_metas = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
				m = cur_metas.get(lb)
				if m is None:
					return False
				if not _has_wall_conflict_for_label_chk(lb):
					return False

				max_overflow_m = 0.0
				max_overflow_ratio = 0.0
				if wall_bounds is not None:
					cx, cy = m["obb"]["center"]
					hw = float(m["half_w"])
					hd = float(m["half_d"])
					wminx = float(wall_bounds["min_x"])
					wmaxx = float(wall_bounds["max_x"])
					wminy = float(wall_bounds["min_y"])
					wmaxy = float(wall_bounds["max_y"])
					over_left = max(0.0, wminx - (float(cx) - hw))
					over_right = max(0.0, (float(cx) + hw) - wmaxx)
					over_bottom = max(0.0, wminy - (float(cy) - hd))
					over_top = max(0.0, (float(cy) + hd) - wmaxy)
					max_overflow_m = max(over_left, over_right, over_bottom, over_top)
					wdepth_x = max(1e-6, wmaxx - wminx)
					wdepth_y = max(1e-6, wmaxy - wminy)
					max_overflow_ratio = max(
						over_left / wdepth_x,
						over_right / wdepth_x,
						over_bottom / wdepth_y,
						over_top / wdepth_y,
					)
				# Overflow is informative but should not be a hard blocker in post-check.

				penetration_ratio = 0.0
				if SHAPELY_AVAILABLE and wall_components:
					oid_lb = label_to_oid.get(lb)
					group_objs = id_to_entry.get(oid_lb, {}).get("group") if oid_lb is not None else None
					near = _nearest_wall_overlap_ratio_for_object(m, wall_components, group_objs=group_objs)
					penetration_ratio = float(near.get("ratio", 0.0))

					strong_pen = penetration_ratio >= float(WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO)
					return bool(strong_pen)

			def _orientation_ok_for_label_chk(lb: int) -> bool:
				cur_metas = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
				m = cur_metas.get(int(lb))
				if m is None:
					return False
				oid = label_to_oid.get(int(lb))
				if oid is None:
					return False
				cat = str(id_to_entry.get(oid, {}).get("category", "")).lower()
				group = id_to_entry.get(oid, {}).get("group", [])
				if _round_like_category(cat):
					return False
				if not group:
					return False
				hx = float(m["obb"].get("hx", m.get("half_w", 0.0)))
				hy = float(m["obb"].get("hy", m.get("half_d", 0.0)))
				if min(hx, hy) < 1e-4:
					return False
				ratio = max(hx, hy) / max(min(hx, hy), 1e-6)
				if ratio > float(ORIENTATION_MAX_ASPECT_RATIO):
					return False
				hz = float(m["obb"].get("hz", m.get("height", hx)))
				if (hx * hy * hz) < float(ORIENTATION_MIN_VOLUME_THRESHOLD):
					return False
				if wall_bounds is None:
					return False
				cx, cy = m["obb"]["center"]
				ds = {
					"W": float(cx - float(wall_bounds["min_x"])),
					"E": float(float(wall_bounds["max_x"]) - float(cx)),
					"S": float(cy - float(wall_bounds["min_y"])),
					"N": float(float(wall_bounds["max_y"]) - float(cy)),
				}
				nearest_side = min(ds.keys(), key=lambda k: ds[k])
				if float(ds[nearest_side]) > float(ORIENTATION_WALL_NEAR_DIST_M):
					return False
				target = (0.0, 1.0) if nearest_side in ("W", "E") else (1.0, 0.0)
				ax = m["obb"].get("ax", (1.0, 0.0))
				ay = m["obb"].get("ay", (0.0, 1.0))
				def _norm2(v):
					try:
						x = float(v[0]); y = float(v[1])
					except Exception:
						return (1.0, 0.0)
					n = math.hypot(x, y)
					if n <= 1e-9:
						return (1.0, 0.0)
					return (x / n, y / n)
				def _ang_err(a, b):
					ax0, ay0 = _norm2(a)
					bx0, by0 = _norm2(b)
					dot = max(-1.0, min(1.0, abs(ax0 * bx0 + ay0 * by0)))
					return math.degrees(math.acos(dot))
				err = min(_ang_err(ax, target), _ang_err(ay, target))
				return bool(err >= float(ORIENTATION_MIN_ANGLE_DELTA_DEG))

			def _label_has_overlap_chk(lb: int) -> bool:
				for other in labels:
					if int(other) == int(lb):
						continue
					if _is_nonwhitelist_collision_pair_chk(int(lb), int(other)):
						return True
				return False

			if main_issue_type_chk == ISSUE_OVERLAP:
				if len(main_issue_labels_chk) < 2:
					raise RuntimeError("Skip: invalid overlap labels after remap")
				a_lb, b_lb = int(main_issue_labels_chk[0]), int(main_issue_labels_chk[1])
				if not _is_nonwhitelist_collision_pair_chk(a_lb, b_lb):
					raise RuntimeError("Skip: overlap inconsistency after remap/injection")
				try:
					op_lb = int(inject_action.get("id"))
				except Exception:
					op_lb = -1
				if op_lb <= 0:
					raise RuntimeError("Skip: overlap case invalid action target label")
				if op_lb not in (a_lb, b_lb):
					raise RuntimeError("Skip: overlap remap inconsistency (action target not in overlap pair)")
				# Operated object cannot have any other error mode.
				if _has_wall_conflict_for_label_chk(op_lb):
					raise RuntimeError("Skip: overlap case introduces wall conflict on operated object")
				if _orientation_ok_for_label_chk(op_lb):
					raise RuntimeError("Skip: overlap case introduces orientation conflict on operated object")
				# Ensure top-view visual overlap is non-trivial, otherwise text/image mismatch is likely.
				if _top_bbox_overlap_area_chk(a_lb, b_lb) < 280.0:
					raise RuntimeError("Skip: overlap too weak in top view")
			elif main_issue_type_chk == ISSUE_WALL:
				if len(main_issue_labels_chk) < 1:
					raise RuntimeError("Skip: wall conflict inconsistency after remap/injection")
				main_lb = int(main_issue_labels_chk[0])
				try:
					op_lb = int(inject_action.get("id"))
				except Exception:
					op_lb = -1
				if op_lb <= 0 or op_lb != main_lb:
					raise RuntimeError("Skip: wall conflict invalid action target label")
				if not _has_wall_conflict_for_label_chk(main_lb):
					raise RuntimeError("Skip: wall conflict inconsistency after remap/injection")
				if not _wall_conflict_visibly_strong_chk(main_lb):
					raise RuntimeError("Skip: wall conflict too weak visually")
				if _label_has_overlap_chk(op_lb):
					raise RuntimeError("Skip: wall conflict case introduces overlap on operated object")
				if _orientation_ok_for_label_chk(op_lb):
					raise RuntimeError("Skip: wall conflict case introduces orientation conflict on operated object")
			elif main_issue_type_chk == ISSUE_ANGLE:
				if len(main_issue_labels_chk) < 1:
					raise RuntimeError("Skip: orientation inconsistency after remap/injection")
				main_lb = int(main_issue_labels_chk[0])
				try:
					op_lb = int(inject_action.get("id"))
				except Exception:
					op_lb = -1
				if op_lb <= 0 or op_lb != main_lb:
					raise RuntimeError("Skip: orientation invalid action target label")
				if not _orientation_ok_for_label_chk(main_lb):
					raise RuntimeError("Skip: orientation inconsistency after remap/injection")
				if _label_has_overlap_chk(op_lb):
					raise RuntimeError("Skip: orientation case introduces overlap on operated object")
				if _has_wall_conflict_for_label_chk(op_lb):
					raise RuntimeError("Skip: orientation case introduces wall conflict on operated object")

		top_abs = _capture_labeled(
			cam_top,
			render_ctx,
			"top",
			id_to_entry,
			label_map,
			sample_dir / ".top_error_raw.png",
			sample_dir / "top_error.png",
			wall_bounds=wall_bounds,
			wall_alpha=TOP_WALL_ALPHA,
			north_world_dir=north_world_dir,
		)
		print(f"[RENDER] Before isometric render: ISO_WALL_ALPHA={ISO_WALL_ALPHA}")
		bpy.context.view_layer.update()
		iso_mode = FIXED_ISOMETRIC_MODE
		issue = (issue_meta.get("issues") or [{}])[0]
		main_issue_type = _issue_main_type(issue)
		issue_subtype = str(issue.get("subtype", "")).strip()
		main_issue_labels = [int(x) for x in issue.get("object_labels", []) if str(x).isdigit()]
		setup_camera_for_mode(cam_iso, cam_iso_data, render_ctx, iso_mode)
		dyn_w_iso, dyn_h_iso = dc_utils.dynamic_resolution_for_mode(render_ctx, iso_mode, BASE_LONG_EDGE, MIN_SHORT_EDGE)
		bpy.context.scene.render.resolution_x = dyn_w_iso
		bpy.context.scene.render.resolution_y = dyn_h_iso
		iso_abs = _capture_labeled(
			cam_iso,
			render_ctx,
			iso_mode,
			id_to_entry,
			label_map,
			sample_dir / ".isometric_error_raw.png",
			sample_dir / "isometric_error.png",
			wall_bounds=wall_bounds,
			wall_alpha=ISO_WALL_ALPHA,
			north_world_dir=north_world_dir,
		)
		correct_action_text = _action_text(_reverse_action(inject_action))

		def _is_nonwhitelist_collision_pair(lb_a: int, lb_b: int) -> bool:
			oid_a = label_to_oid.get(lb_a)
			oid_b = label_to_oid.get(lb_b)
			if oid_a is None or oid_b is None:
				return False
			ca = str(id_to_entry[oid_a].get("category", "")).lower()
			cb = str(id_to_entry[oid_b].get("category", "")).lower()
			if _is_pair_whitelisted(ca, cb, whitelist_pairs):
				return False
			bvh_a = _build_group_bvh(id_to_entry[oid_a]["group"])
			bvh_b = _build_group_bvh(id_to_entry[oid_b]["group"])
			if bvh_a is None or bvh_b is None:
				return False
			return _bvh_overlap_strong(bvh_a, bvh_b)

		def _has_nonwhitelist_collision_for_label(lb: int) -> bool:
			if lb not in label_to_oid:
				return False
			for other in labels:
				if other == lb:
					continue
				if _is_nonwhitelist_collision_pair(lb, other):
					return True
			return False

		def _has_wall_door_path_conflict_for_label(lb: int, include_touch: bool = True) -> bool:
			if lb not in label_to_oid:
				return False
			cur_metas = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
			m = cur_metas.get(lb)
			if m is None:
				return False
			cat_lb = ""
			oid_lb = label_to_oid.get(lb)
			if oid_lb is not None:
				cat_lb = str(id_to_entry.get(oid_lb, {}).get("category", ""))
			if any(k in cat_lb.lower() for k in WALL_CONFLICT_EXCLUDE_TARGET_KEYWORDS):
				return False
			# include_touch controls strictness:
			# - True: light wall contact is considered invalid (for stricter checks)
			# - False: only clear penetration is considered invalid (more tolerant for
			#   distractor filtering).
			min_ratio = (
				float(WALL_CONTACT_BLOCK_MIN_PENETRATION_RATIO)
				if include_touch
				else float(WALL_CONFLICT_VISIBLE_MIN_PENETRATION_RATIO)
			)
			if SHAPELY_AVAILABLE and wall_components:
				group_objs = id_to_entry.get(oid_lb, {}).get("group") if oid_lb is not None else None
				near = _nearest_wall_overlap_ratio_for_object(m, wall_components, group_objs=group_objs)
				if float(near.get("ratio", 0.0)) >= min_ratio:
					return True
				if include_touch and float(near.get("distance", float("inf"))) <= float(WALL_CONTACT_BLOCK_MAX_DISTANCE_M):
					return True
			elif include_touch and wall_bounds is not None:
				cx, cy = m["obb"]["center"]
				ratios = _wall_overlap_depth_ratios(cx, cy, m["half_w"], m["half_d"], wall_bounds)
				if float(ratios.get("max_ratio", 0.0)) >= min_ratio:
					return True
			return False

		def _any_nonwhitelist_collision_exists() -> bool:
			for i in range(len(labels)):
				for j in range(i + 1, len(labels)):
					if _is_nonwhitelist_collision_pair(labels[i], labels[j]):
						return True
			return False

		def _any_wall_door_path_conflict_exists(include_touch: bool = True) -> bool:
			for lb in labels:
				if _has_wall_door_path_conflict_for_label(lb, include_touch=include_touch):
					return True
			return False

		correct_action_obj = _reverse_action(inject_action)

		def _candidate_is_valid_distractor(txt: str) -> bool:
			try:
				act = _parse_action_text(txt)
			except Exception:
				return False
			try:
				target_lb = int(act.get("id", -1))
			except Exception:
				target_lb = -1
			if act.get("op") == "scale":
				# Keep wrong scale options in opposite direction from the correct scale direction.
				if bool(act.get("up", False)) == bool(correct_action_obj.get("up", False)):
					return False
			if act.get("op") == "move" and main_issue_type != ISSUE_WALL:
				if target_lb > 0:
					g_tmp, o_tmp = _apply_action(
						act,
						label_to_oid,
						id_to_entry,
						pivot_by_label=pivot_by_label_post_inject,
					)
					try:
						# Any wall contact or clear penetration invalidates the distractor.
						if _has_wall_door_path_conflict_for_label(target_lb, include_touch=True):
							return False
					finally:
						_restore_group(g_tmp, o_tmp)
			# Re-apply below for normal validation flow.
			g2, o2 = _apply_action(
				act,
				label_to_oid,
				id_to_entry,
				pivot_by_label=pivot_by_label_post_inject,
			)
			try:
				current_resolved = False
				target_orient_lb = -1
				ori_metas = None
				if main_issue_type == ISSUE_OVERLAP:
					if len(main_issue_labels) >= 2:
						a, b = int(main_issue_labels[0]), int(main_issue_labels[1])
						if target_lb == a:
							current_resolved = (not _is_nonwhitelist_collision_pair(a, b))
						elif target_lb == b:
							current_resolved = (not _is_nonwhitelist_collision_pair(b, a))
						else:
							current_resolved = False
					elif len(main_issue_labels) == 1:
						main_lb = int(main_issue_labels[0])
						if target_lb == main_lb:
							current_resolved = (not _has_nonwhitelist_collision_for_label(main_lb))
						else:
							current_resolved = False
					else:
						current_resolved = (not _any_nonwhitelist_collision_exists())
				elif main_issue_type == ISSUE_WALL:
					if len(main_issue_labels) >= 1:
						main_lb = int(main_issue_labels[0])
						if target_lb == main_lb:
							current_resolved = (not _has_wall_door_path_conflict_for_label(main_lb, include_touch=True))
						else:
							current_resolved = False
					else:
						current_resolved = (not _any_wall_door_path_conflict_exists(include_touch=False))
				elif main_issue_type == ISSUE_ANGLE:
					if len(main_issue_labels) >= 1:
						try:
							target_orient_lb = int(main_issue_labels[0])
						except Exception:
							target_orient_lb = -1
					if target_orient_lb > 0:
						ori_metas = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
						if target_lb == target_orient_lb:
							current_resolved = (not _orientation_ok_for_label_chk(target_orient_lb))
						else:
							current_resolved = False
					else:
						current_resolved = False
				else:
					return txt != correct_action_text

				# Wrong option is valid iff:
				# 1) it does NOT fix the current issue, OR
				# 2) it fixes current issue but introduces (or keeps) some other issue.
				if not current_resolved:
					return True

				if main_issue_type == ISSUE_ANGLE:
					if ori_metas is None:
						ori_metas = _rebuild_metas_by_label(labels, label_to_oid, id_to_entry)
					target_has_ori_issue = False
					if target_orient_lb > 0:
						target_has_ori_issue = _orientation_ok_for_label_chk(target_orient_lb)
					return (
						target_has_ori_issue
						or _has_nonwhitelist_collision_for_label(target_orient_lb)
						or _has_wall_door_path_conflict_for_label(target_orient_lb, include_touch=True)
					)

				if target_lb <= 0:
					return False
				has_any_issue = (
					_has_nonwhitelist_collision_for_label(target_lb)
					or _has_wall_door_path_conflict_for_label(target_lb, include_touch=True)
				)
				return has_any_issue
			finally:
				_restore_group(g2, o2)

		# Build options with validation: 1 correct + 3 verified wrong.
		option_texts = [correct_action_text]
		seen = {correct_action_text}
		attempts = 0
		while len(option_texts) < 4 and attempts < 320:
			attempts += 1
			cand = _sample_option_action_text(inject_action)
			if cand in seen:
				continue
			if not _candidate_is_valid_distractor(cand):
				continue
			option_texts.append(cand)
			seen.add(cand)
		if len(option_texts) < 4:
			# Strict mode: skip scene if we cannot find 3 valid distractors.
			raise RuntimeError("Failed to build 3 validated distractors for qa3")
		random.shuffle(option_texts)

		option_images = []
		for i, txt in enumerate(option_texts, start=1):
			action = _parse_action_text(txt)
			g2, o2 = _apply_action(
				action,
				label_to_oid,
				id_to_entry,
				pivot_by_label=pivot_by_label_post_inject,
			)
			try:
				img_abs = _capture_labeled(
					cam_top,
					render_ctx,
					"top",
					id_to_entry,
					label_map,
					ref_dir / f".qa3_option_{i}_raw.png",
					ref_dir / f"qa3_option_{i}.png",
					wall_bounds=wall_bounds,
					wall_alpha=TOP_WALL_ALPHA,
					north_world_dir=north_world_dir,
				)
				option_images.append(img_abs)
			finally:
				_restore_group(g2, o2)

		qa_top_1 = qa1_mcq_what_problem(issue_meta, images=[top_abs], label_category_map=label_category_map)
		qa_top_3 = qa3_fix(
			issue_meta,
			inject_action,
			images=[top_abs],
			option_images=option_images,
			option_texts=option_texts,
			label_category_map=label_category_map,
		)
		qa_top_iso = qa2_mcq_what_problem(issue_meta, images=[top_abs, iso_abs], label_category_map=label_category_map)
	except Exception as e:
		print(f"[DEBUG] skip {glb_name}: post-process exception {e}")
		retry_excluded_labels = set(int(x) for x in (_excluded_inject_labels or set()))
		retry_excluded_actions = set(str(x) for x in (_excluded_inject_action_texts or set()))
		try:
			op_lb_retry = int(inject_action.get("id"))
		except Exception:
			op_lb_retry = -1
		try:
			op_action_text_retry = _action_text(inject_action)
		except Exception:
			op_action_text_retry = ""
		can_retry = (
			_retry_attempt < int(POST_PROCESS_RETRY_MAX_ATTEMPTS_PER_CASE)
			and bool(op_action_text_retry)
			and op_action_text_retry not in retry_excluded_actions
		)
		if can_retry:
			retry_excluded_actions.add(op_action_text_retry)
			if len(retry_excluded_actions) < 64:
				if created_now:
					shutil.rmtree(sample_dir, ignore_errors=True)
				# Avoid restoring stale Blender object handles in outer finally
				# after recursive retry resets the scene.
				group = []
				orig_inject = {}
				print(
					f"[DEBUG] retry {glb_name} case={case_no}: "
					f"exclude action '{op_action_text_retry}', target label {op_lb_retry}, retry={_retry_attempt + 1}"
				)
				return process_scene(
					glb_name=glb_name,
					info=info,
					glb_dir=glb_dir,
					output_dir=output_dir,
					layout_root=layout_root,
					whitelist_pairs=whitelist_pairs,
					sample_idx=sample_idx,
					case_id=case_id,
					fail_context=fail_context,
					_excluded_inject_labels=retry_excluded_labels,
					_excluded_inject_action_texts=retry_excluded_actions,
					_retry_attempt=_retry_attempt + 1,
				)
		_mark_fail("post_process_exception", stage="post_process", message=str(e))
		if created_now:
			shutil.rmtree(sample_dir, ignore_errors=True)
		return None
	finally:
		_restore_group(group, orig_inject)
		# Remove hidden raw intermediates.
		for p in [sample_dir / ".top_error_raw.png", sample_dir / ".isometric_error_raw.png"]:
			try:
				if p.exists():
					p.unlink()
			except Exception:
				pass
		for p in ref_dir.glob(".qa3_option_*_raw.png"):
			try:
				p.unlink()
			except Exception:
				pass

	label_mapping = [
		{"scene_object_id": oid, "label_id": label_map[oid], "instance_index": id_to_entry[oid]["instance_index"]}
		for oid in sorted(selected_oids, key=lambda x: label_map[x])
	]

	return {
		"glb_name": glb_name,
		"scene_name": scene_name,
		"case_id": int(case_no),
		"case_tag": case_tag,
		"object_count": len(valid_ids),
		"labeled_object_count": len(selected_oids),
		"label_mapping": label_mapping,
		"issue_meta": issue_meta,
		"inject_action": inject_action,
		"images": {"top_error": top_abs, "isometric_error": iso_abs},
		"qa": {"top": [qa_top_1, qa_top_3], "top_isometric": [qa_top_iso]},
	}


def _aggregate_results_by_scene(records: list[dict]) -> list[dict]:
	"""Convert case-level records into scene-level records with QA merged by mode."""
	ordered_keys: list[str] = []
	by_scene: dict[str, dict] = {}
	for rec in records:
		gn = str(rec.get("glb_name", "")).strip()
		if not gn:
			continue
		# Already aggregated record: merge directly by scene.
		if "case_id" not in rec:
			if gn not in by_scene:
				ordered_keys.append(gn)
				by_scene[gn] = json.loads(json.dumps(rec, ensure_ascii=False))
			else:
				dst = by_scene[gn]
				dst.setdefault("qa", {}).setdefault("top", []).extend(list(rec.get("qa", {}).get("top", [])))
				dst.setdefault("qa", {}).setdefault("top_isometric", []).extend(list(rec.get("qa", {}).get("top_isometric", [])))
				dst.setdefault("cases", []).extend(list(rec.get("cases", [])))
			continue

		if gn not in by_scene:
			ordered_keys.append(gn)
			by_scene[gn] = {
				"glb_name": gn,
				"scene_name": rec.get("scene_name", gn),
				"object_count": int(rec.get("object_count", 0)),
				"qa": {"top": [], "top_isometric": []},
				"cases": [],
			}
		dst = by_scene[gn]
		dst["object_count"] = max(int(dst.get("object_count", 0)), int(rec.get("object_count", 0)))

		case_info = {
			"case_id": int(rec.get("case_id", 0)),
			"case_tag": str(rec.get("case_tag", "")),
			"labeled_object_count": int(rec.get("labeled_object_count", 0)),
			"label_mapping": rec.get("label_mapping", []),
			"issue_meta": rec.get("issue_meta", {}),
			"inject_action": rec.get("inject_action", {}),
			"images": rec.get("images", {}),
		}
		dst["cases"].append(case_info)

		for q in rec.get("qa", {}).get("top", []):
			qx = json.loads(json.dumps(q, ensure_ascii=False))
			qx["case_id"] = case_info["case_id"]
			qx["case_tag"] = case_info["case_tag"]
			dst["qa"]["top"].append(qx)
		for q in rec.get("qa", {}).get("top_isometric", []):
			qx = json.loads(json.dumps(q, ensure_ascii=False))
			qx["case_id"] = case_info["case_id"]
			qx["case_tag"] = case_info["case_tag"]
			dst["qa"]["top_isometric"].append(qx)

	for gn in ordered_keys:
		try:
			by_scene[gn]["cases"].sort(key=lambda x: int(x.get("case_id", 0)))
		except Exception:
			pass
	return [by_scene[k] for k in ordered_keys if k in by_scene]


def _collect_issue_type_counts(records: list[dict]) -> dict[str, int]:
	out: dict[str, int] = {}
	for rec in records:
		if "issue_meta" in rec:
			issue = (rec.get("issue_meta", {}).get("issues") or [{}])[0]
			t = str(issue.get("type", "unknown"))
			out[t] = out.get(t, 0) + 1
			continue
		for c in rec.get("cases", []):
			issue = (c.get("issue_meta", {}).get("issues") or [{}])[0]
			t = str(issue.get("type", "unknown"))
			out[t] = out.get(t, 0) + 1
	return out


def _build_failure_record(
	glb_name: str,
	case_id: int,
	stage: str,
	reason: str,
	message: str = "",
	extra: dict | None = None,
	exception: str = "",
) -> dict:
	target_issue, target_action_kind = _target_case_by_id(int(case_id))
	case_no = int(case_id) + 1
	rec = {
		"glb_name": str(glb_name),
		"case_id": int(case_no),
		"case_tag": f"case_{case_no}_{target_issue}_{target_action_kind}",
		"target_issue": str(target_issue),
		"target_action_kind": str(target_action_kind),
		"stage": str(stage),
		"reason": str(reason),
	}
	if message:
		rec["message"] = str(message)
	if exception:
		rec["exception"] = str(exception)
	if isinstance(extra, dict) and extra:
		rec["extra"] = extra
	return rec


def _build_failure_stats_payload(
	failure_records: list[dict],
	total_case_attempts: int,
	total_scenes_processed: int,
	mode_name: str,
) -> dict:
	by_reason: dict[str, int] = {}
	by_case: dict[str, dict] = {}
	by_scene: dict[str, int] = {}
	for rec in failure_records:
		rs = str(rec.get("reason", "unknown"))
		by_reason[rs] = by_reason.get(rs, 0) + 1
		case_tag = str(rec.get("case_tag", "unknown_case"))
		if case_tag not in by_case:
			by_case[case_tag] = {"total": 0, "by_reason": {}}
		by_case[case_tag]["total"] = int(by_case[case_tag]["total"]) + 1
		cr = by_case[case_tag]["by_reason"]
		cr[rs] = int(cr.get(rs, 0)) + 1
		gn = str(rec.get("glb_name", ""))
		if gn:
			by_scene[gn] = by_scene.get(gn, 0) + 1
	return {
		"summary": {
			"mode": str(mode_name),
			"total_case_attempts": int(total_case_attempts),
			"total_scenes_processed": int(total_scenes_processed),
			"failed_case_attempts": int(len(failure_records)),
			"success_case_attempts": int(max(0, int(total_case_attempts) - int(len(failure_records)))),
		},
		"by_reason": by_reason,
		"by_case": by_case,
		"by_scene": by_scene,
		"records": failure_records,
	}


def _try_auto_merge_worker_metadata(output_dir: Path, workers: int, all_cases: bool = False) -> tuple[Path | None, bool]:
	if workers <= 1:
		return None, False
	parts = [output_dir / f"metadata_indoor.worker_{i}_of_{workers}.json" for i in range(workers)]
	if not all(p.exists() for p in parts):
		return None, False
	merged: list[dict] = []
	for p in parts:
		try:
			with open(p, "r", encoding="utf-8") as f:
				data = json.load(f)
		except Exception:
			return None, False
		if isinstance(data, list):
			merged.extend(data)
	if all_cases:
		merged_out = _aggregate_results_by_scene(merged)
	else:
		# Deduplicate by (glb_name, case_id) while preserving first-seen order.
		ordered_keys: list[str] = []
		by_key: dict[str, dict] = {}
		for rec in merged:
			gn = str(rec.get("glb_name", ""))
			if not gn:
				continue
			k = f"{gn}::case_{int(rec.get('case_id', -1))}"
			if k not in by_key:
				ordered_keys.append(k)
			by_key[k] = rec
		merged_out = [by_key[k] for k in ordered_keys if k in by_key]
	merged_json = output_dir / "metadata_indoor.json"
	with open(merged_json, "w", encoding="utf-8") as f:
		json.dump(merged_out, f, ensure_ascii=False, indent=2)
	# Delete worker files after merging
	for p in parts:
		try:
			p.unlink()
		except Exception:
			pass
	return merged_json, True


def _cleanup_empty_scene_dir(output_dir: Path, glb_name: str):
	"""Remove per-scene directory if it contains no files (only empty dirs)."""
	scene_dir = output_dir / Path(glb_name).stem
	if not scene_dir.exists() or not scene_dir.is_dir():
		return
	try:
		has_files = any(p.is_file() for p in scene_dir.rglob("*"))
		if not has_files:
			shutil.rmtree(scene_dir, ignore_errors=True)
	except Exception:
		pass


def main():
	args = parse_args()
	random.seed(args.seed)
	selected_case_ids = _parse_case_ids_arg(args.case_ids)

	glb_dir = args.glb_dir.resolve()
	mapping_json = args.mapping_json.resolve()
	layout_root = args.layout_root.resolve()
	whitelist_json = args.whitelist_json.resolve()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not glb_dir.exists():
		raise FileNotFoundError(f"GLB dir not found: {glb_dir}")
	# if not mapping_json.exists():
	# 	fallback = DEFAULT_FALLBACK_MAPPING_JSON.resolve()
	# 	if fallback.exists():
	# 		print(f"[WARN] Mapping json not found: {mapping_json}, fallback to {fallback}")
	# 		mapping_json = fallback
	# 	else:
	# 		raise FileNotFoundError(f"Mapping json not found: {mapping_json}")

	with open(mapping_json, "r", encoding="utf-8") as f:
		scene_mapping = json.load(f)
	whitelist_pairs = load_whitelist_pairs(whitelist_json)
	# Respect whitelist file as the single source of truth (no auto-augmentation).

	glb_files = sorted([p.name for p in glb_dir.glob("*.glb") if p.is_file()])
	# Only keep exact key matches to avoid mixing clean/original scene keys.
	matched_pairs: list[tuple[str, str]] = []
	for k in glb_files:
		if k in scene_mapping:
			matched_pairs.append((k, k))
	scene_keys = [k for k, _ in matched_pairs]
	map_keys = {k: mk for k, mk in matched_pairs}
	if args.region:
		kw = args.region.strip().lower()
		scene_keys = [
			k for k in scene_keys
			if kw in k.lower() or kw in str(scene_mapping.get(map_keys[k], {}).get("scene_name", "")).lower()
		]
	if args.max_scenes > 0:
		scene_keys = scene_keys[: args.max_scenes]
	indexed_scene_keys = list(enumerate(scene_keys))

	# Parent parallel mode: spawn worker processes and merge outputs.
	if args.workers > 1 and args.worker_index < 0:
		script_path = str(Path(__file__).resolve())
		base_args = []
		if args.max_scenes > 0:
			base_args += ["--max-scenes", str(args.max_scenes)]
		if args.region:
			base_args += ["--region", args.region]
		if selected_case_ids is not None:
			base_args += ["--case-ids", ",".join(str(int(x) + 1) for x in selected_case_ids)]
		base_args += [
			"--glb-dir", str(DEFAULT_GLB_DIR),
			"--mapping-json", str(DEFAULT_MAPPING_JSON),
			"--layout-root", str(DEFAULT_LAYOUT_ROOT),
			"--whitelist-json", str(DEFAULT_WHITELIST_JSON),
			"--output-dir", str(output_dir),
			"--seed", str(args.seed),
			"--workers", str(args.workers),
		]
		if args.all_cases:
			base_args += ["--all-cases"]
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

		all_results = []
		all_failure_records: list[dict] = []
		for wi in range(args.workers):
			worker_json = output_dir / f"metadata_indoor.worker_{wi}_of_{args.workers}.json"
			if worker_json.exists():
				with open(worker_json, "r", encoding="utf-8") as f:
					worker_data = json.load(f)
					all_results.extend(worker_data)
				print(f"Loaded {len(worker_data)} results from worker {wi}")
			else:
				print(f"Warning: worker {wi} output not found: {worker_json}")
			worker_fail_json = output_dir / f"failure_stats_indoor.worker_{wi}_of_{args.workers}.json"
			if worker_fail_json.exists():
				try:
					with open(worker_fail_json, "r", encoding="utf-8") as f:
						worker_fail_payload = json.load(f)
					for rec in worker_fail_payload.get("records", []):
						if isinstance(rec, dict):
							all_failure_records.append(rec)
				except Exception:
					pass
		out_records = _aggregate_results_by_scene(all_results) if args.all_cases else all_results
		out_json = output_dir / "metadata_indoor.json"
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(out_records, f, ensure_ascii=False, indent=2)
		issue_mode_counts = _collect_issue_type_counts(out_records if args.all_cases else all_results)
		total_scenes_processed = len(scene_keys)
		case_attempts_per_scene = (
			len(selected_case_ids)
			if selected_case_ids is not None
			else (len(CASE_CYCLE) if args.all_cases else 1)
		)
		total_case_attempts = total_scenes_processed * case_attempts_per_scene
		mode_name = "all-cases" if args.all_cases else "single-case"
		fail_payload = _build_failure_stats_payload(
			failure_records=all_failure_records,
			total_case_attempts=total_case_attempts,
			total_scenes_processed=total_scenes_processed,
			mode_name=mode_name,
		)
		fail_json = output_dir / "failure_stats_indoor.json"
		with open(fail_json, "w", encoding="utf-8") as f:
			json.dump(fail_payload, f, ensure_ascii=False, indent=2)

		for wi in range(args.workers):
			worker_json = output_dir / f"metadata_indoor.worker_{wi}_of_{args.workers}.json"
			try:
				if worker_json.exists():
					worker_json.unlink()
			except Exception:
				pass
			worker_fail_json = output_dir / f"failure_stats_indoor.worker_{wi}_of_{args.workers}.json"
			try:
				if worker_fail_json.exists():
					worker_fail_json.unlink()
			except Exception:
				pass
		print("=" * 80)
		print(f"Parallel done. valid_records={len(out_records)}")
		print(f"Output: {out_json}")
		print(f"Failure stats: {fail_json}")
		print("Merged issue counts by type:")
		for k in sorted(issue_mode_counts.keys()):
			print(f"  {k}: {issue_mode_counts[k]}")
		return

	# Child worker mode: process assigned scenes
	if args.workers > 1 and args.worker_index >= 0:
		indexed_scene_keys = [x for x in indexed_scene_keys if (x[0] % args.workers) == args.worker_index]
		scene_keys = [k for _, k in indexed_scene_keys]
		print(f"[Worker {args.worker_index}/{args.workers}] Processing {len(scene_keys)} scenes")

	print("=" * 80)
	print("Indoor error-mode QA construct")
	print(f"Total glb files: {len(glb_files)}")
	print(f"Matched in mapping: {len(scene_keys)}")
	if args.region:
		print(f"Region filter: {args.region}")
	if selected_case_ids is not None:
		print(f"Case filter: {[int(x) + 1 for x in selected_case_ids]}")

	all_results = []
	failure_records: list[dict] = []
	for local_idx, (global_idx, glb_name) in enumerate(indexed_scene_keys, start=1):
		if selected_case_ids is not None:
			case_iter = selected_case_ids
		else:
			case_iter = range(len(CASE_CYCLE)) if args.all_cases else [int(global_idx % len(CASE_CYCLE))]
		scene_success = 0
		for case_id in case_iter:
			case_no = int(case_id) + 1
			fail_ctx: dict = {}
			try:
				res = process_scene(
					glb_name=glb_name,
					info=scene_mapping[map_keys[glb_name]],
					glb_dir=glb_dir,
					output_dir=output_dir,
					layout_root=layout_root,
					whitelist_pairs=whitelist_pairs,
					sample_idx=global_idx,
					case_id=case_id,
					fail_context=fail_ctx,
				)
			except Exception as e:
				print(f"[{local_idx}/{len(indexed_scene_keys)}] skip {glb_name} case={case_no}: exception {e}")
				failure_records.append(
					_build_failure_record(
						glb_name=glb_name,
						case_id=int(case_id),
						stage="main_loop",
						reason="process_scene_exception",
						message="process_scene raised exception",
						exception=str(e),
					)
				)
				continue
			if not res:
				print(f"[{local_idx}/{len(indexed_scene_keys)}] skip {glb_name} case={case_no}: invalid case")
				failure_records.append(
					_build_failure_record(
						glb_name=glb_name,
						case_id=int(case_id),
						stage=str(fail_ctx.get("stage", "main_loop")),
						reason=str(fail_ctx.get("reason", "invalid_case")),
						message=str(fail_ctx.get("message", "")),
						extra=fail_ctx.get("extra") if isinstance(fail_ctx.get("extra"), dict) else None,
					)
				)
				continue
			all_results.append(res)
			scene_success += 1
			print(
				f"[{local_idx}/{len(indexed_scene_keys)}] done {glb_name} case={case_no}: "
				f"qa_top={len(res['qa']['top'])} qa_top_iso={len(res['qa']['top_isometric'])}"
			)
		if scene_success == 0:
			_cleanup_empty_scene_dir(output_dir, glb_name)
			if args.all_cases:
				print(f"[{local_idx}/{len(indexed_scene_keys)}] skip {glb_name}: all 7 cases failed")
			else:
				print(f"[{local_idx}/{len(indexed_scene_keys)}] skip {glb_name}: single case failed")

	output_records = _aggregate_results_by_scene(all_results) if args.all_cases else all_results
	if args.workers > 1 and args.worker_index >= 0:
		out_json = output_dir / f"metadata_indoor.worker_{args.worker_index}_of_{args.workers}.json"
		fail_json = output_dir / f"failure_stats_indoor.worker_{args.worker_index}_of_{args.workers}.json"
	else:
		out_json = output_dir / "metadata_indoor.json"
		fail_json = output_dir / "failure_stats_indoor.json"
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(output_records, f, ensure_ascii=False, indent=2)

	qa_type_counts: dict[str, int] = {}
	for rec in all_results:
		for q in rec.get("qa", {}).get("top", []):
			t = str(q.get("task_type", "unknown"))
			qa_type_counts[t] = qa_type_counts.get(t, 0) + 1
		for q in rec.get("qa", {}).get("top_isometric", []):
			t = str(q.get("task_type", "unknown"))
			qa_type_counts[t] = qa_type_counts.get(t, 0) + 1
	issue_mode_counts = _collect_issue_type_counts(output_records if args.all_cases else all_results)

	print("=" * 80)
	total_scenes_processed = len(indexed_scene_keys)
	case_attempts_per_scene = (
		len(selected_case_ids)
		if selected_case_ids is not None
		else (len(CASE_CYCLE) if args.all_cases else 1)
	)
	total_case_attempts = total_scenes_processed * case_attempts_per_scene
	mode_name = "all-cases" if args.all_cases else "single-case"
	fail_payload = _build_failure_stats_payload(
		failure_records=failure_records,
		total_case_attempts=total_case_attempts,
		total_scenes_processed=total_scenes_processed,
		mode_name=mode_name,
	)
	with open(fail_json, "w", encoding="utf-8") as f:
		json.dump(fail_payload, f, ensure_ascii=False, indent=2)
	valid_cases = len(all_results) if (not args.all_cases) else sum(len(x.get("cases", [])) for x in output_records)
	print(f"Done. valid_cases={valid_cases}/{total_case_attempts} (scenes={total_scenes_processed}, mode={mode_name})")
	print(f"Output: {out_json}")
	print(f"Failure stats: {fail_json}")
	print("QA counts by task_type:")
	for k in sorted(qa_type_counts.keys()):
		print(f"  {k}: {qa_type_counts[k]}")
	print("Issue counts (concise):")
	for k in sorted(issue_mode_counts.keys()):
		print(f"  {k}: {issue_mode_counts[k]}")


if __name__ == "__main__":
	main()
