#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
	from PIL import Image, ImageDraw, ImageFont
	PIL_AVAILABLE = True
except ImportError:
	PIL_AVAILABLE = False


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path.home() / "SpatialAct")))
INTERNSCENES_ROOT = Path(os.environ.get("INTERNSCENES_ROOT", str(Path.home() / "InternScenes")))
DEFAULT_GLB_DIR = INTERNSCENES_ROOT / "scenes/glb_files_wall_complex-10-15_clean_keep"
DEFAULT_IMAGES_DIR = INTERNSCENES_ROOT / "scenes/images"
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark/data/spatial_orientation/indoor_scenes_complex-10-15"
TMP_ROOT_DIR = PROJECT_ROOT / "tmp" / Path(__file__).stem

BLENDER_BIN = PROJECT_ROOT / "blender-3.2.2-linux-x64/blender"
RENDER_SCRIPT = PROJECT_ROOT / "benchmark/data_construct/utils.py"
SHARED_UTILS_PATH = PROJECT_ROOT / "benchmark/data_construct/utils.py"

_utils_spec = importlib.util.spec_from_file_location("dc_render_utils_task2", SHARED_UTILS_PATH)
if _utils_spec is None or _utils_spec.loader is None:
	raise RuntimeError(f"Cannot load shared utils from {SHARED_UTILS_PATH}")
dc_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(dc_utils)

EXTREME_MARGIN_WORLD = 0.2
TASK1_WORLD_ANGLE_MARGIN_DEG = 8.0
TASK4_ANGLE_MARGIN_DEG = 8.0
TASK_MIN_WORLD_RADIUS = 0.25
QA2_MIN_LABELS = 4
QA2_MAX_LABELS = 5
QA2_MIN_LABEL_DISTANCE_PX = 80.0
FORCE_CANONICAL_VIEW_RENDER = True
KEEP_SCENE_VIEWS = False
LABEL_RADIUS = 13
LABEL_FONT_SIZE = 16
TARGET_A_RADIUS = 15
TARGET_A_FONT_SIZE = 18
MIN_VISIBLE_BBOX_AREA_PX = 800.0
MIN_VISIBLE_BBOX_AREA_RATIO = 0.0005
FIXED_ISOMETRIC_MODE = "isometric_north_ur"
TOP_WALL_ALPHA = 1.0
ISO_WALL_ALPHA = 0.55
LEFT_STRIP_UI_SCALE_MULT = 1.3
LEFT_STRIP_UI_SCALE_MULT_TOP = 1.55
POST_ROTATE_TOP_TO_NORTH_UP = False
TOP_YAW_NEAR_SQUARE_EPS = 0.06
CAMERA_TOP_DIST_SCALE = 2.9
CAMERA_ISO_DIST_SCALE = 3.2
CAMERA_FIT_MARGIN = 1.10
CAMERA_SAFETY_SCALE = 1.03


def _parse_north_world_xy(payload: dict) -> tuple[float, float] | None:
	nw = payload.get("north_world_xy", None) if isinstance(payload, dict) else None
	if not (isinstance(nw, list) and len(nw) == 2):
		return None
	try:
		nx = float(nw[0])
		ny = float(nw[1])
	except Exception:
		return None
	n = math.hypot(nx, ny)
	if n < 1e-8:
		return None
	return (nx / n, ny / n)


def _same_world_north(a: tuple[float, float] | None, b: tuple[float, float] | None, tol: float = 1e-3) -> bool:
	if a is None or b is None:
		return False
	ax, ay = a
	bx, by = b
	return abs(ax - bx) <= tol and abs(ay - by) <= tol


def _resolve_shared_north_world_xy(top_payload: dict, objects: list[dict]) -> tuple[float, float]:
	"""
	Single source of truth for north world direction:
	1) take top payload north_world_xy when available;
	2) fallback to canonical object-extent rule (near-square stabilized).
	"""
	n = _parse_north_world_xy(top_payload)
	if n is not None:
		return n
	return _canonical_north_world_xy_from_objects(objects)


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
	return best_dir, (half - best_diff)


def _mcq_4_from_pool(correct: str, pool: list[str]) -> tuple[list[str], int]:
	pool = list(dict.fromkeys(pool))
	if correct not in pool:
		pool.append(correct)

	adjacent_map = {
		"north": ["northeast", "northwest"],
		"south": ["southeast", "southwest"],
		"east": ["northeast", "southeast"],
		"west": ["northwest", "southwest"],
		"northeast": ["north", "east"],
		"northwest": ["north", "west"],
		"southeast": ["south", "east"],
		"southwest": ["south", "west"],
	}

	distractors = [x for x in pool if x != correct and x not in adjacent_map.get(correct, [])]
	if len(distractors) < 3:
		distractors = [x for x in pool if x != correct]
	distractors = random.sample(distractors, 3)
	options = [correct] + distractors
	random.shuffle(options)
	return options, options.index(correct)


def _load_font(size: int):
	try:
		return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
	except Exception:
		return ImageFont.load_default()


def _draw_north_indicator(drawer, north_vec: tuple[float, float]):
	drawer.rectangle([16, 16, 250, 120], fill=(0, 0, 0))
	cx, cy = 60, 72
	arrow_len = 42
	head_len = 14
	vx, vy = north_vec
	n = math.sqrt(vx * vx + vy * vy)
	if n < 1e-8:
		vx, vy = 0.0, -1.0
	else:
		vx, vy = vx / n, vy / n
	tip = (cx + vx * arrow_len, cy + vy * arrow_len)
	bot = (cx - vx * arrow_len * 0.6, cy - vy * arrow_len * 0.6)
	drawer.line([bot, tip], fill=(255, 0, 0), width=8)
	perp = (-vy, vx)
	wing = (tip[0] - vx * head_len, tip[1] - vy * head_len)
	left = (wing[0] + perp[0] * (head_len * 0.6), wing[1] + perp[1] * (head_len * 0.6))
	right = (wing[0] - perp[0] * (head_len * 0.6), wing[1] - perp[1] * (head_len * 0.6))
	drawer.polygon([tip, left, right], fill=(255, 0, 0))
	drawer.text((105, 55), "NORTH", fill=(255, 0, 0), font=_load_font(30))


def _load_precise_payload(json_path: Path) -> dict:
	if not json_path.exists():
		return {}
	try:
		with open(json_path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}


def _ensure_precise_payload(
	glb_path: Path,
	image_path: Path,
	view_type: str,
	wall_alpha: float = 1.0,
	north_world_xy: tuple[float, float] | None = None,
) -> dict:
	boxes_json = image_path.with_suffix(".boxes.json")
	payload = _load_precise_payload(boxes_json)
	if (
		isinstance(payload, dict)
		and int(payload.get("version", 0)) >= 3
		and "north_screen_vector" in payload
		and "centers_by_instance_index" in payload
	):
		return payload

	if not BLENDER_BIN.exists() or not RENDER_SCRIPT.exists():
		return payload if isinstance(payload, dict) else {}

	env = os.environ.copy()
	env["GLB_PATH"] = str(glb_path)
	env["OUT_PATH"] = str(image_path)
	env["CAMERA_MODE"] = view_type
	env["BOX_JSON_PATH"] = str(boxes_json)
	env["SKIP_RENDER"] = "1"
	env["NORTH_POLICY"] = "top_up"
	env["NORTH_SCREEN_MODE"] = "camera_basis"
	env["SETUP_CAMERA_NORTH_BIND"] = "0"
	env["TOP_DISTANCE_SCALE"] = str(float(CAMERA_TOP_DIST_SCALE))
	env["ISOMETRIC_DISTANCE_SCALE"] = str(float(CAMERA_ISO_DIST_SCALE))
	env["CAMERA_FIT_MARGIN"] = str(float(CAMERA_FIT_MARGIN))
	env["CAMERA_FIT_SAFETY"] = str(float(CAMERA_SAFETY_SCALE))
	env["WALL_ALPHA"] = str(float(wall_alpha))
	env["TOP_WALL_ALPHA"] = str(float(TOP_WALL_ALPHA))
	env["ISO_WALL_ALPHA"] = str(float(ISO_WALL_ALPHA))
	cmd = [str(BLENDER_BIN), "--background", "--python", str(RENDER_SCRIPT)]
	try:
		subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except Exception:
		pass

	payload = _load_precise_payload(boxes_json)
	return payload if isinstance(payload, dict) else {}


def _render_canonical_view(
	glb_path: Path,
	out_image: Path,
	view_type: str,
	wall_alpha: float = 1.0,
	north_world_xy: tuple[float, float] | None = None,
) -> dict:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	boxes_json = out_image.with_suffix(".boxes.json")
	if not BLENDER_BIN.exists() or not RENDER_SCRIPT.exists():
		return _load_precise_payload(boxes_json)

	env = os.environ.copy()
	env["GLB_PATH"] = str(glb_path)
	env["OUT_PATH"] = str(out_image)
	env["CAMERA_MODE"] = view_type
	env["BOX_JSON_PATH"] = str(boxes_json)
	env["SKIP_RENDER"] = "0"
	env["NORTH_POLICY"] = "top_up"
	env["NORTH_SCREEN_MODE"] = "camera_basis"
	env["SETUP_CAMERA_NORTH_BIND"] = "0"
	env["TOP_DISTANCE_SCALE"] = str(float(CAMERA_TOP_DIST_SCALE))
	env["ISOMETRIC_DISTANCE_SCALE"] = str(float(CAMERA_ISO_DIST_SCALE))
	env["CAMERA_FIT_MARGIN"] = str(float(CAMERA_FIT_MARGIN))
	env["CAMERA_FIT_SAFETY"] = str(float(CAMERA_SAFETY_SCALE))
	env["WALL_ALPHA"] = str(float(wall_alpha))
	env["TOP_WALL_ALPHA"] = str(float(TOP_WALL_ALPHA))
	env["ISO_WALL_ALPHA"] = str(float(ISO_WALL_ALPHA))
	cmd = [str(BLENDER_BIN), "--background", "--python", str(RENDER_SCRIPT)]
	try:
		subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except Exception:
		return _load_precise_payload(boxes_json)

	return _load_precise_payload(boxes_json)


def _resolve_scene_center_xy(scene_name: str, glb_name: str, objects: list[dict]) -> tuple[tuple[float, float], str, dict]:
	obj_center, obj_diag = _scene_bbox_center_and_diag_xy(objects)
	detail = {
		"object_bbox_center": [obj_center[0], obj_center[1]],
		"object_bbox_diag": obj_diag,
	}
	return obj_center, "object_bbox_center", detail


def convert_layout_to_objects(layout_info: list) -> list:
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


def assign_independent_labels(objects: list) -> dict[str, int]:
	def scene_id_key(obj):
		sid = str(obj.get("id", ""))
		try:
			return (0, int(sid))
		except ValueError:
			return (1, sid)

	sorted_objs = sorted(objects, key=scene_id_key)
	return {obj["id"]: idx for idx, obj in enumerate(sorted_objs, start=1)}


def _projected_center_by_id(objects: list, payload: dict) -> dict[str, tuple[int, int]]:
	centers = payload.get("centers_by_instance_index", {}) if isinstance(payload, dict) else {}
	out = {}
	for obj in objects:
		inst_idx = str(obj.get("instance_index", -1))
		if inst_idx in centers and isinstance(centers[inst_idx], list) and len(centers[inst_idx]) == 2:
			try:
				out[obj["id"]] = (int(centers[inst_idx][0]), int(centers[inst_idx][1]))
			except Exception:
				continue
	return out


def _visible_ids_from_payload(objects: list, payload: dict) -> set[str]:
	boxes = payload.get("boxes_by_instance_index", {}) if isinstance(payload, dict) else {}
	if not isinstance(boxes, dict):
		return set()
	image_w = float(payload.get("image_width", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
	image_h = float(payload.get("image_height", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
	if image_w <= 1e-6 or image_h <= 1e-6:
		max_x = 0.0
		max_y = 0.0
		for b in boxes.values():
			if isinstance(b, list) and len(b) == 4:
				try:
					max_x = max(max_x, float(b[2]))
					max_y = max(max_y, float(b[3]))
				except Exception:
					pass
		image_w = max(1.0, max_x)
		image_h = max(1.0, max_y)
	img_area = max(1.0, image_w * image_h)
	visible = set()
	for obj in objects:
		inst_idx = str(obj.get("instance_index", -1))
		b = boxes.get(inst_idx)
		if not (isinstance(b, list) and len(b) == 4):
			continue
		try:
			x1, y1, x2, y2 = [float(v) for v in b]
		except Exception:
			continue
		bw = max(0.0, x2 - x1)
		bh = max(0.0, y2 - y1)
		area = bw * bh
		if area >= MIN_VISIBLE_BBOX_AREA_PX and (area / img_area) >= MIN_VISIBLE_BBOX_AREA_RATIO:
			visible.add(obj["id"])
	return visible


def _north_from_payload(payload: dict) -> tuple[float, float]:
	nv = payload.get("north_screen_vector", [0.0, -1.0]) if isinstance(payload, dict) else [0.0, -1.0]
	try:
		return float(nv[0]), float(nv[1])
	except Exception:
		return (0.0, -1.0)


def _dot2(a: tuple[float, float], b: tuple[float, float]) -> float:
	return float(a[0] * b[0] + a[1] * b[1])


def _select_anchor_pair_by_world_north(
	objects: list[dict],
	north_world_xy: tuple[float, float] | None,
) -> tuple[str, str] | None:
	if north_world_xy is None or len(objects) < 2:
		return None
	nx, ny = float(north_world_xy[0]), float(north_world_xy[1])
	n = math.hypot(nx, ny)
	if n < 1e-9:
		return None
	nx, ny = nx / n, ny / n
	proj = []
	for o in objects:
		cw = o.get("center_world")
		oid = o.get("id")
		if not (isinstance(cw, (list, tuple)) and len(cw) >= 2 and isinstance(oid, str)):
			continue
		try:
			wx = float(cw[0])
			wy = float(cw[1])
		except Exception:
			continue
		proj.append((wx * nx + wy * ny, oid))
	if len(proj) < 2:
		return None
	proj.sort(key=lambda x: x[0])
	return proj[0][1], proj[-1][1]


def _north_vec_from_anchor_ids(
	payload: dict,
	objects_by_id: dict[str, dict],
	south_id: str | None,
	north_id: str | None,
) -> tuple[float, float] | None:
	if not (south_id and north_id) or not isinstance(payload, dict):
		return None
	centers = payload.get("centers_by_instance_index", {})
	if not isinstance(centers, dict):
		return None
	os = objects_by_id.get(south_id)
	on = objects_by_id.get(north_id)
	if not os or not on:
		return None
	ks = str(os.get("instance_index", -1))
	kn = str(on.get("instance_index", -1))
	cs = centers.get(ks)
	cn = centers.get(kn)
	if not (isinstance(cs, list) and len(cs) == 2 and isinstance(cn, list) and len(cn) == 2):
		return None
	try:
		vx = float(cn[0]) - float(cs[0])
		vy = float(cn[1]) - float(cs[1])
	except Exception:
		return None
	n = math.hypot(vx, vy)
	if n < 1e-9:
		return None
	return (float(vx / n), float(vy / n))


def _fit_screen_basis_from_objects(
	payload: dict,
	objects: list[dict],
	north_world_xy: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
	"""
	Fit screen east/north basis from visible object correspondences using
	world-basis pair-difference least squares:
	  dS ~= dE * v_e + dN * v_n
	This avoids single-axis confounding and yields a semantic north vector
	anchored to the same world north in both views.
	"""
	if not isinstance(payload, dict):
		return None
	centers = payload.get("centers_by_instance_index", {})
	if not isinstance(centers, dict):
		return None
	nx, ny = float(north_world_xy[0]), float(north_world_xy[1])
	nn = math.hypot(nx, ny)
	if nn < 1e-9:
		return None
	nx, ny = nx / nn, ny / nn
	ex, ey = ny, -nx

	rows = []
	for o in objects:
		cw = o.get("center_world")
		if not (isinstance(cw, (list, tuple)) and len(cw) >= 2):
			continue
		inst_idx = str(o.get("instance_index", -1))
		sc = centers.get(inst_idx)
		if not (isinstance(sc, list) and len(sc) == 2):
			continue
		try:
			wx = float(cw[0])
			wy = float(cw[1])
			sx = float(sc[0])
			sy = float(sc[1])
		except Exception:
			continue
		e = wx * ex + wy * ey
		n = wx * nx + wy * ny
		rows.append((e, n, sx, sy))
	if len(rows) < 4:
		return None

	S_ee = 0.0
	S_nn = 0.0
	S_en = 0.0
	T_ex = 0.0
	T_nx = 0.0
	T_ey = 0.0
	T_ny = 0.0
	pair_cnt = 0
	for i in range(len(rows)):
		ei, ni, sxi, syi = rows[i]
		for j in range(i + 1, len(rows)):
			ej, nj, sxj, syj = rows[j]
			de = ej - ei
			dn = nj - ni
			dsx = sxj - sxi
			dsy = syj - syi
			w = max(1e-6, de * de + dn * dn)
			S_ee += w * de * de
			S_nn += w * dn * dn
			S_en += w * de * dn
			T_ex += w * de * dsx
			T_nx += w * dn * dsx
			T_ey += w * de * dsy
			T_ny += w * dn * dsy
			pair_cnt += 1
	if pair_cnt < 3:
		return None

	det = S_ee * S_nn - S_en * S_en
	if abs(det) < 1e-9:
		return None

	ve_x = (T_ex * S_nn - T_nx * S_en) / det
	vn_x = (S_ee * T_nx - S_en * T_ex) / det
	ve_y = (T_ey * S_nn - T_ny * S_en) / det
	vn_y = (S_ee * T_ny - S_en * T_ey) / det

	ve_n = math.hypot(ve_x, ve_y)
	vn_n = math.hypot(vn_x, vn_y)
	if ve_n < 1e-9 or vn_n < 1e-9:
		return None
	ve = (float(ve_x / ve_n), float(ve_y / ve_n))
	vn = (float(vn_x / vn_n), float(vn_y / vn_n))
	# keep right-handed orientation in screen coords (+y down)
	if _dot2(ve, (-vn[1], vn[0])) < 0:
		ve = (-ve[0], -ve[1])
	return ve, vn


def _semantic_north_from_objects(
	payload: dict,
	objects: list[dict] | None,
	north_world_xy: tuple[float, float] | None,
) -> tuple[float, float] | None:
	if not objects or north_world_xy is None:
		return None
	basis = _fit_screen_basis_from_objects(payload, objects, north_world_xy)
	if basis is None:
		return None
	_, vn = basis
	return vn


def _norm2(vx: float, vy: float, fallback: tuple[float, float] = (0.0, -1.0)) -> tuple[float, float]:
	n = math.hypot(vx, vy)
	if n < 1e-8:
		return fallback
	return (vx / n, vy / n)


def _rotate_vec_quarter(vx: float, vy: float, deg: int) -> tuple[float, float]:
	d = int(deg) % 360
	if d == 0:
		return (vx, vy)
	if d == 90:
		# Image CCW 90°
		return (vy, -vx)
	if d == 270:
		# Image CW 90° (equivalent to -90°)
		return (-vy, vx)
	if d == 180:
		return (-vx, -vy)
	return (vx, vy)


def _top_northup_quarter_turn(payload: dict) -> int:
	"""
	Return the nearest quarter-turn (deg) to rotate top image so north points up.
	"""
	nx, ny = _north_from_payload(payload)
	nx, ny = _norm2(nx, ny)
	target = (0.0, -1.0)
	cands = [0, 90, 270, 180]
	best_deg = 0
	best_dot = -1e9
	for deg in cands:
		rx, ry = _rotate_vec_quarter(nx, ny, deg)
		dot = float(rx * target[0] + ry * target[1])
		if dot > best_dot:
			best_dot = dot
			best_deg = deg
	return int(best_deg)


def _rotate_point_quarter(x: float, y: float, w: int, h: int, deg: int) -> tuple[float, float]:
	d = int(deg) % 360
	if d == 0:
		return (x, y)
	if d == 90:
		# CCW 90
		return (y, float(w - 1) - x)
	if d == 270:
		# CW 90
		return (float(h - 1) - y, x)
	if d == 180:
		return (float(w - 1) - x, float(h - 1) - y)
	return (x, y)


def _payload_image_size(payload: dict) -> tuple[int, int]:
	w = int(payload.get("width", 0) or 0) if isinstance(payload, dict) else 0
	h = int(payload.get("height", 0) or 0) if isinstance(payload, dict) else 0
	if w > 0 and h > 0:
		return w, h
	iw = int(payload.get("image_width", 0) or 0) if isinstance(payload, dict) else 0
	ih = int(payload.get("image_height", 0) or 0) if isinstance(payload, dict) else 0
	if iw > 0 and ih > 0:
		return iw, ih
	boxes = payload.get("boxes_by_instance_index", {}) if isinstance(payload, dict) else {}
	max_x = 0
	max_y = 0
	if isinstance(boxes, dict):
		for b in boxes.values():
			if isinstance(b, list) and len(b) == 4:
				try:
					max_x = max(max_x, int(round(float(b[2]))))
					max_y = max(max_y, int(round(float(b[3]))))
				except Exception:
					pass
	return max(1, max_x + 1), max(1, max_y + 1)


def _rotate_payload_quarter(payload: dict, deg: int) -> dict:
	d = int(deg) % 360
	if d == 0:
		return payload

	try:
		out = json.loads(json.dumps(payload))
	except Exception:
		out = dict(payload)

	w, h = _payload_image_size(out)
	new_w, new_h = (h, w) if d in {90, 270} else (w, h)

	centers = out.get("centers_by_instance_index", {})
	if isinstance(centers, dict):
		for k, c in list(centers.items()):
			if isinstance(c, list) and len(c) == 2:
				try:
					x, y = float(c[0]), float(c[1])
					rx, ry = _rotate_point_quarter(x, y, w, h, d)
					centers[k] = [int(round(rx)), int(round(ry))]
				except Exception:
					pass

	boxes = out.get("boxes_by_instance_index", {})
	if isinstance(boxes, dict):
		for k, b in list(boxes.items()):
			if isinstance(b, list) and len(b) == 4:
				try:
					x1, y1, x2, y2 = [float(v) for v in b]
					pts = [
						_rotate_point_quarter(x1, y1, w, h, d),
						_rotate_point_quarter(x2, y1, w, h, d),
						_rotate_point_quarter(x2, y2, w, h, d),
						_rotate_point_quarter(x1, y2, w, h, d),
					]
					xs = [p[0] for p in pts]
					ys = [p[1] for p in pts]
					boxes[k] = [
						int(round(min(xs))),
						int(round(min(ys))),
						int(round(max(xs))),
						int(round(max(ys))),
					]
				except Exception:
					pass

	polys = out.get("polygons_by_instance_index", {})
	if isinstance(polys, dict):
		for k, poly in list(polys.items()):
			if isinstance(poly, list):
				new_poly = []
				for p in poly:
					if isinstance(p, list) and len(p) == 2:
						try:
							x, y = float(p[0]), float(p[1])
							rx, ry = _rotate_point_quarter(x, y, w, h, d)
							new_poly.append([int(round(rx)), int(round(ry))])
						except Exception:
							continue
				polys[k] = new_poly

	sc = out.get("scene_center_screen", None)
	if isinstance(sc, list) and len(sc) == 2:
		try:
			x, y = float(sc[0]), float(sc[1])
			rx, ry = _rotate_point_quarter(x, y, w, h, d)
			out["scene_center_screen"] = [int(round(rx)), int(round(ry))]
		except Exception:
			pass

	nx, ny = _north_from_payload(out)
	rx, ry = _rotate_vec_quarter(nx, ny, d)
	rx, ry = _norm2(rx, ry)
	out["north_screen_vector"] = [float(rx), float(ry)]
	out["width"] = int(new_w)
	out["height"] = int(new_h)
	if "image_width" in out or "image_height" in out:
		out["image_width"] = int(new_w)
		out["image_height"] = int(new_h)
	return out


def _rotate_image_quarter(src_image: Path, deg: int, out_image: Path) -> bool:
	if int(deg) % 360 == 0:
		if src_image.resolve() != out_image.resolve():
			try:
				shutil.copy2(src_image, out_image)
				return True
			except Exception:
				return False
		return True
	if (not PIL_AVAILABLE) or (not src_image.exists()):
		return False
	try:
		img = Image.open(src_image).convert("RGBA")
		rot90 = getattr(getattr(Image, "Transpose", Image), "ROTATE_90", getattr(Image, "ROTATE_90", None))
		rot180 = getattr(getattr(Image, "Transpose", Image), "ROTATE_180", getattr(Image, "ROTATE_180", None))
		rot270 = getattr(getattr(Image, "Transpose", Image), "ROTATE_270", getattr(Image, "ROTATE_270", None))
		d = int(deg) % 360
		if d == 90:
			img2 = img.transpose(rot90) if rot90 is not None else img.rotate(90, expand=True)
		elif d == 270:
			img2 = img.transpose(rot270) if rot270 is not None else img.rotate(-90, expand=True)
		elif d == 180:
			img2 = img.transpose(rot180) if rot180 is not None else img.rotate(180, expand=True)
		else:
			img2 = img
		out_image.parent.mkdir(parents=True, exist_ok=True)
		img2.save(out_image)
		return True
	except Exception:
		return False


def _enforce_top_north_up(top_image: Path, top_payload: dict) -> tuple[Path, dict, int]:
	"""
	Normalize top view so north arrow always points upward in image coordinates.
	Returns (normalized_top_image_path, normalized_payload, applied_quarter_turn_deg).
	"""
	deg = _top_northup_quarter_turn(top_payload)
	if deg % 360 == 0:
		return top_image, top_payload, 0
	rot_top = top_image.with_name(f"{top_image.stem}_northup{top_image.suffix}")
	ok = _rotate_image_quarter(top_image, deg, rot_top)
	if not ok:
		return top_image, top_payload, 0
	new_payload = _rotate_payload_quarter(top_payload, deg)
	return rot_top, new_payload, int(deg)


def _north_ur_score(payload: dict, north_vec: tuple[float, float] | None = None) -> float:
	nx, ny = north_vec if north_vec is not None else _north_from_payload(payload)
	ok_ur = (float(nx) > 1e-6) and (float(ny) < -1e-6)
	return (1000.0 if ok_ur else 0.0) + float(nx) - float(ny)


def _angle_between_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
	ax, ay = _norm2(float(a[0]), float(a[1]))
	bx, by = _norm2(float(b[0]), float(b[1]))
	c = max(-1.0, min(1.0, ax * bx + ay * by))
	return float(math.degrees(math.acos(c)))


def _map_top_north_to_iso_stats(
	top_payload: dict,
	iso_payload: dict,
) -> tuple[tuple[float, float] | None, dict]:
	"""
	Estimate isometric north as the mapped direction of top north using
	shared instance centers between top and isometric payloads:
	  d_iso ~= dE_top * v_e + dN_top * v_n
	then v_n is the target semantic north in isometric.
	"""
	if not (isinstance(top_payload, dict) and isinstance(iso_payload, dict)):
		return None, {"pair_count": 0, "confidence": 0.0, "reason": "invalid_payload"}
	tc = top_payload.get("centers_by_instance_index", {})
	ic = iso_payload.get("centers_by_instance_index", {})
	if not (isinstance(tc, dict) and isinstance(ic, dict)):
		return None, {"pair_count": 0, "confidence": 0.0, "reason": "missing_centers"}

	keys = [k for k in tc.keys() if k in ic]
	if len(keys) < 4:
		return None, {"pair_count": 0, "confidence": 0.0, "reason": "shared_keys_lt4"}

	top_n = _norm2(*_north_from_payload(top_payload))
	top_e = _screen_east_from_north(top_n)

	rows = []
	for k in keys:
		pt = tc.get(k)
		pi = ic.get(k)
		if not (isinstance(pt, list) and len(pt) == 2 and isinstance(pi, list) and len(pi) == 2):
			continue
		try:
			tx, ty = float(pt[0]), float(pt[1])
			ix, iy = float(pi[0]), float(pi[1])
		except Exception:
			continue
			rows.append((tx, ty, ix, iy))
	if len(rows) < 4:
		return None, {"pair_count": 0, "confidence": 0.0, "reason": "usable_rows_lt4"}

	S_ee = 0.0
	S_nn = 0.0
	S_en = 0.0
	T_ex = 0.0
	T_nx = 0.0
	T_ey = 0.0
	T_ny = 0.0
	pair_cnt = 0

	for i in range(len(rows)):
		txi, tyi, ixi, iyi = rows[i]
		for j in range(i + 1, len(rows)):
			txj, tyj, ixj, iyj = rows[j]
			dtx = txj - txi
			dty = tyj - tyi
			de = dtx * top_e[0] + dty * top_e[1]
			dn = dtx * top_n[0] + dty * top_n[1]
			dix = ixj - ixi
			diy = iyj - iyi
			w = max(1e-6, de * de + dn * dn)
			S_ee += w * de * de
			S_nn += w * dn * dn
			S_en += w * de * dn
			T_ex += w * de * dix
			T_nx += w * dn * dix
			T_ey += w * de * diy
			T_ny += w * dn * diy
			pair_cnt += 1

	if pair_cnt < 3:
		return None, {"pair_count": int(pair_cnt), "confidence": 0.0, "reason": "pair_cnt_lt3"}
	det = S_ee * S_nn - S_en * S_en
	if abs(det) < 1e-9:
		return None, {"pair_count": int(pair_cnt), "confidence": 0.0, "reason": "degenerate_det"}

	ve_x = (T_ex * S_nn - T_nx * S_en) / det
	ve_y = (T_ey * S_nn - T_ny * S_en) / det
	vn_x = (S_ee * T_nx - S_en * T_ex) / det
	vn_y = (S_ee * T_ny - S_en * T_ey) / det
	ve_n = math.hypot(ve_x, ve_y)
	vn_n = math.hypot(vn_x, vn_y)
	if ve_n < 1e-9 or vn_n < 1e-9:
		return None, {"pair_count": int(pair_cnt), "confidence": 0.0, "reason": "zero_basis_norm"}

	# Weighted fit quality in pair-difference space.
	sum_w = 0.0
	sum_wc = 0.0
	for i in range(len(rows)):
		txi, tyi, ixi, iyi = rows[i]
		for j in range(i + 1, len(rows)):
			txj, tyj, ixj, iyj = rows[j]
			dtx = txj - txi
			dty = tyj - tyi
			de = dtx * top_e[0] + dty * top_e[1]
			dn = dtx * top_n[0] + dty * top_n[1]
			dix = ixj - ixi
			diy = iyj - iyi
			w = max(1e-6, de * de + dn * dn)
			px = de * ve_x + dn * vn_x
			py = de * ve_y + dn * vn_y
			pn = math.hypot(px, py)
			on = math.hypot(dix, diy)
			if pn < 1e-9 or on < 1e-9:
				continue
			cosv = max(-1.0, min(1.0, (px * dix + py * diy) / (pn * on)))
			sum_w += w
			sum_wc += w * cosv
	mean_cos = (sum_wc / sum_w) if sum_w > 1e-9 else 0.0

	# Geometry conditioning confidence.
	trace = S_ee + S_nn
	disc = max(0.0, trace * trace - 4.0 * det)
	sdisc = math.sqrt(disc)
	lmax = 0.5 * (trace + sdisc)
	lmin = max(1e-9, 0.5 * (trace - sdisc))
	cond = float(lmax / lmin)

	cnt_conf = min(1.0, float(pair_cnt) / 24.0)
	fit_conf = max(0.0, min(1.0, 0.5 * (mean_cos + 1.0)))
	geom_conf = 1.0 / (1.0 + max(0.0, cond - 1.0) / 25.0)
	conf = max(0.0, min(1.0, cnt_conf * fit_conf * geom_conf))

	vn = (float(vn_x / vn_n), float(vn_y / vn_n))
	dbg = {
		"pair_count": int(pair_cnt),
		"mean_cos": float(mean_cos),
		"cond": float(cond),
		"confidence": float(conf),
		"reason": "ok",
	}
	return vn, dbg


def _map_top_north_to_iso(
	top_payload: dict,
	iso_payload: dict,
) -> tuple[float, float] | None:
	vn, _ = _map_top_north_to_iso_stats(top_payload, iso_payload)
	return vn


def _auto_correct_90_if_needed(
	render_n: tuple[float, float],
	mapped_n: tuple[float, float] | None,
	map_conf: float,
) -> tuple[tuple[float, float], dict]:
	rn = _norm2(float(render_n[0]), float(render_n[1]))
	if mapped_n is None:
		return rn, {"applied": False, "reason": "no_mapped_n", "delta_deg": None, "map_conf": float(map_conf)}
	mn = _norm2(float(mapped_n[0]), float(mapped_n[1]))
	delta = _angle_between_deg(rn, mn)
	# High-confidence and close to 90 deg => likely axis swap / quarter-turn mismatch.
	if (map_conf >= 0.35) and (abs(delta - 90.0) <= 20.0):
		c1 = _norm2(*_rotate_vec_quarter(rn[0], rn[1], 90))
		c2 = _norm2(*_rotate_vec_quarter(rn[0], rn[1], 270))
		d1 = _angle_between_deg(c1, mn)
		d2 = _angle_between_deg(c2, mn)
		use = c1 if d1 <= d2 else c2
		return use, {
			"applied": True,
			"reason": "delta_near_90_high_conf",
			"delta_deg": float(delta),
			"map_conf": float(map_conf),
			"after_delta_deg": float(min(d1, d2)),
		}
	return rn, {"applied": False, "reason": "not_90_or_low_conf", "delta_deg": float(delta), "map_conf": float(map_conf)}


def _select_isometric_north_ur(
	glb_path: Path,
	work_dir: Path,
	stem: str,
	top_payload: dict,
	wall_alpha: float = 1.0,
	north_world_xy: tuple[float, float] | None = None,
) -> tuple[Path, dict, str]:
	"""
	Render 4 fixed isometric quarter-turn candidates with the SAME world north,
	then choose the one whose north points to upper-right on screen.
	"""
	best_image = work_dir / f"{stem}_isometric.png"
	best_payload: dict = {}
	best_mode = "isometric"
	best_score = -1e9

	for mode in ["isometric", "isometric_cw90", "isometric_ccw90", "isometric_180"]:
		cand_img = work_dir / f"{stem}_{mode}.png"
		cand_payload = _render_canonical_view(
			glb_path,
			cand_img,
			mode,
			wall_alpha=wall_alpha,
			north_world_xy=north_world_xy,
		)
		render_n = _norm2(*_north_from_payload(cand_payload))
		mapped_n, map_dbg = _map_top_north_to_iso_stats(top_payload, cand_payload)
		use_n, auto_dbg = _auto_correct_90_if_needed(render_n, mapped_n, float(map_dbg.get("confidence", 0.0)))
		if mapped_n is not None and float(map_dbg.get("confidence", 0.0)) >= 0.35:
			# Prefer semantic mapping when confidence is sufficient.
			use_n = _norm2(mapped_n[0], mapped_n[1])
			auto_dbg["reason"] = "use_mapped_high_conf"
			auto_dbg["applied"] = False
		cand_payload["north_screen_vector"] = [float(use_n[0]), float(use_n[1])]
		cand_payload["_north_auto_debug"] = {
			"mode": mode,
			"render_n": [float(render_n[0]), float(render_n[1])],
			"mapped_n": [float(mapped_n[0]), float(mapped_n[1])] if mapped_n is not None else None,
			"map_dbg": map_dbg,
			"auto_dbg": auto_dbg,
		}
		score = _north_ur_score(cand_payload, north_vec=use_n)
		if score > best_score:
			best_score = score
			best_image = cand_img
			best_payload = cand_payload
			best_mode = mode

	return best_image, best_payload, best_mode


def _compose_with_left_strip_scaled(
	img,
	north_vec: tuple[float, float],
	px_per_unit: float,
	ui_scale_mult: float = 1.0,
	model_pad: int = 6,
	gap: int = 6,
):
	"""
	Task-local variant with adjustable UI scale multiplier,
	used to keep N arrow and 1m marker style aligned with task0.
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


def _annotate_top_with_numbers(base_image: Path, out_image: Path, objects: list, label_map: dict[str, int], payload: dict) -> str:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	centers = _projected_center_by_id(objects, payload)

	north_vec = _north_from_payload(payload)
	ppm = float(payload.get("pixels_per_meter", 70.0)) if isinstance(payload, dict) else 70.0

	w, h = img.size
	for obj in objects:
		oid = obj["id"]
		if oid not in centers:
			continue
		cx, cy = centers[oid]
		cx = max(14, min(w - 14, cx))
		cy = max(14, min(h - 14, cy))
		r = LABEL_RADIUS
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 60), outline=(255, 255, 255), width=1)
		txt = str(label_map[oid])
		font = _load_font(LABEL_FONT_SIZE)
		try:
			draw.text((cx, cy), txt, fill=(255, 255, 255), font=font, anchor="mm")
		except TypeError:
			bbox = draw.textbbox((0, 0), txt, font=font)
			draw.text((cx - (bbox[2] - bbox[0]) // 2, cy - (bbox[3] - bbox[1]) // 2), txt, fill=(255, 255, 255), font=font)

	img = _compose_with_left_strip_scaled(img, north_vec, ppm, ui_scale_mult=LEFT_STRIP_UI_SCALE_MULT_TOP)
	img.save(out_image)
	return str(out_image.resolve())


def _annotate_top_plain(base_image: Path, out_image: Path, payload: dict) -> str:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	north_vec = _north_from_payload(payload)
	ppm = float(payload.get("pixels_per_meter", 70.0)) if isinstance(payload, dict) else 70.0
	img = _compose_with_left_strip_scaled(img, north_vec, ppm, ui_scale_mult=LEFT_STRIP_UI_SCALE_MULT_TOP)
	img.save(out_image)
	return str(out_image.resolve())


def _choose_sparse_label_ids(
	objects: list,
	target_oid: str,
	centers: dict[str, tuple[int, int]],
	min_labels: int = QA2_MIN_LABELS,
	max_labels: int = QA2_MAX_LABELS,
	min_dist: float = QA2_MIN_LABEL_DISTANCE_PX,
) -> list[str]:
	available = [o["id"] for o in objects if o["id"] in centers]
	if target_oid not in available:
		return []

	def dist(a: str, b: str) -> float:
		ax, ay = centers[a]
		bx, by = centers[b]
		return math.dist((ax, ay), (bx, by))

	target_labels = min(max_labels, len(available))
	thresholds = [min_dist, min_dist * 0.8, min_dist * 0.6, min_dist * 0.4, 0.0]

	for th in thresholds:
		selected = [target_oid]
		cands = [oid for oid in available if oid != target_oid]
		random.shuffle(cands)
		cands.sort(key=lambda oid: dist(oid, target_oid), reverse=True)

		for oid in cands:
			if all(dist(oid, s) >= th for s in selected):
				selected.append(oid)
				if len(selected) >= target_labels:
					break

		if len(selected) >= min(min_labels, len(available)):
			return selected

	return [target_oid]


def _annotate_top_sparse_numbers(
	base_image: Path,
	out_image: Path,
	objects: list,
	label_map: dict[str, int],
	payload: dict,
	target_oid: str,
) -> tuple[str, list[str]]:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve()), []
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve()), []

	centers = _projected_center_by_id(objects, payload)
	selected_ids = _choose_sparse_label_ids(objects, target_oid, centers)

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	north_vec = _north_from_payload(payload)
	ppm = float(payload.get("pixels_per_meter", 70.0)) if isinstance(payload, dict) else 70.0

	w, h = img.size
	for oid in selected_ids:
		if oid not in centers or oid not in label_map:
			continue
		cx, cy = centers[oid]
		cx = max(14, min(w - 14, cx))
		cy = max(14, min(h - 14, cy))
		r = LABEL_RADIUS
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 60), outline=(255, 255, 255), width=1)
		txt = str(label_map[oid])
		font = _load_font(LABEL_FONT_SIZE)
		try:
			draw.text((cx, cy), txt, fill=(255, 255, 255), font=font, anchor="mm")
		except TypeError:
			bbox = draw.textbbox((0, 0), txt, font=font)
			draw.text((cx - (bbox[2] - bbox[0]) // 2, cy - (bbox[3] - bbox[1]) // 2), txt, fill=(255, 255, 255), font=font)

	img = _compose_with_left_strip_scaled(img, north_vec, ppm, ui_scale_mult=LEFT_STRIP_UI_SCALE_MULT_TOP)
	img.save(out_image)
	return str(out_image.resolve()), selected_ids


def _annotate_isometric_a(base_image: Path, out_image: Path, target_oid: str, objects: list, payload: dict) -> str:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	centers = _projected_center_by_id(objects, payload)
	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)

	north_vec = _north_from_payload(payload)
	ppm = float(payload.get("pixels_per_meter", 70.0)) if isinstance(payload, dict) else 70.0

	if target_oid in centers:
		cx, cy = centers[target_oid]
		w, h = img.size
		cx = max(14, min(w - 14, cx))
		cy = max(14, min(h - 14, cy))
		r = TARGET_A_RADIUS
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0), outline=(255, 255, 255), width=2)
		font = _load_font(TARGET_A_FONT_SIZE)
		try:
			draw.text((cx, cy), "A", fill=(255, 255, 255), font=font, anchor="mm")
		except TypeError:
			bbox = draw.textbbox((0, 0), "A", font=font)
			draw.text((cx - (bbox[2] - bbox[0]) // 2, cy - (bbox[3] - bbox[1]) // 2), "A", fill=(255, 255, 255), font=font)

	img = _compose_with_left_strip_scaled(img, north_vec, ppm, ui_scale_mult=LEFT_STRIP_UI_SCALE_MULT)
	img.save(out_image)
	return str(out_image.resolve())


def _annotate_isometric_with_north(base_image: Path, out_image: Path, payload: dict) -> str:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	north_vec = _north_from_payload(payload)
	ppm = float(payload.get("pixels_per_meter", 70.0)) if isinstance(payload, dict) else 70.0
	img = _compose_with_left_strip_scaled(img, north_vec, ppm, ui_scale_mult=LEFT_STRIP_UI_SCALE_MULT)
	img.save(out_image)
	return str(out_image.resolve())


def _estimate_scene_center_screen(
	objects: list[dict],
	payload: dict,
	scene_center_xy: tuple[float, float],
) -> tuple[int, int] | None:
	if isinstance(payload, dict):
		sc = payload.get("scene_center_screen")
		if isinstance(sc, list) and len(sc) == 2:
			try:
				return int(sc[0]), int(sc[1])
			except Exception:
				pass

	centers_by_id = _projected_center_by_id(objects, payload)
	if len(centers_by_id) < 3:
		return None

	cx, cy = scene_center_xy
	weighted_x = 0.0
	weighted_y = 0.0
	weight_sum = 0.0
	for obj in objects:
		oid = obj["id"]
		if oid not in centers_by_id:
			continue
		wx, wy = _obj_xy(obj)
		sx, sy = centers_by_id[oid]
		d = math.hypot(wx - cx, wy - cy)
		w = 1.0 / (d + 1e-3)
		weighted_x += sx * w
		weighted_y += sy * w
		weight_sum += w

	if weight_sum <= 1e-8:
		return None
	return int(round(weighted_x / weight_sum)), int(round(weighted_y / weight_sum))


def _make_camera_rotated_debug_image(
	base_image: Path,
	out_image: Path,
	scene_center_screen: tuple[int, int] | None,
	rot_dir: str,
) -> str:
	out_image.parent.mkdir(parents=True, exist_ok=True)
	if not base_image.exists():
		return str(base_image.resolve())
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_image)
		return str(out_image.resolve())

	img = Image.open(base_image).convert("RGB")
	w, h = img.size
	if scene_center_screen is None:
		center = (w // 2, h // 2)
	else:
		center = (
			max(0, min(w - 1, int(scene_center_screen[0]))),
			max(0, min(h - 1, int(scene_center_screen[1]))),
		)

	if rot_dir == "cw90":
		tag = "camera CW90"
	elif rot_dir == "ccw90":
		tag = "camera CCW90"
	else:
		raise ValueError(rot_dir)

	canvas = img.copy()
	draw = ImageDraw.Draw(canvas)
	r = 10
	cx, cy = center
	draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 215, 0), outline=(0, 0, 0), width=2)
	draw.line([(cx - 18, cy), (cx + 18, cy)], fill=(0, 0, 0), width=2)
	draw.line([(cx, cy - 18), (cx, cy + 18)], fill=(0, 0, 0), width=2)
	draw.rectangle([16, 16, 280, 110], fill=(0, 0, 0))
	draw.text((28, 26), "SCENE CENTER", fill=(255, 215, 0), font=_load_font(22))
	draw.text((28, 62), tag, fill=(255, 255, 255), font=_load_font(22))

	canvas.save(out_image)
	return str(out_image.resolve())


def _normalize2(vx: float, vy: float) -> tuple[float, float]:
	n = math.hypot(vx, vy)
	if n < 1e-8:
		return 0.0, 1.0
	return vx / n, vy / n


def _canonical_north_world_xy_from_objects(objects: list[dict]) -> tuple[float, float]:
	if not objects:
		return (0.0, 1.0)
	xs = [float(o["center_world"][0]) for o in objects if isinstance(o.get("center_world"), list) and len(o["center_world"]) >= 2]
	ys = [float(o["center_world"][1]) for o in objects if isinstance(o.get("center_world"), list) and len(o["center_world"]) >= 2]
	if not xs or not ys:
		return (0.0, 1.0)
	ext_x = max(xs) - min(xs)
	ext_y = max(ys) - min(ys)
	# Same canonical rule as shared renderer/task1.
	if ext_y > ext_x * (1.0 + TOP_YAW_NEAR_SQUARE_EPS):
		return (-1.0, 0.0)
	return (0.0, 1.0)


def _get_world_basis(payload: dict, objects: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
	nw = payload.get("north_world_xy", None) if isinstance(payload, dict) else None
	if isinstance(nw, list) and len(nw) == 2:
		try:
			nx, ny = float(nw[0]), float(nw[1])
		except Exception:
			nx, ny = 0.0, 1.0
	else:
		nx, ny = 0.0, 1.0
	north = _normalize2(nx, ny)
	east = _normalize2(north[1], -north[0])
	return north, east


def _project_world_xy_to_basis(
	world_xy: tuple[float, float],
	north_vec: tuple[float, float],
	east_vec: tuple[float, float],
) -> tuple[float, float]:
	wx, wy = float(world_xy[0]), float(world_xy[1])
	x_new = wx * east_vec[0] + wy * east_vec[1]
	y_new = wx * north_vec[0] + wy * north_vec[1]
	return float(x_new), float(y_new)


def _screen_east_from_north(north_screen: tuple[float, float]) -> tuple[float, float]:
	# Screen coordinates: +x right, +y down.
	# If north is up (0,-1), east should be right (1,0).
	nx, ny = north_screen
	ex, ey = -ny, nx
	return _norm2(ex, ey, fallback=(1.0, 0.0))


def _calibrate_north_screen_quarter_turn(
	payload: dict,
	objects: list[dict],
	world_north: tuple[float, float],
	world_east: tuple[float, float],
) -> tuple[float, float]:
	"""
	Robustly calibrate north_screen_vector by quarter-turn search using
	world-space object deltas vs screen-space projected deltas.
	This guards against rare 90° drift on near-square scenes.
	"""
	base_n = _norm2(*_north_from_payload(payload))
	centers = _projected_center_by_id(objects, payload)
	valid = []
	for o in objects:
		oid = o.get("id")
		cw = o.get("center_world")
		if oid not in centers or not (isinstance(cw, (list, tuple)) and len(cw) >= 2):
			continue
		sx, sy = centers[oid]
		valid.append((float(cw[0]), float(cw[1]), float(sx), float(sy)))
	if len(valid) < 3:
		return base_n

	def _score(ncand: tuple[float, float]) -> tuple[float, int]:
		ecand = _screen_east_from_north(ncand)
		total = 0.0
		cnt = 0
		for i in range(len(valid)):
			for j in range(i + 1, len(valid)):
				wxi, wyi, sxi, syi = valid[i]
				wxj, wyj, sxj, syj = valid[j]
				dwx = wxj - wxi
				dwy = wyj - wyi
				de = dwx * world_east[0] + dwy * world_east[1]
				dn = dwx * world_north[0] + dwy * world_north[1]
				if math.hypot(de, dn) < 1e-6:
					continue
				px = de * ecand[0] + dn * ncand[0]
				py = de * ecand[1] + dn * ncand[1]
				sx = sxj - sxi
				sy = syj - syi
				pn = math.hypot(px, py)
				sn = math.hypot(sx, sy)
				if pn < 1e-6 or sn < 1e-6:
					continue
				cosv = (px * sx + py * sy) / (pn * sn)
				total += float(cosv) * max(1.0, math.hypot(de, dn))
				cnt += 1
		return total, cnt

	cands = [
		base_n,
		_norm2(*_rotate_vec_quarter(base_n[0], base_n[1], 90)),
		_norm2(*_rotate_vec_quarter(base_n[0], base_n[1], 180)),
		_norm2(*_rotate_vec_quarter(base_n[0], base_n[1], 270)),
	]
	best_n = base_n
	best_score = -1e18
	best_cnt = -1
	for n in cands:
		sc, cnt = _score(n)
		if cnt > best_cnt or (cnt == best_cnt and sc > best_score):
			best_n = n
			best_score = sc
			best_cnt = cnt
	return best_n


def _apply_calibrated_north_to_payload(
	payload: dict,
	objects: list[dict],
	world_north: tuple[float, float],
	world_east: tuple[float, float],
) -> dict:
	nx, ny = _calibrate_north_screen_quarter_turn(payload, objects, world_north, world_east)
	try:
		payload["north_screen_vector"] = [float(nx), float(ny)]
	except Exception:
		pass
	return payload


def _obj_xy(obj: dict) -> tuple[float, float]:
	"""
	QA world-basis frame:
	- x: world-east axis derived from shared north_world_xy
	- y: world-north axis derived from shared north_world_xy
	If unavailable, fall back to world XY.
	"""
	nu = obj.get("center_northup")
	if isinstance(nu, (list, tuple)) and len(nu) == 2:
		try:
			return float(nu[0]), float(nu[1])
		except Exception:
			pass
	cw = obj.get("center_world")
	if isinstance(cw, (list, tuple)) and len(cw) >= 2:
		return float(cw[0]), float(cw[1])
	return 0.0, 0.0


def _scene_bbox_center_xy(objects: list[dict]) -> tuple[float, float]:
	xs = [float(_obj_xy(o)[0]) for o in objects]
	ys = [float(_obj_xy(o)[1]) for o in objects]
	return float((min(xs) + max(xs)) / 2.0), float((min(ys) + max(ys)) / 2.0)


def _scene_bbox_center_and_diag_xy(objects: list[dict]) -> tuple[tuple[float, float], float]:
	xs = [float(_obj_xy(o)[0]) for o in objects]
	ys = [float(_obj_xy(o)[1]) for o in objects]
	min_x, max_x = min(xs), max(xs)
	min_y, max_y = min(ys), max(ys)
	center = (float((min_x + max_x) / 2.0), float((min_y + max_y) / 2.0))
	diag = float(math.hypot(max_x - min_x, max_y - min_y))
	return center, diag


def _extreme_object_world(objects: list[dict], extreme_type: str) -> tuple[str, float]:
	"""
	Return:
	- object id of the target extreme object
	- directional component margin to 2nd best (must be >= EXTREME_MARGIN_WORLD)

	Directional component definition:
	- northmost: +y
	- southmost: -y
	- eastmost: +x
	- westmost: -x
	"""
	if len(objects) < 2:
		oid = objects[0]["id"] if objects else ""
		return oid, 0.0

	def _dir_component(o: dict) -> float:
		x, y = _obj_xy(o)
		if extreme_type == "northmost":
			return float(y)
		if extreme_type == "southmost":
			return float(-y)
		if extreme_type == "eastmost":
			return float(x)
		if extreme_type == "westmost":
			return float(-x)
		raise ValueError(f"Unknown extreme_type: {extreme_type}")

	scored = [(_dir_component(o), o["id"]) for o in objects]
	scored.sort(key=lambda t: t[0], reverse=True)  # larger directional component is better
	best = scored[0]
	second = scored[1]
	return best[1], float(best[0] - second[0])


def _task1_ok_with_retry(objects: list[dict], scene_center: tuple[float, float] | None = None) -> tuple[bool, str]:
	candidates = ["northmost", "southmost", "eastmost", "westmost"]
	random.shuffle(candidates)
	cx, cy = scene_center if scene_center is not None else _scene_bbox_center_xy(objects)

	for extreme_type in candidates:
		oid, margin = _extreme_object_world(objects, extreme_type)
		if margin < EXTREME_MARGIN_WORLD:
			continue

		target = next(o for o in objects if o["id"] == oid)
		tx, ty = _obj_xy(target)
		vx = tx - cx
		vy = ty - cy
		if math.hypot(vx, vy) < TASK_MIN_WORLD_RADIUS:
			continue
		theta = math.atan2(vy, vx)
		_, dist_to_boundary = _world_dir_8way(theta)
		if dist_to_boundary >= math.radians(TASK1_WORLD_ANGLE_MARGIN_DEG):
			return True, extreme_type

	return False, ""


def _task4_ok_with_retry(objects: list[dict], scene_center: tuple[float, float] | None = None) -> tuple[bool, str]:
	candidates = ["northmost", "southmost", "eastmost", "westmost"]
	random.shuffle(candidates)
	cx, cy = scene_center if scene_center is not None else _scene_bbox_center_xy(objects)

	for extreme_type in candidates:
		oid, margin = _extreme_object_world(objects, extreme_type)
		if margin < EXTREME_MARGIN_WORLD:
			continue

		target = next(o for o in objects if o["id"] == oid)
		tx, ty = _obj_xy(target)
		vx = tx - cx
		vy = ty - cy
		if math.hypot(vx, vy) < TASK_MIN_WORLD_RADIUS:
			continue

		ok = True
		for rot_dir in ("cw90", "ccw90"):
			if rot_dir == "cw90":
				new_north = (1.0, 0.0)
				new_east = (0.0, -1.0)
			else:
				new_north = (-1.0, 0.0)
				new_east = (0.0, 1.0)

			x_new = vx * new_east[0] + vy * new_east[1]
			y_new = vx * new_north[0] + vy * new_north[1]
			theta = math.atan2(y_new, x_new)
			_, dist_to_boundary = _world_dir_8way(theta)
			if dist_to_boundary < math.radians(TASK4_ANGLE_MARGIN_DEG):
				ok = False
				break

		if ok:
			return True, extreme_type

	return False, ""


def qa_iso_task1_extreme_object_top_direction_mcq(objects: list[dict], extreme_type: str, scene_center: tuple[float, float] | None = None) -> dict:
	target_id, _ = _extreme_object_world(objects, extreme_type)
	cx, cy = scene_center if scene_center is not None else _scene_bbox_center_xy(objects)
	t = next(o for o in objects if o["id"] == target_id)
	tx, ty = _obj_xy(t)
	vx = tx - cx
	vy = ty - cy
	theta = math.atan2(vy, vx)
	correct, _ = _world_dir_8way(theta)

	options, correct_idx = _mcq_4_from_pool(correct, ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"])
	labels = ["A", "B", "C", "D"]
	choices = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

	desc = {
		"northmost": "the northernmost",
		"southmost": "the southernmost",
		"eastmost": "the easternmost",
		"westmost": "the westernmost",
	}[extreme_type]

	q = (
		"There are two images: an isometric image and a top-view image.\n"
		"- In both images, the red arrow indicates North, and both arrows represent the same world North direction.\n"
		"- East/West/South are defined from this North direction in a standard right-handed way.\n"
		"- The reference center is the center of the bounding rectangle of all objects in the top-view image.\n"
		"- Object directions are determined by comparing object center points.\n\n"
		f"Consider {desc} object in the isometric image. Relative to the reference center, "
		"which direction is that object located in the top-view image? Choose one option.\n\n"
		"Isometric image: <image>\n"
		"Top-view image: <image>\n"
	)
	return {
		"task_type": "top_isometric_direction_consistent",
		"question": q + "\n".join(choices),
		"answer": labels[correct_idx],
		"answer_text": options[correct_idx],
		"images": ["isometric.png", "top.png"],
		"meta": {"extreme_type": extreme_type, "target_object_id": target_id},
	}


def qa_iso_task2_objectA_maps_to_top_id(object_a_label: int, candidate_labels: list[int]) -> dict:
	cand_sorted = sorted(set(candidate_labels))
	options_text = "Candidates: " + ", ".join(str(x) for x in cand_sorted)
	q = (
		"There are two images: an isometric image where only one object is labeled 'A', and a top-view image where only a few objects are numbered. "
		"Which numbered object is 'A' in the top-view image? Answer with one number from the candidate labels.\n\n"
		f"{options_text}\n\n"
		"Isometric image: <image>\nTop-view image: <image>"
	)
	return {
		"task_type": "top_isometric_A_consistent",
		"question": q,
		"answer": str(object_a_label),
		"images": ["isometric_A.png", "top_qa2.png"],
		"meta": {"objectA_label": object_a_label, "candidate_labels": cand_sorted},
	}


def qa_iso_task4_extreme_object_after_rot90_mcq(
	objects: list[dict],
	rot_dir: str,
	extreme_type: str,
	scene_center: tuple[float, float] | None = None,
) -> dict:
	target_id, _ = _extreme_object_world(objects, extreme_type)
	cx, cy = scene_center if scene_center is not None else _scene_bbox_center_xy(objects)
	t = next(o for o in objects if o["id"] == target_id)
	tx, ty = _obj_xy(t)
	vx = tx - cx
	vy = ty - cy

	if rot_dir not in {"cw90", "ccw90"}:
		raise ValueError(rot_dir)

	if rot_dir == "cw90":
		new_north = (1.0, 0.0)
		new_east = (0.0, -1.0)
	else:
		new_north = (-1.0, 0.0)
		new_east = (0.0, 1.0)
	x_new = vx * new_east[0] + vy * new_east[1]
	y_new = vx * new_north[0] + vy * new_north[1]
	theta = math.atan2(y_new, x_new)
	correct, _ = _world_dir_8way(theta)

	options, correct_idx = _mcq_4_from_pool(correct, ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"])
	labels = ["A", "B", "C", "D"]
	choices = [f"{labels[i]}. {opt}" for i, opt in enumerate(options)]

	rot_text = "CLOCKWISE" if rot_dir == "cw90" else "COUNTER-CLOCKWISE"
	desc = {
		"northmost": "the northernmost",
		"southmost": "the southernmost",
		"eastmost": "the easternmost",
		"westmost": "the westernmost",
	}[extreme_type]

	q = (
		"There is an isometric image of an indoor scene. In this image, the red arrow indicates the North direction.\n"
		f"Identify {desc} object in the image.\n\n"
		f"Now imagine that the camera rotates {rot_text} by 90 degrees around the center of the bounding rectangle of all objects. "
		"When the camera rotates, the cardinal directions (north, east, south, west) rotate together with the view.\n\n"
		"After this rotation, where will that object be located relative to the reference center in the rotated coordinate frame?\n"
		"Choose one option.\n\n"
		"Isometric image: <image>\n"
	)
	return {
		"task_type": "isometric_camera_rotate",
		"question": q + "\n".join(choices),
		"answer": labels[correct_idx],
		"answer_text": options[correct_idx],
		"images": ["isometric.png"],
		"meta": {
			"extreme_type": extreme_type,
			"target_object_id": target_id,
			"rotation": rot_dir,
			"x_new": x_new,
			"y_new": y_new,
			"relative_angle_deg": math.degrees(theta),
		},
	}


def _qa_images_to_abs(qa_items: list[dict], sample_dir: Path) -> list[dict]:
	for item in qa_items:
		images = item.get("images")
		if images:
			item["images"] = [str((sample_dir / p).resolve()) for p in images]
		ref_images = item.get("reference_images")
		if ref_images:
			item["reference_images"] = [str((sample_dir / p).resolve()) for p in ref_images]
	return qa_items


def build_qa_isometric(
	objects: list[dict],
	object_a_label: int,
	task1_extreme: str,
	task4_extreme: str,
	qa2_candidate_labels: list[int],
	scene_center: tuple[float, float] | None = None,
) -> dict:
	top_isometric = [
		qa_iso_task1_extreme_object_top_direction_mcq(objects, task1_extreme, scene_center),
		qa_iso_task2_objectA_maps_to_top_id(object_a_label, qa2_candidate_labels),
	]
	isometric = [
		qa_iso_task4_extreme_object_after_rot90_mcq(objects, "cw90", task4_extreme, scene_center),
		qa_iso_task4_extreme_object_after_rot90_mcq(objects, "ccw90", task4_extreme, scene_center),
	]
	return {"top_isometric": top_isometric, "isometric": isometric}


def main():
	parser = argparse.ArgumentParser(description="Construct indoor spatial orientation QA")
	parser.add_argument("--glb-dir", type=Path, default=DEFAULT_GLB_DIR)
	parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
	parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--seed", type=int, default=30)
	parser.add_argument("--max-scenes", type=int, default=0, help="0 means all")
	args = parser.parse_args()

	random.seed(args.seed)

	glb_dir = args.glb_dir.resolve()
	images_dir = args.images_dir.resolve()
	mapping_json = args.mapping_json.resolve()
	output_dir = args.output_dir.resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	TMP_ROOT_DIR.mkdir(parents=True, exist_ok=True)

	if not glb_dir.exists():
		raise FileNotFoundError(f"GLB dir not found: {glb_dir}")
	if not mapping_json.exists():
		raise FileNotFoundError(f"Mapping json not found: {mapping_json}")

	with open(mapping_json, "r", encoding="utf-8") as f:
		scene_mapping = json.load(f)

	glb_files = sorted([p.name for p in glb_dir.glob("*.glb") if p.is_file()])
	scene_keys = [k for k in glb_files if k in scene_mapping]
	if args.max_scenes > 0:
		scene_keys = scene_keys[: args.max_scenes]

	print(f"Total glb files: {len(glb_files)}")
	print(f"Matched in mapping: {len(scene_keys)}")

	all_results = []
	for idx, glb_name in enumerate(scene_keys, start=1):
		info = scene_mapping[glb_name]
		scene_name = info.get("scene_name", glb_name)
		layout_info = info.get("layout_info", [])
		objects = convert_layout_to_objects(layout_info)
		if len(objects) < 4:
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: objects<4")
			continue

		scene_center, scene_center_source, scene_center_detail = _resolve_scene_center_xy(scene_name, glb_name, objects)

		stem = Path(glb_name).stem
		glb_path = glb_dir / glb_name
		sample_dir = output_dir / stem
		temp_ctx = None
		if FORCE_CANONICAL_VIEW_RENDER:
				if KEEP_SCENE_VIEWS:
					canonical_dir = sample_dir / "scene_views"
					canonical_dir.mkdir(parents=True, exist_ok=True)
				else:
					temp_ctx = tempfile.TemporaryDirectory(prefix=f"{stem}_scene_views_", dir=str(TMP_ROOT_DIR))
					canonical_dir = Path(temp_ctx.name)
		else:
			canonical_dir = images_dir / stem

		if FORCE_CANONICAL_VIEW_RENDER:
			top_img = canonical_dir / f"{stem}_top.png"
			iso_img = canonical_dir / f"{stem}_isometric.png"
			top_payload = _render_canonical_view(glb_path, top_img, "top", wall_alpha=TOP_WALL_ALPHA)
			iso_payload = _render_canonical_view(
				glb_path,
				iso_img,
				FIXED_ISOMETRIC_MODE,
				wall_alpha=ISO_WALL_ALPHA,
			)
		else:
			scene_img_dir = images_dir / stem
			top_img = scene_img_dir / f"{stem}_top.png"
			iso_img = scene_img_dir / f"{stem}_isometric.png"
			top_payload = _ensure_precise_payload(glb_path, top_img, "top", wall_alpha=TOP_WALL_ALPHA)
			iso_payload = _ensure_precise_payload(
				glb_path,
				iso_img,
				FIXED_ISOMETRIC_MODE,
				wall_alpha=ISO_WALL_ALPHA,
			)
		if not top_img.exists() or not iso_img.exists():
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: missing top/isometric image")
			continue

		# Task1-aligned north logic:
		# keep renderer north as single source of truth for both top/isometric.
		top_northup_rotate_deg = 0
		selected_iso_mode = FIXED_ISOMETRIC_MODE

		if not top_img.exists() or not iso_img.exists():
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: normalized top/isometric image missing")
			continue

		top_centers = _projected_center_by_id(objects, top_payload)
		iso_centers = _projected_center_by_id(objects, iso_payload)
		if len(top_centers) < 4 or len(iso_centers) < 4:
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: projected centers<4")
			continue

		top_visible_ids = _visible_ids_from_payload(objects, top_payload)
		iso_visible_ids = _visible_ids_from_payload(objects, iso_payload)
		valid_id_set = set(top_centers.keys()) & set(iso_centers.keys()) & top_visible_ids & iso_visible_ids
		valid_objects = [o for o in objects if o["id"] in valid_id_set]
		if len(valid_objects) < 4:
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: valid projected objects<4")
			continue

		# Build QA frame from shared world north/east basis (task1-aligned),
		# independent of any image-space orientation assumptions.
		world_north, world_east = _get_world_basis(top_payload, valid_objects)
		for o in valid_objects:
			cw = o.get("center_world")
			if isinstance(cw, (list, tuple)) and len(cw) >= 2:
				xb, yb = _project_world_xy_to_basis((float(cw[0]), float(cw[1])), world_north, world_east)
				o["center_northup"] = [float(xb), float(yb)]

		scene_center, scene_center_source, scene_center_detail = _resolve_scene_center_xy(scene_name, glb_name, valid_objects)
		task1_ok, task1_extreme = _task1_ok_with_retry(valid_objects, scene_center)
		task4_ok, task4_extreme = _task4_ok_with_retry(valid_objects, scene_center)
		if not task1_ok or not task4_ok:
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: task margin constraints not met")
			continue

		label_map = assign_independent_labels(valid_objects)
		obj_a = random.choice(valid_objects)

		sample_dir.mkdir(parents=True, exist_ok=True)

		top_plain_out = sample_dir / "top_plain.png"
		top_qa2_out = sample_dir / "top_qa2.png"
		iso_out = sample_dir / "isometric.png"
		iso_a_out = sample_dir / "isometric_A.png"
		iso_rot_cw_out = sample_dir / "isometric_rot_cw90_center.png"
		iso_rot_ccw_out = sample_dir / "isometric_rot_ccw90_center.png"
		iso_rot_cw_raw = sample_dir / "isometric_cw90.png"
		iso_rot_ccw_raw = sample_dir / "isometric_ccw90.png"

		top_plain_abs = _annotate_top_plain(top_img, top_plain_out, top_payload)
		qa2_centers = _projected_center_by_id(valid_objects, top_payload)
		qa2_candidate_ids = _choose_sparse_label_ids(valid_objects, obj_a["id"], qa2_centers)
		qa2_local_map = assign_independent_labels([{"id": oid} for oid in qa2_candidate_ids])
		if obj_a["id"] not in qa2_local_map:
			qa2_candidate_ids.append(obj_a["id"])
			qa2_local_map = assign_independent_labels([{"id": oid} for oid in qa2_candidate_ids])
		top_qa2_abs, qa2_candidate_ids = _annotate_top_sparse_numbers(
			top_img, top_qa2_out, valid_objects, qa2_local_map, top_payload, obj_a["id"]
		)
		iso_abs = _annotate_isometric_with_north(iso_img, iso_out, iso_payload)
		iso_a_abs = _annotate_isometric_a(iso_img, iso_a_out, obj_a["id"], valid_objects, iso_payload)
		iso_rot_cw_payload = _render_canonical_view(
			glb_path,
			iso_rot_cw_raw,
			"isometric_cw90",
			wall_alpha=ISO_WALL_ALPHA,
		)
		iso_rot_ccw_payload = _render_canonical_view(
			glb_path,
			iso_rot_ccw_raw,
			"isometric_ccw90",
			wall_alpha=ISO_WALL_ALPHA,
		)
		iso_rot_cw_visible_ids = _visible_ids_from_payload(valid_objects, iso_rot_cw_payload)
		iso_rot_ccw_visible_ids = _visible_ids_from_payload(valid_objects, iso_rot_ccw_payload)
		scene_center_screen_iso = _estimate_scene_center_screen(valid_objects, iso_payload, scene_center)
		scene_center_screen_cw = _estimate_scene_center_screen(valid_objects, iso_rot_cw_payload, scene_center)
		scene_center_screen_ccw = _estimate_scene_center_screen(valid_objects, iso_rot_ccw_payload, scene_center)
		iso_rot_cw_abs = _make_camera_rotated_debug_image(iso_rot_cw_raw, iso_rot_cw_out, scene_center_screen_cw, "cw90")
		iso_rot_ccw_abs = _make_camera_rotated_debug_image(iso_rot_ccw_raw, iso_rot_ccw_out, scene_center_screen_ccw, "ccw90")
		try:
			if iso_rot_cw_raw.exists():
				iso_rot_cw_raw.unlink()
			if iso_rot_ccw_raw.exists():
				iso_rot_ccw_raw.unlink()
		except Exception:
			pass
		task1_target_id, _ = _extreme_object_world(valid_objects, task1_extreme)
		task4_target_id, _ = _extreme_object_world(valid_objects, task4_extreme)
		if (
			obj_a["id"] not in top_visible_ids
			or obj_a["id"] not in iso_visible_ids
			or task1_target_id not in top_visible_ids
			or task1_target_id not in iso_visible_ids
			or task4_target_id not in iso_visible_ids
			or task4_target_id not in iso_rot_cw_visible_ids
			or task4_target_id not in iso_rot_ccw_visible_ids
		):
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: visibility constraints not met")
			continue
		qa2_candidate_labels = [qa2_local_map[oid] for oid in qa2_candidate_ids if oid in qa2_local_map]
		object_a_label = qa2_local_map[obj_a["id"]]
		if object_a_label not in qa2_candidate_labels:
			qa2_candidate_labels.append(object_a_label)

		qa_groups = build_qa_isometric(
			valid_objects,
			object_a_label,
			task1_extreme,
			task4_extreme,
			qa2_candidate_labels,
			scene_center,
		)
		qa_groups["top_isometric"][0]["images"] = ["isometric.png", "top_plain.png"]
		qa_groups["top_isometric"][1]["images"] = ["isometric_A.png", "top_qa2.png"]
		qa_groups["isometric"][0]["images"] = ["isometric.png"]
		qa_groups["isometric"][0]["reference_images"] = ["isometric_rot_cw90_center.png"]
		qa_groups["isometric"][1]["images"] = ["isometric.png"]
		qa_groups["isometric"][1]["reference_images"] = ["isometric_rot_ccw90_center.png"]
		qa_groups["top_isometric"] = _qa_images_to_abs(qa_groups["top_isometric"], sample_dir)
		qa_groups["isometric"] = _qa_images_to_abs(qa_groups["isometric"], sample_dir)

		label_mapping = [
			{"scene_object_id": o["id"], "label_id": label_map[o["id"]]}
			for o in sorted(valid_objects, key=lambda x: label_map[x["id"]])
		]

		all_results.append(
			{
				"glb_name": glb_name,
				"scene_name": scene_name,
				"object_count": len(valid_objects),
				"label_mapping": label_mapping,
				"images": {
					"top_plain": top_plain_abs,
					"top_qa2": top_qa2_abs,
					"isometric": iso_abs,
					"isometric_A": iso_a_abs,
					"isometric_rot_cw90_center": iso_rot_cw_abs,
					"isometric_rot_ccw90_center": iso_rot_ccw_abs,
				},
				"qa": {
					"top_isometric": qa_groups["top_isometric"],
					"isometric": qa_groups["isometric"],
				},
					"special_refs": {
						"objectA_label": object_a_label,
						"extreme_margin_world": EXTREME_MARGIN_WORLD,
						"top_northup_rotate_deg": int(top_northup_rotate_deg),
						"selected_isometric_mode": str(selected_iso_mode),
						"top_north_screen_vector": top_payload.get("north_screen_vector"),
						"isometric_north_screen_vector": iso_payload.get("north_screen_vector"),
						"north_world_xy": top_payload.get("north_world_xy"),
						"qa_world_north_xy": [float(world_north[0]), float(world_north[1])],
						"qa_world_east_xy": [float(world_east[0]), float(world_east[1])],
						"scene_center_definition": "center of the bounding rectangle of all objects (x/y min-max midpoint)",
					"scene_center_source": scene_center_source,
					"scene_center_xy": [scene_center[0], scene_center[1]],
					"scene_center_screen_isometric": [scene_center_screen_iso[0], scene_center_screen_iso[1]] if scene_center_screen_iso is not None else None,
					"scene_center_screen_isometric_cw90": [scene_center_screen_cw[0], scene_center_screen_cw[1]] if scene_center_screen_cw is not None else None,
					"scene_center_screen_isometric_ccw90": [scene_center_screen_ccw[0], scene_center_screen_ccw[1]] if scene_center_screen_ccw is not None else None,
					"scene_center_detail": scene_center_detail,
					"task1_world_angle_margin_deg": TASK1_WORLD_ANGLE_MARGIN_DEG,
					"task4_angle_margin_deg": TASK4_ANGLE_MARGIN_DEG,
					"task1_extreme_type": task1_extreme,
					"task4_extreme_type": task4_extreme,
				},
			}
		)
		print(f"[{idx}/{len(scene_keys)}] done {glb_name}: qa_top_iso={len(qa_groups['top_isometric'])} qa_iso={len(qa_groups['isometric'])}")
		if temp_ctx is not None:
			temp_ctx.cleanup()

	out_json = output_dir / "metadata_indoor.json"
	with open(out_json, "w", encoding="utf-8") as f:
		json.dump(all_results, f, ensure_ascii=False, indent=2)

	print("=" * 80)
	print(f"Done. valid_scenes={len(all_results)}/{len(scene_keys)}")
	print(f"Output: {out_json}")


if __name__ == "__main__":
	main()
