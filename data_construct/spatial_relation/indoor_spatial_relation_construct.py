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
DEFAULT_GLB_DIR = INTERNSCENES_ROOT / "scenes/glb_files_wall_simple-5-10_clean_keep"
DEFAULT_IMAGES_DIR = INTERNSCENES_ROOT / "scenes/images"
DEFAULT_MAPPING_JSON = DEFAULT_GLB_DIR / "scene_layout_mapping_seed_30_indoor_clean.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark/data/spatial_relation/indoor_scenes_simple-5-10"
TMP_ROOT_DIR = PROJECT_ROOT / "tmp" / Path(__file__).stem
BLENDER_BIN = PROJECT_ROOT / "blender-3.2.2-linux-x64/blender"
RENDER_SCRIPT = PROJECT_ROOT / "benchmark/data_construct/utils.py"
SHARED_UTILS_PATH = PROJECT_ROOT / "benchmark/data_construct/utils.py"

_utils_spec = importlib.util.spec_from_file_location("dc_render_utils_task1_construct", SHARED_UTILS_PATH)
if _utils_spec is None or _utils_spec.loader is None:
	raise RuntimeError(f"Cannot load shared utils from {SHARED_UTILS_PATH}")
dc_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(dc_utils)

DIRECTION_ANGLE_MARGIN_DEG = 8.0
SIDE_COUNT_THRESHOLD_PX = 2.0
SIDE_COUNT_THRESHOLD_WORLD_FALLBACK = 0.08
MIN_LABEL_SEPARATION_PX = 24.0
CLOSEST_MARGIN_THRESHOLD_PX = 18.0
FARTHEST_MARGIN_THRESHOLD_PX = 18.0
FORCE_CANONICAL_VIEW_RENDER = True
KEEP_SCENE_VIEWS = False
LABEL_RADIUS = 13
LABEL_FONT_SIZE = 16
TARGET_A_RADIUS = 15
TARGET_A_FONT_SIZE = 18
MIN_VISIBLE_BBOX_AREA_PX = 800.0
MIN_VISIBLE_BBOX_AREA_RATIO = 0.0015
FIXED_ISOMETRIC_MODE = "isometric_north_ur"
TOP_WALL_ALPHA = 1.0
ISO_WALL_ALPHA = 0.55
LEFT_STRIP_UI_SCALE_MULT = 1.3
LEFT_STRIP_UI_SCALE_MULT_TOP = 1.55


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


def get_direction_with_confidence(a_pos, b_pos, min_margin_deg: float = 8.0) -> tuple[str, float, bool]:
	dx = a_pos[0] - b_pos[0]
	dy = a_pos[1] - b_pos[1]
	theta = math.atan2(dy, dx)
	direction, dist_to_boundary = _world_dir_8way(theta)
	margin_deg = math.degrees(dist_to_boundary)
	is_clear = margin_deg >= min_margin_deg
	return direction, margin_deg, is_clear


def _direction_pool() -> list[str]:
	return ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]


def make_direction_mcq(correct: str, k: int = 4) -> tuple[list[str], int]:
	pool = _direction_pool()
	if correct not in pool:
		correct = random.choice(pool)

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
	if len(distractors) < k - 1:
		distractors = [x for x in pool if x != correct]
	distractors = random.sample(distractors, k - 1)
	options = [correct] + distractors
	random.shuffle(options)
	return options, options.index(correct)


def _labels_options(options: list[str], correct_idx: int) -> tuple[str, str]:
	labels = ["A", "B", "C", "D"]
	choice_lines = [f"{labels[i]}. {options[i]}" for i in range(len(options))]
	return "\n".join(choice_lines), labels[correct_idx]


def _load_precise_payload(json_path: Path) -> dict:
	if not json_path.exists():
		return {}
	try:
		with open(json_path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}


def _ensure_precise_payload(glb_path: Path, image_path: Path, view_type: str, wall_alpha: float = 1.0) -> dict:
	boxes_json = image_path.with_suffix(".boxes.json")
	payload = _load_precise_payload(boxes_json)
	if (
		isinstance(payload, dict)
		and int(payload.get("version", 0)) >= 3
		and "north_screen_vector" in payload
		and "polygons_by_instance_index" in payload
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


def _render_canonical_view(glb_path: Path, out_image: Path, view_type: str, wall_alpha: float = 1.0) -> dict:
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
	env["WALL_ALPHA"] = str(float(wall_alpha))
	env["TOP_WALL_ALPHA"] = str(float(TOP_WALL_ALPHA))
	env["ISO_WALL_ALPHA"] = str(float(ISO_WALL_ALPHA))
	cmd = [str(BLENDER_BIN), "--background", "--python", str(RENDER_SCRIPT)]
	try:
		subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except Exception:
		return _load_precise_payload(boxes_json)

	return _load_precise_payload(boxes_json)


def _load_font(size: int):
	try:
		return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
	except Exception:
		return ImageFont.load_default()


def _rect_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	if ix2 <= ix1 or iy2 <= iy1:
		return 0
	return (ix2 - ix1) * (iy2 - iy1)


def _find_best_anchor(
	img_w: int,
	img_h: int,
	box_w: int,
	box_h: int,
	occupied: list[tuple[int, int, int, int]],
	avoid: list[tuple[int, int, int, int]] | None = None,
	margin: int = 16,
) -> tuple[int, int, int, int]:
	avoid = avoid or []
	cands = [
		(margin, margin),
		(img_w - margin - box_w, margin),
		(margin, img_h - margin - box_h),
		(img_w - margin - box_w, img_h - margin - box_h),
		((img_w - box_w) // 2, margin),
		((img_w - box_w) // 2, img_h - margin - box_h),
	]
	best_rect = (margin, margin, margin + box_w, margin + box_h)
	best_score = float("inf")
	for x, y in cands:
		x = max(margin, min(img_w - margin - box_w, x))
		y = max(margin, min(img_h - margin - box_h, y))
		r = (x, y, x + box_w, y + box_h)
		score = 0.0
		for o in occupied:
			score += _rect_intersection_area(r, o)
		for a in avoid:
			score += _rect_intersection_area(r, a) * 3.0
		if score < best_score:
			best_score = score
			best_rect = r
	return best_rect


def _draw_north_indicator(drawer, north_vec: tuple[float, float], panel_rect: tuple[int, int, int, int]):
	x1, y1, x2, y2 = panel_rect
	drawer.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 130))
	cx, cy = x1 + 32, y1 + 38
	arrow_len = 22
	head_len = 8
	vx, vy = north_vec
	n = math.sqrt(vx * vx + vy * vy)
	if n < 1e-8:
		vx, vy = 0.0, -1.0
	else:
		vx, vy = vx / n, vy / n
	tip = (cx + vx * arrow_len, cy + vy * arrow_len)
	bot = (cx - vx * arrow_len * 0.6, cy - vy * arrow_len * 0.6)
	drawer.line([bot, tip], fill=(255, 0, 0, 230), width=5)
	perp = (-vy, vx)
	wing = (tip[0] - vx * head_len, tip[1] - vy * head_len)
	left = (wing[0] + perp[0] * (head_len * 0.6), wing[1] + perp[1] * (head_len * 0.6))
	right = (wing[0] - perp[0] * (head_len * 0.6), wing[1] - perp[1] * (head_len * 0.6))
	drawer.polygon([tip, left, right], fill=(255, 0, 0, 230))
	drawer.text((x1 + 66, y1 + 22), "N", fill=(255, 0, 0, 240), font=_load_font(22))


def _draw_scale_bar(drawer, panel_rect: tuple[int, int, int, int], px_per_meter: float):
	x1, y1, x2, y2 = panel_rect
	drawer.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 130))
	bar_px = int(round(max(14.0, min(140.0, px_per_meter if px_per_meter > 0 else 70.0))))
	by = y1 + (y2 - y1) // 2 + 2
	bx1 = x1 + 16
	bx2 = bx1 + bar_px
	drawer.line([(bx1, by), (bx2, by)], fill=(255, 255, 255, 245), width=4)
	drawer.line([(bx1, by - 5), (bx1, by + 5)], fill=(255, 255, 255, 245), width=2)
	drawer.line([(bx2, by - 5), (bx2, by + 5)], fill=(255, 255, 255, 245), width=2)
	drawer.text((bx1, y1 + 6), "1m", fill=(255, 255, 255, 245), font=_load_font(16))


def _centroid(poly: list[tuple[int, int]]) -> tuple[int, int]:
	x = int(round(sum(p[0] for p in poly) / len(poly)))
	y = int(round(sum(p[1] for p in poly) / len(poly)))
	return x, y


def _annotate_with_numbers(
	base_image: str,
	objects: list,
	highlight_ids: list[str],
	label_map: dict[str, int],
	payload: dict,
	out_image: str,
	target_id: str | None = None,
) -> str:
	out_path = Path(out_image)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	if not os.path.exists(base_image):
		return os.path.abspath(base_image)
	if not PIL_AVAILABLE:
		shutil.copy2(base_image, out_path)
		return str(out_path.resolve())

	img = Image.open(base_image).convert("RGBA")
	draw = ImageDraw.Draw(img)
	polys = payload.get("polygons_by_instance_index", {}) if isinstance(payload, dict) else {}
	boxes = payload.get("boxes_by_instance_index", {}) if isinstance(payload, dict) else {}
	centers = payload.get("centers_by_instance_index", {}) if isinstance(payload, dict) else {}
	px_per_meter = 80.0
	try:
		ppm = float(payload.get("pixels_per_meter", 80.0)) if isinstance(payload, dict) else 80.0
		if ppm > 1e-6:
			px_per_meter = ppm
	except Exception:
		pass
	nv = payload.get("north_screen_vector", [0.0, -1.0]) if isinstance(payload, dict) else [0.0, -1.0]
	try:
		north_vec = (float(nv[0]), float(nv[1]))
	except Exception:
		north_vec = (0.0, -1.0)

	w, h = img.size
	occupied: list[tuple[int, int, int, int]] = []
	for b in boxes.values():
		if isinstance(b, list) and len(b) == 4:
			try:
				x1, y1, x2, y2 = [int(v) for v in b]
				occupied.append((x1, y1, x2, y2))
			except Exception:
				pass

	obj_map = {o["id"]: o for o in objects}
	label_points: list[tuple[int, int]] = []
	for oid in highlight_ids:
		if oid not in obj_map or oid not in label_map:
			continue
		inst_idx = str(obj_map[oid].get("instance_index", -1))
		cx, cy = None, None
		if inst_idx in centers and isinstance(centers[inst_idx], list) and len(centers[inst_idx]) == 2:
			cx, cy = int(centers[inst_idx][0]), int(centers[inst_idx][1])
		elif inst_idx in polys and isinstance(polys[inst_idx], list) and len(polys[inst_idx]) >= 3:
			poly = [(int(p[0]), int(p[1])) for p in polys[inst_idx] if isinstance(p, list) and len(p) == 2]
			if len(poly) >= 3:
				cx, cy = _centroid(poly)
		elif inst_idx in boxes and isinstance(boxes[inst_idx], list) and len(boxes[inst_idx]) == 4:
			x1, y1, x2, y2 = [int(v) for v in boxes[inst_idx]]
			cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
		else:
			# fallback: skip this id if precise projection missing
			continue

		cx = max(16, min(w - 16, cx))
		cy = max(16, min(h - 16, cy))
		label_points.append((cx, cy))
		r = LABEL_RADIUS
		draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 60), outline=(255, 255, 255), width=1)
		text = str(label_map[oid])
		font = _load_font(LABEL_FONT_SIZE)
		try:
			draw.text((cx, cy), text, fill=(255, 255, 255), font=font, anchor="mm")
		except TypeError:
			bbox = draw.textbbox((0, 0), text, font=font)
			tw = bbox[2] - bbox[0]
			th = bbox[3] - bbox[1]
			draw.text((cx - tw // 2, cy - th // 2), text, fill=(255, 255, 255), font=font)

	if target_id is not None and target_id in obj_map:
		inst_idx = str(obj_map[target_id].get("instance_index", -1))
		cx, cy = None, None
		if inst_idx in centers and isinstance(centers[inst_idx], list) and len(centers[inst_idx]) == 2:
			cx, cy = int(centers[inst_idx][0]), int(centers[inst_idx][1])
		elif inst_idx in polys and isinstance(polys[inst_idx], list) and len(polys[inst_idx]) >= 3:
			poly = [(int(p[0]), int(p[1])) for p in polys[inst_idx] if isinstance(p, list) and len(p) == 2]
			if len(poly) >= 3:
				cx, cy = _centroid(poly)
		elif inst_idx in boxes and isinstance(boxes[inst_idx], list) and len(boxes[inst_idx]) == 4:
			x1, y1, x2, y2 = [int(v) for v in boxes[inst_idx]]
			cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

		if cx is not None and cy is not None:
			cx = max(16, min(w - 16, cx))
			cy = max(16, min(h - 16, cy))
			r = TARGET_A_RADIUS
			draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0), outline=(255, 255, 255), width=2)
			font = _load_font(TARGET_A_FONT_SIZE)
			try:
				draw.text((cx, cy), "A", fill=(255, 255, 255), font=font, anchor="mm")
			except TypeError:
				bbox = draw.textbbox((0, 0), "A", font=font)
				tw = bbox[2] - bbox[0]
				th = bbox[3] - bbox[1]
				draw.text((cx - tw // 2, cy - th // 2), "A", fill=(255, 255, 255), font=font)

	base_stem = Path(base_image).stem.lower()
	scale_mult = LEFT_STRIP_UI_SCALE_MULT_TOP if "top" in base_stem else LEFT_STRIP_UI_SCALE_MULT
	img = _compose_with_left_strip_scaled(
		img,
		north_vec,
		px_per_meter,
		ui_scale_mult=scale_mult,
	)
	img.save(out_path)
	return str(out_path.resolve())


def _compose_with_left_strip_scaled(
	img,
	north_vec: tuple[float, float],
	px_per_unit: float,
	ui_scale_mult: float = 1.0,
	model_pad: int = 6,
	gap: int = 6,
):
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


def _pair_dist(a: dict, b: dict) -> float:
	return math.dist(a["center2d"], b["center2d"])


def _min_pairwise_distance(objects: list[dict]) -> float:
	if len(objects) < 2:
		return float("inf")
	min_dist = float("inf")
	for i in range(len(objects)):
		for j in range(i + 1, len(objects)):
			d = _pair_dist(objects[i], objects[j])
			if d < min_dist:
				min_dist = d
	return min_dist


def _highlight_well_separated(objects: list[dict], ids: list[str], min_sep: float) -> bool:
	by_id = {o["id"]: o for o in objects}
	selected = [by_id[oid] for oid in ids if oid in by_id]
	return _min_pairwise_distance(selected) >= min_sep


def _normalize2(vx: float, vy: float) -> tuple[float, float]:
	n = math.hypot(vx, vy)
	if n < 1e-8:
		return 0.0, -1.0
	return vx / n, vy / n


def _canonical_north_world_xy_from_objects(objects: list[dict]) -> tuple[float, float]:
	if not objects:
		return (0.0, 1.0)
	xs = [float(o["center"][0]) for o in objects if isinstance(o.get("center"), list) and len(o["center"]) >= 2]
	ys = [float(o["center"][1]) for o in objects if isinstance(o.get("center"), list) and len(o["center"]) >= 2]
	if not xs or not ys:
		return (0.0, 1.0)
	ext_x = max(xs) - min(xs)
	ext_y = max(ys) - min(ys)
	# Keep same rule as utils.canonical_north_world(): if Y-span dominates, north is -X; else +Y.
	if ext_y > ext_x:
		return (-1.0, 0.0)
	return (0.0, 1.0)


def _get_world_basis(payload: dict, objects: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
	nw = payload.get("north_world_xy", None) if isinstance(payload, dict) else None
	if isinstance(nw, list) and len(nw) == 2:
		try:
			nx, ny = float(nw[0]), float(nw[1])
		except Exception:
			nx, ny = _canonical_north_world_xy_from_objects(objects)
	else:
		nx, ny = _canonical_north_world_xy_from_objects(objects)
	north = _normalize2(nx, ny)
	# Same as task567 _set_move_basis_from_north(): right-hand 90° rotation in XY.
	east = _normalize2(north[1], -north[0])
	return north, east


def _side_threshold_world_from_payload(payload: dict) -> float:
	try:
		ppm = float(payload.get("pixels_per_meter", 0.0))
	except Exception:
		ppm = 0.0
	if ppm > 1e-6:
		return max(0.0, float(SIDE_COUNT_THRESHOLD_PX) / ppm)
	return SIDE_COUNT_THRESHOLD_WORLD_FALLBACK


def _build_view_objects(objects: list, payload: dict) -> list:
	centers = payload.get("centers_by_instance_index", {}) if isinstance(payload, dict) else {}
	view_objects = []
	for obj in objects:
		inst_idx = str(obj.get("instance_index", -1))
		if inst_idx in centers and isinstance(centers[inst_idx], list) and len(centers[inst_idx]) == 2:
			try:
				cx = float(centers[inst_idx][0])
				cy = float(centers[inst_idx][1])
			except Exception:
				continue
		else:
			continue
		item = dict(obj)
		item["center2d"] = [cx, cy]
		view_objects.append(item)
	return view_objects


def _visible_ids_from_payload(objects: list, payload: dict) -> set[str]:
	boxes = payload.get("boxes_by_instance_index", {}) if isinstance(payload, dict) else {}
	if not isinstance(boxes, dict):
		return set()
	image_w = float(payload.get("image_width", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
	image_h = float(payload.get("image_height", 0.0) or 0.0) if isinstance(payload, dict) else 0.0
	if image_w <= 1e-6 or image_h <= 1e-6:
		# fallback from box extents if explicit image size absent
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


def _relative_direction_8way_world(a_obj: dict, b_obj: dict, north_vec, east_vec, min_margin_deg: float = 8.0) -> tuple[str, float, bool]:
	dx = float(a_obj["center"][0]) - float(b_obj["center"][0])
	dy = float(a_obj["center"][1]) - float(b_obj["center"][1])
	x = dx * east_vec[0] + dy * east_vec[1]
	y = dx * north_vec[0] + dy * north_vec[1]
	theta = math.atan2(y, x)
	direction, dist_to_boundary = _world_dir_8way(theta)
	margin_deg = math.degrees(dist_to_boundary)
	return direction, margin_deg, margin_deg >= min_margin_deg


def qa_task_direction(objects: list, view_type: str, north_vec, east_vec) -> dict:
	for _ in range(120):
		a, b = random.sample(objects, 2)
		if _pair_dist(a, b) < MIN_LABEL_SEPARATION_PX:
			continue
		dir_name, _, clear = _relative_direction_8way_world(a, b, north_vec, east_vec, DIRECTION_ANGLE_MARGIN_DEG)
		if clear:
			qa_label_map = assign_independent_labels([{"id": a["id"]}, {"id": b["id"]}])
			options, correct_idx = make_direction_mcq(dir_name)
			choices_text, answer_label = _labels_options(options, correct_idx)
			prefix = (
				"In this image, North is shown by the red arrow. "
				"Direction is determined by comparing object center points. "
			)
			question = (
				f"{prefix}Which side is object {qa_label_map[a['id']]} relative to object {qa_label_map[b['id']]}?\n"
				+ "Image: <image>\n"
				+ choices_text
			)
			return {
				"question": question,
				"answer": answer_label,
				"answer_text": options[correct_idx],
				"task_type": f"{view_type}_direction",
				"highlight_ids": [a["id"], b["id"]],
				"qa_label_map": qa_label_map,
			}
	# fallback
	return None


def qa_task_closest(objects: list, view_type: str) -> dict:
	pairs = []
	for i in range(len(objects)):
		for j in range(i + 1, len(objects)):
			pairs.append((_pair_dist(objects[i], objects[j]), i, j))
	pairs.sort(key=lambda x: x[0])
	for best_idx, (best_dist, i, j) in enumerate(pairs):
		if best_dist < MIN_LABEL_SEPARATION_PX:
			continue
		a, b = objects[i], objects[j]
		correct_pair = tuple(sorted([a["id"], b["id"]]))

		candidate_distractors: list[tuple[str, str]] = []
		for dist, u_idx, v_idx in pairs:
			if {u_idx, v_idx} == {i, j}:
				continue
			if dist - best_dist < CLOSEST_MARGIN_THRESHOLD_PX:
				continue
			u, v = objects[u_idx], objects[v_idx]
			pair_ids = tuple(sorted([u["id"], v["id"]]))
			if pair_ids not in candidate_distractors and pair_ids != correct_pair:
				candidate_distractors.append(pair_ids)
			if len(candidate_distractors) >= 6:
				break

		if len(candidate_distractors) < 3:
			continue

		distractors = random.sample(candidate_distractors, 3)
		options = [correct_pair] + distractors
		random.shuffle(options)
		correct_idx = options.index(correct_pair)

		highlight = set()
		for opt in options:
			highlight.add(opt[0])
			highlight.add(opt[1])
		highlight_ids = sorted(highlight)
		if not _highlight_well_separated(objects, highlight_ids, MIN_LABEL_SEPARATION_PX):
			continue

		qa_label_map = assign_independent_labels([{"id": oid} for oid in highlight_ids])
		rendered_options = [f"{qa_label_map[pair[0]]} and {qa_label_map[pair[1]]}" for pair in options]
		choices_text, answer_label = _labels_options(rendered_options, correct_idx)
		question = "Which two numbered objects are closest to each other?\nImage: <image>\n" + choices_text

		return {
			"question": question,
			"answer": answer_label,
			"answer_text": rendered_options[correct_idx],
			"task_type": f"{view_type}_closest",
			"highlight_ids": highlight_ids,
			"qa_label_map": qa_label_map,
		}

	return None


def qa_task_farthest(objects: list, view_type: str) -> dict:
	targets = objects[:]
	random.shuffle(targets)
	for target in targets:
		others = [o for o in objects if o["id"] != target["id"]]
		ranked = sorted([(_pair_dist(target, o), o) for o in others], key=lambda x: x[0], reverse=True)
		if len(ranked) < 4:
			continue

		best_dist, correct_obj = ranked[0]
		second_dist, _ = ranked[1]
		if best_dist < MIN_LABEL_SEPARATION_PX:
			continue
		if best_dist - second_dist < FARTHEST_MARGIN_THRESHOLD_PX:
			continue

		correct_id = correct_obj["id"]
		distractor_pool = [o for _, o in ranked[1:] if best_dist - _pair_dist(target, o) >= FARTHEST_MARGIN_THRESHOLD_PX]
		if len(distractor_pool) < 3:
			continue
		distractors = [o["id"] for o in distractor_pool[:3]]
		options = [correct_id] + distractors
		random.shuffle(options)
		correct_idx = options.index(correct_id)
		highlight = {target["id"]}
		for opt in options:
			highlight.add(opt)
		highlight_ids = sorted(highlight)
		if not _highlight_well_separated(objects, highlight_ids, MIN_LABEL_SEPARATION_PX):
			continue

		qa_label_map = assign_independent_labels([{"id": oid} for oid in highlight_ids])
		rendered_options = [str(qa_label_map[oid]) for oid in options]
		choices_text, answer_label = _labels_options(rendered_options, correct_idx)
		question = (
			f"Which numbered object is farthest from object {qa_label_map[target['id']]}?\n"
			"Image: <image>\n"
			+ choices_text
		)

		return {
			"question": question,
			"answer": answer_label,
			"answer_text": rendered_options[correct_idx],
			"task_type": f"{view_type}_farthest",
			"highlight_ids": highlight_ids,
			"qa_label_map": qa_label_map,
		}

	return None


def qa_task_side_count(objects: list, view_type: str, north_vec, east_vec, side_threshold_world: float) -> dict:
	if len(objects) < 6:
		return None
	sides = ["north", "south", "east", "west"]
	targets = objects[:]
	random.shuffle(targets)
	for target in targets:
		others = [o for o in objects if o["id"] != target["id"]]
		if len(others) < 5:
			continue
		tx, ty = float(target["center"][0]), float(target["center"][1])
		per_obj = []
		for obj in others:
			dx = float(obj["center"][0]) - tx
			dy = float(obj["center"][1]) - ty
			proj_n = dx * north_vec[0] + dy * north_vec[1]
			proj_e = dx * east_vec[0] + dy * east_vec[1]
			per_obj.append((obj, proj_n, proj_e))
		if len(per_obj) < 5:
			continue

		for side in random.sample(sides, len(sides)):
			chosen = None
			fallback_candidate = None
			for _ in range(60):
				candidate = random.sample(per_obj, 5)
				fallback_candidate = candidate
				candidate_ids = [x[0]["id"] for x in candidate]
				if _highlight_well_separated(objects, candidate_ids + [target["id"]], MIN_LABEL_SEPARATION_PX):
					chosen = candidate
					break
			if chosen is None:
				# Fallback: keep generation stable in crowded views where strict separation
				# is hard to satisfy for all 5 candidates plus target A.
				if fallback_candidate is None:
					continue
				chosen = fallback_candidate

			def _is_on_side(row) -> bool:
				if side == "north":
					return row[1] > side_threshold_world
				if side == "south":
					return row[1] < -side_threshold_world
				if side == "east":
					return row[2] > side_threshold_world
				return row[2] < -side_threshold_world

			qa_label_map = assign_independent_labels([{"id": x[0]["id"]} for x in chosen])
			correct_ids = [x[0]["id"] for x in chosen if _is_on_side(x)]
			correct_numbers = sorted(qa_label_map[oid] for oid in correct_ids)
			answer_text = ",".join(str(v) for v in correct_numbers) if correct_numbers else "none"
			question = (
				"In this image, North is shown by the red arrow. "
				"Direction is determined by comparing object center points. "
				"An object can be on multiple sides at once (e.g., northwest counts as both north and west). "
				f"Object A is the black marker. Among the five red labeled objects, which are on the {side} side of object A? "
				"Answer using label numbers in ascending order, separated by commas (e.g., 1,3). If none, answer none.\n"
				"Image: <image>"
			)
			return {
				"question": question,
				"answer": answer_text,
				"task_type": f"{view_type}_side_count",
				"highlight_ids": [x[0]["id"] for x in chosen],
				"target_id": target["id"],
				"qa_label_map": qa_label_map,
			}
	return None


def build_qa_for_view(objects: list, view_type: str, north_vec, east_vec, side_threshold_world: float) -> list:
	direction_qa = qa_task_direction(objects, view_type, north_vec, east_vec)
	closest_qa = qa_task_closest(objects, view_type)
	farthest_qa = qa_task_farthest(objects, view_type)
	side_count_qa = qa_task_side_count(objects, view_type, north_vec, east_vec, side_threshold_world)
	if direction_qa is None or closest_qa is None or farthest_qa is None or side_count_qa is None:
		return None
	return [direction_qa, closest_qa, farthest_qa, side_count_qa]


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
				"center": [float(bbox[0]), float(bbox[1]), float(bbox[2])],
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
	label_map = {}
	for idx, obj in enumerate(sorted_objs, start=1):
		label_map[obj["id"]] = idx
	return label_map


def main():
	parser = argparse.ArgumentParser(description="Construct indoor spatial relation QA with independent numeric labels")
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

		stem = Path(glb_name).stem
		glb_path = glb_dir / glb_name
		scene_img_dir = images_dir / stem
		region_dir = output_dir / stem
		temp_ctx = None

		if FORCE_CANONICAL_VIEW_RENDER:
				if KEEP_SCENE_VIEWS:
					canonical_dir = region_dir / "scene_views"
					canonical_dir.mkdir(parents=True, exist_ok=True)
				else:
					temp_ctx = tempfile.TemporaryDirectory(prefix=f"{stem}_scene_views_", dir=str(TMP_ROOT_DIR))
					canonical_dir = Path(temp_ctx.name)
		else:
			canonical_dir = scene_img_dir

		if FORCE_CANONICAL_VIEW_RENDER:
			top_img = canonical_dir / f"{stem}_top.png"
			iso_img = canonical_dir / f"{stem}_isometric.png"
			top_payload = _render_canonical_view(glb_path, top_img, "top", wall_alpha=TOP_WALL_ALPHA)
			iso_payload = _render_canonical_view(glb_path, iso_img, FIXED_ISOMETRIC_MODE, wall_alpha=ISO_WALL_ALPHA)
		else:
			top_img = scene_img_dir / f"{stem}_top.png"
			iso_img = scene_img_dir / f"{stem}_isometric.png"
			top_payload = _ensure_precise_payload(glb_path, top_img, "top", wall_alpha=TOP_WALL_ALPHA)
			iso_payload = _ensure_precise_payload(glb_path, iso_img, FIXED_ISOMETRIC_MODE, wall_alpha=ISO_WALL_ALPHA)

		if not top_img.exists() or not iso_img.exists():
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: missing top/isometric image")
			continue

		top_view_objects_all = _build_view_objects(objects, top_payload)
		iso_view_objects_all = _build_view_objects(objects, iso_payload)
		top_visible_ids = _visible_ids_from_payload(objects, top_payload)
		iso_visible_ids = _visible_ids_from_payload(objects, iso_payload)
		top_ids = {o["id"] for o in top_view_objects_all}
		iso_ids = {o["id"] for o in iso_view_objects_all}
		valid_id_set = top_ids & iso_ids & top_visible_ids & iso_visible_ids
		valid_objects = [o for o in objects if o["id"] in valid_id_set]
		if len(valid_objects) < 4:
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: valid projected objects<4")
			continue

		top_view_objects = [o for o in top_view_objects_all if o["id"] in valid_id_set]
		iso_view_objects = [o for o in iso_view_objects_all if o["id"] in valid_id_set]

		label_map = assign_independent_labels(valid_objects)
		label_mapping_list = [
			{"scene_object_id": obj["id"], "label_id": label_map[obj["id"]]}
			for obj in sorted(valid_objects, key=lambda o: label_map[o["id"]])
		]

		world_north, world_east = _get_world_basis(top_payload, valid_objects)
		top_side_threshold_world = _side_threshold_world_from_payload(top_payload)
		iso_side_threshold_world = _side_threshold_world_from_payload(iso_payload)

		qa_top = build_qa_for_view(top_view_objects, "top", world_north, world_east, top_side_threshold_world)
		qa_iso = build_qa_for_view(iso_view_objects, "isometric", world_north, world_east, iso_side_threshold_world)
		if qa_top is None or qa_iso is None:
			if temp_ctx is not None:
				temp_ctx.cleanup()
			print(f"[{idx}/{len(scene_keys)}] skip {glb_name}: cannot satisfy QA distance/margin thresholds")
			continue

		if FORCE_CANONICAL_VIEW_RENDER and not KEEP_SCENE_VIEWS:
			meta_top = str((scene_img_dir / f"{stem}_top.png").resolve())
			meta_iso = str((scene_img_dir / f"{stem}_isometric.png").resolve())
		else:
			meta_top = str(top_img.resolve())
			meta_iso = str(iso_img.resolve())

		anno_dir = region_dir / "qa_labels"
		anno_dir.mkdir(parents=True, exist_ok=True)

		for i, qa in enumerate(qa_top):
			out_img = anno_dir / f"{stem}_top__qa{i+1}_{qa['task_type']}.png"
			qa_label_map = qa.get("qa_label_map", {})
			qa["images"] = [
				_annotate_with_numbers(
					str(top_img),
					valid_objects,
					qa.get("highlight_ids", []),
					qa_label_map,
					top_payload,
					str(out_img),
					qa.get("target_id"),
				)
			]
			qa.pop("highlight_ids", None)
			if qa.get("task_type") == "top_side_count":
				qa["target_object_id"] = qa.get("target_id")
			qa.pop("target_id", None)
			if qa.get("task_type") != "top_side_count":
				qa.pop("qa_label_map", None)

		for i, qa in enumerate(qa_iso):
			out_img = anno_dir / f"{stem}_isometric__qa{i+1}_{qa['task_type']}.png"
			qa_label_map = qa.get("qa_label_map", {})
			qa["images"] = [
				_annotate_with_numbers(
					str(iso_img),
					valid_objects,
					qa.get("highlight_ids", []),
					qa_label_map,
					iso_payload,
					str(out_img),
					qa.get("target_id"),
				)
			]
			qa.pop("highlight_ids", None)
			if qa.get("task_type") == "isometric_side_count":
				qa["target_object_id"] = qa.get("target_id")
			qa.pop("target_id", None)
			if qa.get("task_type") != "isometric_side_count":
				qa.pop("qa_label_map", None)

		all_results.append(
			{
				"glb_name": glb_name,
				"scene_name": scene_name,
				"object_count": len(valid_objects),
				"label_mapping": label_mapping_list,
				"images": {
					"top": meta_top,
					"isometric": meta_iso,
					"qa_labels_dir": os.path.abspath(str(anno_dir)),
				},
				"qa": {
					"top": qa_top,
					"isometric": qa_iso,
				},
			}
		)
		print(f"[{idx}/{len(scene_keys)}] done {glb_name}: qa_top={len(qa_top)} qa_iso={len(qa_iso)}")
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
