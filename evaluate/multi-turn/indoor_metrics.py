#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BLENDER_DETECT_SCRIPT = r'''
import bpy
import json
import math
import re
import sys
from collections import Counter
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import convex_hull_2d
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union


def world_aabb(obj):
    mw = obj.matrix_world
    verts = []
    if obj.type == 'MESH' and obj.data and hasattr(obj.data, 'vertices') and obj.data.vertices:
        verts = [mw @ v.co for v in obj.data.vertices]
    elif hasattr(obj, 'bound_box') and obj.bound_box:
        verts = [mw @ Vector(co) for co in obj.bound_box]
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    return {
        'min_x': float(min(xs)), 'max_x': float(max(xs)),
        'min_y': float(min(ys)), 'max_y': float(max(ys)),
        'min_z': float(min(zs)), 'max_z': float(max(zs)),
    }


def merge_aabb(aabbs):
    if not aabbs:
        return None
    return {
        'min_x': float(min(bb['min_x'] for bb in aabbs)),
        'max_x': float(max(bb['max_x'] for bb in aabbs)),
        'min_y': float(min(bb['min_y'] for bb in aabbs)),
        'max_y': float(max(bb['max_y'] for bb in aabbs)),
        'min_z': float(min(bb['min_z'] for bb in aabbs)),
        'max_z': float(max(bb['max_z'] for bb in aabbs)),
    }


def parse_obj_id_and_category(name: str):
    s = str(name or '')
    m = re.match(r'^(\d+)_([^@/]+)', s)
    if not m:
        return None, ''
    oid = str(m.group(1))
    cat = str(m.group(2)).strip().lower().replace('_', ' ').replace('-', ' ')
    return oid, cat


def category_tokens(cat: str):
    s = str(cat or '').lower().replace('_', ' ').replace('-', ' ')
    return {x for x in s.split() if x}


def load_whitelist_pairs(path: str):
    if not path:
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return set()
    out = set()
    for x in data.get('allowed_overlap_pairs', []) or []:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            continue
        a = str(x[0]).strip().lower()
        b = str(x[1]).strip().lower()
        out.add(tuple(sorted((a, b))))
    return out


def is_pair_whitelisted(cat_a: str, cat_b: str, whitelist_pairs):
    a = str(cat_a).strip().lower()
    b = str(cat_b).strip().lower()
    if tuple(sorted((a, b))) in whitelist_pairs:
        return True
    ta = category_tokens(a)
    tb = category_tokens(b)
    for xa in ta:
        for xb in tb:
            if tuple(sorted((xa, xb))) in whitelist_pairs:
                return True
    return False


def mesh_polygon_xy(obj, max_points=5000):
    if obj is None or obj.type != 'MESH' or obj.data is None or not obj.data.vertices:
        return None
    mw = obj.matrix_world
    verts = obj.data.vertices
    step = max(1, len(verts) // max_points)
    pts = []
    for i in range(0, len(verts), step):
        w = mw @ verts[i].co
        pts.append(Vector((float(w.x), float(w.y))))
    if len(pts) < 3:
        return None
    try:
        hull_idx = convex_hull_2d(pts)
        poly = Polygon([(float(pts[i].x), float(pts[i].y)) for i in hull_idx])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def build_group_bvh(group_objs):
    verts = []
    tris = []
    v_ofs = 0
    for obj in group_objs:
        if obj.type != 'MESH' or obj.data is None:
            continue
        mw = obj.matrix_world
        mesh = obj.data
        local_to_global = {}
        for vi, v in enumerate(mesh.vertices):
            local_to_global[vi] = v_ofs
            verts.append(mw @ v.co)
            v_ofs += 1
        for poly in mesh.polygons:
            if len(poly.vertices) < 3:
                continue
            ids = [local_to_global[int(i)] for i in poly.vertices]
            for k in range(1, len(ids) - 1):
                tris.append((ids[0], ids[k], ids[k + 1]))
    if not tris or not verts:
        return None
    try:
        return BVHTree.FromPolygons(verts, tris, all_triangles=True)
    except Exception:
        return None


def _xy_penetration_depth(aabb_a, aabb_b):
    dx = min(float(aabb_a['max_x']), float(aabb_b['max_x'])) - max(float(aabb_a['min_x']), float(aabb_b['min_x']))
    dy = min(float(aabb_a['max_y']), float(aabb_b['max_y'])) - max(float(aabb_a['min_y']), float(aabb_b['min_y']))
    return min(dx, dy)


def _normalize2(v):
    l = math.hypot(float(v[0]), float(v[1]))
    if l < 1e-9:
        return (1.0, 0.0)
    return (float(v[0]) / l, float(v[1]) / l)


def _build_group_obb_xy(poly, aabb):
    if poly is not None and (not poly.is_empty):
        try:
            mrr = poly.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)
            edges = []
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                dx, dy = float(x2 - x1), float(y2 - y1)
                ln = math.hypot(dx, dy)
                if ln > 1e-8:
                    edges.append((ln, (dx / ln, dy / ln)))
            if edges:
                edges.sort(key=lambda x: x[0], reverse=True)
                major_len, ax = edges[0]
                minor_len = edges[-1][0]
                ay = (-ax[1], ax[0])
                center = (float(mrr.centroid.x), float(mrr.centroid.y))
                return {
                    'center': center,
                    'ax': _normalize2(ax),
                    'ay': _normalize2(ay),
                    'hx': float(major_len) * 0.5,
                    'hy': float(minor_len) * 0.5,
                }
        except Exception:
            pass
    if aabb is None:
        return None
    cx = 0.5 * (float(aabb['min_x']) + float(aabb['max_x']))
    cy = 0.5 * (float(aabb['min_y']) + float(aabb['max_y']))
    hx = 0.5 * (float(aabb['max_x']) - float(aabb['min_x']))
    hy = 0.5 * (float(aabb['max_y']) - float(aabb['min_y']))
    return {'center': (cx, cy), 'ax': (1.0, 0.0), 'ay': (0.0, 1.0), 'hx': hx, 'hy': hy}


def _compute_pca_obb_xy(points_xy):
    if not points_xy:
        return None
    n = len(points_xy)
    mx = sum(float(p[0]) for p in points_xy) / float(n)
    my = sum(float(p[1]) for p in points_xy) / float(n)
    xx = sum((float(p[0]) - mx) ** 2 for p in points_xy)
    yy = sum((float(p[1]) - my) ** 2 for p in points_xy)
    xy = sum((float(p[0]) - mx) * (float(p[1]) - my) for p in points_xy)

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
        dx = float(x) - mx
        dy = float(y) - my
        rx = ca * dx + sa * dy
        ry = -sa * dx + ca * dy
        rot.append((rx, ry))
    if not rot:
        return None

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
        'ax': (float(ax[0]), float(ax[1])),
        'ay': (float(ay[0]), float(ay[1])),
        'hx': float(hx),
        'hy': float(hy),
        'center': (float(cx), float(cy)),
    }


def _project_obb(center, ax, ay, hx, hy, axis):
    dots = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            px = center[0] + ax[0] * (sx * hx) + ay[0] * (sy * hy)
            py = center[1] + ax[1] * (sx * hx) + ay[1] * (sy * hy)
            dots.append(px * axis[0] + py * axis[1])
    return min(dots), max(dots)


def _obb_penetration_depth_xy(obb_a, obb_b):
    if not obb_a or not obb_b:
        return 0.0
    ca = tuple(obb_a['center'][:2])
    cb = tuple(obb_b['center'][:2])
    ax_a, ay_a = tuple(obb_a['ax']), tuple(obb_a['ay'])
    ax_b, ay_b = tuple(obb_b['ax']), tuple(obb_b['ay'])
    hx_a, hy_a = float(obb_a['hx']), float(obb_a['hy'])
    hx_b, hy_b = float(obb_b['hx']), float(obb_b['hy'])

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


def _angle_diff_mod_180(a, b):
    d = abs((float(a) - float(b)) % 180.0)
    if d > 90.0:
        d = 180.0 - d
    return d


def _group_main_angle(poly):
    if poly is None or poly.is_empty:
        return None
    mrr = poly.minimum_rotated_rectangle
    if mrr is None or mrr.is_empty:
        return None
    coords = list(mrr.exterior.coords)
    if len(coords) < 3:
        return None
    best_len = -1.0
    best_ang = None
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        seg_len = math.hypot(dx, dy)
        if seg_len > best_len + 1e-9:
            best_len = seg_len
            best_ang = math.degrees(math.atan2(dy, dx))
    return best_ang


def _major_axis_from_obb(obb):
    if not obb:
        return (1.0, 0.0)
    hx = float(obb.get('hx', 0.0))
    hy = float(obb.get('hy', 0.0))
    axis = obb.get('ax', (1.0, 0.0)) if hx >= hy else obb.get('ay', (0.0, 1.0))
    return _normalize2((float(axis[0]), float(axis[1])))


def _is_round_like_category(cat: str):
    t = str(cat or '').lower()
    keys = ['round', 'circle', 'circular', 'cylinder', 'column', 'pillar', 'stool', 'lamp', 'vase', 'basket', 'bucket']
    return any(k in t for k in keys)


WALL_CONFLICT_EXCLUDE_TARGET_KEYWORDS = {
    'window', 'window frame', 'door', 'door frame', 'curtain', 'blinds',
    'painting', 'picture', 'picture frame', 'photo', 'poster', 'wall art', 'artwork', 'mirror',
}
REGION_BOUNDARY_RELAX_KEYWORDS = {
    'window', 'window frame', 'door', 'door frame', 'curtain', 'blinds',
    'shelf', 'cabinet', 'toilet', 'sink', 'counter', 'bathtub', 'shower',
}
WALL_CONFLICT_DEPTH_RATIO_THRESHOLD = 0.12
# Overlap area threshold on XY footprint intersection (m^2).
# Keep a tiny positive floor to avoid counting numerical-touch cases.
OVERLAP_MIN_INTERSECTION_AREA = 1e-4


def _contains_any_keyword(text: str, keys):
    t = str(text or '').lower()
    return any(k in t for k in keys)


def _is_region_boundary_relaxed_category(cat: str):
    return _contains_any_keyword(cat, REGION_BOUNDARY_RELAX_KEYWORDS)


def _inside_wall_region_by_category_bbox(cx, cy, hw, hd, cat, wall_bounds, ratio_default=0.98, ratio_relaxed=0.70):
    if wall_bounds is None:
        return True
    thr = float(ratio_relaxed) if _is_region_boundary_relaxed_category(cat) else float(ratio_default)
    obj_min_x = float(cx) - float(hw)
    obj_max_x = float(cx) + float(hw)
    obj_min_y = float(cy) - float(hd)
    obj_max_y = float(cy) + float(hd)
    obj_area = max(1e-9, (obj_max_x - obj_min_x) * (obj_max_y - obj_min_y))

    wminx = float(wall_bounds['min_x'])
    wmaxx = float(wall_bounds['max_x'])
    wminy = float(wall_bounds['min_y'])
    wmaxy = float(wall_bounds['max_y'])

    if not (wminx <= float(cx) <= wmaxx and wminy <= float(cy) <= wmaxy):
        return False
    ix = max(0.0, min(wmaxx, obj_max_x) - max(wminx, obj_min_x))
    iy = max(0.0, min(wmaxy, obj_max_y) - max(wminy, obj_min_y))
    inter_area = float(ix * iy)
    ratio = inter_area / obj_area
    return bool(ratio >= float(thr))


def _closest_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float):
    vx, vy = (bx - ax), (by - ay)
    den = vx * vx + vy * vy
    if den <= 1e-12:
        return (ax, ay)
    t = ((px - ax) * vx + (py - ay) * vy) / den
    t = max(0.0, min(1.0, t))
    return (ax + t * vx, ay + t * vy)


def _extract_linestrings(geom):
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == 'LineString':
        return [geom]
    if gt == 'MultiLineString':
        return [g for g in geom.geoms if g is not None and (not g.is_empty)]
    if gt == 'GeometryCollection':
        out = []
        for g in geom.geoms:
            out.extend(_extract_linestrings(g))
        return out
    return []


def _collect_geom_coords_xy(geom):
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == 'Polygon':
        return [(float(x), float(y)) for x, y in geom.exterior.coords]
    if gt == 'MultiPolygon':
        out = []
        for g in geom.geoms:
            out.extend(_collect_geom_coords_xy(g))
        return out
    if gt == 'LineString':
        return [(float(x), float(y)) for x, y in geom.coords]
    if gt == 'MultiLineString':
        out = []
        for g in geom.geoms:
            out.extend(_collect_geom_coords_xy(g))
        return out
    if gt == 'Point':
        return [(float(geom.x), float(geom.y))]
    if gt == 'MultiPoint':
        return [(float(g.x), float(g.y)) for g in geom.geoms]
    if gt == 'GeometryCollection':
        out = []
        for g in geom.geoms:
            out.extend(_collect_geom_coords_xy(g))
        return out
    return []


def _nearest_wall_overlap_ratio(obj_poly, wall_components):
    if obj_poly is None or obj_poly.is_empty or (not wall_components):
        return {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float('inf')}

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
        return {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float('inf')}

    try:
        _p_obj, p_wall = nearest_points(obj_poly, best.boundary)
    except Exception:
        return {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float(best_dist)}
    px, py = float(p_wall.x), float(p_wall.y)

    coords = list(best.exterior.coords) if hasattr(best, 'exterior') else []
    if len(coords) < 2:
        return {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float(best_dist)}

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
        return {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float(best_dist)}

    ax, ay, bx, by = best_seg
    tx, ty = (bx - ax), (by - ay)
    tl = math.hypot(tx, ty)
    if tl < 1e-9:
        return {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float(best_dist)}
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
            'ratio': 0.0,
            'overlap_depth': 0.0,
            'wall_thickness': float(wall_thickness),
            'distance': float(best_dist),
        }
    coords_ov = _collect_geom_coords_xy(ov)
    if not coords_ov:
        return {
            'ratio': 0.0,
            'overlap_depth': 0.0,
            'wall_thickness': float(wall_thickness),
            'distance': float(best_dist),
        }
    projs = [float(x) * float(nx) + float(y) * float(ny) for x, y in coords_ov]
    overlap_depth = max(0.0, max(projs) - min(projs))
    ratio = overlap_depth / max(wall_thickness, 1e-9)
    return {
        'ratio': float(ratio),
        'overlap_depth': float(overlap_depth),
        'wall_thickness': float(wall_thickness),
        'distance': float(best_dist),
    }


def _rectangularity_and_aspect(poly):
    if poly is None or poly.is_empty:
        return 0.0, 1e9
    area = float(getattr(poly, 'area', 0.0) or 0.0)
    if area <= 1e-8:
        return 0.0, 1e9
    mrr = poly.minimum_rotated_rectangle
    mrr_area = float(getattr(mrr, 'area', 0.0) or 0.0) if mrr is not None else 0.0
    if mrr_area <= 1e-8:
        return 0.0, 1e9
    rect_ratio = area / mrr_area
    aspect = 1e9
    try:
        coords = list(mrr.exterior.coords)
        lens = []
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len > 1e-8:
                lens.append(float(seg_len))
        if lens:
            aspect = max(lens) / max(min(lens), 1e-8)
    except Exception:
        aspect = 1e9
    return float(rect_ratio), float(aspect)


def detect_overlap(groups, whitelist_pairs, label_of_oid):
    pair_count = 0
    involved = set()
    involved_labels = set()
    pairs = []

    keys = sorted(groups.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    for i in range(len(keys)):
        a = keys[i]
        ga = groups[a]
        for j in range(i + 1, len(keys)):
            b = keys[j]
            gb = groups[b]

            if is_pair_whitelisted(ga.get('category', ''), gb.get('category', ''), whitelist_pairs):
                continue

            bvh_a = ga.get('bvh')
            bvh_b = gb.get('bvh')
            hits = 0
            if bvh_a is not None and bvh_b is not None:
                try:
                    hits = len(bvh_a.overlap(bvh_b))
                except Exception:
                    hits = 0

            overlap_area = 0.0
            poly_a = ga.get('polygon')
            poly_b = gb.get('polygon')
            if poly_a is not None and poly_b is not None:
                try:
                    if poly_a.intersects(poly_b):
                        overlap_area = float(poly_a.intersection(poly_b).area)
                except Exception:
                    overlap_area = 0.0

    
            strong_overlap = (hits >= 5 and overlap_area >= float(OVERLAP_MIN_INTERSECTION_AREA))
            if not strong_overlap:
                continue

            pair_count += 1
            involved.add(a)
            involved.add(b)
            la = label_of_oid.get(str(a))
            lb = label_of_oid.get(str(b))
            if isinstance(la, int):
                involved_labels.add(la)
            if isinstance(lb, int):
                involved_labels.add(lb)

            pairs.append({
                'object_ids': [str(a), str(b)],
                'label_ids': [la, lb],
                'object_label_ids': [la, lb],
                'reason': f'3D overlap detected (bvh_hits={int(hits)}, xy_overlap_area={float(overlap_area):.6f})',
            })

    return {
        'count': int(pair_count),
        'involved_object_ids': sorted(involved),
        'involved_building_label_ids': sorted(x for x in involved_labels if isinstance(x, int)),
        'problem_pairs': pairs,
    }


def detect_wall_conflict(groups, wall_bounds, label_of_oid, wall_components=None):
    if (not wall_bounds) and (not wall_components):
        return {
            'count': 0,
            'involved_object_ids': [],
            'involved_building_label_ids': [],
            'problems': [],
            'wall_bounds': None,
        }

    involved = []
    involved_labels = []
    problems = []

    for oid, g in groups.items():
        cat = str(g.get('category', '')).lower()
        skip_out_of_region = _contains_any_keyword(cat, WALL_CONFLICT_EXCLUDE_TARGET_KEYWORDS)
        bb = g.get('aabb')
        if not bb:
            continue
        cx = 0.5 * (bb['min_x'] + bb['max_x'])
        cy = 0.5 * (bb['min_y'] + bb['max_y'])
        hw = 0.5 * (bb['max_x'] - bb['min_x'])
        hd = 0.5 * (bb['max_y'] - bb['min_y'])
        out_of_region = False
        if wall_bounds and (not skip_out_of_region):
            inside = _inside_wall_region_by_category_bbox(cx, cy, hw, hd, cat, wall_bounds, ratio_default=0.98, ratio_relaxed=0.70)
            out_of_region = (not inside)

        near = {'ratio': 0.0, 'overlap_depth': 0.0, 'wall_thickness': 1.0, 'distance': float('inf')}
        penetrate_wall = False
        poly = g.get('polygon')
        if poly is not None and (not poly.is_empty) and wall_components:
            near = _nearest_wall_overlap_ratio(poly, wall_components)
            penetrate_wall = float(near.get('ratio', 0.0)) > float(WALL_CONFLICT_DEPTH_RATIO_THRESHOLD)

        if (not out_of_region) and (not penetrate_wall):
            continue

        lb = label_of_oid.get(str(oid))
        involved.append(str(oid))
        if isinstance(lb, int):
            involved_labels.append(lb)
        if out_of_region and penetrate_wall:
            reason = 'object footprint extends outside wall bounds and penetrates wall thickness'
        elif out_of_region:
            reason = 'object footprint extends outside wall bounds'
        else:
            reason = 'object penetrates wall thickness'
        problems.append({
            'object_id': str(oid),
            'label_id': lb,
            'object_label_id': lb,
            'reason': reason,
            'out_of_wall_bounds': bool(out_of_region),
            'penetrate_wall': bool(penetrate_wall),
            'wall_overlap_ratio': float(near.get('ratio', 0.0)),
            'wall_overlap_depth': float(near.get('overlap_depth', 0.0)),
            'wall_thickness': float(near.get('wall_thickness', 1.0)),
            'wall_distance': float(near.get('distance', float('inf'))),
            'penetration_ratio_threshold': float(WALL_CONFLICT_DEPTH_RATIO_THRESHOLD),
        })

    return {
        'count': int(len(involved)),
        'involved_object_ids': sorted(set(involved)),
        'involved_building_label_ids': sorted(set(int(x) for x in involved_labels if isinstance(x, int))),
        'problems': problems,
        'wall_bounds': wall_bounds,
    }


def detect_orientation(groups, wall_bounds, label_of_oid, overlap_object_ids=None, wall_object_ids=None):
    if not wall_bounds:
        return {
            'count': 0,
            'involved_object_ids': [],
            'involved_building_label_ids': [],
            'problems': [],
            'threshold_deg': 12.0,
        }

    overlap_object_ids = set(str(x) for x in (overlap_object_ids or []))
    wall_object_ids = set(str(x) for x in (wall_object_ids or []))
    involved = []
    involved_labels = []
    problems = []
    candidates = []

    for oid, g in groups.items():
        poly = g.get('polygon')
        if poly is None or poly.is_empty:
            continue

        lb = label_of_oid.get(str(oid))

        if _is_round_like_category(g.get('category', '')):
            continue

        rect_ratio, aspect = _rectangularity_and_aspect(poly)
        if rect_ratio < 0.86:
            continue

        bb = g.get('aabb')
        if not bb:
            continue
        hx = 0.5 * (bb['max_x'] - bb['min_x'])
        hy = 0.5 * (bb['max_y'] - bb['min_y'])
        hz = max(1e-6, float(bb['max_z'] - bb['min_z']))
        volume = float(hx * hy * hz)
        if volume < 0.08:
            continue

        cx = 0.5 * (bb['min_x'] + bb['max_x'])
        cy = 0.5 * (bb['min_y'] + bb['max_y'])
        dists = {
            'W': float(cx - wall_bounds['min_x']),
            'E': float(wall_bounds['max_x'] - cx),
            'S': float(cy - wall_bounds['min_y']),
            'N': float(wall_bounds['max_y'] - cy),
        }
        dist_wall = min(float(v) for v in dists.values())
        if dist_wall > 1.6:
            continue

        obb = g.get('obb')
        if not obb:
            continue

        axis = _major_axis_from_obb(obb)
        ang = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
        d = min(_angle_diff_mod_180(ang, 0.0), _angle_diff_mod_180(ang, 90.0))
        candidates.append({
            'oid': str(oid),
            'label_id': lb,
            'dist_by_side': dists,
            'dist_wall': float(dist_wall),
            'angle_dev': float(d),
        })

    if candidates:
        for c in candidates:
            # All wall-near candidates are eligible (no nearest-side gate).
            if c['angle_dev'] <= 12.0:
                continue

            lb = c['label_id']
            involved.append(c['oid'])
            if isinstance(lb, int):
                involved_labels.append(lb)
            problems.append({
                'object_id': c['oid'],
                'label_id': lb,
                'object_label_id': lb,
                'reason': f'orientation deviation to axis is {float(c["angle_dev"]):.2f} deg (>12 deg)',
            })

    return {
        'count': int(len(involved)),
        'involved_object_ids': sorted(set(involved)),
        'involved_building_label_ids': sorted(set(int(x) for x in involved_labels if isinstance(x, int))),
        'problems': problems,
        'threshold_deg': 12.0,
    }


def _import_and_get_wall_bounds_and_components(wall_glb_path: str):
    if not wall_glb_path:
        return None, []
    if not bpy.path.abspath(wall_glb_path):
        return None, []
    import os
    if not os.path.exists(wall_glb_path):
        return None, []

    before = {o.name for o in bpy.data.objects}
    after_objs = []
    try:
        bpy.ops.import_scene.gltf(filepath=str(wall_glb_path))
    except Exception:
        return None, []
    after_objs = [o for o in bpy.data.objects if o.name not in before and o.type == 'MESH']
    if not after_objs:
        return None, []

    try:
        aabbs = [world_aabb(o) for o in after_objs]
        aabbs = [x for x in aabbs if x is not None]
        wall_bounds = merge_aabb(aabbs)

        wall_components = []
        polys = []
        for obj in after_objs:
            mesh = obj.data
            if mesh is None:
                continue
            mw = obj.matrix_world
            nm = mw.to_3x3()
            for poly in mesh.polygons:
                try:
                    wn = nm @ poly.normal
                except Exception:
                    continue
                if abs(float(wn.z)) < 0.7:
                    continue
                pts = []
                for vid in poly.vertices:
                    w = mw @ mesh.vertices[vid].co
                    pts.append((float(w.x), float(w.y)))
                if len(pts) < 3:
                    continue
                try:
                    p = Polygon(pts)
                except Exception:
                    continue
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty or float(p.area) <= 1e-6:
                    continue
                polys.append(p)
        if polys:
            try:
                merged = unary_union(polys)
            except Exception:
                merged = None
            if merged is not None and (not merged.is_empty):
                geoms = list(getattr(merged, 'geoms', [merged]))
                for g in geoms:
                    if g is None or g.is_empty:
                        continue
                    if not g.is_valid:
                        g = g.buffer(0)
                    if g.is_empty:
                        continue
                    if g.geom_type == 'Polygon':
                        if float(g.area) > 1e-5:
                            wall_components.append(g)
                    elif g.geom_type == 'MultiPolygon':
                        for gg in g.geoms:
                            if gg is not None and (not gg.is_empty) and float(gg.area) > 1e-5:
                                wall_components.append(gg)
        return wall_bounds, wall_components
    finally:
        for obj in after_objs:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass


def main():
    argv = sys.argv
    if '--' not in argv:
        raise RuntimeError('Expected args after --: <output_json> <scene_path> <whitelist_json> <label_map_json> <wall_glb_path>')
    user_args = argv[argv.index('--') + 1:]
    if len(user_args) < 2:
        raise RuntimeError('Missing args')

    output_path = user_args[0]
    scene_path = user_args[1]
    whitelist_json = user_args[2] if len(user_args) > 2 else ''
    label_map_json = user_args[3] if len(user_args) > 3 else ''
    wall_glb_path = user_args[4] if len(user_args) > 4 else ''

    label_of_oid = {}
    if label_map_json:
        try:
            with open(label_map_json, 'r', encoding='utf-8') as f:
                d = json.load(f)
            if isinstance(d, dict):
                for k, v in d.items():
                    if str(v).isdigit():
                        label_of_oid[str(k)] = int(v)
        except Exception:
            label_of_oid = {}

    whitelist_pairs = load_whitelist_pairs(whitelist_json)

    scene_path_l = str(scene_path).lower()
    if scene_path_l.endswith('.blend'):
        bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(scene_path))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    wall_bounds, wall_components = _import_and_get_wall_bounds_and_components(wall_glb_path)

    groups = {}
    for obj in meshes:
        oid, cat = parse_obj_id_and_category(obj.name)
        if not oid:
            continue
        g = groups.setdefault(oid, {'id': oid, 'category_votes': [], 'objs': []})
        if cat:
            g['category_votes'].append(cat)
        g['objs'].append(obj)

    obj_groups = {}
    for oid, g in groups.items():
        cat_votes = g.get('category_votes', [])
        cat = Counter(cat_votes).most_common(1)[0][0] if cat_votes else ''

        polys = []
        aabbs = []
        points_xy = []
        for obj in g.get('objs', []):
            p = mesh_polygon_xy(obj)
            if p is not None and (not p.is_empty):
                polys.append(p)
            bb = world_aabb(obj)
            if bb is not None:
                aabbs.append(bb)
            if obj.type == 'MESH' and obj.data is not None and obj.data.vertices:
                mw = obj.matrix_world
                verts = obj.data.vertices
                step = max(1, len(verts) // 1200)
                for i in range(0, len(verts), step):
                    w = mw @ verts[i].co
                    points_xy.append((float(w.x), float(w.y)))

        if not aabbs:
            continue

        try:
            poly_u = unary_union(polys) if polys else None
        except Exception:
            poly_u = None
        if poly_u is not None and hasattr(poly_u, 'is_empty') and poly_u.is_empty:
            poly_u = None
        merged_aabb = merge_aabb(aabbs)
        obb_pca = _compute_pca_obb_xy(points_xy)

        obj_groups[oid] = {
            'id': oid,
            'category': cat,
            'polygon': poly_u,
            'aabb': merged_aabb,
            'obb': obb_pca if obb_pca is not None else _build_group_obb_xy(poly_u, merged_aabb),
            'bvh': build_group_bvh(g.get('objs', [])),
        }

    overlap = detect_overlap(obj_groups, whitelist_pairs, label_of_oid)
    wall_conflict = detect_wall_conflict(obj_groups, wall_bounds, label_of_oid, wall_components=wall_components)
    overlap_obj_set = set(str(x) for x in (overlap.get('involved_object_ids') or []))
    wall_obj_set = set(str(x) for x in (wall_conflict.get('involved_object_ids') or []))
    orientation = detect_orientation(obj_groups, wall_bounds, label_of_oid, overlap_object_ids=overlap_obj_set, wall_object_ids=wall_obj_set)

    payload = {
        'object_stats': {
            'mesh_count': len(meshes),
            'group_count': len(obj_groups),
            'wall_bounds_available': bool(wall_bounds),
            'wall_components_available': bool(wall_components),
        },
        'selected_object_ids': sorted(obj_groups.keys(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))),
        'error_type_counts': {
            'Overlap': int(overlap.get('count', 0)),
            'WallConflict': int(wall_conflict.get('count', 0)),
            'Orientation': int(orientation.get('count', 0)),
        },
        'details': {
            'Overlap': overlap,
            'WallConflict': wall_conflict,
            'Orientation': orientation,
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
'''


DEFAULT_TYPES = ["Overlap", "WallConflict", "Orientation"]
DEFAULT_WHITELIST_JSON = "InternScenes/scenes/normal_overlap_whitelist.json"
DEFAULT_LAYOUT_ROOT = "InternScenes/data/Layout_info"
DEFAULT_BLENDER_BIN = "SpatialAct/blender-3.2.2-linux-x64/blender"


@dataclass
class BlendMetrics:
    scene_path: Optional[str]
    error_type_counts: Dict[str, int]
    details: Dict
    object_stats: Dict


@dataclass
class SceneMetrics:
    glb_name: str
    scene_name: str
    scene_dir: str
    before: BlendMetrics
    after: BlendMetrics
    expected_error_type_counts: Dict[str, int]
    step_validation: List[Dict]
    metadata_matched: bool


def _resolve_blender_path(project_root: str, explicit: Optional[str] = None) -> str:
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.append(DEFAULT_BLENDER_BIN)
    candidates.append(os.path.join(project_root, "blender-3.2.2-linux-x64", "blender"))
    candidates.append(os.path.join(project_root, "blender-3.2.2-linux-x64"))
    env_blender = os.environ.get("BLENDER_PATH") or os.environ.get("BLENDER_BIN")
    if env_blender:
        candidates.append(env_blender)
    candidates.append("blender")

    for path in candidates:
        if path == "blender":
            return path
        if os.path.isdir(path):
            candidate = os.path.join(path, "blender")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(" Blender  BLENDER_PATH/BLENDER_BIN")


def _issue_main_type(issue: dict) -> str:
    main_t = str(issue.get("main_type", "")).strip()
    if main_t:
        return main_t
    t = str(issue.get("type", "")).strip()
    if " by " in t:
        t = t.split(" by ", 1)[0].strip()
    return t


def _map_issue_type_to_metric_key(main_t: str) -> Optional[str]:
    t = str(main_t or "").strip().lower()
    if t == "overlap":
        return "Overlap"
    if t in {"wall_conflict", "wall_door_conflict", "door_conflict", "path_conflict"}:
        return "WallConflict"
    if t == "orientation":
        return "Orientation"
    return None


def _expected_counts_from_record(rec: dict) -> Dict[str, int]:
    counts = {k: 0 for k in DEFAULT_TYPES}
    for st in rec.get("anomalies", []) or []:
        issue = ((st.get("issue_meta") or {}).get("issues") or [{}])[0]
        key = _map_issue_type_to_metric_key(_issue_main_type(issue))
        if key:
            counts[key] += 1
    return counts


def _label_map_from_record(rec: dict) -> Dict[str, int]:
    out = {}
    for row in rec.get("label_mapping", []) or []:
        lb = row.get("label_id")
        if lb is None or not str(lb).isdigit():
            continue
        lb_i = int(lb)

        oid = row.get("instance_index")
        if oid is None or str(oid).strip() == "":
            oid = row.get("scene_object_id")
        if oid is None:
            continue

        soid = str(oid).strip()
        if not soid:
            continue
        if soid.isdigit():
            soid = str(int(soid))
        out[soid] = lb_i
    return out


def _selected_object_ids_from_record(rec: dict) -> List[str]:
    keys = list(_label_map_from_record(rec).keys())
    return sorted(keys, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


def _find_error_glbs(workspace_root: str) -> List[str]:
    out = []
    if not os.path.isdir(workspace_root):
        return out
    for root, _, files in os.walk(workspace_root):
        if "error_scene.glb" in files:
            out.append(os.path.abspath(os.path.join(root, "error_scene.glb")))
    return sorted(out)


def _resolve_scene_glb_path(rec: dict, workspace_root: str, steps: str) -> Optional[str]:
    images = rec.get("images") or {}
    img_glb = images.get("error_scene_glb")
    if img_glb:
        p = os.path.abspath(str(img_glb))
        if os.path.isfile(p):
            return p

    glb_name = str(rec.get("glb_name", "")).strip()
    if not glb_name:
        return None
    stem = Path(glb_name).stem

    candidates = [
        os.path.join(workspace_root, stem, "error_scene.glb"),
        os.path.join(workspace_root, f"{stem}_{steps}steps", "error_scene.glb"),
        os.path.join(workspace_root, stem, f"multi_step_{steps}steps", "error_scene.glb"),
    ]

    rec_steps = rec.get("steps")
    if isinstance(rec_steps, int) and rec_steps > 0 and str(rec_steps) != str(steps):
        candidates.append(os.path.join(workspace_root, f"{stem}_{rec_steps}steps", "error_scene.glb"))
        candidates.append(os.path.join(workspace_root, stem, f"multi_step_{rec_steps}steps", "error_scene.glb"))

    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _build_wall_path(glb_name: str, layout_root: str) -> Optional[str]:
    stem = Path(glb_name).stem
    if stem.endswith("_clean"):
        stem = stem[:-6]

    root = Path(layout_root)
    cands: List[Path] = []

    if stem.startswith("3rscan__"):
        sid = stem.split("__", 1)[1]
        cands += [
            root / "3rscan" / sid / "StructureMesh/wall.glb",
            root / "3rscan" / sid / "wall.glb",
        ]
    elif stem.startswith("arkitscenes__"):
        parts = stem.split("__")
        if len(parts) >= 3:
            cands += [
                root / "arkitscenes" / parts[1] / parts[2] / "StructureMesh/wall.glb",
                root / "arkitscenes" / parts[1] / parts[2] / "wall.glb",
            ]
    elif stem.startswith("scannet__"):
        sid = stem.split("__", 1)[1]
        cands += [
            root / "scannet" / sid / "StructureMesh/wall.glb",
            root / "scannet" / sid / "wall.glb",
        ]
    elif stem.startswith("matterport3d__"):
        parts = stem.split("__")
        if len(parts) >= 3:
            cands += [
                root / "matterport3d" / parts[1] / parts[2] / "StructureMesh/wall.glb",
                root / "matterport3d" / parts[1] / parts[2] / "wall.glb",
            ]

    for p in cands:
        if p.exists():
            return str(p.resolve())
    return None


def _run_blender_detect(
    blender_bin: str,
    scene_path: str,
    whitelist_json: str,
    label_map: Dict[str, int],
    wall_glb_path: Optional[str],
) -> Dict:
    with tempfile.TemporaryDirectory(prefix="indoor_metrics_") as tmpdir:
        script_path = os.path.join(tmpdir, "detect.py")
        output_path = os.path.join(tmpdir, "detect_output.json")
        label_map_path = os.path.join(tmpdir, "label_map.json")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(BLENDER_DETECT_SCRIPT)
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False)

        cmd = [
            blender_bin,
            "--background",
            "--python",
            script_path,
            "--",
            output_path,
            scene_path,
            whitelist_json or "",
            label_map_path,
            wall_glb_path or "",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Blender :\n"
                f"cmd={' '.join(cmd)}\n"
                f"stdout_tail={proc.stdout[-1200:]}\n"
                f"stderr_tail={proc.stderr[-1200:]}"
            )

        if not os.path.exists(output_path):
            raise RuntimeError(
                "Blender  JSON\n"
                f"cmd={' '.join(cmd)}\n"
                f"stdout_tail={proc.stdout[-1200:]}\n"
                f"stderr_tail={proc.stderr[-1200:]}"
            )

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)


def _empty_detect_payload() -> Dict:
    return {
        "object_stats": {"mesh_count": 0, "group_count": 0, "wall_bounds_available": False},
        "selected_object_ids": [],
        "error_type_counts": {k: 0 for k in DEFAULT_TYPES},
        "details": {
            "Overlap": {"count": 0, "problem_pairs": [], "involved_object_ids": [], "involved_building_label_ids": []},
            "WallConflict": {"count": 0, "problems": [], "involved_object_ids": [], "involved_building_label_ids": [], "wall_bounds": None},
            "Orientation": {"count": 0, "problems": [], "involved_object_ids": [], "involved_building_label_ids": [], "threshold_deg": 12.0},
        },
    }


def _normalize_counts(payload: Dict) -> Dict[str, int]:
    raw = payload.get("error_type_counts", {}) or {}
    return {k: int(raw.get(k, 0) or 0) for k in DEFAULT_TYPES}


def _extract_step_expectations(rec: dict) -> List[Dict]:
    steps = []
    for st in rec.get("anomalies", []) or []:
        issue = ((st.get("issue_meta") or {}).get("issues") or [{}])[0]
        key = _map_issue_type_to_metric_key(_issue_main_type(issue))
        labels = []
        for x in issue.get("object_labels", []) or []:
            sx = str(x).strip()
            if sx.isdigit():
                labels.append(int(sx))
        steps.append(
            {
                "step_id": int(st.get("step_id", 0) or 0),
                "case_tag": str(st.get("case_tag", "")),
                "type": key,
                "object_label_ids": labels,
                "description": str(issue.get("description", "")),
            }
        )
    return steps


def _validate_steps(step_expectations: List[Dict], detected_details: Dict) -> Tuple[List[Dict], Dict[str, int]]:
    overlap_problem_pairs = ((detected_details.get("Overlap") or {}).get("problem_pairs") or [])
    wall_problems = ((detected_details.get("WallConflict") or {}).get("problems") or [])
    ori_problems = ((detected_details.get("Orientation") or {}).get("problems") or [])

    overlap_pairs = set()
    for p in overlap_problem_pairs:
        lbs = [x for x in (p.get("object_label_ids") or p.get("label_ids") or []) if isinstance(x, int)]
        if len(lbs) >= 2:
            overlap_pairs.add(tuple(sorted(lbs[:2])))

    wall_labels = set(int(x) for x in ((detected_details.get("WallConflict") or {}).get("involved_building_label_ids") or []) if str(x).isdigit())
    ori_labels = set(int(x) for x in ((detected_details.get("Orientation") or {}).get("involved_building_label_ids") or []) if str(x).isdigit())

    validations: List[Dict] = []
    aligned = {k: 0 for k in DEFAULT_TYPES}

    for st in step_expectations:
        t = st.get("type")
        lbs = [int(x) for x in st.get("object_label_ids", []) if str(x).isdigit()]
        ok = False
        reason = ""
        matched_details: List[Dict] = []

        if t == "Overlap":
            if len(lbs) >= 2:
                key = tuple(sorted(lbs[:2]))
                ok = key in overlap_pairs
                reason = ("target overlap pair detected" if ok else "target overlap pair not found in detected overlap pairs")
                if ok:
                    for p in overlap_problem_pairs:
                        p_lbs = [x for x in (p.get("object_label_ids") or p.get("label_ids") or []) if isinstance(x, int)]
                        if len(p_lbs) >= 2 and tuple(sorted(p_lbs[:2])) == key:
                            matched_details.append(
                                {
                                    "object_label_ids": p_lbs[:2],
                                    "reason": str(p.get("reason", "")),
                                }
                            )
            else:
                ok = False
                reason = "invalid overlap labels in metadata"
        elif t == "WallConflict":
            ok = any(lb in wall_labels for lb in lbs)
            reason = ("target label appears in wall-conflict set" if ok else "target label not found in wall-conflict set")
            if ok:
                for p in wall_problems:
                    lb = p.get("object_label_id", p.get("label_id"))
                    if str(lb).isdigit() and int(lb) in lbs:
                        matched_details.append(
                            {
                                "object_label_id": int(lb),
                                "reason": str(p.get("reason", "")),
                            }
                        )
        elif t == "Orientation":
            ok = any(lb in ori_labels for lb in lbs)
            reason = ("target label appears in orientation-conflict set" if ok else "target label not found in orientation-conflict set")
            if ok:
                for p in ori_problems:
                    lb = p.get("object_label_id", p.get("label_id"))
                    if str(lb).isdigit() and int(lb) in lbs:
                        matched_details.append(
                            {
                                "object_label_id": int(lb),
                                "reason": str(p.get("reason", "")),
                            }
                        )
        else:
            ok = False
            reason = "unknown or unsupported issue type"

        if ok and t in aligned:
            aligned[t] += 1

        validations.append(
            {
                "step_id": int(st.get("step_id", 0) or 0),
                "case_tag": st.get("case_tag", ""),
                "type": t,
                "object_label_ids": lbs,
                "passed": bool(ok),
                "reason": reason,
                "matched_problem_reasons": matched_details,
                "expected_description": st.get("description", ""),
            }
        )

    return validations, aligned


def _counter_from_counts(items: List[SceneMetrics], attr: str) -> Counter:
    acc: Counter = Counter()
    for row in items:
        d = getattr(row, attr, None)
        if isinstance(d, dict):
            acc.update(d)
    return acc


def _safe_int(x) -> Optional[int]:
    try:
        s = str(x).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
    except Exception:
        return None
    return None


def _label_focused_details(details: Dict) -> Dict:
    details = details or {}
    out: Dict = {}

    overlap = details.get("Overlap") or {}
    overlap_pairs = []
    overlap_labels = set()
    for p in (overlap.get("problem_pairs") or []):
        raw_lbs = p.get("object_label_ids") or p.get("label_ids") or []
        raw_oids = p.get("object_ids") or []
        lbs = []
        for x in raw_lbs:
            xi = _safe_int(x)
            if xi is not None:
                lbs.append(xi)
        oids = [str(x) for x in raw_oids if str(x).strip()]
        if lbs:
            overlap_labels.update(lbs)
        overlap_pairs.append(
            {
                "object_ids": oids,
                "object_label_ids": lbs,
                "reason": str(p.get("reason", "")),
            }
        )
    for x in (overlap.get("involved_building_label_ids") or []):
        xi = _safe_int(x)
        if xi is not None:
            overlap_labels.add(xi)
    out["Overlap"] = {
        "count": int(overlap.get("count", 0) or 0),
        "involved_building_label_ids": sorted(overlap_labels),
        "problem_pairs": overlap_pairs,
    }

    wall = details.get("WallConflict") or {}
    wall_probs = []
    wall_labels = set()
    for p in (wall.get("problems") or []):
        lb = _safe_int(p.get("object_label_id", p.get("label_id")))
        oid = str(p.get("object_id", "")).strip()
        if lb is not None:
            wall_labels.add(lb)
        wall_probs.append(
            {
                "object_id": oid,
                "object_label_id": lb,
                "reason": str(p.get("reason", "")),
            }
        )
    for x in (wall.get("involved_building_label_ids") or []):
        xi = _safe_int(x)
        if xi is not None:
            wall_labels.add(xi)
    out["WallConflict"] = {
        "count": int(wall.get("count", 0) or 0),
        "involved_building_label_ids": sorted(wall_labels),
        "problems": wall_probs,
        "wall_bounds": wall.get("wall_bounds"),
    }

    ori = details.get("Orientation") or {}
    ori_probs = []
    ori_labels = set()
    for p in (ori.get("problems") or []):
        lb = _safe_int(p.get("object_label_id", p.get("label_id")))
        oid = str(p.get("object_id", "")).strip()
        if lb is not None:
            ori_labels.add(lb)
        ori_probs.append(
            {
                "object_id": oid,
                "object_label_id": lb,
                "reason": str(p.get("reason", "")),
            }
        )
    for x in (ori.get("involved_building_label_ids") or []):
        xi = _safe_int(x)
        if xi is not None:
            ori_labels.add(xi)
    out["Orientation"] = {
        "count": int(ori.get("count", 0) or 0),
        "involved_building_label_ids": sorted(ori_labels),
        "problems": ori_probs,
        "threshold_deg": ori.get("threshold_deg", 12.0),
    }
    return out


def _build_metadata_indexes(records: List[dict]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    by_glb_name: Dict[str, dict] = {}
    by_scene_stem: Dict[str, dict] = {}
    for rec in records:
        glb_name = str(rec.get("glb_name", "")).strip()
        if glb_name:
            by_glb_name[glb_name] = rec
            by_scene_stem[Path(glb_name).stem] = rec
    return by_glb_name, by_scene_stem


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_if_exists(path: Optional[str]):
    if not path:
        return None
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        return None
    try:
        return _load_json(p)
    except Exception:
        return None


def _empty_blend_metrics(scene_path: Optional[str] = None) -> BlendMetrics:
    return BlendMetrics(
        scene_path=scene_path,
        error_type_counts={k: 0 for k in DEFAULT_TYPES},
        details={
            "Overlap": {"count": 0, "involved_building_label_ids": [], "problem_pairs": []},
            "WallConflict": {"count": 0, "involved_building_label_ids": [], "problems": [], "wall_bounds": None},
            "Orientation": {"count": 0, "involved_building_label_ids": [], "problems": [], "threshold_deg": 12.0},
        },
        object_stats={"mesh_count": 0, "group_count": 0, "wall_bounds_available": False},
    )


def _detect_scene_metrics(
    blender_bin: str,
    scene_path: Optional[str],
    whitelist_json: str,
    label_map: Dict[str, int],
    wall_glb_path: Optional[str],
) -> Tuple[BlendMetrics, Dict]:
    if not scene_path or not os.path.isfile(scene_path):
        empty = _empty_blend_metrics(scene_path=scene_path)
        return empty, _empty_detect_payload()

    payload = _run_blender_detect(
        blender_bin=blender_bin,
        scene_path=scene_path,
        whitelist_json=whitelist_json,
        label_map=label_map,
        wall_glb_path=wall_glb_path,
    )
    metrics = BlendMetrics(
        scene_path=scene_path,
        error_type_counts=_normalize_counts(payload),
        details=_label_focused_details(payload.get("details", {}) or {}),
        object_stats=payload.get("object_stats", {}) or {},
    )
    return metrics, payload


def _iter_scene_dirs(workspace_root: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(workspace_root):
        return out
    for name in sorted(os.listdir(workspace_root)):
        if name.startswith("."):
            continue
        scene_dir = os.path.join(workspace_root, name)
        if not os.path.isdir(scene_dir):
            continue
        files = set(os.listdir(scene_dir))
        has_signal = (
            "region_info.json" in files
            or "labels.json" in files
            or any(fn.startswith("final_scene") and fn.endswith(".blend") for fn in files)
            or "error_scene.glb" in files
            or "error_scene.blend" in files
        )
        if has_signal:
            out.append(scene_dir)
    return out


def _load_region_info(scene_dir: str) -> Dict:
    p = os.path.join(scene_dir, "region_info.json")
    if not os.path.isfile(p):
        return {}
    try:
        d = _load_json(p)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _normalize_oid(text) -> str:
    s = str(text or "").strip()
    if s.isdigit():
        return str(int(s))
    return s


def _load_label_map_from_scene_dir(scene_dir: str) -> Dict[str, int]:
    p = os.path.join(scene_dir, "labels.json")
    if not os.path.isfile(p):
        return {}
    try:
        data = _load_json(p)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        if not str(k).isdigit():
            continue
        oid = _normalize_oid(v)
        if not oid:
            continue
        out[oid] = int(k)
    return out


def _resolve_before_scene_path(scene_dir: str, region_info: Dict, steps: str) -> Optional[str]:
    cands: List[str] = []
    src_path = str(((region_info.get("source") or {}).get("source_scene_path") or "")).strip()
    if src_path:
        cands.append(src_path)
    if steps:
        cands.append(os.path.join(scene_dir, f"error_scene_steps{steps}.blend"))
    cands.append(os.path.join(scene_dir, "error_scene.blend"))
    cands.append(os.path.join(scene_dir, "error_scene.glb"))

    for p in cands:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _resolve_after_scene_path(scene_dir: str, steps: str) -> Optional[str]:
    cands: List[str] = []
    if steps:
        cands.append(os.path.join(scene_dir, f"final_scene_steps{steps}.blend"))
    cands.append(os.path.join(scene_dir, "final_scene.blend"))
    cands.append(os.path.join(scene_dir, "final_scene.glb"))
    for p in cands:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _resolve_metadata_record(
    scene_dir: str,
    region_info: Dict,
    metadata_by_name: Dict[str, dict],
    metadata_by_stem: Dict[str, dict],
) -> Optional[dict]:
    glb_name = str(region_info.get("glb_name", "")).strip()
    if glb_name and glb_name in metadata_by_name:
        return metadata_by_name[glb_name]
    if glb_name:
        stem = Path(glb_name).stem
        if stem in metadata_by_stem:
            return metadata_by_stem[stem]

    dir_name = os.path.basename(scene_dir)
    aliases = [dir_name, dir_name.replace("__", "_"), dir_name.replace("_", "__")]
    for stem in aliases:
        rec = metadata_by_stem.get(stem)
        if rec:
            return rec
    return None


def _build_scene_row(
    scene_dir: str,
    steps: str,
    blender_bin: str,
    whitelist_json: str,
    layout_root: str,
    metadata_by_name: Dict[str, dict],
    metadata_by_stem: Dict[str, dict],
    strict_missing_before: bool,
    strict_missing_after: bool,
) -> SceneMetrics:
    region_info = _load_region_info(scene_dir)
    rec = _resolve_metadata_record(scene_dir, region_info, metadata_by_name, metadata_by_stem)

    if rec:
        glb_name = str(rec.get("glb_name", "")).strip()
        scene_name = str(rec.get("scene_name", "")).strip() or os.path.basename(scene_dir)
        expected = _expected_counts_from_record(rec)
        step_expect = _extract_step_expectations(rec)
    else:
        glb_name = str(region_info.get("glb_name", "")).strip() or f"{os.path.basename(scene_dir)}.glb"
        scene_name = str(region_info.get("scene_name", "")).strip() or os.path.basename(scene_dir)
        expected = {k: 0 for k in DEFAULT_TYPES}
        step_expect = []

    label_map = _load_label_map_from_scene_dir(scene_dir)
    if (not label_map) and rec:
        label_map = _label_map_from_record(rec)

    wall_path = _build_wall_path(glb_name, layout_root) if glb_name else None
    before_path = _resolve_before_scene_path(scene_dir, region_info, steps)
    after_path = _resolve_after_scene_path(scene_dir, steps)

    if strict_missing_before and (not before_path):
        raise FileNotFoundError(f" before : {scene_dir}")
    if strict_missing_after and (not after_path):
        raise FileNotFoundError(f" final_scene : {scene_dir}")

    before_metrics, _before_payload = _detect_scene_metrics(
        blender_bin=blender_bin,
        scene_path=before_path,
        whitelist_json=whitelist_json,
        label_map=label_map,
        wall_glb_path=wall_path,
    )
    after_metrics, after_payload = _detect_scene_metrics(
        blender_bin=blender_bin,
        scene_path=after_path,
        whitelist_json=whitelist_json,
        label_map=label_map,
        wall_glb_path=wall_path,
    )

    if rec:
        step_validation, _ = _validate_steps(step_expect, (after_payload.get("details", {}) or {}))
    else:
        step_validation = []

    return SceneMetrics(
        glb_name=glb_name,
        scene_name=scene_name,
        scene_dir=scene_dir,
        before=before_metrics,
        after=after_metrics,
        expected_error_type_counts=expected,
        step_validation=step_validation,
        metadata_matched=bool(rec),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indoor  before/final  metadata ")
    parser.add_argument("--workspace-root", default=None, help="run_iteration  indoor  --blend-path ")
    parser.add_argument("--blend-path", default=None, help=".blend/.glb --workspace-root ")
    parser.add_argument("--scene-dir", default=None, help=" region_info/labels")
    parser.add_argument("--metadata", default=None, help="metadata_indoor_stepsN.json  expected  detect")
    parser.add_argument("--steps", default=os.environ.get("STEPS", os.environ.get("ERROR_STEPS", "3")), help=" N")
    parser.add_argument("--layout-root", default=DEFAULT_LAYOUT_ROOT, help="Layout_info  wall.glb")
    parser.add_argument("--whitelist-json", default=DEFAULT_WHITELIST_JSON, help="overlap  JSON ")
    parser.add_argument("--max-scenes", type=int, default=0, help="0=")
    parser.add_argument("--keyword", default="", help=" scene_dir/glb_name/scene_name ")
    parser.add_argument("--strict-missing-before", action="store_true", help="before ")
    parser.add_argument("--strict-missing-after", action="store_true", help="final_scene ")
    parser.add_argument("--blender", default=DEFAULT_BLENDER_BIN, help="Blender ")
    parser.add_argument("--output", default=None, help=" JSON ")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.workspace_root and not args.blend_path:
        raise ValueError(" --workspace-root  --blend-path ")
    if args.workspace_root and args.blend_path:
        raise ValueError("--workspace-root  --blend-path ")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    steps = str(args.steps).strip() if args.steps is not None else "3"
    blender_bin = _resolve_blender_path(project_root, args.blender)

    workspace_root = os.path.abspath(args.workspace_root) if args.workspace_root else None
    metadata_path = None
    metadata_records: List[dict] = []
    metadata_by_name: Dict[str, dict] = {}
    metadata_by_stem: Dict[str, dict] = {}

    metadata_candidate = None
    if args.metadata:
        metadata_candidate = os.path.abspath(args.metadata)
    elif workspace_root:
        metadata_candidate = os.path.join(workspace_root, f"metadata_indoor_steps{steps}.json")
    if metadata_candidate and os.path.isfile(metadata_candidate):
        maybe_records = _load_json_if_exists(metadata_candidate)
        if isinstance(maybe_records, list):
            metadata_records = maybe_records
            metadata_by_name, metadata_by_stem = _build_metadata_indexes(metadata_records)
            metadata_path = os.path.abspath(metadata_candidate)

    rows: List[SceneMetrics] = []
    mode = "single_blend"

    if args.blend_path:
        scene_path = os.path.abspath(args.blend_path)
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(f": {scene_path}")

        scene_dir = os.path.abspath(args.scene_dir) if args.scene_dir else os.path.abspath(os.path.dirname(scene_path))
        row = _build_scene_row(
            scene_dir=scene_dir,
            steps=steps,
            blender_bin=blender_bin,
            whitelist_json=args.whitelist_json,
            layout_root=args.layout_root,
            metadata_by_name=metadata_by_name,
            metadata_by_stem=metadata_by_stem,
            strict_missing_before=args.strict_missing_before,
            strict_missing_after=False,  #  scene_dir  final_scene
        )

        # Force single-file "after" evaluation to use user-provided file.
        label_map = _load_label_map_from_scene_dir(scene_dir)
        if (not label_map) and row.metadata_matched:
            rec = _resolve_metadata_record(scene_dir, _load_region_info(scene_dir), metadata_by_name, metadata_by_stem)
            if rec:
                label_map = _label_map_from_record(rec)
        wall_path = _build_wall_path(row.glb_name, args.layout_root) if row.glb_name else None
        after_metrics, after_payload = _detect_scene_metrics(
            blender_bin=blender_bin,
            scene_path=scene_path,
            whitelist_json=args.whitelist_json,
            label_map=label_map,
            wall_glb_path=wall_path,
        )
        row.after = after_metrics
        if row.metadata_matched:
            rec = _resolve_metadata_record(scene_dir, _load_region_info(scene_dir), metadata_by_name, metadata_by_stem)
            if rec:
                row.step_validation, _ = _validate_steps(_extract_step_expectations(rec), (after_payload.get("details", {}) or {}))
        rows = [row]
        mode = "single_blend"
    else:
        if not workspace_root or (not os.path.isdir(workspace_root)):
            raise FileNotFoundError(f"workspace_root : {workspace_root}")
        scene_dirs = _iter_scene_dirs(workspace_root)
        if not scene_dirs:
            raise RuntimeError(f"workspace_root : {workspace_root}")

        for scene_dir in scene_dirs:
            row = _build_scene_row(
                scene_dir=scene_dir,
                steps=steps,
                blender_bin=blender_bin,
                whitelist_json=args.whitelist_json,
                layout_root=args.layout_root,
                metadata_by_name=metadata_by_name,
                metadata_by_stem=metadata_by_stem,
                strict_missing_before=args.strict_missing_before,
                strict_missing_after=args.strict_missing_after,
            )
            if args.keyword:
                kw = str(args.keyword).strip().lower()
                hay = " ".join([os.path.basename(scene_dir), row.glb_name, row.scene_name]).lower()
                if kw not in hay:
                    continue
            rows.append(row)
            if args.max_scenes > 0 and len(rows) >= int(args.max_scenes):
                break
        mode = "workspace"

    if not rows:
        raise RuntimeError("")

    before_total_by_type: Counter = Counter()
    after_total_by_type: Counter = Counter()
    expected_total_by_type: Counter = Counter()
    delta_after_before_by_type: Counter = Counter()
    delta_after_expected_by_type: Counter = Counter()

    for r in rows:
        before_total_by_type.update(r.before.error_type_counts)
        after_total_by_type.update(r.after.error_type_counts)
        expected_total_by_type.update(r.expected_error_type_counts)
        delta_after_before_by_type.update(
            {k: int(r.after.error_type_counts.get(k, 0) - r.before.error_type_counts.get(k, 0)) for k in DEFAULT_TYPES}
        )
        delta_after_expected_by_type.update(
            {k: int(r.after.error_type_counts.get(k, 0) - r.expected_error_type_counts.get(k, 0)) for k in DEFAULT_TYPES}
        )

    before_total_errors = int(sum(before_total_by_type.values()))
    after_total_errors = int(sum(after_total_by_type.values()))
    expected_total_errors = int(sum(expected_total_by_type.values()))

    out = {
        "workspace_root": workspace_root,
        "blend_path": os.path.abspath(args.blend_path) if args.blend_path else None,
        "metadata": metadata_path,
        "blender": blender_bin,
        "whitelist_json": os.path.abspath(args.whitelist_json) if args.whitelist_json else None,
        "layout_root": os.path.abspath(args.layout_root) if args.layout_root else None,
        "mode": mode,
        "steps": steps,
        "summary": {
            "scenes_total": len(rows),
            "before_total_errors": before_total_errors,
            "after_total_errors": after_total_errors,
            "expected_total_errors": expected_total_errors,
            "delta_after_vs_before_total_errors": after_total_errors - before_total_errors,
            "delta_after_vs_expected_total_errors": after_total_errors - expected_total_errors,
            "before_error_type_counts": dict(sorted(before_total_by_type.items(), key=lambda kv: kv[0])),
            "after_error_type_counts": dict(sorted(after_total_by_type.items(), key=lambda kv: kv[0])),
            "expected_error_type_counts": dict(sorted(expected_total_by_type.items(), key=lambda kv: kv[0])),
            "delta_after_vs_before_error_type_counts": dict(sorted(delta_after_before_by_type.items(), key=lambda kv: kv[0])),
            "delta_after_vs_expected_error_type_counts": dict(sorted(delta_after_expected_by_type.items(), key=lambda kv: kv[0])),
            "metadata_matched_scenes": int(sum(1 for r in rows if r.metadata_matched)),
            "rule": (
                "Detection is geometry-only from scene files (before/final). "
                "Metadata is used only for expected-count comparison and step-level reference validation."
            ),
        },
        "scenes": [],
    }

    for r in rows:
        before_total = int(sum(r.before.error_type_counts.values()))
        after_total = int(sum(r.after.error_type_counts.values()))
        expected_total = int(sum(r.expected_error_type_counts.values()))
        out["scenes"].append(
            {
                "scene_dir": r.scene_dir,
                "scene_name": r.scene_name,
                "glb_name": r.glb_name,
                "metadata_matched": bool(r.metadata_matched),
                "before": {
                    "scene_path": r.before.scene_path,
                    "total_errors": before_total,
                    "error_type_counts": r.before.error_type_counts,
                    "object_stats": r.before.object_stats,
                    "details": r.before.details,
                },
                "after": {
                    "scene_path": r.after.scene_path,
                    "total_errors": after_total,
                    "error_type_counts": r.after.error_type_counts,
                    "object_stats": r.after.object_stats,
                    "details": r.after.details,
                },
                "expected": {
                    "total_errors": expected_total,
                    "error_type_counts": r.expected_error_type_counts,
                },
                "delta_after_vs_before_total_errors": after_total - before_total,
                "delta_after_vs_before_error_type_counts": {
                    k: int(r.after.error_type_counts.get(k, 0) - r.before.error_type_counts.get(k, 0))
                    for k in DEFAULT_TYPES
                },
                "delta_after_vs_expected_total_errors": after_total - expected_total,
                "delta_after_vs_expected_error_type_counts": {
                    k: int(r.after.error_type_counts.get(k, 0) - r.expected_error_type_counts.get(k, 0))
                    for k in DEFAULT_TYPES
                },
                "step_validation": r.step_validation,
            }
        )

    step_suffix = str(steps).strip()
    if args.output:
        output_path = os.path.abspath(args.output)
    elif workspace_root:
        output_path = os.path.join(
            workspace_root,
            (f"indoor_metrics_summary_blend_steps{step_suffix}.json" if step_suffix else "indoor_metrics_summary_blend.json"),
        )
    else:
        base_dir = os.path.dirname(os.path.abspath(args.blend_path))
        output_path = os.path.join(
            base_dir,
            (f"indoor_metrics_single_blend_steps{step_suffix}.json" if step_suffix else "indoor_metrics_single_blend.json"),
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("idx\tbefore\tafter\texpected\tdelta_after_before\tdelta_after_expected\tglb_name")
    for idx, r in enumerate(rows, start=1):
        before_total = int(sum(r.before.error_type_counts.values()))
        after_total = int(sum(r.after.error_type_counts.values()))
        expected_total = int(sum(r.expected_error_type_counts.values()))
        print(
            f"{idx}\t{before_total}\t{after_total}\t{expected_total}\t"
            f"{after_total-before_total}\t{after_total-expected_total}\t{r.glb_name}"
        )

    print(f"\n[OK] indoor metrics written to: {output_path}")


if __name__ == "__main__":
    main()
