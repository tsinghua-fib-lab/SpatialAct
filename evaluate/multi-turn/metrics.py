#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


BLENDER_DETECT_SCRIPT = r'''
import bpy
import json
import math
import sys
from mathutils import Vector
from mathutils.geometry import convex_hull_2d
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

ORIENTATION_NEAREST_ROAD_RADIUS = 500.0
ORIENTATION_ROAD_EDGE_SAMPLE_MAX = 1200
ORIENTATION_MAIN_ROAD_LOCAL_RADIUS = 80.0
ORIENTATION_ERROR_THRESHOLD_DEG = 15.0
ORIENTATION_CENTERLINE_WINDOW_RADIUS = 45.0


def is_building(name: str) -> bool:
    token = (name or '').lower()
    exclude = ['vegetation', 'water', 'forest', 'terrain', 'road', 'ground', 'tree', 'bush', 'label']
    include = ['building', 'buildings', 'osm_buildings']
    if any(k in token for k in exclude):
        return False
    return any(k in token for k in include)


def is_road(name: str) -> bool:
    token = (name or '').lower()
    keys = ['road', 'street', 'way', 'avenue', 'lane', 'drive']
    return any(k in token for k in keys)


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
        'min_x': min(xs), 'max_x': max(xs),
        'min_y': min(ys), 'max_y': max(ys),
        'min_z': min(zs), 'max_z': max(zs),
    }


def rect_intersects(a, b):
    return not (
        a['max_x'] < b['min_x'] or
        a['min_x'] > b['max_x'] or
        a['max_y'] < b['min_y'] or
        a['min_y'] > b['max_y']
    )


def normalize_angle_180(deg):
    x = abs(float(deg)) % 180.0
    if x > 90.0:
        x = 180.0 - x
    return x


def extract_footprint_polygon_xy(obj, max_points=4000):
    if obj is None or obj.type != 'MESH' or obj.data is None or not obj.data.vertices:
        return []

    mw = obj.matrix_world
    verts = obj.data.vertices
    step = max(1, len(verts) // max_points)

    pts = []
    for i in range(0, len(verts), step):
        w = mw @ verts[i].co
        pts.append(Vector((float(w.x), float(w.y))))

    if len(pts) < 3:
        return []

    hull_idx = convex_hull_2d(pts)
    poly = [(float(pts[i].x), float(pts[i].y)) for i in hull_idx]
    return poly


def get_building_footprint(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    try:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
    except Exception:
        return None

    if mesh is None or len(mesh.vertices) < 3:
        return None

    world_matrix = obj.matrix_world
    transform = world_matrix.to_3x3()
    valid_polygons = []

    for poly in mesh.polygons:
        world_coords = [world_matrix @ mesh.vertices[idx].co for idx in poly.vertices]
        if len(world_coords) < 3:
            continue

        world_normal = transform @ poly.normal
        if abs(world_normal.z) < 0.7:
            continue

        coords_2d = [(v.x, v.y) for v in world_coords]
        face_poly = Polygon(coords_2d)
        if not face_poly.is_valid:
            face_poly = face_poly.buffer(0)
        if not face_poly.is_empty and face_poly.area > 0.01:
            valid_polygons.append(face_poly)

    obj_eval.to_mesh_clear()

    if not valid_polygons:
        return None

    try:
        footprint = unary_union(valid_polygons)
        if footprint.is_empty:
            return None
        return footprint
    except Exception:
        return None


def _get_building_polygon(obj):
    poly = get_building_footprint(obj)
    if poly is None or poly.is_empty:
        pts = extract_footprint_polygon_xy(obj)
        if len(pts) >= 3:
            try:
                poly = Polygon(pts)
            except Exception:
                poly = None
    if poly is None or poly.is_empty:
        return None
    return poly


def get_road_faces_in_region(road_obj, region_bounds=None, buffer=0.0):
    if region_bounds is None:
        region_bounds = {'min_x': -1e6, 'max_x': 1e6, 'min_y': -1e6, 'max_y': 1e6}
    if road_obj.type != 'MESH' or road_obj.data is None:
        return []

    road_faces = []
    mesh = road_obj.data
    mw = road_obj.matrix_world
    world_verts = [mw @ v.co for v in mesh.vertices]
    for poly in mesh.polygons:
        face_vs = [world_verts[i] for i in poly.vertices]
        coords_2d = [(v.x, v.y) for v in face_vs]
        if not coords_2d:
            continue

        cx = sum(v[0] for v in coords_2d) / len(coords_2d)
        cy = sum(v[1] for v in coords_2d) / len(coords_2d)
        if not (
            region_bounds['min_x'] - buffer <= cx <= region_bounds['max_x'] + buffer and
            region_bounds['min_y'] - buffer <= cy <= region_bounds['max_y'] + buffer
        ):
            continue

        if len(coords_2d) < 3:
            continue
        face_poly = Polygon(coords_2d)
        if not face_poly.is_valid:
            face_poly = face_poly.buffer(0)
        if face_poly.is_empty:
            continue
        road_faces.append({'geom': face_poly, 'verts': coords_2d})

    return road_faces


def detect_overlap(buildings):
    count = 0
    involved = set()
    for i in range(len(buildings)):
        a_obj = buildings[i]
        poly_a = _get_building_polygon(a_obj)
        if poly_a is None:
            continue
        for j in range(i + 1, len(buildings)):
            b_obj = buildings[j]
            poly_b = _get_building_polygon(b_obj)
            if poly_b is None:
                continue
            if not poly_a.intersects(poly_b):
                continue
            try:
                inter_area = poly_a.intersection(poly_b).area
            except Exception:
                inter_area = 0.0
            if inter_area > 0.02:
                count += 1
                involved.add(a_obj.name)
                involved.add(b_obj.name)
    return {'count': count, 'involved_object_ids': sorted(involved)}


def detect_road_conflict(buildings, roads, threshold_area=0.1):
    if not roads or not buildings:
        return {'count': 0, 'involved_object_ids': []}

    min_x, min_y = 1e9, 1e9
    max_x, max_y = -1e9, -1e9
    for b in buildings:
        bb = world_aabb(b)
        if not bb:
            continue
        min_x = min(min_x, bb['min_x'])
        min_y = min(min_y, bb['min_y'])
        max_x = max(max_x, bb['max_x'])
        max_y = max(max_y, bb['max_y'])

    region_bounds = {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}
    road_polys = []
    for r_obj in roads:
        faces = get_road_faces_in_region(r_obj, region_bounds=region_bounds, buffer=0.0)
        for f in faces:
            road_polys.append(f['geom'])

    if not road_polys:
        return {'count': 0, 'involved_object_ids': []}

    road_union = unary_union(road_polys)
    if road_union.is_empty:
        return {'count': 0, 'involved_object_ids': []}

    involved = []
    for b in buildings:
        poly_b = _get_building_polygon(b)
        if poly_b is None:
            continue
        if not poly_b.intersects(road_union):
            continue
        try:
            area = poly_b.intersection(road_union).area
        except Exception:
            area = 0.0
        if area > threshold_area:
            involved.append(b.name)

    return {'count': len(involved), 'involved_object_ids': sorted(involved)}


def _check_nearby_road_orthogonality(target_obj, road_objs, radius=40.0):
    if not road_objs:
        return False

    center = target_obj.location
    nearby_vecs = []
    for r_obj in road_objs:
        dist = (Vector((r_obj.location.x, r_obj.location.y, 0)) - Vector((center.x, center.y, 0))).length
        if dist > radius + max(r_obj.dimensions):
            continue
        try:
            mesh = r_obj.data
            indices = range(0, len(mesh.edges), max(1, len(mesh.edges) // 20))
            for i in indices:
                edge = mesh.edges[i]
                v1_w = r_obj.matrix_world @ mesh.vertices[edge.vertices[0]].co
                v2_w = r_obj.matrix_world @ mesh.vertices[edge.vertices[1]].co
                mid = (v1_w + v2_w) / 2
                if (mid - center).length < radius:
                    vec = v2_w - v1_w
                    vec.z = 0
                    if vec.length > 0.5:
                        nearby_vecs.append(vec.normalized())
        except Exception:
            pass
        if len(nearby_vecs) > 30:
            break

    if len(nearby_vecs) < 2:
        return False

    for i in range(len(nearby_vecs)):
        for j in range(i + 1, len(nearby_vecs)):
            angle = nearby_vecs[i].angle(nearby_vecs[j])
            deg = math.degrees(angle)
            deg = deg % 180
            if deg > 90:
                deg = 180 - deg
            if abs(deg - 90) < 15:
                return True
    return False


def _is_orientation_candidate_shape(target_obj):
    if target_obj is None:
        return False

    try:
        footprint = get_building_footprint(target_obj)
        if footprint is None or footprint.is_empty:
            pts = extract_footprint_polygon_xy(target_obj)
            if len(pts) >= 3:
                footprint = Polygon(pts)
    except Exception:
        return False

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

    def _angle_diff_mod_180(a, b):
        d = abs((a - b) % 180.0)
        if d > 90.0:
            d = 180.0 - d
        return d

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

    # Moderately simplify contour to suppress tiny mesh zig-zags while
    # avoiding over-smoothing true shape features.
    simple = footprint.simplify(0.25, preserve_topology=True)
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
    # Keep a conservative lower bound: allow mildly fragmented orthogonal
    # footprints, but avoid admitting clearly irregular outlines.
    if right_ratio < 0.18:
        return False

    is_rectangle = (corner_count <= 5) and (rect_ratio >= 0.82) and (convex_ratio >= 0.95)
    # Slightly widen L-shape envelope for rectilinear outlines with one to
    # three extra corners from mesh tessellation.
    is_l_shape = (6 <= corner_count <= 11) and (0.45 <= rect_ratio <= 0.85) and (0.55 <= convex_ratio <= 0.94)

    return is_rectangle or is_l_shape


def _normalize_angle_180(deg):
    d = float(deg) % 180.0
    if d < 0.0:
        d += 180.0
    return d


def _angle_diff_180(a_deg, b_deg):
    a = _normalize_angle_180(a_deg)
    b = _normalize_angle_180(b_deg)
    d = abs(a - b)
    return min(d, 180.0 - d)


def _axis_error_deg(curr_deg, axis_deg):
    return min(_angle_diff_180(curr_deg, axis_deg), _angle_diff_180(curr_deg, axis_deg + 90.0))


def _polygon_major_axis_deg(poly):
    if poly is None or poly.is_empty:
        return None
    try:
        mrr = poly.minimum_rotated_rectangle
    except Exception:
        return None
    if mrr is None or mrr.is_empty:
        return None
    try:
        coords = list(mrr.exterior.coords)
    except Exception:
        return None
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        if math.hypot(dx, dy) > 1e-6:
            return _normalize_angle_180(math.degrees(math.atan2(dy, dx)))
    return None


def _largest_polygon(poly):
    if poly is None or poly.is_empty:
        return None
    if getattr(poly, "geom_type", "") == "MultiPolygon":
        polys = [g for g in poly.geoms if g and (not g.is_empty)]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    return poly


def _extract_orientation_footprint(target_obj):
    if target_obj is None:
        return None
    footprint = get_building_footprint(target_obj)
    if footprint is None or footprint.is_empty:
        pts = extract_footprint_polygon_xy(target_obj)
        if len(pts) >= 3:
            try:
                footprint = Polygon(pts)
            except Exception:
                footprint = None
    return _largest_polygon(footprint)


def _building_edge_axis_deg(target_obj):
    """
    Building direction from a real footprint edge:
    - rectangle: usually picks one long edge
    - L-shape: picks dominant outer edge direction
    """
    footprint = _extract_orientation_footprint(target_obj)
    if footprint is None or footprint.is_empty:
        return None

    simple = footprint.simplify(0.15, preserve_topology=True)
    if simple is None or simple.is_empty or not hasattr(simple, "exterior"):
        simple = footprint

    coords = list(simple.exterior.coords) if hasattr(simple, "exterior") else []
    if len(coords) < 4:
        return _polygon_major_axis_deg(footprint)

    best_len = -1.0
    best_ang = None
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx, dy = (x2 - x1), (y2 - y1)
        seg_len = math.hypot(dx, dy)
        if seg_len < 0.5:
            continue
        ang = _normalize_angle_180(math.degrees(math.atan2(dy, dx)))
        if seg_len > best_len:
            best_len = seg_len
            best_ang = ang

    if best_ang is not None:
        return best_ang
    return _polygon_major_axis_deg(footprint)


_ROAD_EDGE_SAMPLE_CACHE = {}


def _get_sampled_road_edges(road_obj):
    """
    Cache sampled road edges: [(mid_x, mid_y, angle_deg, length), ...]
    """
    if road_obj is None or road_obj.type != "MESH" or road_obj.data is None:
        return []
    try:
        key = (
            road_obj.name_full,
            len(road_obj.data.vertices),
            len(road_obj.data.edges),
        )
    except Exception:
        key = (road_obj.name, 0, 0)
    cached = _ROAD_EDGE_SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached

    samples = []
    try:
        mesh = road_obj.data
        if len(mesh.edges) <= 0:
            _ROAD_EDGE_SAMPLE_CACHE[key] = samples
            return samples

        step = max(1, len(mesh.edges) // max(1, ORIENTATION_ROAD_EDGE_SAMPLE_MAX))
        for i in range(0, len(mesh.edges), step):
            edge = mesh.edges[i]
            v1_w = road_obj.matrix_world @ mesh.vertices[edge.vertices[0]].co
            v2_w = road_obj.matrix_world @ mesh.vertices[edge.vertices[1]].co
            dx = float(v2_w.x - v1_w.x)
            dy = float(v2_w.y - v1_w.y)
            seg_len = math.hypot(dx, dy)
            if seg_len <= 0.5:
                continue
            ang = _normalize_angle_180(math.degrees(math.atan2(dy, dx)))
            mid_x = float((v1_w.x + v2_w.x) * 0.5)
            mid_y = float((v1_w.y + v2_w.y) * 0.5)
            samples.append((mid_x, mid_y, ang, seg_len))
    except Exception:
        samples = []

    _ROAD_EDGE_SAMPLE_CACHE[key] = samples
    return samples


def _nearest_road_axis_deg(target_obj, road_objs, radius=None):
    """
    Find nearest main-road axis around object in XY plane.
    1) Locate nearest road edge sample.
    2) On local area, use multiple nearby road edges to estimate a weighted dominant axis.
    """
    if target_obj is None or not road_objs:
        return None
    if radius is None:
        radius = ORIENTATION_NEAREST_ROAD_RADIUS

    center = target_obj.location
    center_x = float(center.x)
    center_y = float(center.y)
    best_dist = float("inf")
    nearest_road = None
    nearest_edge_ang = None

    for r_obj in road_objs:
        if r_obj is None or r_obj.type != "MESH":
            continue
        edge_samples = _get_sampled_road_edges(r_obj)
        if not edge_samples:
            continue
        for mid_x, mid_y, ang, _seg_len in edge_samples:
            d = math.hypot(mid_x - center_x, mid_y - center_y)
            if d > radius:
                continue
            if d < best_dist:
                best_dist = d
                nearest_road = r_obj
                nearest_edge_ang = ang

    if nearest_road is None:
        return None

    # Centerline-like local axis:
    # - Find nearest sampled road point to anchor a local road segment.
    # - Around that anchor, fit a weighted PCA direction on nearby edge midpoints
    #   (approximating local centerline tangent), then map to [0, 180).
    # - Fallback to weighted edge-angle averaging when PCA is unstable.
    local_edges = []
    local_radius = max(15.0, float(ORIENTATION_MAIN_ROAD_LOCAL_RADIUS))
    for r_obj in road_objs:
        if r_obj is None or r_obj.type != "MESH":
            continue
        for mid_x, mid_y, ang, seg_len in _get_sampled_road_edges(r_obj):
            d = math.hypot(mid_x - center_x, mid_y - center_y)
            if d > local_radius:
                continue
            if seg_len <= 0.5:
                continue
            # Closer/longer edges contribute more.
            w = float(seg_len) / (1.0 + d)
            if w <= 0.0:
                continue
            local_edges.append((float(ang), float(w)))

    local_best_ang = None

    def _fit_axis_from_points(points_xy, weights):
        if not points_xy or len(points_xy) < 2:
            return None
        w_sum = float(sum(weights))
        if w_sum <= 1e-9:
            return None
        mx = sum(w * p[0] for p, w in zip(points_xy, weights)) / w_sum
        my = sum(w * p[1] for p, w in zip(points_xy, weights)) / w_sum
        sxx = sum(w * (p[0] - mx) * (p[0] - mx) for p, w in zip(points_xy, weights)) / w_sum
        syy = sum(w * (p[1] - my) * (p[1] - my) for p, w in zip(points_xy, weights)) / w_sum
        sxy = sum(w * (p[0] - mx) * (p[1] - my) for p, w in zip(points_xy, weights)) / w_sum
        if (sxx + syy) <= 1e-12:
            return None
        theta = 0.5 * math.atan2(2.0 * sxy, (sxx - syy))
        return _normalize_angle_180(math.degrees(theta))

    # Build local centerline points around nearest sampled road point.
    centerline_points = []
    centerline_weights = []
    if nearest_road is not None:
        anchor = None
        best_anchor_dist = float("inf")
        for mid_x, mid_y, _ang, seg_len in _get_sampled_road_edges(nearest_road):
            d = math.hypot(mid_x - center_x, mid_y - center_y)
            if d < best_anchor_dist:
                best_anchor_dist = d
                anchor = (mid_x, mid_y, seg_len)
        if anchor is not None:
            ax, ay, _ = anchor
            window = max(20.0, float(ORIENTATION_CENTERLINE_WINDOW_RADIUS))
            sigma = max(8.0, 0.5 * window)
            for r_obj in road_objs:
                if r_obj is None or r_obj.type != "MESH":
                    continue
                for mid_x, mid_y, _ang, seg_len in _get_sampled_road_edges(r_obj):
                    d_anchor = math.hypot(mid_x - ax, mid_y - ay)
                    if d_anchor > window:
                        continue
                    d_center = math.hypot(mid_x - center_x, mid_y - center_y)
                    if d_center > local_radius:
                        continue
                    base = float(seg_len) / (1.0 + d_center)
                    w = base * math.exp(-(d_anchor * d_anchor) / (2.0 * sigma * sigma))
                    if w <= 0.0:
                        continue
                    centerline_points.append((float(mid_x), float(mid_y)))
                    centerline_weights.append(float(w))

    if centerline_points:
        local_best_ang = _fit_axis_from_points(centerline_points, centerline_weights)


    if local_edges:
        bin_deg = 10.0
        bins = {}
        for ang, w in local_edges:
            idx = int(_normalize_angle_180(ang) // bin_deg)
            bins[idx] = bins.get(idx, 0.0) + w
        if bins:
            peak_idx = max(bins.items(), key=lambda kv: kv[1])[0]
            peak_center = (peak_idx + 0.5) * bin_deg
            selected = []
            for ang, w in local_edges:
                if _angle_diff_180(ang, peak_center) <= 15.0:
                    selected.append((ang, w))
            if not selected:
                selected = local_edges

            sx = 0.0
            sy = 0.0
            for ang, w in selected:
                rad2 = math.radians(2.0 * _normalize_angle_180(ang))
                sx += w * math.cos(rad2)
                sy += w * math.sin(rad2)
            if abs(sx) > 1e-9 or abs(sy) > 1e-9:
                local_best_ang = _normalize_angle_180(0.5 * math.degrees(math.atan2(sy, sx)))

    if local_best_ang is None:
        local_best_ang = nearest_edge_ang

    if local_best_ang is None:
        return None
    return _normalize_angle_180(local_best_ang)


def _check_orientation_issue(target_obj, road_objs=None, threshold_deg=ORIENTATION_ERROR_THRESHOLD_DEG, original_rot_z=None):
    if target_obj is None:
        return False

    current_axis_deg = _building_edge_axis_deg(target_obj)
    if current_axis_deg is None:
        current_rot_z = target_obj.rotation_euler.z if target_obj.rotation_euler else 0.0
        current_axis_deg = _normalize_angle_180(math.degrees(current_rot_z))

    if original_rot_z is not None:
        current_rot_z = target_obj.rotation_euler.z if target_obj.rotation_euler else 0.0
        current_rot_deg = math.degrees(current_rot_z)
        ref_rot_deg = math.degrees(float(original_rot_z))
        delta_rot = current_rot_deg - ref_rot_deg
        ref_axis_deg = _normalize_angle_180(current_axis_deg - delta_rot)
        err = _axis_error_deg(current_axis_deg, ref_axis_deg)
        return err >= float(threshold_deg)

    if not road_objs:
        return False

    nearest_axis_deg = _nearest_road_axis_deg(target_obj, road_objs, radius=ORIENTATION_NEAREST_ROAD_RADIUS)
    if nearest_axis_deg is None:
        return False

    err = _axis_error_deg(current_axis_deg, nearest_axis_deg)
    return err >= float(threshold_deg)


def detect_orientation(buildings, roads):
    orthogonal_any = False
    count = 0
    involved = []
    for b in buildings:
        if not _is_orientation_candidate_shape(b):
            continue
        is_bad = _check_orientation_issue(
            b,
            road_objs=roads,
            threshold_deg=ORIENTATION_ERROR_THRESHOLD_DEG,
            original_rot_z=None,
        )
        if is_bad:
            count += 1
            involved.append(b.name)
        if not orthogonal_any and _check_nearby_road_orthogonality(b, roads):
            orthogonal_any = True

    return {
        'count': count,
        'involved_object_ids': sorted(involved),
        'road_grid_orthogonal': orthogonal_any,
    }


def main():
    argv = sys.argv
    if '--' not in argv:
        raise RuntimeError('Expected args after --: <output_json_path>')
    user_args = argv[argv.index('--') + 1:]
    if len(user_args) < 1:
        raise RuntimeError('Missing output json path')

    output_path = user_args[0]
    labels_json = user_args[1] if len(user_args) > 1 else ''
    scene_path = user_args[2] if len(user_args) > 2 else ''
    road_conflict_threshold_area = 0.25
    if len(user_args) > 3:
        try:
            road_conflict_threshold_area = float(user_args[3])
        except Exception:
            road_conflict_threshold_area = 0.25

    if scene_path:
        scene_l = str(scene_path).lower()
        if scene_l.endswith('.blend'):
            bpy.ops.wm.open_mainfile(filepath=str(scene_path))
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.gltf(filepath=str(scene_path))

    target_building_names = set()
    selected_label_ids = []
    object_id_to_label_id = {}
    if labels_json:
        try:
            with open(labels_json, 'r', encoding='utf-8') as f:
                labels_map = json.load(f)
            if isinstance(labels_map, dict):
                target_building_names = {str(v) for v in labels_map.values()}
                selected_label_ids = sorted(int(k) for k in labels_map.keys() if str(k).isdigit())
                object_id_to_label_id = {str(v): int(k) for k, v in labels_map.items() if str(k).isdigit()}
        except Exception:
            target_building_names = set()
            selected_label_ids = []
            object_id_to_label_id = {}

    all_buildings = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and is_building(obj.name)]
    if target_building_names:
        buildings = [obj for obj in all_buildings if obj.name in target_building_names]
    else:
        buildings = all_buildings

    all_roads = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and is_road(obj.name)]
    if buildings:
        region_boxes = [world_aabb(b) for b in buildings]
        region_boxes = [bb for bb in region_boxes if bb is not None]
        if region_boxes:
            min_x = min(bb['min_x'] for bb in region_boxes) - 10.0
            max_x = max(bb['max_x'] for bb in region_boxes) + 10.0
            min_y = min(bb['min_y'] for bb in region_boxes) - 10.0
            max_y = max(bb['max_y'] for bb in region_boxes) + 10.0
            region_box = {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}

            roads = []
            for r in all_roads:
                rbb = world_aabb(r)
                if rbb and rect_intersects(rbb, region_box):
                    roads.append(r)
        else:
            roads = []
    else:
        roads = []

    overlap = detect_overlap(buildings)
    road_conflict = detect_road_conflict(buildings, roads, threshold_area=road_conflict_threshold_area)
    orientation = detect_orientation(buildings, roads)

    def _to_label_ids(detail):
        obj_ids = detail.pop('involved_object_ids', [])
        label_ids = [object_id_to_label_id[obj_id] for obj_id in obj_ids if obj_id in object_id_to_label_id]
        detail['involved_building_label_ids'] = sorted(set(label_ids))
        return detail

    overlap = _to_label_ids(overlap)
    road_conflict = _to_label_ids(road_conflict)
    orientation = _to_label_ids(orientation)

    payload = {
        'object_stats': {
            'building_count': len(buildings),
            'road_count': len(roads),
        },
        'selected_building_label_ids': selected_label_ids,
        'selected_building_object_ids': sorted([b.name for b in buildings]),
        'error_type_counts': {
            'Overlap': int(overlap['count']),
            'RoadConflict': int(road_conflict['count']),
            'Orientation': int(orientation['count']),
        },
        'details': {
            'Overlap': overlap,
            'RoadConflict': road_conflict,
            'Orientation': orientation,
            'thresholds': {
                'road_conflict_area': float(road_conflict_threshold_area),
            },
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
'''


@dataclass
class BlendMetrics:
    blend_path: Optional[str]
    error_type_counts: Dict[str, int]
    details: Dict
    object_stats: Dict[str, int]
    selected_building_label_ids: List[int]
    selected_building_object_ids: List[str]


@dataclass
class RegionEntry:
    region_key: str
    region_dir: str
    region_id: Optional[int]
    source_region_name: Optional[str]
    source_region_id: Optional[int]


@dataclass
class RegionMetrics:
    region_key: str
    region_id: Optional[int]
    region_dir: str
    source_region_name: Optional[str]
    source_region_id: Optional[int]
    selected_building_label_ids: List[int]
    selected_building_object_ids: List[str]
    before: BlendMetrics
    after: BlendMetrics
    delta_error_type_counts: Dict[str, int]
    before_total_errors: int
    after_total_errors: int
    delta_total_errors: int


DEFAULT_TYPES = ["Overlap", "RoadConflict", "Orientation"]
DEFAULT_INPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/multi_step_error"))
KNOWN_REGIONS = ["region_a", "region_b", "default_region"]
DEFAULT_BLENDER_BIN = "SpatialAct/blender-3.2.2-linux-x64/blender"
DEFAULT_ROAD_CONFLICT_THRESHOLD_AREA = 0.25


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _load_json_dict_if_exists(path: str) -> Dict:
    if not path or (not os.path.isfile(path)):
        return {}
    try:
        data = _load_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_region_id_from_text(text: str) -> Optional[int]:
    token = str(text or "")
    m = re.search(r"(?:^|[^0-9])region[_-]?(\d+)(?:[^0-9]|$)", token)
    if m:
        return int(m.group(1))
    return None


def _extract_source_region(region_info: Dict) -> Tuple[Optional[str], Optional[int]]:
    scene_name = str(region_info.get("scene_name", "") or "")
    m = re.search(r"([^/\\]+)[/\\]region_(\d+)$", scene_name)
    if m:
        return str(m.group(1)), int(m.group(2))

    src_path = str((((region_info.get("source") or {}).get("source_scene_path")) or "")).replace("\\", "/")
    m_name = re.search(r"/([^/]+)_steps\d+_regions(?:/|$)", src_path)
    region_name = str(m_name.group(1)) if m_name else None
    region_id = _safe_int((region_info.get("source") or {}).get("region_id"))
    return region_name, region_id


def _build_region_entry(region_dir: str) -> RegionEntry:
    dir_name = os.path.basename(region_dir.rstrip(os.sep))
    region_info = _load_json_dict_if_exists(os.path.join(region_dir, "region_info.json"))

    region_id = _safe_int(region_info.get("region_id"))
    if region_id is None:
        region_id = _parse_region_id_from_text(dir_name)

    source_region_name, source_region_id = _extract_source_region(region_info)

    if source_region_name and source_region_id is not None:
        region_key = f"{source_region_name}__region_{source_region_id}"
    elif source_region_name and region_id is not None:
        region_key = f"{source_region_name}__sample_{region_id}"
    elif region_id is not None:
        region_key = f"region_{region_id}"
    else:
        region_key = dir_name

    return RegionEntry(
        region_key=region_key,
        region_dir=region_dir,
        region_id=region_id,
        source_region_name=source_region_name,
        source_region_id=source_region_id,
    )


def _iter_region_dirs(workspace_root: str) -> List[RegionEntry]:
    entries: List[RegionEntry] = []
    if not os.path.isdir(workspace_root):
        return entries

    child_names = sorted(os.listdir(workspace_root))
    for name in child_names:
        region_dir = os.path.join(workspace_root, name)
        if not os.path.isdir(region_dir):
            continue
        try:
            files = set(os.listdir(region_dir))
        except Exception:
            files = set()
        has_signal = (
            "labels.json" in files
            or "region_info.json" in files
            or any(fn.startswith("final_scene") and (fn.endswith(".blend") or fn.endswith(".glb")) for fn in files)
            or any(fn.startswith("error_scene") and (fn.endswith(".blend") or fn.endswith(".glb")) for fn in files)
        )
        if not has_signal:
            continue
        entries.append(_build_region_entry(region_dir))

    dedup_count: Dict[str, int] = {}
    out: List[RegionEntry] = []
    for e in entries:
        n = dedup_count.get(e.region_key, 0) + 1
        dedup_count[e.region_key] = n
        if n > 1:
            e = RegionEntry(
                region_key=f"{e.region_key}__dup{n}",
                region_dir=e.region_dir,
                region_id=e.region_id,
                source_region_name=e.source_region_name,
                source_region_id=e.source_region_id,
            )
        out.append(e)
    return out


def _resolve_blender_path(project_root: str, explicit: Optional[str] = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(DEFAULT_BLENDER_BIN)
    candidates.append(os.path.join(project_root, "blender-3.2.2-linux-x64", "blender"))
    candidates.append(os.path.join(project_root, "blender-3.2.2-linux-x64"))
    env_blender = os.environ.get("BLENDER_PATH") or os.environ.get("BLENDER_BIN")
    if env_blender:
        candidates.append(env_blender)
    candidates.extend(
        [
            "blender",
        ]
    )

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


def _resolve_labels_json(
    region_dir: str,
    region_id: Optional[int],
    region_name: Optional[str],
    source_region_name: Optional[str],
    source_region_id: Optional[int],
    steps: Optional[str],
    input_root: str,
    explicit_labels_json: Optional[str] = None,
) -> Optional[str]:
    if explicit_labels_json:
        p = os.path.abspath(explicit_labels_json)
        if os.path.isfile(p):
            return p

    local_labels = os.path.join(region_dir, "labels.json")
    if os.path.isfile(local_labels):
        return local_labels

    step_suffix = str(steps).strip() if steps is not None else ""

    candidates: List[str] = []
    if source_region_name and source_region_id is not None:
        if step_suffix:
            candidates.append(
                os.path.join(input_root, f"{source_region_name}_steps{step_suffix}_regions", f"region_{source_region_id}", "labels.json")
            )
        candidates.append(os.path.join(input_root, f"{source_region_name}_regions", f"region_{source_region_id}", "labels.json"))

    if region_id is not None and region_name:
        if step_suffix:
            candidates.append(os.path.join(input_root, f"{region_name}_steps{step_suffix}_regions", f"region_{region_id}", "labels.json"))
        candidates.append(os.path.join(input_root, f"{region_name}_regions", f"region_{region_id}", "labels.json"))

    for labels_path in candidates:
        if os.path.isfile(labels_path):
            return labels_path
    return None


def _infer_region_from_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    token = str(path_value).lower()
    hits = [region for region in KNOWN_REGIONS if region in token]
    if len(hits) == 1:
        return hits[0]
    return None


def _empty_blend_metrics(blend_path: Optional[str]) -> BlendMetrics:
    return BlendMetrics(
        blend_path=blend_path,
        error_type_counts={k: 0 for k in DEFAULT_TYPES},
        details={},
        object_stats={"building_count": 0, "road_count": 0},
        selected_building_label_ids=[],
        selected_building_object_ids=[],
    )


def _load_region_labels(labels_path: Optional[str]) -> Tuple[List[int], List[str]]:
    if not labels_path or not os.path.exists(labels_path):
        return [], []
    try:
        payload = _load_json(labels_path)
    except Exception:
        return [], []
    if not isinstance(payload, dict):
        return [], []

    label_ids: List[int] = []
    object_ids: List[str] = []
    for key, value in payload.items():
        if str(key).isdigit():
            label_ids.append(int(key))
        if value is not None:
            object_ids.append(str(value))
    return sorted(label_ids), sorted(set(object_ids))


def _run_blender_detect(
    blender_bin: str,
    blend_path: str,
    labels_path: Optional[str],
    road_conflict_threshold_area: float,
) -> BlendMetrics:
    with tempfile.TemporaryDirectory(prefix="blend_metrics_") as tmpdir:
        script_path = os.path.join(tmpdir, "detect.py")
        output_path = os.path.join(tmpdir, "detect_output.json")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(BLENDER_DETECT_SCRIPT)

        blend_or_scene_path = os.path.abspath(blend_path)
        is_blend = blend_or_scene_path.lower().endswith(".blend")
        cmd = [blender_bin, "--background"]
        if is_blend:
            cmd.append(blend_or_scene_path)
        cmd.extend(
            [
                "--python",
                script_path,
                "--",
                output_path,
                labels_path or "",
                blend_or_scene_path if (not is_blend) else "",
                str(float(road_conflict_threshold_area)),
            ]
        )
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

        payload = _load_json(output_path)
        counts = payload.get("error_type_counts", {}) or {}
        out_counts = {k: int(counts.get(k, 0) or 0) for k in DEFAULT_TYPES}

        return BlendMetrics(
            blend_path=blend_path,
            error_type_counts=out_counts,
            details=payload.get("details", {}) or {},
            object_stats=payload.get("object_stats", {}) or {},
            selected_building_label_ids=sorted(int(x) for x in (payload.get("selected_building_label_ids", []) or []) if str(x).isdigit()),
            selected_building_object_ids=sorted(str(x) for x in (payload.get("selected_building_object_ids", []) or [])),
        )


def _resolve_region_blend_paths(
    region_dir: str,
    steps: Optional[str],
    after_blend_override: Optional[str] = None,
) -> Tuple[str, str]:
    step_suffix = str(steps).strip() if steps is not None else ""
    workspace_root = os.path.dirname(region_dir)
    region_info = _load_json_dict_if_exists(os.path.join(region_dir, "region_info.json"))
    source_scene_path = str((((region_info.get("source") or {}).get("source_scene_path")) or "")).strip()

    candidates_before: List[str] = []
    candidates_after: List[str] = []

    if step_suffix:
        candidates_before.append(os.path.join(region_dir, f"error_scene_steps{step_suffix}.blend"))
        candidates_before.append(os.path.join(region_dir, f"error_scene_steps{step_suffix}.glb"))
        candidates_before.append(os.path.join(workspace_root, f"error_scene_steps{step_suffix}.blend"))
        candidates_before.append(os.path.join(workspace_root, f"error_scene_steps{step_suffix}.glb"))
        candidates_after.append(os.path.join(region_dir, f"final_scene_steps{step_suffix}.blend"))
        candidates_after.append(os.path.join(region_dir, f"final_scene_steps{step_suffix}.glb"))

    candidates_before.append(os.path.join(region_dir, "error_scene.blend"))
    candidates_before.append(os.path.join(region_dir, "error_scene.glb"))
    candidates_before.append(os.path.join(workspace_root, "error_scene.blend"))
    candidates_before.append(os.path.join(workspace_root, "error_scene.glb"))
    if source_scene_path:
        candidates_before.append(source_scene_path)
    candidates_after.append(os.path.join(region_dir, "final_scene.blend"))
    candidates_after.append(os.path.join(region_dir, "final_scene.glb"))

    before_blend = next((path for path in candidates_before if os.path.isfile(path)), candidates_before[0])
    if after_blend_override:
        after_blend = os.path.abspath(after_blend_override)
    else:
        after_blend = next((path for path in candidates_after if os.path.isfile(path)), candidates_after[0])
    return before_blend, after_blend


def _compute_region_metrics(
    region_entry: RegionEntry,
    blender_bin: str,
    steps: Optional[str],
    region_name: Optional[str],
    input_root: str,
    labels_json_override: Optional[str],
    allow_missing_labels: bool,
    road_conflict_threshold_area: float,
    after_blend_override: Optional[str] = None,
) -> RegionMetrics:
    before_blend, after_blend = _resolve_region_blend_paths(
        region_entry.region_dir,
        steps,
        after_blend_override=after_blend_override,
    )
    labels_path = _resolve_labels_json(
        region_dir=region_entry.region_dir,
        region_id=region_entry.region_id,
        region_name=region_name,
        source_region_name=region_entry.source_region_name,
        source_region_id=region_entry.source_region_id,
        steps=steps,
        input_root=input_root,
        explicit_labels_json=labels_json_override,
    )
    if (not labels_path) and (not allow_missing_labels):
        raise FileNotFoundError(
            f" labels.json“”region_key={region_entry.region_key}, region_dir={region_entry.region_dir}"
        )
    region_label_ids, region_object_ids = _load_region_labels(labels_path)

    before_metrics = _run_blender_detect(
        blender_bin, before_blend, labels_path, road_conflict_threshold_area
    ) if os.path.exists(before_blend) else _empty_blend_metrics(None)
    after_metrics = _run_blender_detect(
        blender_bin, after_blend, labels_path, road_conflict_threshold_area
    ) if os.path.exists(after_blend) else _empty_blend_metrics(None)

    delta_counts = {
        k: int(after_metrics.error_type_counts.get(k, 0) - before_metrics.error_type_counts.get(k, 0))
        for k in DEFAULT_TYPES
    }

    before_total = sum(before_metrics.error_type_counts.values())
    after_total = sum(after_metrics.error_type_counts.values())

    return RegionMetrics(
        region_key=region_entry.region_key,
        region_id=region_entry.region_id,
        region_dir=region_entry.region_dir,
        source_region_name=region_entry.source_region_name,
        source_region_id=region_entry.source_region_id,
        selected_building_label_ids=region_label_ids,
        selected_building_object_ids=region_object_ids,
        before=before_metrics,
        after=after_metrics,
        delta_error_type_counts=delta_counts,
        before_total_errors=before_total,
        after_total_errors=after_total,
        delta_total_errors=after_total - before_total,
    )


def _build_output(workspace_root: str, blender_bin: str, metrics: List[RegionMetrics], steps: Optional[str]) -> Dict:
    before_total_by_type = Counter()
    after_total_by_type = Counter()
    delta_total_by_type = Counter()

    for row in metrics:
        before_total_by_type.update(row.before.error_type_counts)
        after_total_by_type.update(row.after.error_type_counts)
        delta_total_by_type.update(row.delta_error_type_counts)

    before_total_errors = sum(before_total_by_type.values())
    after_total_errors = sum(after_total_by_type.values())

    return {
        "workspace_root": workspace_root,
        "blender": blender_bin,
        "steps": str(steps).strip() if steps is not None else "",
        "summary": {
            "regions_total": len(metrics),
            "before_total_errors": before_total_errors,
            "after_total_errors": after_total_errors,
            "delta_total_errors": after_total_errors - before_total_errors,
            "before_error_type_counts": dict(sorted(before_total_by_type.items(), key=lambda kv: kv[0])),
            "after_error_type_counts": dict(sorted(after_total_by_type.items(), key=lambda kv: kv[0])),
            "delta_error_type_counts": dict(sorted(delta_total_by_type.items(), key=lambda kv: kv[0])),
            "rule": "Directly read blend files and detect 3 modes (Overlap, RoadConflict, Orientation) with Blender-side geometric checks.",
        },
        "regions": [
            {
                "region_key": row.region_key,
                "region_id": row.region_id,
                "region_dir": row.region_dir,
                "source_region_name": row.source_region_name,
                "source_region_id": row.source_region_id,
                "selected_building_label_ids": row.selected_building_label_ids,
                "selected_building_object_ids": row.selected_building_object_ids,
                "before": {
                    "blend_path": row.before.blend_path,
                    "total_errors": row.before_total_errors,
                    "error_type_counts": row.before.error_type_counts,
                    "object_stats": row.before.object_stats,
                    "selected_building_label_ids": row.before.selected_building_label_ids,
                    "selected_building_object_ids": row.before.selected_building_object_ids,
                    "details": row.before.details,
                },
                "after": {
                    "blend_path": row.after.blend_path,
                    "total_errors": row.after_total_errors,
                    "error_type_counts": row.after.error_type_counts,
                    "object_stats": row.after.object_stats,
                    "selected_building_label_ids": row.after.selected_building_label_ids,
                    "selected_building_object_ids": row.after.selected_building_object_ids,
                    "details": row.after.details,
                },
                "delta_total_errors": row.delta_total_errors,
                "delta_error_type_counts": row.delta_error_type_counts,
            }
            for row in metrics
        ],
    }


def _build_single_blend_output(blender_bin: str, blend_metrics: BlendMetrics, blend_path: str, steps: Optional[str]) -> Dict:
    total_errors = sum((blend_metrics.error_type_counts or {}).values())
    return {
        "workspace_root": None,
        "blend_path": blend_path,
        "blender": blender_bin,
        "steps": str(steps).strip() if steps is not None else "",
        "summary": {
            "regions_total": 1,
            "before_total_errors": 0,
            "after_total_errors": total_errors,
            "delta_total_errors": total_errors,
            "before_error_type_counts": {k: 0 for k in DEFAULT_TYPES},
            "after_error_type_counts": dict(sorted((blend_metrics.error_type_counts or {}).items(), key=lambda kv: kv[0])),
            "delta_error_type_counts": dict(sorted((blend_metrics.error_type_counts or {}).items(), key=lambda kv: kv[0])),
            "rule": "Directly read one blend file and detect 3 modes (Overlap, RoadConflict, Orientation) with Blender-side geometric checks.",
        },
        "single_blend": {
            "blend_path": blend_path,
            "total_errors": total_errors,
            "error_type_counts": blend_metrics.error_type_counts,
            "object_stats": blend_metrics.object_stats,
            "selected_building_label_ids": blend_metrics.selected_building_label_ids,
            "selected_building_object_ids": blend_metrics.selected_building_object_ids,
            "details": blend_metrics.details,
        },
    }


def _print_console(metrics: List[RegionMetrics]) -> None:
    print("region_key\tregion_id\tbefore_total\tafter_total\tdelta\tdelta_by_type")
    for row in metrics:
        print(
            f"{row.region_key}\t{row.region_id}\t{row.before_total_errors}\t{row.after_total_errors}\t{row.delta_total_errors}\t"
            f"{json.dumps(row.delta_error_type_counts, ensure_ascii=False)}"
        )


def parse_region_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=" blend ")
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="run_iteration  --blend-path / --region-dir ",
    )
    parser.add_argument(
        "--region-dir",
        default=None,
        help=" .../region_xxx --workspace-root / --blend-path ",
    )
    parser.add_argument(
        "--blend-path",
        default=None,
        help=" blend  --workspace-root / --region-dir ",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help=" labels.json",
    )
    parser.add_argument(
        "--region",
        default=None,
        help=" default_region workspace-root/blend-path ",
    )
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help=" region + steps  labels.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=" JSON  <workspace-root>/metrics_summary_blend.json",
    )
    parser.add_argument(
        "--steps",
        default=os.environ.get("STEPS", os.environ.get("ERROR_STEPS", "3")),
        help=" error_scene_stepsX/final_scene_stepsX",
    )
    parser.add_argument(
        "--blender",
        default=DEFAULT_BLENDER_BIN,
        help="Blender  blender-3.2.2",
    )
    parser.add_argument(
        "--after-blend-path",
        default=None,
        help=" workspace/region-dir  after",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help=" labels ",
    )
    parser.add_argument(
        "--road-conflict-threshold-area",
        type=float,
        default=DEFAULT_ROAD_CONFLICT_THRESHOLD_AREA,
        help="RoadConflict  0.25",
    )
    return parser.parse_args(argv)


def _main_region(argv: Optional[List[str]] = None) -> None:
    args = parse_region_args(argv)
    mode_count = int(bool(args.workspace_root)) + int(bool(args.blend_path)) + int(bool(args.region_dir))
    if mode_count == 0:
        raise ValueError(" --workspace-root  --blend-path  --region-dir ")
    if mode_count > 1:
        raise ValueError("--workspace-root / --blend-path / --region-dir ")

    region_name = args.region or _infer_region_from_path(args.workspace_root or args.blend_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    blender_bin = _resolve_blender_path(project_root, args.blender)
    step_suffix = str(args.steps).strip() if args.steps is not None else ""

    if args.blend_path:
        blend_path = os.path.abspath(args.blend_path)
        if not os.path.isfile(blend_path):
            raise FileNotFoundError(f"blend : {blend_path}")
        parent = os.path.dirname(blend_path)
        entry = _build_region_entry(parent)
        labels_path = _resolve_labels_json(
            region_dir=parent,
            region_id=entry.region_id,
            region_name=region_name,
            source_region_name=entry.source_region_name,
            source_region_id=entry.source_region_id,
            steps=args.steps,
            input_root=os.path.abspath(args.input_root),
            explicit_labels_json=args.labels_json,
        )
        if (not labels_path) and (not args.allow_missing_labels):
            raise FileNotFoundError(
                " labels.json"
                "  --labels-json  labels.json"
            )

        single = _run_blender_detect(
            blender_bin,
            blend_path,
            labels_path,
            float(args.road_conflict_threshold_area),
        )
        out = _build_single_blend_output(blender_bin, single, blend_path, args.steps)
        default_output_name = f"metrics_single_blend_steps{step_suffix}.json" if step_suffix else "metrics_single_blend.json"
        output_path = args.output or os.path.join(os.path.dirname(blend_path), default_output_name)
    else:
        input_root = os.path.abspath(args.input_root)
        if args.region_dir:
            region_dir = os.path.abspath(args.region_dir)
            if not os.path.isdir(region_dir):
                raise FileNotFoundError(f"region_dir : {region_dir}")
            region_entry = _build_region_entry(region_dir)
            metrics = [
                _compute_region_metrics(
                    region_entry=region_entry,
                    blender_bin=blender_bin,
                    steps=args.steps,
                    region_name=region_name,
                    input_root=input_root,
                    labels_json_override=args.labels_json,
                    allow_missing_labels=args.allow_missing_labels,
                    road_conflict_threshold_area=float(args.road_conflict_threshold_area),
                    after_blend_override=args.after_blend_path,
                )
            ]
            workspace_root = os.path.dirname(region_dir)
            out = _build_output(workspace_root, blender_bin, metrics, args.steps)
            default_output_name = f"metrics_region_blend_steps{step_suffix}.json" if step_suffix else "metrics_region_blend.json"
            output_path = args.output or os.path.join(region_dir, default_output_name)
        else:
            workspace_root = os.path.abspath(args.workspace_root)
            if not os.path.isdir(workspace_root):
                raise FileNotFoundError(f"workspace_root : {workspace_root}")

            region_entries = _iter_region_dirs(workspace_root)
            if not region_entries:
                raise RuntimeError(f"workspace_root : {workspace_root}")
            metrics = [
                _compute_region_metrics(
                    region_entry=e,
                    blender_bin=blender_bin,
                    steps=args.steps,
                    region_name=region_name,
                    input_root=input_root,
                    labels_json_override=args.labels_json,
                    allow_missing_labels=args.allow_missing_labels,
                    road_conflict_threshold_area=float(args.road_conflict_threshold_area),
                    after_blend_override=args.after_blend_path,
                )
                for e in region_entries
            ]
            out = _build_output(workspace_root, blender_bin, metrics, args.steps)
            default_output_name = f"metrics_summary_blend_steps{step_suffix}.json" if step_suffix else "metrics_summary_blend.json"
            output_path = args.output or os.path.join(workspace_root, default_output_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)

    if args.workspace_root:
        _print_console(metrics)
    print(f"\n[OK] metrics written to: {output_path}")


def _parse_mode(argv: List[str]) -> Tuple[str, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["region", "indoor"], default="region")
    ns, rest = parser.parse_known_args(argv)
    return str(ns.mode), rest


def _main_indoor(argv: Optional[List[str]] = None) -> None:
    args = argv or []
    try:
        import indoor_metrics
    except Exception as exc:
        raise RuntimeError(f" indoor_metrics.py: {exc}") from exc
    indoor_metrics.main(args)


def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    mode, rest = _parse_mode(raw_argv)
    if mode == "indoor":
        _main_indoor(rest)
        return
    _main_region(rest)


if __name__ == "__main__":
    main()
