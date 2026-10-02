"""Fit primitive candidates (plane, cylinder, box) to FBX meshes per moniker.

FBX primitive fitting is an independent reverse-engineering reference: it
proposes which basic geometry could explain each mesh.  It does not decode VUE
bytes, does not assign business component names, and does not feed vertices
back into the VUE decoder.  Calibration evidence (2026-09-08, 063E):

- Assimp ``mesh.vertices`` are already in world coordinates; multiplying the
  node-chain transform moves them away from the VUE world frame (pipe 430 raw
  bbox matches the VUE record bbox; the transformed one does not).
- FBX vertices are float32; at ~9000 m the quantization step is ~0.98 mm, so
  geometric tolerances must stay above that reference noise.

Result states follow VUE_GROUP_ELEMENTS_RESEARCH.md section 10.3.3:
primitive_combination / partial / unresolved_mesh / ambiguous /
reference_missing / processing_failed.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

VERSION = "0.1.3"
SEED = 20260908
WELD_DECIMALS = 5
DEFAULT_TOL_M = 0.0015
PLANE_ANGLE_TOL_DEG = 1.0
PATCH_NORMAL_SPREAD_DEG = 10.0
OFFSET_TOL_M = 0.002
MIN_PATCH_AREA_M2 = 1.0e-7
MAX_PRIMITIVES = 12
EXPLAINED_RATIO_SUCCESS = 0.98
EXPLAINED_RATIO_PARTIAL = 0.2
CYL_SIDE_COVERAGE_MIN = 0.5
MIN_LOOP_VERTICES_FOR_CIRCLE = 8
CIRCLE_AREA_RATIO_MIN = 0.75
CIRCLE_AREA_RATIO_MAX = 1.20
REBUILD_SEGMENT_COUNT = 48
REBUILD_SAMPLE_COUNT = 20000

DEFAULT_MONIKERS = {
    "063E": [
        "0027-80012-301858369261339043",
        "0027-80012-301858369032094224",
        "0027-80005-306080471679897572",
        "0027-80005-312835864004529065",
        "0027-80005-316776537703780116",
        "0027-80005-316776537703779569",
        "0027-80005-318746879423021721",
        "0027-80005-302702794249864811",
    ],
    "15300": [
        "0027-80005-301576559404805715",
        "0027-80005-304391331410043561",
    ],
}


def canonical_moniker(value):
    return value.replace("@a=", "", 1).replace("!!", "-", 1).replace("##", "-", 1)


# --------------------------------------------------------------------------
# mesh extraction
# --------------------------------------------------------------------------

def triangulate_faces(raw_faces):
    faces = []
    sources = []
    for face_id, face in enumerate(raw_faces):
        indices = [int(i) for i in face if int(i) >= 0]
        if len(indices) < 3:
            continue
        for k in range(1, len(indices) - 1):
            faces.append((indices[0], indices[k], indices[k + 1]))
            sources.append(face_id)
    return faces, sources


def collect_meshes(fbx_path, wanted):
    import pyassimp

    found = {key: [] for key in wanted}
    with pyassimp.load(str(fbx_path)) as scene:
        def walk(node, owner=None, path_parts=()):
            name = str(node.name)
            current_path = path_parts + (name,)
            if name.startswith(("0027-", "0028-")):
                owner = canonical_moniker(name)
            if owner in found:
                for mesh in node.meshes:
                    vertices = np.asarray(mesh.vertices, dtype=np.float64)
                    raw_faces = np.asarray(mesh.faces)
                    faces, sources = triangulate_faces(raw_faces)
                    found[owner].append({
                        "node_path": "/" + "/".join(current_path),
                        "is_insulation": "Insulation" in "/".join(current_path),
                        "vertices": vertices,
                        "faces": faces,
                        "face_sources": sources,
                    })
            for child in node.children:
                walk(child, owner, current_path)

        walk(scene.rootnode)
    return found


def weld_vertices(vertices, faces):
    keys = [tuple(np.round(v, decimals=WELD_DECIMALS)) for v in vertices]
    remap = {}
    vertex_map = np.empty(len(vertices), dtype=np.int64)
    unique_indices = []
    for index, key in enumerate(keys):
        target = remap.get(key)
        if target is None:
            target = len(unique_indices)
            remap[key] = target
            unique_indices.append(index)
        vertex_map[index] = target
    unique_vertices = vertices[unique_indices]
    all_remapped = [tuple(int(vertex_map[i]) for i in face) for face in faces]
    degenerate = set()
    for index, face in enumerate(all_remapped):
        if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
            degenerate.add(index)
            continue
        a, b, c = unique_vertices[list(face)]
        if np.linalg.norm(np.cross(b - a, c - a)) <= 1.0e-14:
            # FBX exports can retain a zero-area triangle even when its three
            # vertex indices remain distinct.  It must not seed a plane fit.
            degenerate.add(index)
    new_faces = [face for index, face in enumerate(all_remapped)
                 if index not in degenerate]
    return unique_vertices, new_faces, vertex_map, sorted(degenerate)


def connected_components(face_count, faces):
    parent = list(range(face_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for k in range(3):
            a, b = face[k], face[(k + 1) % 3]
            edge_faces[(min(a, b), max(a, b))].append(fi)
    for members in edge_faces.values():
        for other in members[1:]:
            union(members[0], other)
    groups = defaultdict(list)
    for fi in range(face_count):
        groups[find(fi)].append(fi)
    return sorted(groups.values(), key=len, reverse=True)


def triangle_normals_areas(vertices, faces):
    faces_arr = np.asarray(faces, dtype=np.int64)
    a = vertices[faces_arr[:, 0]]
    b = vertices[faces_arr[:, 1]]
    c = vertices[faces_arr[:, 2]]
    cross = np.cross(b - a, c - a)
    norms = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norms
    normals = np.zeros_like(cross)
    good = norms > 0
    normals[good] = cross[good] / norms[good, None]
    centroids = (a + b + c) / 3.0
    return normals, areas, centroids


def face_area(vertices, faces, fi):
    face = faces[fi]
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


# --------------------------------------------------------------------------
# plane patches
# --------------------------------------------------------------------------

def cluster_directions(directions, weights, angle_tol_deg):
    cos_tol = math.cos(math.radians(angle_tol_deg))
    order = np.argsort(-np.asarray(weights))
    labels = np.full(len(directions), -1, dtype=np.int64)
    representatives = []
    for idx in order:
        direction = directions[idx]
        best_label, best_dot = -1, cos_tol
        for label, rep in enumerate(representatives):
            dot = float(np.dot(direction, rep))
            if dot > best_dot:
                best_label, best_dot = label, dot
        if best_label < 0:
            representatives.append(direction / np.linalg.norm(direction))
            best_label = len(representatives) - 1
        labels[idx] = best_label
    return labels


def boundary_loops(face_ids, faces):
    edge_count = defaultdict(int)
    for fi in face_ids:
        face = faces[int(fi)]
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            edge_count[(min(a, b), max(a, b))] += 1
    boundary = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary:
        return [], "closed_surface"
    adjacency = defaultdict(list)
    for a, b in boundary:
        adjacency[a].append(b)
        adjacency[b].append(a)
    used = set()
    loops = []
    for a, b in boundary:
        edge = (min(a, b), max(a, b))
        if edge in used:
            continue
        used.add(edge)
        loop = [a, b]
        previous, current = a, b
        while True:
            candidates = [n for n in adjacency[current]
                          if n != previous and (min(current, n), max(current, n)) not in used]
            if not candidates:
                break
            nxt = candidates[0]
            used.add((min(current, nxt), max(current, nxt)))
            if nxt == loop[0]:
                break
            loop.append(nxt)
            previous, current = current, nxt
        if len(loop) >= 3:
            loops.append(loop)
    status = "ok" if loops else "boundary_walk_failed"
    return loops, status


def face_adjacency(faces, component_face_ids):
    component_set = set(component_face_ids)
    edge_faces = defaultdict(list)
    for fi in component_face_ids:
        face = faces[fi]
        for k in range(3):
            a, b = int(face[k]), int(face[(k + 1) % 3])
            edge_faces[(min(a, b), max(a, b))].append(fi)
    adjacency = defaultdict(list)
    for edge, members in edge_faces.items():
        inside = [fi for fi in members if fi in component_set]
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                adjacency[inside[i]].append(inside[j])
                adjacency[inside[j]].append(inside[i])
    return adjacency, edge_faces


def fit_plane_from_region(vertices, faces, region, areas):
    vertex_weight = defaultdict(float)
    for fi in region:
        face = faces[fi]
        w = areas[fi] / 3.0
        for vi in face:
            vertex_weight[int(vi)] += w
    points = np.array([vertices[vi] for vi in sorted(vertex_weight)])
    weights = np.array([vertex_weight[vi] for vi in sorted(vertex_weight)])
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm
    return {"normal": normal, "point": center}


def extract_plane_patches(vertices, faces, component_face_ids, normals, areas,
                          centroids, tol):
    if not component_face_ids:
        return []
    adjacency, _ = face_adjacency(faces, component_face_ids)
    component_set = set(component_face_ids)
    unvisited = set(component_face_ids)
    seeds = sorted(component_face_ids, key=lambda fi: -float(areas[fi]))
    patches = []
    for seed in seeds:
        if seed not in unvisited:
            continue
        region = {seed}
        plane = fit_plane_from_region(vertices, faces, region, areas)
        if plane is None:
            continue
        queue = [seed]
        in_region = {seed}
        while queue:
            fi = queue.pop()
            for nb in adjacency.get(fi, ()):
                if nb in in_region or nb not in unvisited:
                    continue
                face = faces[nb]
                shared = set(faces[fi]) & set(face)
                if len(shared) != 2:
                    continue
                third = next(int(v) for v in face if int(v) not in shared)
                distance = abs(float((vertices[third] - plane["point"]) @ plane["normal"]))
                if distance > tol:
                    continue
                region.add(nb)
                in_region.add(nb)
                queue.append(nb)
                refined = fit_plane_from_region(vertices, faces, region, areas)
                if refined is not None:
                    plane = refined
        if len(region) < 2:
            continue
        face_ids = sorted(region)
        normal = plane["normal"]
        point = plane["point"]
        residual = float(np.max(np.abs((centroids[face_ids] - point) @ normal)))
        area = float(np.sum(areas[face_ids]))
        if area < MIN_PATCH_AREA_M2 or residual > tol:
            continue
        mean_normal = np.average(normals[face_ids], axis=0, weights=areas[face_ids])
        norm_len = np.linalg.norm(mean_normal)
        if norm_len < 1e-12:
            continue
        mean_normal = mean_normal / norm_len
        cos_spread = math.cos(math.radians(PATCH_NORMAL_SPREAD_DEG))
        dots = np.abs(normals[face_ids] @ mean_normal)
        if float(np.min(dots)) < cos_spread:
            continue
        unvisited -= in_region
        loops, loop_status = boundary_loops(face_ids, faces)
        patches.append({
            "normal": normal,
            "point": point,
            "offset": float(point @ normal),
            "residual_max_m": residual,
            "face_ids": face_ids,
            "area": area,
            "boundary_loops": loops,
            "boundary_status": loop_status,
        })
    patches.sort(key=lambda p: -p["area"])
    return patches


# --------------------------------------------------------------------------
# circle and cylinder fitting
# --------------------------------------------------------------------------

def fit_circle_pratt(points):
    points = np.asarray(points, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    Amat = np.c_[2 * x, 2 * y, np.ones(len(points))]
    bvec = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r_sq = sol[2] + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return None
    radius = math.sqrt(r_sq)
    distances = np.hypot(x - cx, y - cy)
    return {
        "center_2d": (float(cx), float(cy)),
        "radius": float(radius),
        "residual_max_m": float(np.max(np.abs(distances - radius))),
    }


def plane_frame(normal):
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(normal @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def circle_from_patch(vertices, patch):
    loops = patch.get("boundary_loops") or []
    if len(loops) != 1:
        return None
    loop = loops[0]
    if len(loop) < MIN_LOOP_VERTICES_FOR_CIRCLE:
        return None
    points = vertices[loop]
    origin = points.mean(axis=0)
    u, v = plane_frame(patch["normal"])
    local = np.c_[(points - origin) @ u, (points - origin) @ v]
    circle = fit_circle_pratt(local)
    if circle is None:
        return None
    # A curved side is sometimes split into a small, nearly planar strip.  Its
    # boundary can still produce a numerically good circle, but the patch area
    # is only a fraction of the fitted disk area.  Require the finite patch to
    # be disk-like before using it as a cylinder cap.
    disk_area = math.pi * circle["radius"] ** 2
    area_ratio = float(patch.get("area", 0.0)) / disk_area if disk_area > 0 else 0.0
    if not CIRCLE_AREA_RATIO_MIN <= area_ratio <= CIRCLE_AREA_RATIO_MAX:
        return None
    center_3d = origin + u * circle["center_2d"][0] + v * circle["center_2d"][1]
    return {
        "center": center_3d,
        "axis": np.asarray(patch["normal"], dtype=float),
        "radius": circle["radius"],
        "residual_max_m": circle["residual_max_m"],
        "loop_vertex_count": len(loop),
        "area_ratio": area_ratio,
    }


def cylinder_support_faces(vertices, faces, candidate_faces, axis, center, radius, length, tol):
    support = []
    half = length / 2.0
    rel = vertices - center
    t = rel @ axis
    radial = np.linalg.norm(rel - np.outer(t, axis), axis=1)
    for fi in candidate_faces:
        ids = list(faces[fi])
        if np.all(np.abs(radial[ids] - radius) <= tol) and \
                np.all(np.abs(t[ids]) <= half + tol):
            support.append(fi)
    return support


def patch_adjacency(patches, faces):
    edge_patch = defaultdict(list)
    for index, patch in enumerate(patches):
        local_count = defaultdict(int)
        for fi in patch["face_ids"]:
            face = faces[fi]
            for k in range(3):
                a, b = int(face[k]), int(face[(k + 1) % 3])
                local_count[(min(a, b), max(a, b))] += 1
        for edge, count in local_count.items():
            if count == 1:
                edge_patch[edge].append(index)
    adjacency = defaultdict(set)
    for edge, members in edge_patch.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                adjacency[members[i]].add(members[j])
                adjacency[members[j]].add(members[i])
    return adjacency


def canonical_axis(direction):
    direction = np.asarray(direction, dtype=float)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def fit_cylinders_from_sides(vertices, faces, patches, component_face_ids, tol, claimed):
    adjacency = patch_adjacency(patches, faces)
    axis_candidates = []
    for index, neighbors in adjacency.items():
        for other in neighbors:
            if other <= index:
                continue
            n1 = patches[index]["normal"]
            n2 = patches[other]["normal"]
            cos_angle = float(abs(np.dot(n1, n2)))
            if cos_angle > math.cos(math.radians(3.0)):
                continue
            if cos_angle < math.cos(math.radians(75.0)):
                continue
            cross = np.cross(n1, n2)
            norm = np.linalg.norm(cross)
            if norm < 1e-12:
                continue
            axis_candidates.append((canonical_axis(cross),
                                    min(patches[index]["area"], patches[other]["area"])))
    if not axis_candidates:
        return []
    reps = []
    axes = []
    for direction, weight in axis_candidates:
        best, best_dot = -1, math.cos(math.radians(5.0))
        for label, rep in enumerate(reps):
            dot = abs(float(np.dot(direction, rep)))
            if dot > best_dot:
                best, best_dot = label, dot
        if best < 0:
            reps.append(direction)
            axes.append([0.0, []])
            best = len(reps) - 1
        sign = 1.0 if float(np.dot(direction, reps[best])) >= 0 else -1.0
        axes[best][0] += weight
        axes[best][1].append(direction * sign * weight)
    cylinders = []
    for total_weight, directions in axes:
        if total_weight <= 0 or not directions:
            continue
        stacked = np.sum(directions, axis=0)
        norm = np.linalg.norm(stacked)
        if norm < 1e-12:
            continue
        axis = stacked / norm
        side_patches = [p for p in patches
                        if abs(float(np.dot(p["normal"], axis))) < 0.35
                        and not (set(p["face_ids"]) & claimed)]
        if len(side_patches) < 3:
            continue
        vertex_ids = sorted({vi for p in side_patches for fi in p["face_ids"]
                             for vi in faces[fi]})
        points = vertices[vertex_ids]
        origin = points.mean(axis=0)
        u, v = plane_frame(axis)
        local2d = np.c_[(points - origin) @ u, (points - origin) @ v]
        circle = fit_circle_pratt(local2d)
        if circle is None or circle["residual_max_m"] > tol:
            continue
        radius = circle["radius"]
        center_2d = circle["center_2d"]
        center_line = origin + u * center_2d[0] + v * center_2d[1]
        rel = vertices - center_line
        t_all = rel @ axis
        radial_all = np.linalg.norm(rel - np.outer(t_all, axis), axis=1)
        support = []
        for fi in component_face_ids:
            if fi in claimed:
                continue
            ids = list(faces[fi])
            if np.all(np.abs(radial_all[ids] - radius) <= tol):
                support.append(fi)
        if not support:
            continue
        t_min = float(np.min([np.min(t_all[list(faces[fi])]) for fi in support]))
        t_max = float(np.max([np.max(t_all[list(faces[fi])]) for fi in support]))
        length = t_max - t_min
        if length <= 2.0 * tol:
            continue
        support_area = float(sum(face_area(vertices, faces, fi) for fi in support))
        expected_area = 2.0 * math.pi * radius * length
        coverage = support_area / expected_area if expected_area > 0 else 0.0
        if coverage < CYL_SIDE_COVERAGE_MIN:
            continue
        center = center_line + axis * (t_min + t_max) / 2.0
        cylinders.append({
            "kind": "cylinder",
            "subtype": "cylinder",
            "axis": axis,
            "center": center,
            "radius_m": float(radius),
            "length_m": length,
            "cap_face_ids": [],
            "side_face_ids": support,
            "face_ids": list(support),
            "cap_residual_max_m": None,
            "side_coverage": coverage,
            "has_caps": False,
            "axis_vote_weight": float(total_weight),
        })
    return cylinders


def fit_cylinders(vertices, faces, patches, component_face_ids, tol):
    caps = []
    for patch in patches:
        circle = circle_from_patch(vertices, patch)
        if circle is None or circle["residual_max_m"] > tol:
            continue
        caps.append({"patch": patch, "circle": circle})
    cylinders = []
    used_caps = set()
    claimed_faces = set()
    for i, cap_a in enumerate(caps):
        if i in used_caps:
            continue
        paired = False
        for j in range(i + 1, len(caps)):
            if j in used_caps:
                continue
            cap_b = caps[j]
            axis_a = cap_a["circle"]["axis"]
            axis_b = cap_b["circle"]["axis"]
            if abs(float(np.dot(axis_a, axis_b))) < 1.0 - math.sin(math.radians(2.0)):
                continue
            r_a = cap_a["circle"]["radius"]
            r_b = cap_b["circle"]["radius"]
            if abs(r_a - r_b) > tol:
                continue
            c_a = cap_a["circle"]["center"]
            c_b = cap_b["circle"]["center"]
            delta = c_b - c_a
            length = float(abs(delta @ axis_a))
            if length <= 2.0 * tol:
                continue
            direction = axis_a if (delta @ axis_a) >= 0 else -axis_a
            center = (c_a + c_b) / 2.0
            cap_face_ids = list(cap_a["patch"]["face_ids"]) + list(cap_b["patch"]["face_ids"])
            candidate_faces = [fi for fi in component_face_ids
                               if fi not in claimed_faces and fi not in set(cap_face_ids)]
            support = cylinder_support_faces(vertices, faces, candidate_faces,
                                             direction, center, r_a, length, tol)
            if not support:
                continue
            side_area = float(sum(face_area(vertices, faces, fi) for fi in support))
            expected_area = 2.0 * math.pi * r_a * length
            coverage = side_area / expected_area if expected_area > 0 else 0.0
            if coverage < CYL_SIDE_COVERAGE_MIN:
                continue
            cylinders.append({
                "kind": "cylinder",
                "subtype": "cylinder",
                "axis": direction,
                "center": center,
                "radius_m": float(r_a),
                "length_m": length,
                "cap_face_ids": cap_face_ids,
                "side_face_ids": support,
                "face_ids": cap_face_ids + support,
                "cap_residual_max_m": max(cap_a["circle"]["residual_max_m"],
                                          cap_b["circle"]["residual_max_m"]),
                "side_coverage": coverage,
                "has_caps": True,
            })
            claimed_faces.update(cap_face_ids)
            claimed_faces.update(support)
            used_caps.update({i, j})
            paired = True
            break
        if not paired:
            circle = cap_a["circle"]
            patch = cap_a["patch"]
            cap_face_ids = list(patch["face_ids"])
            candidate_faces = [fi for fi in component_face_ids
                               if fi not in claimed_faces and fi not in set(cap_face_ids)]
            support = cylinder_support_faces(vertices, faces, candidate_faces,
                                             circle["axis"], circle["center"],
                                             circle["radius"], 0.0, tol)
            cylinders.append({
                "kind": "cylinder",
                "subtype": "disk" if support else "open_disk",
                "axis": circle["axis"],
                "center": circle["center"],
                "radius_m": float(circle["radius"]),
                "length_m": 0.0,
                "cap_face_ids": cap_face_ids,
                "side_face_ids": support,
                "face_ids": cap_face_ids + support,
                "cap_residual_max_m": circle["residual_max_m"],
                "side_coverage": None,
                "has_caps": True,
            })
            claimed_faces.update(cap_face_ids)
            claimed_faces.update(support)
    for side_cylinder in fit_cylinders_from_sides(vertices, faces, patches,
                                                  component_face_ids, tol,
                                                  claimed_faces):
        cylinders.append(side_cylinder)
        claimed_faces.update(side_cylinder["face_ids"])
    return cylinders


# --------------------------------------------------------------------------
# box fitting
# --------------------------------------------------------------------------

def canonical_direction(normal):
    normal = np.asarray(normal, dtype=float)
    pivot = int(np.argmax(np.abs(normal)))
    if normal[pivot] < 0:
        normal = -normal
    return normal


def fit_boxes(vertices, faces, patches, component_face_ids, tol):
    if len(patches) < 6:
        return []
    canonical = [canonical_direction(p["normal"]) for p in patches]
    labels = cluster_directions(np.array(canonical),
                                np.array([p["area"] for p in patches]),
                                2.0)
    direction_groups = defaultdict(list)
    for patch, label in zip(patches, labels):
        direction_groups[int(label)].append(patch)
    axis_info = {}
    for label, members in direction_groups.items():
        avg = np.average([canonical_direction(p["normal"]) for p in members],
                         axis=0, weights=[p["area"] for p in members])
        norm = np.linalg.norm(avg)
        if norm < 1e-12:
            continue
        axis_info[label] = (avg / norm, members)
    labels_list = list(axis_info)
    boxes = []
    cos_perp = math.sin(math.radians(2.0))
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            for k in range(j + 1, len(labels_list)):
                n1 = axis_info[labels_list[i]][0]
                n2 = axis_info[labels_list[j]][0]
                n3 = axis_info[labels_list[k]][0]
                if abs(float(np.dot(n1, n2))) > cos_perp:
                    continue
                if abs(float(np.dot(n1, n3))) > cos_perp:
                    continue
                if abs(float(np.dot(n2, n3))) > cos_perp:
                    continue
                axes = []
                pairs_per_axis = []
                along_per_axis = []
                ok = True
                for label in (labels_list[i], labels_list[j], labels_list[k]):
                    axis, members = axis_info[label]
                    along = []
                    for patch in members:
                        sign = 1.0 if float(np.dot(patch["normal"], axis)) >= 0 else -1.0
                        # ``patch["offset"]`` is measured against the PCA
                        # normal, whose sign is arbitrary.  Use the common
                        # candidate axis directly so opposite box faces have
                        # opposite signed positions.
                        along.append((patch, float(np.dot(patch["point"], axis))))
                    along.sort(key=lambda item: item[1])
                    used = [False] * len(along)
                    pairs = []
                    for a_idx in range(len(along)):
                        if used[a_idx]:
                            continue
                        for b_idx in range(a_idx + 1, len(along)):
                            if used[b_idx]:
                                continue
                            span = abs(along[a_idx][1] - along[b_idx][1])
                            if span > 2.0 * tol:
                                pairs.append((along[a_idx][0], along[b_idx][0], span,
                                              along[a_idx][1], along[b_idx][1]))
                                used[a_idx] = used[b_idx] = True
                                break
                    if len(pairs) != 1 or any(not u for u in used):
                        ok = False
                        break
                    axes.append(axis)
                    pairs_per_axis.append(pairs[0])
                    along_per_axis.append((pairs[0][3], pairs[0][4]))
                if not ok:
                    continue
                R = np.asarray(axes)
                centers_along = np.array([
                    (along_per_axis[i][0] + along_per_axis[i][1]) / 2.0
                    for i in range(3)])
                extents = np.array([p[2] for p in pairs_per_axis])
                if float(np.dot(np.cross(R[0], R[1]), R[2])) < 0:
                    R = np.asarray([R[0], R[1], -R[2]])
                    centers_along[2] = -centers_along[2]
                # The FBX float32 quantization makes the three recovered
                # normals very slightly non-orthogonal.  At world coordinates
                # around 9000 m, ``R.T @ centers`` amplifies that tiny angular
                # error into metres.  Intersect the three measured planes
                # instead; retain R as the local frame for the small residual
                # check and for the rebuild reference.
                try:
                    center = np.linalg.solve(R, centers_along)
                except np.linalg.LinAlgError:
                    continue
                component_vertex_ids = sorted({vi for fi in component_face_ids
                                               for vi in faces[fi]})
                local = (vertices[component_vertex_ids] - center) @ R.T
                if not np.all(np.abs(local) <= extents / 2.0 + tol):
                    continue
                box_face_ids = sorted({fi for pair in pairs_per_axis
                                       for p in pair[:2] for fi in p["face_ids"]})
                boxes.append({
                    "kind": "box",
                    "center": center,
                    "axes": R,
                    "half_extents_m": [float(e) / 2.0 for e in extents],
                    "face_ids": box_face_ids,
                })
    # Several adjacent patches can yield the same box through different
    # three-group combinations.  Keep one stable candidate for a geometry;
    # otherwise MAX_PRIMITIVES would spend the budget on duplicates and make a
    # complex component look more confidently explained than it is.
    unique = {}
    for box in boxes:
        signature = (
            tuple(np.round(box["center"], 3)),
            tuple(sorted(round(float(x), 4) for x in box["half_extents_m"])),
            tuple(np.round(np.abs(np.asarray(box["axes"])), 3).ravel()),
        )
        previous = unique.get(signature)
        if previous is None or len(box["face_ids"]) > len(previous["face_ids"]):
            unique[signature] = box
    boxes = list(unique.values())
    boxes.sort(key=lambda b: -len(b["face_ids"]))
    return boxes


# --------------------------------------------------------------------------
# rebuild and comparison
# --------------------------------------------------------------------------

def rebuild_mesh(primitives):
    all_vertices = []
    all_faces = []
    offset = 0
    for primitive in primitives:
        kind = primitive["kind"]
        if kind == "cylinder":
            verts, faces = _rebuild_cylinder(primitive["axis"], primitive["center"],
                                             primitive["radius_m"], primitive["length_m"],
                                             primitive.get("has_caps", True))
        elif kind == "box":
            verts, faces = _rebuild_box(np.asarray(primitive["axes"], dtype=float),
                                        np.asarray(primitive["center"], dtype=float),
                                        primitive["half_extents_m"])
        else:
            continue
        all_vertices.append(verts)
        all_faces.append(np.asarray(faces, dtype=np.int64) + offset)
        offset += len(verts)
    if not all_vertices:
        return None, None
    return np.vstack(all_vertices), np.vstack(all_faces)


def _rebuild_cylinder(axis, center, radius, length, with_caps=True):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    center = np.asarray(center, dtype=float)
    u, v = plane_frame(axis)
    angles = np.linspace(0.0, 2.0 * math.pi, REBUILD_SEGMENT_COUNT, endpoint=False)
    ring = center + (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)) * radius
    bottom = ring - axis * (length / 2.0)
    top = ring + axis * (length / 2.0)
    verts = np.vstack([bottom, top, center - axis * (length / 2.0), center + axis * (length / 2.0)])
    faces = []
    n = REBUILD_SEGMENT_COUNT
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
        if with_caps:
            faces.append((2 * n, i, j))
            faces.append((2 * n + 1, n + i, n + j))
    return verts, faces


def _rebuild_box(rotation, center, half_extents):
    hx, hy, hz = half_extents
    signs = np.array([[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)])
    verts = center + signs @ np.diag([hx, hy, hz]) @ rotation
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = []
    for quad in quads:
        faces.append((quad[0], quad[1], quad[2]))
        faces.append((quad[0], quad[2], quad[3]))
    return verts, faces


def point_triangle_distances(points, mesh_vertices, mesh_faces):
    """Exact closest-point distances from points to a triangle mesh.

    The area samples used by the report are stochastic, but the distance query
    itself must be to finite triangles rather than to sparse mesh vertices;
    otherwise a long, low-resolution cylinder reports an artificial axial
    error.
    """
    points = np.asarray(points, dtype=float)
    mesh_faces = np.asarray(mesh_faces, dtype=np.int64)
    best = np.full(len(points), np.inf, dtype=float)
    for ia, ib, ic in mesh_faces:
        a, b, c = (mesh_vertices[ia], mesh_vertices[ib], mesh_vertices[ic])
        ab = b - a
        ac = c - a
        ap = points - a
        d1 = ap @ ab
        d2 = ap @ ac
        closest = np.empty_like(points)
        assigned = np.zeros(len(points), dtype=bool)

        mask = (d1 <= 0.0) & (d2 <= 0.0)
        closest[mask] = a
        assigned |= mask

        bp = points - b
        d3 = bp @ ab
        d4 = bp @ ac
        mask = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
        closest[mask] = b
        assigned |= mask

        vc = d1 * d4 - d3 * d2
        mask = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        denom = d1 - d3
        v = np.divide(d1, denom, out=np.zeros_like(d1), where=np.abs(denom) > 1e-15)
        closest[mask] = a + v[mask, None] * ab
        assigned |= mask

        cp = points - c
        d5 = cp @ ab
        d6 = cp @ ac
        mask = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
        closest[mask] = c
        assigned |= mask

        vb = d5 * d2 - d1 * d6
        mask = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        denom = d2 - d6
        w = np.divide(d2, denom, out=np.zeros_like(d2), where=np.abs(denom) > 1e-15)
        closest[mask] = a + w[mask, None] * ac
        assigned |= mask

        va = d3 * d6 - d5 * d4
        mask = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        denom = (d4 - d3) + (d5 - d6)
        w = np.divide(d4 - d3, denom, out=np.zeros_like(d4), where=np.abs(denom) > 1e-15)
        closest[mask] = b + w[mask, None] * (c - b)
        assigned |= mask

        mask = ~assigned
        denom = va + vb + vc
        v_face = np.divide(vb, denom, out=np.zeros_like(vb), where=np.abs(denom) > 1e-15)
        w_face = np.divide(vc, denom, out=np.zeros_like(vc), where=np.abs(denom) > 1e-15)
        closest[mask] = (a + v_face[mask, None] * ab + w_face[mask, None] * ac)

        distances = np.linalg.norm(points - closest, axis=1)
        best = np.minimum(best, distances)
    return best


def compare_surfaces(vertices, original_faces, primitives, rng):
    rebuild_v, rebuild_f = rebuild_mesh(primitives)
    if rebuild_v is None or len(rebuild_f) == 0:
        return {"status": "rebuild_failed"}
    sample_count = min(REBUILD_SAMPLE_COUNT, max(len(rebuild_f), 1) * 20)

    def sample_surface(mesh_vertices, mesh_faces, count):
        tri_vertices = mesh_vertices[mesh_faces]
        cross = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0],
                         tri_vertices[:, 2] - tri_vertices[:, 0])
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        good = areas > 0.0
        if not np.any(good):
            return np.empty((0, 3), dtype=float)
        good_faces = mesh_faces[good]
        probabilities = areas[good] / areas[good].sum()
        indices = rng.choice(len(good_faces), size=count, p=probabilities)
        tri = good_faces[indices]
        bary = rng.random((count, 2))
        flip = (bary[:, 0] + bary[:, 1]) > 1.0
        bary[flip] = 1.0 - bary[flip]
        a, b, c = (mesh_vertices[tri[:, 0]], mesh_vertices[tri[:, 1]],
                   mesh_vertices[tri[:, 2]])
        return a + bary[:, 0, None] * (b - a) + bary[:, 1, None] * (c - a)

    original_faces = np.asarray(original_faces, dtype=np.int64)
    original_samples = sample_surface(vertices, original_faces, sample_count)
    rebuild_samples = sample_surface(rebuild_v, rebuild_f, sample_count)
    if len(original_samples) == 0 or len(rebuild_samples) == 0:
        return {"status": "sampling_failed"}
    d_orig_to_rebuild = point_triangle_distances(original_samples, rebuild_v, rebuild_f)
    d_rebuild_to_orig = point_triangle_distances(rebuild_samples, vertices, original_faces)
    return {
        "status": "ok",
        "method": "bidirectional area samples with exact point-to-triangle distance",
        "orig_to_rebuild_p95_m": float(np.percentile(d_orig_to_rebuild, 95)),
        "orig_to_rebuild_max_m": float(np.max(d_orig_to_rebuild)),
        "rebuild_to_orig_p95_m": float(np.percentile(d_rebuild_to_orig, 95)),
        "rebuild_to_orig_max_m": float(np.max(d_rebuild_to_orig)),
        "sample_count": int(sample_count),
    }


# --------------------------------------------------------------------------
# per-component fitting
# --------------------------------------------------------------------------

def assign_faces(component_face_ids, candidates):
    order = sorted(range(len(candidates)),
                   key=lambda idx: -len(candidates[idx]["face_ids"]))
    claimed = set()
    kept = []
    for idx in order:
        candidate = candidates[idx]
        mine = [fi for fi in candidate["face_ids"] if fi not in claimed]
        if not mine:
            continue
        candidate["face_ids"] = mine
        claimed.update(mine)
        kept.append(candidate)
    residual = [fi for fi in component_face_ids if fi not in claimed]
    return kept, residual


def fit_component(vertices, faces, component_face_ids, tol, rng, candidate_kinds=None):
    candidate_kinds = set(candidate_kinds or ("plane", "cylinder", "box"))
    normals, areas, centroids = triangle_normals_areas(vertices, faces)
    patches = extract_plane_patches(vertices, faces, component_face_ids, normals,
                                    areas, centroids, tol)
    cylinders = (fit_cylinders(vertices, faces, patches, component_face_ids, tol)
                 if "cylinder" in candidate_kinds else [])
    claimed = set()
    for cylinder in cylinders:
        claimed.update(cylinder["face_ids"])
    remaining_faces = [fi for fi in component_face_ids if fi not in claimed]
    remaining_patches = [p for p in patches
                         if set(p["face_ids"]) & set(remaining_faces)]
    boxes = (fit_boxes(vertices, faces, remaining_patches, remaining_faces, tol)
             if "box" in candidate_kinds else [])
    for box in boxes:
        claimed.update(box["face_ids"])
    candidates = list(cylinders) + list(boxes)
    for patch in remaining_patches:
        if "plane" not in candidate_kinds:
            continue
        if set(patch["face_ids"]) & claimed:
            continue
        candidates.append({
            "kind": "plane_patch",
            "normal": patch["normal"],
            "point": patch["point"],
            "face_ids": list(patch["face_ids"]),
            "area_m2": patch["area"],
            "residual_max_m": patch["residual_max_m"],
            "boundary_loop_count": len(patch.get("boundary_loops", [])),
        })
    kept, residual = assign_faces(component_face_ids, candidates[:MAX_PRIMITIVES])
    dropped = [c["kind"] for c in candidates[MAX_PRIMITIVES:]]
    total_area = float(np.sum(areas[component_face_ids]))
    explained_area = float(sum(face_area(vertices, faces, fi)
                               for c in kept for fi in c["face_ids"]))
    residual_area = float(np.sum(areas[residual])) if residual else 0.0
    ratio = explained_area / total_area if total_area > 0 else 0.0
    if ratio >= EXPLAINED_RATIO_SUCCESS:
        status = "primitive_combination"
    elif ratio >= EXPLAINED_RATIO_PARTIAL:
        status = "partial"
    else:
        status = "unresolved_mesh"
    component_faces = np.asarray([faces[fi] for fi in component_face_ids], dtype=np.int64)
    comparison = (compare_surfaces(vertices, component_faces, kept, rng)
                  if kept else {"status": "no_primitives"})
    return {
        "status": status,
        "primitives": kept,
        "dropped_overflow_kinds": dropped,
        "residual_face_ids": residual,
        "total_area_m2": total_area,
        "explained_area_m2": explained_area,
        "residual_area_m2": residual_area,
        "explained_ratio": ratio,
        "surface_comparison": comparison,
    }


# --------------------------------------------------------------------------
# object-level pipeline
# --------------------------------------------------------------------------

def serialize_primitive(primitive, source_face_ids=None):
    welded_face_ids = [int(fi) for fi in primitive["face_ids"]]
    if source_face_ids is None:
        source_face_ids = list(range(max(welded_face_ids, default=-1) + 1))
    out = {
        "kind": primitive["kind"],
        "face_count": len(primitive["face_ids"]),
        "face_ids": [int(source_face_ids[fi]) for fi in welded_face_ids],
        "welded_face_ids": welded_face_ids,
    }
    if primitive["kind"] == "cylinder":
        out.update({
            "subtype": primitive.get("subtype"),
            "axis": [float(x) for x in primitive["axis"]],
            "center_m": [float(x) for x in primitive["center"]],
            "radius_m": float(primitive["radius_m"]),
            "length_m": float(primitive["length_m"]),
            "has_caps": bool(primitive.get("has_caps", True)),
            "cap_residual_max_m": (None if primitive.get("cap_residual_max_m") is None
                                   else float(primitive["cap_residual_max_m"])),
            "side_coverage": primitive.get("side_coverage"),
        })
    elif primitive["kind"] == "box":
        out.update({
            "center_m": [float(x) for x in primitive["center"]],
            "axes": np.asarray(primitive["axes"], dtype=float).tolist(),
            "half_extents_m": [float(x) for x in primitive["half_extents_m"]],
        })
    elif primitive["kind"] == "plane_patch":
        out.update({
            "normal": [float(x) for x in primitive["normal"]],
            "point_m": [float(x) for x in primitive["point"]],
            "area_m2": float(primitive.get("area_m2", 0.0)),
            "residual_max_m": float(primitive.get("residual_max_m", 0.0)),
            "boundary_loop_count": int(primitive.get("boundary_loop_count", 0)),
        })
    return out


def process_moniker(moniker, meshes, tol, rng, candidate_kinds=None):
    if not meshes:
        return {"moniker": moniker, "status": "reference_missing",
                "reason": "no_fbx_mesh", "meshes": []}
    mesh_results = []
    for mesh_index, mesh in enumerate(meshes):
        original_face_sources = list(mesh.get("face_sources", range(len(mesh["faces"]))))
        vertices, faces, vertex_map, degenerate = weld_vertices(
            mesh["vertices"], mesh["faces"])
        degenerate_set = set(degenerate)
        source_face_ids = [source for index, source in enumerate(original_face_sources)
                           if index not in degenerate_set]
        components = connected_components(len(faces), faces)
        comp_results = []
        for comp_index, comp_faces in enumerate(components):
            comp_vertex_count = len({vi for fi in comp_faces for vi in faces[fi]})
            try:
                fit = fit_component(vertices, faces, comp_faces, tol, rng, candidate_kinds)
            except Exception as exc:
                comp_results.append({
                    "component_index": comp_index,
                    "triangle_count": len(comp_faces),
                    "status": "processing_failed",
                    "error": repr(exc),
                })
                continue
            comp_results.append({
                "component_index": comp_index,
                "vertex_count": comp_vertex_count,
                "triangle_count": len(comp_faces),
                "status": fit["status"],
                "primitives": [serialize_primitive(p, source_face_ids)
                               for p in fit["primitives"]],
                "residual_face_ids": [int(source_face_ids[fi])
                                      for fi in fit["residual_face_ids"]],
                "residual_welded_face_ids": fit["residual_face_ids"],
                "residual_area_m2": fit["residual_area_m2"],
                "explained_area_m2": fit["explained_area_m2"],
                "total_area_m2": fit["total_area_m2"],
                "explained_ratio": fit["explained_ratio"],
                "dropped_overflow_kinds": fit["dropped_overflow_kinds"],
                "surface_comparison": fit["surface_comparison"],
            })
        mesh_results.append({
            "mesh_index": mesh_index,
            "node_path": mesh["node_path"],
            "is_insulation": mesh["is_insulation"],
            "vertex_count_raw": int(len(mesh["vertices"])),
            "vertex_count_welded": int(len(vertices)),
            "triangle_count": len(faces),
            "degenerate_faces_after_weld": len(degenerate),
            "component_count": len(components),
            "components": comp_results,
        })
    statuses = [c["status"] for m in mesh_results for c in m["components"]]
    if statuses and all(s == "primitive_combination" for s in statuses):
        object_status = "primitive_combination"
    elif "processing_failed" in statuses:
        object_status = "processing_failed"
    elif "primitive_combination" in statuses or "partial" in statuses:
        object_status = "partial"
    else:
        object_status = "unresolved_mesh"
    return {
        "moniker": moniker,
        "status": object_status,
        "meshes": mesh_results,
    }


def write_rebuild_obj(path, row):
    lines = ["# fbx_primitive_fit rebuild geometry; research reference only"]
    vertex_offset = 1
    written = 0
    for mesh in row["meshes"]:
        for comp in mesh["components"]:
            rebuild = []
            for primitive in comp.get("primitives", []):
                if primitive["kind"] == "cylinder":
                    rebuild.append({"kind": "cylinder",
                                    "axis": primitive["axis"],
                                    "center": primitive["center_m"],
                                    "radius_m": primitive["radius_m"],
                                    "length_m": primitive["length_m"],
                                    "has_caps": primitive.get("has_caps", True)})
                elif primitive["kind"] == "box":
                    rebuild.append({"kind": "box",
                                    "axes": np.asarray(primitive["axes"], dtype=float),
                                    "center": primitive["center_m"],
                                    "half_extents_m": primitive["half_extents_m"]})
            verts, faces = rebuild_mesh(rebuild)
            if verts is None:
                continue
            lines.append(f"o {row['moniker']}_m{mesh['mesh_index']}_c{comp['component_index']}")
            for v in verts:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            for face in faces:
                a, b, c = face + vertex_offset
                lines.append(f"f {a} {b} {c}")
            vertex_offset += len(verts)
            written += 1
    if written:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def summarize(rows):
    summary = []
    for row in rows:
        counts = defaultdict(int)
        explained = 0.0
        total = 0.0
        residual = 0.0
        for mesh in row["meshes"]:
            for comp in mesh["components"]:
                for primitive in comp.get("primitives", []):
                    counts[primitive["kind"]] += 1
                explained += comp.get("explained_area_m2", 0.0)
                total += comp.get("total_area_m2", 0.0)
                residual += comp.get("residual_area_m2", 0.0)
        summary.append({
            "moniker": row["moniker"],
            "status": row["status"],
            "mesh_count": len(row["meshes"]),
            "cylinder_count": counts["cylinder"],
            "box_count": counts["box"],
            "plane_patch_count": counts["plane_patch"],
            "explained_area_m2": round(explained, 6),
            "residual_area_m2": round(residual, 6),
            "total_area_m2": round(total, 6),
            "explained_ratio": round(explained / total, 4) if total > 0 else 0.0,
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=str, help="sample stem, e.g. 063E or 15300")
    parser.add_argument("--moniker", action="append", dest="monikers",
                        help="object moniker; may be repeated")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL_M,
                        help="geometric tolerance in meters (>= FBX float32 quantization)")
    parser.add_argument("--candidate", action="append", dest="candidates",
                        choices=("plane", "cylinder", "box"),
                        help="enable one candidate family; may be repeated (default: all)")
    parser.add_argument("--run-id", type=str, default="baseline1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-obj", action="store_true",
                        help="write rebuild OBJ files for fitted objects")
    args = parser.parse_args()

    vue_path = Path(args.sample) if args.sample.endswith(".vue") else ROOT / f"{args.sample}.vue"
    fbx_path = vue_path.with_suffix(".fbx")
    if not fbx_path.exists():
        raise SystemExit(f"missing FBX file: {fbx_path}")
    wanted = [canonical_moniker(m) for m in args.monikers] if args.monikers \
        else DEFAULT_MONIKERS.get(args.sample, [])
    if not wanted:
        raise SystemExit(f"no monikers selected for sample {args.sample}")
    output_dir = args.output_dir or (ROOT / "outputs" / "primitive_research" /
                                     args.run_id / args.sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rng = np.random.default_rng(SEED)
    candidate_kinds = args.candidates or ["plane", "cylinder", "box"]
    meshes_by_moniker = collect_meshes(fbx_path, wanted)
    rows = []
    residual_rows = []
    for moniker in wanted:
        row = process_moniker(moniker, meshes_by_moniker.get(moniker, []),
                              args.tol, rng, candidate_kinds)
        rows.append(row)
        for mesh in row["meshes"]:
            for comp in mesh["components"]:
                if comp["status"] == "primitive_combination" and not comp.get("residual_face_ids"):
                    continue
                residual_rows.append({
                    "moniker": moniker,
                    "mesh_index": mesh["mesh_index"],
                    "node_path": mesh["node_path"],
                    "component_index": comp["component_index"],
                    "status": comp["status"],
                    "triangle_count": comp.get("triangle_count"),
                    "residual_face_ids": comp.get("residual_face_ids", []),
                    "residual_welded_face_ids": comp.get("residual_welded_face_ids", []),
                    "residual_area_m2": comp.get("residual_area_m2"),
                    "dropped_overflow_kinds": comp.get("dropped_overflow_kinds", []),
                    "error": comp.get("error"),
                })

    fbx_sha256 = hashlib.sha256(fbx_path.read_bytes()).hexdigest()
    fits_path = output_dir / "fbx_primitive_fits.jsonl"
    with fits_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "script_version": VERSION,
                "sample": args.sample,
                "fbx_sha256": fbx_sha256,
                "tol_m": args.tol,
                "candidate_kinds": candidate_kinds,
                "weld_decimals": WELD_DECIMALS,
                "seed": SEED,
                "record": row,
            }, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary_path = output_dir / "fbx_primitive_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "moniker", "status", "mesh_count", "cylinder_count", "box_count",
            "plane_patch_count", "explained_area_m2", "residual_area_m2",
            "total_area_m2",
            "explained_ratio"])
        writer.writeheader()
        writer.writerows(summary)

    residuals_path = output_dir / "fbx_residuals.jsonl"
    with residuals_path.open("w", encoding="utf-8") as handle:
        for item in residual_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.rebuild_obj:
        for row in rows:
            if row["status"] in ("primitive_combination", "partial"):
                obj_path = output_dir / f"rebuild_{row['moniker']}.obj"
                if write_rebuild_obj(obj_path, row) == 0:
                    obj_path.unlink(missing_ok=True)

    elapsed = time.time() - started
    print(json.dumps({
        "sample": args.sample,
        "fbx": fbx_path.name,
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "objects": len(rows),
        "status_counts": {status: sum(1 for r in rows if r["status"] == status)
                          for status in sorted({r["status"] for r in rows})},
        "elapsed_s": round(elapsed, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
