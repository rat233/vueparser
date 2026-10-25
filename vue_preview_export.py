"""Export all currently usable VUE geometry, including tentative partial objects.

No FBX input, fitting, business-type filtering or per-object acceptance gate.
Outputs GLB (Y up by CLI default), FBX and OBJ (source Z up), in metres.
"""
import argparse
from collections import Counter, defaultdict, OrderedDict
from functools import lru_cache
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import trimesh
import vue_primitive_geometry as primitive_geometry

from vue_record_parser import read_records
from vue_pipe_parser import VuePipeParser
from vue_block_sequence_probe import walk_group_elements, tag_block_views
from vue_primitive_compare import worldize_vue_blocks
from vue_root_transform_audit import read_matrix_pair
from vue_scene_tables import read_scene_tables
from vue_runtime import helper_command
from vue_export_options import (ExportOptions, add_export_arguments, options_from_args,
                                DEFAULT_SEGMENTS, SEGMENTS_ALIGNMENT, SEGMENTS_MINIMUM,
                                SEGMENTS_MAXIMUM, SEGMENTATION_POLICY_VERSION,
                                arc_path_segments, validate_segments,
                                is_full_turn, normalize_sweep, full_turn_chord_error)
from vue_progress import ExportProgress, console_progress


def canonical(s):
    return s.strip().replace('@a=', '', 1).replace('!!', '-', 1).replace('##', '-', 1)


def vue_signature_label(signature):
    """Return the compact VUE block signature used in exported node names."""
    if signature is None:
        return ''
    return '0x%x/%d' % (int(signature[0]), int(signature[1]))


def vue_signature_suffix(signature):
    label = vue_signature_label(signature)
    return '__' + label if label else ''


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-15:
        raise ValueError('invalid direction')
    return v / n


def float_view(data, offset, count):
    """Zero-copy read-only view of ``count`` stored little-endian doubles.

    Replaces the struct.unpack_from tuple + np.array double copy at every
    whole-float-array read site; cast/copy explicitly only when a writable
    array is required downstream.
    """
    return np.frombuffer(data, dtype='<f8', count=count, offset=offset)


HARD_EDGE_ANGLE_DEG = 30.0

# P1 segmentation (0928 plan §2.2).  The single production default lives in
# vue_export_options (ExportOptions.segments == DEFAULT_SEGMENTS == 16) and is
# always passed explicitly by export_sample below.  The generators accept an
# explicit ``segments`` argument; ``None`` keeps the historical direct-call
# sampling so low-level callers and the topology tests are unaffected, while
# ``export_sample`` never relies on it.  ``ELLIPSOID_MERIDIAN_SAMPLES`` stays
# fixed in P1 (only the circumferential direction is parameterized).
ELLIPSOID_MERIDIAN_SAMPLES = 24
LEGACY_CYLINDER_SEGMENTS = 32
LEGACY_TORUS_RING_SEGMENTS = 96
LEGACY_TORUS_SECTION_SEGMENTS = 24
LEGACY_CONE_SEGMENTS = 96
LEGACY_ELLIPSOID_SEGMENTS = 96
LEGACY_REVOLVED_PROFILE_SEGMENTS = 96
LEGACY_ARC_PATH_DIVISIONS = 72
LEGACY_ARC_SECTION_SEGMENTS = 24

# Keep every published format in the source VUE coordinate frame.  The VUE
# samples and the compatible FBX both use E,N,EL metres with Z up; applying a
# separate GLB-only axis swap made node translations disagree across formats.
VUE_TO_EXPORT_MATRIX = np.eye(4)

# Display-only configuration: the VUE payload carries no alpha for the
# insulation display group (display code 69); Navisworks renders that
# group translucent while the shell shares the object's own diffuse RGB.
INSULATION_OPACITY = 0.5


def piece_color(meta, group):
    """Piece RGBA: VUE material colour, translucent for display group 69."""
    color = [*meta['diffuse_rgb'], meta['opacity']]
    if group == 69:
        color[3] = min(color[3], INSULATION_OPACITY)
    return color


def split_hard_edges(vertices, faces, hard_edge_angle_deg=HARD_EDGE_ANGLE_DEG):
    """Split vertices at sharp edges and return explicit corner normals.

    VUE preview primitives share rim/corner vertices for compactness.  That
    topology is not enough to represent both a smooth cylinder wall and a
    flat end cap, so a format exporter would otherwise average their normals.
    Faces whose geometric normals differ by more than ``hard_edge_angle_deg``
    are kept in separate smoothing islands at each vertex.  The returned
    vertices and normals are therefore suitable for GLB, OBJ and FBX without
    relying on a consumer's smoothing-group defaults.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('vertices must be an Nx3 array')
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('faces must be an Mx3 triangle array')
    if len(faces) == 0:
        return vertices.copy(), faces.copy(), np.empty((0, 3), dtype=float)
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('face index out of range')

    edge_vectors = vertices[faces[:, 1:]] - vertices[faces[:, :1]]
    # Hand-written cross product is bit-identical to np.cross for these
    # operands (verified exhaustively) and avoids its moveaxis overhead.
    ea, eb = edge_vectors[:, 0], edge_vectors[:, 1]
    face_cross = np.stack([ea[:, 1] * eb[:, 2] - ea[:, 2] * eb[:, 1],
                           ea[:, 2] * eb[:, 0] - ea[:, 0] * eb[:, 2],
                           ea[:, 0] * eb[:, 1] - ea[:, 1] * eb[:, 0]], axis=1)
    face_lengths = np.linalg.norm(face_cross, axis=1)
    face_unit = np.zeros_like(face_cross)
    good = face_lengths > 1.0e-15
    face_unit[good] = face_cross[good] / face_lengths[good, None]
    cosine_limit = math.cos(math.radians(float(hard_edge_angle_deg)))

    # Group corners by their original vertex, then process equal-valence
    # vertices together. Connectivity still uses ALL incident faces, including
    # faces which share only a vertex, and remains transitively closed.
    flat = faces.reshape(-1)
    order = np.argsort(flat, kind='stable')
    counts = np.bincount(flat, minlength=len(vertices))
    starts = np.cumsum(counts) - counts
    keys = np.empty(len(flat), dtype=np.int64)
    for count in np.unique(counts[counts > 0]):
        owners = np.flatnonzero(counts == count)
        if count > 128:
            # Fan centres can have thousands of incident triangles. Traverse
            # their normal similarity in bounded chunks (never materialising
            # the O(k^2) edge set) and label by smallest member index.
            for owner in owners:
                corners = order[starts[owner]:starts[owner] + count]
                ids, inverse_ids = np.unique(corners // 3, return_inverse=True)
                unit = face_unit[ids]
                valid = good[ids]
                labels = np.full(len(ids), -1, dtype=np.int64)
                for seed in range(len(ids)):
                    if labels[seed] >= 0:
                        continue
                    labels[seed] = seed
                    frontier = np.array([seed]) if valid[seed] else np.empty(0, dtype=int)
                    while len(frontier):
                        remaining = np.flatnonzero((labels < 0) & valid)
                        if not len(remaining):
                            break
                        hit = np.zeros(len(remaining), dtype=bool)
                        for begin in range(0, len(remaining), 512):
                            chunk = remaining[begin:begin + 512]
                            hit[begin:begin + 512] = (
                                unit[chunk] @ unit[frontier].T).max(axis=1) >= cosine_limit
                        frontier = remaining[hit]
                        labels[frontier] = seed
                keys[corners] = starts[owner] + labels[inverse_ids]
            continue
        # Bound temporary adjacency/label arrays even for large input meshes.
        batch_size = max(1, 262144 // (int(count) ** 2))
        for begin in range(0, len(owners), batch_size):
            batch = owners[begin:begin + batch_size]
            corners = order[starts[batch, None] + np.arange(count)]
            ids = corners // 3
            normals = face_unit[ids]
            adjacent = (normals @ np.swapaxes(normals, 1, 2)) >= cosine_limit
            valid = good[ids]
            adjacent &= valid[:, :, None] & valid[:, None, :]
            # Repeated corners of one degenerate face belong to one island.
            adjacent |= ids[:, :, None] == ids[:, None, :]
            labels = np.broadcast_to(np.arange(count), ids.shape).copy()
            while True:
                propagated = np.where(adjacent, labels[:, None, :], count).min(axis=2)
                if np.array_equal(propagated, labels):
                    break
                labels = propagated
            keys[corners] = starts[batch, None] + labels

    # O(n) equivalent of np.unique(keys, return_index=True, return_inverse=True)
    # for in-range integer keys (each key lies in a disjoint per-vertex corner
    # range). Reproduces the exact values, first-occurrence indices and inverse
    # mapping (verified bit-exact) but avoids the full sort.
    used = np.zeros(len(flat), dtype=bool)
    used[keys] = True
    remap = np.cumsum(used) - 1
    inverse = remap[keys]
    first = np.empty(len(flat), dtype=np.int64)
    first[keys[::-1]] = np.arange(len(flat) - 1, -1, -1)
    first = first[used]
    new_vertices = vertices[flat[first]]
    new_faces = inverse.reshape((-1, 3))
    new_normals = np.empty((len(first), 3), dtype=float)
    # Per-column bincount accumulates in the same order as np.add.at and is
    # bit-identical (verified), but runs at C speed.
    corner_weight_columns = np.repeat(face_cross, 3, axis=0)
    for axis in range(3):
        new_normals[:, axis] = np.bincount(
            inverse, weights=corner_weight_columns[:, axis], minlength=len(first))
    normal_lengths = np.linalg.norm(new_normals, axis=1)
    valid_normals = normal_lengths > 1.0e-15
    new_normals[valid_normals] /= normal_lengths[valid_normals, None]
    new_normals[~valid_normals] = (0.0, 0.0, 1.0)
    # Release the per-corner scratch before returning so a batched caller's
    # peak does not stack it underneath the per-entry result copies.
    del edge_vectors, face_cross, face_lengths, face_unit, good
    del flat, order, counts, starts, keys
    del used, remap, inverse, first
    del corner_weight_columns, normal_lengths, valid_normals
    return new_vertices, new_faces, new_normals


def split_scratch_estimate(nv, nf):
    """Conservative peak-scratch UPPER bound for one split batch.

    Calibration (tracemalloc peaks against the previous 84*nv + 560*nf pair,
    itself raised from 24/160 after a 3.5x underestimate): grid200 0.594,
    grid350 0.583, grid500 0.572, 200k-triangle fan centre 0.733, 8xgrid200
    batch 0.632.  The coefficients below raise both a further ~14% (96/640)
    so unmeasured valence mixes stay under 1.0x, well inside the required
    batch-peak <= 1.2 x budget gate.
    """
    return 96 * nv + 640 * nf


def split_hard_edges_many(entries, budget_bytes=64 * 1024 * 1024, progress=None,
                          downcast=False):
    """Split several meshes in one call; identical results to per-mesh calls.

    ``entries`` is a sequence of (vertices, faces).  The computation above is
    purely per-vertex, so concatenating the meshes produces the same islands,
    vertices and normals; only the Python/numpy call overhead disappears.
    Results are split back per input entry.  Entries without faces are
    returned as the identity copy, matching ``split_hard_edges``.

    Entries are grouped into batches bounded by ``budget_bytes`` (a hard
    peak-scratch upper bound), so concatenated inputs and per-corner
    temporaries never span the whole scene; oversized meshes form their own
    batch.  The returned slices are copies so batch-wide arrays can be freed
    immediately.

    With ``downcast=True`` the returned local vertices/normals are float32 and
    the faces uint32 -- the exact precision every downstream consumer (GLB
    buffer, ASCII FBX) stores.  World placement stays double precision in
    ``origin``; only these local arrays are reduced.

    Calibration (tracemalloc peaks against 84*nv + 560*nf, the previous
    coefficient pair, itself raised from 24/160 after a 3.5x underestimate):
    grid200 0.594, grid350 0.583, grid500 0.572, 200k-triangle fan centre
    0.733, 8xgrid200 batch 0.632.  The bound below raises both coefficients a
    further ~14% (96/640) so unmeasured valence mixes stay under 1.0x, well
    inside the required batch-peak <= 1.2 x budget gate.
    """
    prepared = [(np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64))
                for v, f in entries]
    results = [None] * len(prepared)

    def estimate(v, f):
        # Conservative peak-scratch UPPER bound (concatenated inputs plus the
        # edge/cross/normal/weight temporaries and their output copies).
        return split_scratch_estimate(len(v), len(f))

    batches, current, cost = [], [], 0
    for index, (v, f) in enumerate(prepared):
        if not len(f):
            results[index] = (v.copy(), f.copy(), np.empty((0, 3), dtype=float))
            continue
        size = estimate(v, f)
        if current and cost + size > budget_bytes:
            batches.append(current)
            current, cost = [], 0
        current.append(index)
        cost += size
        if cost >= budget_bytes:
            batches.append(current)
            current, cost = [], 0
    if current:
        batches.append(current)
    if progress is not None:
        for batch in batches:
            if len(batch) == 1 and estimate(*prepared[batch[0]]) >= budget_bytes:
                progress('超大网格独立成批, 预计峰值内存 ~%.0f MB, 批预算无法约束其连通性拆分'
                         % (estimate(*prepared[batch[0]]) / 2**20))

    for batch in batches:
        solid = [(index, prepared[index]) for index in batch]
        vertex_offsets = np.cumsum([0] + [len(v) for _, (v, _) in solid])
        face_offsets = np.cumsum([0] + [len(f) for _, (_, f) in solid])
        vertices_all = np.vstack([v for _, (v, _) in solid])
        faces_all = np.vstack([f for _, (_, f) in solid])
        faces_all += np.repeat(vertex_offsets[:-1], np.diff(face_offsets))[:, None]
        new_vertices, new_faces, new_normals = split_hard_edges(vertices_all, faces_all)
        del vertices_all, faces_all
        if downcast:
            # B3: intermediate pieces live at GLB precision from here on;
            # the split computation above stays double precision, so the
            # cast reproduces the previous single f64->f32 rounding exactly.
            new_vertices = new_vertices.astype('<f4')
            new_normals = new_normals.astype('<f4')
        # New vertices keep vertex order: each mesh contributes one
        # contiguous range, recoverable from its face indices.
        next_vertex = 0
        for slot, (index, _) in enumerate(solid):
            block = new_faces[face_offsets[slot]:face_offsets[slot + 1]]
            count = int(block.max()) + 1 - next_vertex
            results[index] = (new_vertices[next_vertex:next_vertex + count].copy(),
                              (block - next_vertex).astype('<u4') if downcast
                              else (block - next_vertex).copy(),
                              new_normals[next_vertex:next_vertex + count].copy())
            del block
            next_vertex += count
        del solid, vertex_offsets, face_offsets
        del new_vertices, new_faces, new_normals
    return results


@lru_cache(maxsize=128)
def _cylinder_faces(segments, disk, separate_caps):
    """Preallocated topology; reuse segment indices across cylinder instances."""
    i = np.arange(segments)
    j = (i + 1) % segments
    if disk:
        faces = np.empty((segments, 3), dtype=np.int64)
        faces[:, 0], faces[:, 1], faces[:, 2] = segments, i, j
    else:
        faces = np.empty((segments, 4, 3), dtype=np.int64)
        faces[:, 0, :] = np.column_stack((i, j, segments + j))
        faces[:, 1, :] = np.column_stack((i, segments + j, segments + i))
        bottom = 2 * segments if separate_caps else 0
        top = 3 * segments if separate_caps else segments
        center = 4 * segments if separate_caps else 2 * segments
        faces[:, 2, 0], faces[:, 2, 1], faces[:, 2, 2] = center, bottom + j, bottom + i
        faces[:, 3, 0], faces[:, 3, 1], faces[:, 3, 2] = center + 1, top + i, top + j
        faces = faces.reshape(-1, 3)
    faces.setflags(write=False)
    return faces


def cylinder(block, segments=LEGACY_CYLINDER_SEGMENTS, *, with_normals=False):
    p, a = np.asarray(block['point']), np.asarray(block['axis'])
    h, r = np.linalg.norm(a), block['radius']
    z = unit(a if h > 1e-12 else block['dirv'])
    x = unit(np.cross(z, np.eye(3)[np.argmin(np.abs(z))]))
    y = np.cross(z, x)
    t = np.arange(segments) * (2 * math.pi / segments)
    radial = np.cos(t)[:, None] * x + np.sin(t)[:, None] * y
    disk = h <= 1e-10
    count = segments + 1 if disk else (4 if with_normals else 2) * segments + 2
    vertices = np.empty((count, 3), dtype=float)
    vertices[:segments] = p + r * radial
    if disk:
        vertices[-1] = p
    else:
        vertices[segments:2 * segments] = p + a + r * radial
        if with_normals:
            vertices[2 * segments:4 * segments] = vertices[:2 * segments]
        vertices[-2], vertices[-1] = p, p + a
    faces = _cylinder_faces(segments, disk, with_normals).copy()
    if not with_normals:
        return vertices, faces
    normals = np.empty_like(vertices)
    if disk:
        normals[:] = z
    else:
        normals[:segments] = normals[segments:2 * segments] = radial
        normals[2 * segments:3 * segments] = normals[-2] = -z
        normals[3 * segments:4 * segments] = normals[-1] = z
    return vertices, faces, normals


def _orient_generated_solid(vertices, faces):
    """Orient a generated, already consistently wound closed surface.

    This is not a replacement for repairing arbitrary input meshes. Only
    topology constructed by the primitive generators below uses this path.
    Subtract an origin before evaluating signed volume to retain precision.
    """
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces] - vertices[0]
    volume6 = np.einsum('ij,ij->i', triangles[:, 0],
                        np.cross(triangles[:, 1], triangles[:, 2])).sum()
    if volume6 < 0:
        faces = faces[:, ::-1].copy()
    elif not faces.flags.writeable:
        faces = faces.copy()
    return vertices, faces


def _orient_generated_surface(vertices, faces):
    """``fix_normals`` equivalent specialised for generator topology.

    Reproduces trimesh's visual result: winding is propagated consistent
    within each connected component (``fix_winding``), then the whole mesh
    flips outward when closed and inside-out (``fix_inversion``).  Avoids
    trimesh's heavy graph/mass-properties path while keeping the looks.
    """
    faces = np.asarray(faces, dtype=np.int64).copy()
    if len(faces) == 0:
        return vertices, faces
    directed = np.stack([faces[:, [0, 1]], faces[:, [1, 2]],
                         faces[:, [2, 0]]], axis=1).reshape(-1, 2)
    face_ids = np.repeat(np.arange(len(faces)), 3)
    key = np.sort(directed, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    skey = key[order]
    starts = np.flatnonzero(np.r_[True, np.any(skey[1:] != skey[:-1], axis=1)])
    lengths = np.diff(np.r_[starts, len(skey)])
    manifold = lengths == 2
    first = order[starts[manifold]]
    second = order[starts[manifold] + 1]
    # Same-direction use of a shared edge means one face is wound wrongly.
    disagree = ((directed[first, 0] == directed[second, 0]) &
                (directed[first, 1] == directed[second, 1]))
    adjacency = [[] for _ in range(len(faces))]
    for a, b, bad in zip(face_ids[first], face_ids[second], disagree):
        adjacency[a].append((int(b), bool(bad)))
        adjacency[b].append((int(a), bool(bad)))
    flip = np.zeros(len(faces), dtype=bool)
    component = np.full(len(faces), -1, dtype=np.int64)
    for seed in range(len(faces)):
        if component[seed] >= 0:
            continue
        component[seed] = seed
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour, bad in adjacency[current]:
                if component[neighbour] < 0:
                    component[neighbour] = seed
                    flip[neighbour] = flip[current] ^ bad
                    stack.append(neighbour)
    faces[flip] = faces[flip, ::-1]
    # fix_inversion equivalent, PER connected component (matching trimesh's
    # body-wise behaviour): closed shells orient outward by signed volume,
    # open shells keep the propagated winding (their starting orientation is
    # fixed by the generator construction contract).  A whole-mesh volume
    # would hide reversed shells behind larger positive ones.
    group = np.empty(len(order), dtype=np.int64)
    group[order] = np.repeat(np.arange(len(starts)), lengths)
    face_closed = (lengths[group] == 2).reshape(-1, 3).all(axis=1)
    sorted_faces = np.argsort(component, kind='stable')
    comp_sorted = component[sorted_faces]
    splits = np.flatnonzero(np.r_[True, comp_sorted[1:] != comp_sorted[:-1]])
    for members in np.split(sorted_faces, splits[1:]):
        if not face_closed[members].all():
            continue
        triangles = vertices[faces[members]] - vertices[0]
        volume6 = np.einsum('ij,ij->i', triangles[:, 0],
                            np.cross(triangles[:, 1], triangles[:, 2])).sum()
        if volume6 < 0:
            faces[members] = faces[members][:, ::-1]
    return vertices, faces


def extrusion(profile, vector, matrix, *, with_normals=False):
    profile=np.asarray(profile,dtype=float)
    vector=np.asarray(vector,dtype=float)
    if vector.ndim==0: vector=np.array([float(vector),0.,0.])
    if profile.shape!=(5,3) or vector.shape!=(3,) or not np.isfinite(np.r_[profile.ravel(),vector]).all():
        raise ValueError('unusable extrusion profile')
    if np.linalg.norm(profile[0]-profile[-1])>1e-8 or np.linalg.norm(vector)<=1e-10:
        raise ValueError('open profile or zero extrusion vector')
    profile=profile[:4];edges=np.roll(profile,-1,axis=0)-profile
    lengths=np.linalg.norm(edges,axis=1)
    if min(lengths)<=1e-10 or np.max(abs(edges[:2]+edges[2:]))>1e-8:
        raise ValueError('nonrectangular extrusion profile')
    if abs(edges[0]@edges[1])>1e-8*lengths[0]*lengths[1]:
        raise ValueError('nonrectangular extrusion profile')
    if np.max(abs(edges@vector)/(lengths*np.linalg.norm(vector)))>1e-7:
        raise ValueError('oblique extrusion outside supported right prisms')
    local=np.vstack([profile,profile+vector])
    v=(np.c_[local,np.ones(8)]@np.asarray(matrix).reshape(4,4))[:,:3]
    if with_normals:
        return primitive_geometry.prism(v[:4],v[4:],np.array([[0,2,1],[0,3,2]]))
    f=np.empty((12,3),dtype=np.int64)
    f[:4]=[(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
    f[4:]=primitive_geometry.ring_faces(2,4)
    return _orient_generated_solid(v, f)


def _bspline_basis(index, degree, parameter, knots):
    """Cox-de Boor basis evaluation used by the native VUE surface decoder."""
    if degree == 0:
        return float(knots[index] <= parameter < knots[index + 1])
    left = 0.0
    if knots[index + degree] > knots[index]:
        left = ((parameter - knots[index]) /
                (knots[index + degree] - knots[index]) *
                _bspline_basis(index, degree - 1, parameter, knots))
    right = 0.0
    if knots[index + degree + 1] > knots[index + 1]:
        right = ((knots[index + degree + 1] - parameter) /
                 (knots[index + degree + 1] - knots[index + 1]) *
                 _bspline_basis(index + 1, degree - 1, parameter, knots))
    return left + right


def _bspline_basis_row(parameter, degree, knots, count):
    """Return all basis values, including the clamped right endpoint."""
    return _bspline_basis_matrix((parameter,), degree, knots, count)[0].copy()


def _bspline_basis_matrix(parameters, degree, knots, count):
    return _cached_bspline_basis(tuple(parameters), degree, tuple(knots), count)


@lru_cache(maxsize=256)
def _cached_bspline_basis(parameters, degree, knots, count):
    """Iterative Cox-de Boor evaluation; cached arrays are immutable."""
    knots = np.asarray(knots, dtype=float)
    values = np.asarray(parameters, dtype=float)[:, None]
    basis = ((values >= knots[:-1]) & (values < knots[1:])).astype(float)
    for level in range(1, degree + 1):
        width = len(knots) - level - 1
        left_den = knots[level:level + width] - knots[:width]
        right_den = knots[level + 1:level + 1 + width] - knots[1:1 + width]
        left = np.divide(values - knots[:width], left_den,
                         out=np.zeros((len(values), width)), where=left_den > 0)
        right = np.divide(knots[level + 1:level + 1 + width] - values, right_den,
                          out=np.zeros((len(values), width)), where=right_den > 0)
        basis = left * basis[:, :width] + right * basis[:, 1:width + 1]
    basis = basis[:, :count].copy()
    endpoint = values[:, 0] >= knots[-1] - 1e-12
    basis[endpoint] = 0
    basis[endpoint, -1] = 1
    basis.setflags(write=False)
    return basis


def _surface_parameters(knots, samples_per_span):
    """Sample every non-zero knot span without duplicating internal seams."""
    values = []
    for low, high in zip(knots[:-1], knots[1:]):
        if high - low <= 1e-12:
            continue
        values.extend(np.linspace(low, high, samples_per_span,
                                  endpoint=False).tolist())
    values.append(float(knots[-1]))
    return values


def nurbs_surface_from_vue(data, offset, matrix=None):
    """Decode the observed 0x18 rational/non-rational VUE surface payloads.

    The control net, weights and knot vectors are all stored in the block.  No
    reference geometry or primitive fitting is involved.  The 752-byte
    variant is non-rational; the 760-, 2128- and 2464-byte variants carry one
    weight per control point.  Their control-net dimensions and knot counts are
    declared at +0x68..+0x77.
    """
    tag, size = struct.unpack_from('<2I', data, offset)
    if tag != 0x18 or size not in (752, 760, 2128, 2464, 2288, 2488):
        raise ValueError('unsupported 0x18 surface layout')
    nu, nv, knot_u_count, knot_v_count = struct.unpack_from(
        '<4I', data, offset + 0x68)
    if min(nu, nv, knot_u_count, knot_v_count) <= 0:
        raise ValueError('invalid VUE surface dimensions')
    point_count = int(nu * nv)
    point_base = offset + 0xc0
    point_end = point_base + 24 * point_count
    if point_end > offset + size:
        raise ValueError('truncated VUE surface control net')
    control = np.frombuffer(data, dtype='<f8', count=3*point_count,
                            offset=point_base).reshape(nv, nu, 3)

    rational = size in (760, 2128, 2464, 2288, 2488)
    if rational:
        weight_base = point_end
        weight_end = weight_base + 8 * point_count
        if weight_end > offset + size:
            raise ValueError('truncated VUE surface weights')
        weights = np.frombuffer(data, dtype='<f8', count=point_count,
                                offset=weight_base).reshape(nv, nu)
        knot_base = weight_end
    else:
        weights = np.ones((nv, nu), dtype=float)
        knot_base = point_end
    knot_end = knot_base + 8 * (knot_u_count + knot_v_count)
    if knot_end > offset + size:
        raise ValueError('truncated VUE surface knot vectors')
    knots_u = np.frombuffer(data, dtype='<f8', count=knot_u_count, offset=knot_base)
    knots_v = np.frombuffer(data, dtype='<f8', count=knot_v_count,
                            offset=knot_base + 8*knot_u_count)
    if not np.isfinite(np.r_[control.ravel(), weights.ravel(), knots_u, knots_v]).all():
        raise ValueError('nonfinite VUE surface payload')
    if np.any(weights <= 0) or np.any(np.diff(knots_u) < -1e-12) or np.any(np.diff(knots_v) < -1e-12):
        raise ValueError('invalid VUE surface weights or knots')
    degree_u = knot_u_count - nu - 1
    degree_v = knot_v_count - nv - 1
    if not (1 <= degree_u < nu and 1 <= degree_v < nv):
        raise ValueError('unsupported VUE surface degree')

    # The observed stair surfaces are compact, low-degree splines.  Keeping
    # several samples in each knot span preserves the native curved profile
    # while avoiding the much larger tessellation used by the reference FBX.
    u_values = _surface_parameters(knots_u, 12 if size != 752 else 8)
    v_values = _surface_parameters(knots_v, 4 if size == 2464 else 2)
    u_basis = _bspline_basis_matrix(u_values, degree_u, knots_u, nu)
    v_basis = _bspline_basis_matrix(v_values, degree_v, knots_v, nv)
    # Keep the original control-point reduction order. Reassociating the sum
    # as matrix products perturbs normals on extremely thin world-space faces.
    rational_basis = v_basis[:, None, :, None] * u_basis[None, :, None, :] * weights
    denominator = rational_basis.sum(axis=(2, 3))
    if np.any(denominator <= 1e-15):
        raise ValueError('zero VUE surface basis denominator')
    numerator = (rational_basis[:, :, :, :, None] * control).sum(axis=(2, 3))
    vertices = (numerator / denominator[:, :, None]).reshape(-1, 3)
    u_count = len(u_values)
    faces = []
    for j in range(len(v_values) - 1):
        for i in range(u_count - 1):
            a = j * u_count + i
            b, c, d = a + 1, a + u_count + 1, a + u_count
            faces.extend(((a, b, c), (a, c, d)))
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError('empty VUE surface')

    # Most observed 0x18 stair blocks already contain world coordinates.  A
    # root matrix is still considered for the other valid placement variant;
    # the block's own VUE bounds select the correct native interpretation.
    candidates = [vertices]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        candidates.append((np.c_[vertices, np.ones(len(vertices))] @ m)[:, :3])
    source_box = float_view(data, offset + 8, 6)
    vertices = min(candidates, key=lambda candidate: bounds_error(candidate, source_box))
    if size in (2288, 2488):
        # Inline array offsets are relative to the surface header at +0x50.
        actual = (point_base, point_end, knot_base, knot_base+8*knot_u_count)
        declared = struct.unpack_from('<4I', data, offset+0x50)
        if any(offset+0x50+p != a for p,a in zip(declared,actual)) or knot_end != offset+size-8:
            raise ValueError('inconsistent native surface array pointers')
        # These rotated rail variants have truncated cached bounds. Their
        # complete native control nets and knot domains define the surface.
    elif bounds_error(vertices, source_box) > 0.005:
        raise ValueError('VUE surface bounds mismatch')
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.fix_normals()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _polygon_area(points):
    return 0.5 * float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                               points[:, 1] * np.roll(points[:, 0], -1)))


def _split_planar_edges_at_vertices(points, faces):
    """Restore collinear boundary vertices omitted by earcut hole bridges.

    Split only existing triangle edges, using original loop vertices. This
    prevents T-junctions at aligned holes without moving a boundary or
    introducing points on independently decoded wall edges.
    """
    cache, result = {}, []
    pending = [tuple(map(int, face)) for face in faces]
    while pending:
        triangle = pending.pop()
        abc = points[list(triangle)]
        ab, ac = abc[1]-abc[0], abc[2]-abc[0]
        longest = max(np.linalg.norm(ab), np.linalg.norm(ac), np.linalg.norm(abc[2]-abc[1]))
        if longest <= 1e-12 or abs(ab[0]*ac[1]-ab[1]*ac[0])/longest <= 1e-9:
            # Earcut may leave a numerically thin bridge between collinear
            # hole edges. It has no surface area at the boundary tolerance.
            continue
        for i in range(3):
            a, b, c = triangle[i], triangle[(i+1)%3], triangle[(i+2)%3]
            key = (min(a,b), max(a,b))
            if key not in cache:
                start, stop = points[key[0]], points[key[1]]
                vector = stop-start
                length = np.linalg.norm(vector)
                if length <= 1e-12:
                    raise ValueError('degenerate planar triangulation edge')
                delta = points-start
                parameter = delta @ vector / length**2
                distance = abs(delta[:,0]*vector[1]-delta[:,1]*vector[0])/length
                indices = np.flatnonzero((parameter*length > 1e-9) &
                    ((1-parameter)*length > 1e-9) & (distance < 1e-9))
                cache[key] = indices[np.argsort(parameter[indices])].tolist()
            middle = cache[key]
            if middle:
                ordered = [a] + (middle if a==key[0] else middle[::-1]) + [b]
                pending.extend((first, second, c) for first, second in zip(ordered, ordered[1:]))
                break
        else:
            result.append(triangle)
    return np.asarray(result, dtype=np.int64)


def _triangulate_planar_loops(loops, normal):
    """Triangulate one planar outer loop and zero or more inner loops.

    The VUE face and wall blocks keep the original boundary vertices.  Earcut
    is used here with all of those vertices intact, so the resulting cap can
    share its square/circular boundary edges with the independently decoded
    hole walls instead of introducing T-junctions.
    """
    loops = [np.asarray(loop, dtype=float) for loop in loops]
    if len(loops) < 1 or any(loop.ndim != 2 or loop.shape[1] != 3 or
                              len(loop) < 3 for loop in loops):
        raise ValueError('invalid planar loop set')
    normal = unit(normal)
    if any(not np.isfinite(loop).all() for loop in loops):
        raise ValueError('nonfinite planar loop')
    origin = loops[0].mean(axis=0)
    if any(np.max(np.abs((loop - origin) @ normal)) > 1e-7
           for loop in loops):
        raise ValueError('planar loops are not coplanar')

    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    axis_u = unit(np.cross(normal, helper))
    axis_v = np.cross(normal, axis_u)

    def flatten(points):
        relative = points - origin
        return np.c_[relative @ axis_u, relative @ axis_v]

    flat_loops = [flatten(loop) for loop in loops]
    # mapbox-earcut accepts the ring ends, with the first ring as the exterior
    # and the remaining rings as holes.  Keep every original ring vertex.
    if _polygon_area(flat_loops[0]) < 0:
        flat_loops[0] = flat_loops[0][::-1]
    for index in range(1, len(flat_loops)):
        if _polygon_area(flat_loops[index]) < 0:
            flat_loops[index] = flat_loops[index][::-1]
    flat = np.ascontiguousarray(np.vstack(flat_loops), dtype=np.float64)
    ends = np.cumsum([len(loop) for loop in flat_loops], dtype=np.uint32)
    import mapbox_earcut
    faces = mapbox_earcut.triangulate_float64(flat, ends).reshape(-1, 3)
    if len(faces) == 0:
        raise ValueError('empty planar triangulation')
    faces = _split_planar_edges_at_vertices(flat, faces)
    areas = [_polygon_area(loop) for loop in flat_loops]
    expected = abs(areas[0]) - sum(abs(value) for value in areas[1:])
    edge1 = flat[faces[:, 1]] - flat[faces[:, 0]]
    edge2 = flat[faces[:, 2]] - flat[faces[:, 0]]
    signed = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if (expected <= 0 or np.any(np.abs(signed) < 1e-16) or
            abs(np.abs(signed).sum() / 2 - expected) >
            max(1e-12, expected * 1e-8)):
        raise ValueError('invalid planar hole triangulation')
    vertices = origin + flat[:, 0, None] * axis_u + flat[:, 1, None] * axis_v
    faces = _orient_triangles(vertices, faces,
                              np.tile(normal, (len(faces), 1)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _native_boundary_loop(data, descriptor, count, record_start, block_start, block_end):
    """Follow counted native boundary references; validate every edge join."""
    if not 1 <= count <= 10000 or not block_start <= descriptor <= block_end - 8*count:
        raise ValueError('invalid native boundary descriptor range')
    pieces = []
    for index in range(count):
        slot = descriptor + 8*index
        pointer, kind = struct.unpack_from('<2I', data, slot)
        target = record_start + 0x70 + pointer
        length = {3: 64, 16: 96, 15: 80, 4: 16}.get(kind)
        if length is None or not descriptor + 8*count <= target <= block_end - length:
            raise ValueError('unsupported or out-of-block native boundary reference')
        if kind == 4:
            if count != 1:
                raise ValueError('polyline must be the sole loop descriptor')
            return _native_polyline(data, slot, record_start, block_end)
        if kind == 15:
            if count != 1:
                raise ValueError('circle must be the sole loop descriptor')
            center, axis, radial = native_circle(data, target, np.eye(4))
            angles = np.arange(96) * (2*np.pi/96)
            return center + np.cos(angles)[:, None]*radial + np.sin(angles)[:, None]*np.cross(axis, radial)
        piece = _profile_points(_profile_entity(data, target, kind))
        if pieces and np.linalg.norm(pieces[-1][-1] - piece[0]) > 1e-7:
            raise ValueError('disconnected native boundary edges')
        pieces.append(piece)
    if np.linalg.norm(pieces[-1][-1] - pieces[0][0]) > 1e-7:
        raise ValueError('open native boundary loop')
    return np.vstack([piece[:-1] for piece in pieces])


def planar_loops_from_vue(data, offset, record_start, matrix=None):
    """Decode native face loop tables, including polygon and circular holes."""
    tag, size = struct.unpack_from('<2I', data, offset)
    end = offset + size - 8
    if tag != 14 or size < 232 or end > len(data):
        raise ValueError('unsupported native planar loop layout')
    pointer, count = struct.unpack_from('<2I', data, offset + 0x98)
    table = record_start + 0x70 + pointer
    if table != offset + 0xc8 or not 1 <= count <= 1000 or table + 12*count > end:
        raise ValueError('invalid native face loop table')
    loops = []
    for index in range(count):
        pointer, edges, flags = struct.unpack_from('<3I', data, table + 12*index)
        descriptor = record_start + 0x70 + pointer
        if descriptor < table + 12*count or flags != 256:
            raise ValueError('invalid native face loop entry')
        loops.append(_native_boundary_loop(data, descriptor, edges, record_start, offset, end))
    normal = unit(_profile_vector(data, offset + 0x68))
    vertices, faces = _triangulate_planar_loops(loops, normal)
    plane = _profile_vector(data, offset + 0x50)
    if np.max(abs((vertices-plane) @ normal)) > 1e-7:
        raise ValueError('native boundary outside declared face plane')
    source_box = float_view(data, offset + 8, 6)
    world, error = _choose_placement(vertices, matrix, source_box)
    if error > 1e-5:
        # Some cap caches omit outer corners; the native closed loops and
        # owning record bounds still constrain the complete face.
        record_box = float_view(data, record_start+8, 6)
        if np.any(world.min(0)<record_box[:3]-1e-7) or np.any(world.max(0)>record_box[3:]+1e-7):
            raise ValueError('native planar loop bounds mismatch')
    return world, faces


def native_profile_extrusion_from_vue(data, offset, record_start, matrix=None):
    """Decode direct polylines and counted compound extrusion boundaries."""
    tag, size = struct.unpack_from('<2I', data, offset)
    end = offset + size - 8
    if tag != 31 or size < 200 or end > len(data):
        raise ValueError('unsupported native profile extrusion')
    pointer, flags = struct.unpack_from('<2I', data, offset + 0x68)
    if record_start + 0x70 + pointer != offset + 0x70:
        raise ValueError('invalid extrusion profile reference')
    pointer, kind = struct.unpack_from('<2I', data, offset + 0x70)
    if kind == 4:
        profile = _native_polyline(data, offset + 0x70, record_start, end)
    elif kind == 12:
        if record_start + 0x70 + pointer != offset + 0x78:
            raise ValueError('invalid compound extrusion boundary')
        pointer, count, loop_flags = struct.unpack_from('<3I', data, offset + 0x78)
        if loop_flags != 256:
            raise ValueError('unsupported extrusion boundary flags')
        profile = _native_boundary_loop(data, record_start + 0x70 + pointer, count,
                                        record_start, offset, end)
    else:
        raise ValueError('unsupported extrusion profile kind')
    # Remove only zero-area A-B-A spikes and consecutive duplicate vertices.
    # A finite notch or concavity must retain its native boundary.
    changed = True
    while changed and len(profile) >= 3:
        changed = False
        for i in range(len(profile)):
            if (np.linalg.norm(profile[i] - profile[(i+1) % len(profile)]) < 1e-10 or
                    np.linalg.norm(profile[i-1] - profile[(i+1) % len(profile)]) < 1e-10):
                profile = np.delete(profile, i, axis=0); changed = True; break
    vector = _profile_vector(data, offset + 0x50)
    unit(vector)  # Reject a zero or non-finite extrusion vector.
    if len(profile) < 3:
        raise ValueError('invalid extrusion profile plane')
    centered = profile - profile[0]
    normal = unit(np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0))
    if np.max(abs(centered @ normal)) > 1e-7 or abs(normal @ vector) < 1e-12:
        raise ValueError('invalid extrusion profile plane')
    # Native vectors can describe oblique extrusion; preserve their direction.
    count = len(profile)
    vertices = np.vstack((profile, profile + vector))
    faces = []
    for i in range(count):
        j = (i+1) % count
        faces.extend(((i,j,j+count), (i,j+count,i+count)))
    if flags & 0x10000:
        cap = _triangulate_simple_polygon(profile-profile[0])
        faces.extend(cap); faces.extend(cap+count)
    faces = np.asarray(faces, dtype=np.int64)
    source_box = float_view(data, offset + 8, 6)
    world, error = _choose_placement(vertices, matrix, source_box)
    # Observed caches extend 5 mm beyond each extrusion end.
    if error > .005 + 1e-8:
        raise ValueError('native extrusion bounds mismatch')
    mesh = trimesh.Trimesh(world-world[0], faces, process=False)
    if flags & 0x10000:
        mesh.fix_normals()
        if not mesh.is_watertight or mesh.volume <= 0:
            raise ValueError('native extrusion solid did not close')
    return world, np.asarray(mesh.faces)


def planar_boundary_3528_from_vue(data, offset, record_start, matrix=None):
    """Decode the seven-straight-edge boundary in the observed 0xe/3528.

    In the slab layout the block is one of the two zero-thickness cap faces;
    its seven inline kind-3 records form the stepped outer perimeter.  Hole
    loops are supplied by the sibling wall blocks and are assembled by
    ``floor_shell_context_from_vue``.
    """
    if struct.unpack_from('<2I', data, offset) != (0x0e, 3528):
        raise ValueError('unsupported 0xe/3528 planar boundary')
    if struct.unpack_from('<2I', data, offset + 0x98)[1] != 13 or \
            struct.unpack_from('<2I', data, offset + 0xc8)[1] != 7:
        raise ValueError('unsupported 0xe/3528 boundary counts')
    points = []
    for index in range(7):
        pointer, kind = struct.unpack_from('<2I', data, offset + 0x168 + 8 * index)
        target = record_start + 0x70 + pointer
        expected = offset + 0x1a0 + 0x40 * index
        if kind != 3 or target != expected:
            raise ValueError('invalid 0xe/3528 straight-edge reference')
        values = float_view(data, target, 8)
        if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:6]) - 1) > 1e-6:
            raise ValueError('invalid 0xe/3528 edge parameters')
        points.append(values[:3])
        end = values[:3] + values[3:6] * values[7]
        if index < 6:
            next_values = struct.unpack_from('<3d', data, target + 0x40)
            if np.linalg.norm(end - next_values) > 1e-6:
                raise ValueError('0xe/3528 boundary chain is open')
        else:
            if np.linalg.norm(end - points[0]) > 1e-6:
                raise ValueError('0xe/3528 boundary does not close')
    points = np.asarray(points, dtype=float)
    normal = unit(struct.unpack_from('<3d', data, offset + 0x68))
    if np.max(np.abs((points - points[0]) @ normal)) > 1e-7:
        raise ValueError('0xe/3528 boundary is nonplanar')
    source_box = float_view(data, offset + 8, 6)
    world, error = _choose_placement(points, matrix, source_box)
    if error > 1e-5:
        raise ValueError('0xe/3528 boundary bounds mismatch')
    if matrix is not None:
        normal = unit(normal @ np.asarray(matrix).reshape(4, 4)[:3, :3])
    return world, normal


def _inline_cylinder_440_layout(data, offset, record_start):
    """Whether 0x17/440 stores two inline equal-radius circle boundaries."""
    if struct.unpack_from('<2I', data, offset) != (0x17, 440):
        return False
    for descriptor, target in ((0x100, 0x108), (0x158, 0x160)):
        pointer, kind = struct.unpack_from('<2I', data, offset + descriptor)
        if kind != 15 or record_start + 0x70 + pointer != offset + target:
            return False
    return True


def cylindrical_surface_440_from_vue(data, offset, record_start, matrix=None, *, with_normals=False):
    """Decode the inline-circle 0x17/440 cylindrical hole-wall variant."""
    if not _inline_cylinder_440_layout(data, offset, record_start):
        raise ValueError('unsupported inline-cylinder 0x17/440 layout')

    def circle(base):
        center = float_view(data, offset + base, 3)
        axis = unit(struct.unpack_from('<3d', data, offset + base + 0x18))
        radial = float_view(data, offset + base + 0x30, 3)
        ratio = struct.unpack_from('<d', data, offset + base + 0x48)[0]
        if (not np.isfinite(np.r_[center, axis, radial, ratio]).all() or
                np.linalg.norm(radial) <= 1e-10 or abs(axis @ radial) > 1e-7 or
                abs(ratio - 1.0) > 1e-7):
            raise ValueError('invalid inline-cylinder circle')
        return center, axis, radial

    first = circle(0x108)
    second = circle(0x160)
    center1, axis1, radial1 = first
    center2, axis2, radial2 = second
    radius1, radius2 = np.linalg.norm(radial1), np.linalg.norm(radial2)
    delta = center2 - center1
    if (abs(radius1 - radius2) > 1e-7 or
            np.linalg.norm(axis1 + axis2) > 1e-7 or
            abs(float(delta @ axis1)) <= 1e-10 or
            np.linalg.norm(delta - float(delta @ axis1) * axis1) > 1e-7):
        raise ValueError('inline-cylinder circles are not coaxial')
    matrix = np.eye(4) if matrix is None else np.asarray(matrix).reshape(4, 4)
    if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError('nonrigid inline-cylinder transform')
    centers = (np.c_[np.array([center1, center2]), np.ones(2)] @ matrix)[:, :3]
    axis = unit(axis1 @ matrix[:3, :3])
    radial = radial1 @ matrix[:3, :3]
    radius = np.linalg.norm(radial)
    radial /= radius
    tangent = unit(np.cross(axis, radial))
    angles = np.arange(96) * (2 * math.pi / 96)
    rings = [center + radius * (np.cos(angles)[:, None] * radial +
                                np.sin(angles)[:, None] * tangent)
             for center in centers]
    vertices = np.vstack(rings)
    faces = primitive_geometry.ring_faces(2,96).copy()
    # A hole wall's outward normal points toward the hole centre, opposite to
    # the radial direction of the solid slab.
    references = np.tile(centers.mean(0), (len(faces), 1)) - vertices[faces].mean(1)
    faces = _orient_triangles(vertices, faces, references)
    source_box = float_view(data, offset + 8, 6)
    if bounds_error(vertices, source_box) > 1e-5:
        raise ValueError('inline-cylinder bounds mismatch')
    if with_normals:
        normals=-np.vstack([ring-center for ring,center in zip(rings,centers)])
        return primitive_geometry.finish(vertices,faces,normals)
    return vertices, faces


def _segment_key(point):
    return tuple(np.round(np.asarray(point, dtype=float)[:2], 8))


def _quad_xy_segment(vertices):
    points = []
    for point in np.asarray(vertices, dtype=float):
        if not any(np.linalg.norm(point[:2] - previous[:2]) <= 1e-7
                   for previous in points):
            points.append(point)
    if len(points) != 2:
        raise ValueError('planar quad is not a vertical boundary wall')
    return np.asarray(points)[:, :2]


def _closed_segment_loops(segments):
    """Join exact XY wall segments into ordered polygon loops."""
    adjacency = {}
    unused = set()
    for index, (first, second) in enumerate(segments):
        a, b = _segment_key(first), _segment_key(second)
        if a == b:
            raise ValueError('zero-length floor hole segment')
        adjacency.setdefault(a, []).append((index, b))
        adjacency.setdefault(b, []).append((index, a))
        unused.add(index)
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        raise ValueError('floor hole wall graph is not a set of closed loops')
    loops = []
    while unused:
        first_index = min(unused)
        first, second = segments[first_index]
        start = _segment_key(first)
        current, previous = start, None
        loop = []
        while True:
            loop.append(current)
            candidates = [(edge, neighbour) for edge, neighbour in adjacency[current]
                          if edge in unused and neighbour != previous]
            if not candidates:
                raise ValueError('floor hole wall loop traversal failed')
            edge, following = candidates[0]
            unused.remove(edge)
            previous, current = current, following
            if current == start:
                break
        if len(loop) < 3:
            raise ValueError('floor hole loop has fewer than three edges')
        loops.append(np.asarray(loop, dtype=float))
    return loops


def floor_shell_context_from_vue(data, elements, record_start, matrix):
    """Build the observed slab cap/wall assembly for 0xe/3528 floors.

    This is deliberately strict: it activates only when two seven-edge cap
    boundaries, four inline circular walls, seven matching outer wall quads,
    and eight closed four-edge square wall loops are all present.  Other
    variable-length planar faces continue through the diagnostic placeholder
    path until their native layout is understood.
    """
    cap_elements = [e for e in elements if e.get('kind') == 'tag_block' and
                    (e['type_raw'], e['size_raw']) == (0x0e, 3528)]
    quad_elements = [e for e in elements if e.get('kind') == 'tag_block' and
                     (e['type_raw'], e['size_raw']) == (0x0e, 512)]
    circle_elements = [e for e in elements if e.get('kind') == 'tag_block' and
                       (e['type_raw'], e['size_raw']) == (0x17, 440)]
    if len(cap_elements) != 2 or len(circle_elements) != 4 or len(quad_elements) < 15:
        return None
    cap_boundaries = {}
    for element in cap_elements:
        offset = record_start + element['off_rel_record']
        cap_boundaries[element['off_rel_record']] = planar_boundary_3528_from_vue(
            data, offset, record_start, matrix)
    circle_meshes = {}
    for element in circle_elements:
        offset = record_start + element['off_rel_record']
        circle_meshes[element['off_rel_record']] = cylindrical_surface_440_from_vue(
            data, offset, record_start, matrix)

    outer = next(iter(cap_boundaries.values()))[0]
    outer_xy = outer[:, :2]
    outer_segments = [(outer_xy[i], outer_xy[(i + 1) % len(outer_xy)])
                      for i in range(len(outer_xy))]
    hole_segments = []
    outer_wall_count = 0
    for element in quad_elements:
        offset = record_start + element['off_rel_record']
        vertices, _ = planar_quad(data, offset, record_start, matrix)
        segment = _quad_xy_segment(vertices)
        is_outer = any(
            (np.linalg.norm(segment[0] - edge[0]) <= 1e-7 and
             np.linalg.norm(segment[1] - edge[1]) <= 1e-7) or
            (np.linalg.norm(segment[0] - edge[1]) <= 1e-7 and
             np.linalg.norm(segment[1] - edge[0]) <= 1e-7)
            for edge in outer_segments)
        if is_outer:
            outer_wall_count += 1
        else:
            hole_segments.append(tuple(segment))
    if outer_wall_count != 7:
        raise ValueError('0xe/3528 floor outer wall count mismatch')
    square_loops = _closed_segment_loops(hole_segments)
    if len(square_loops) != 8 or any(len(loop) != 4 for loop in square_loops):
        raise ValueError('0xe/3528 floor square-hole layout mismatch')

    normal = cap_boundaries[next(iter(cap_boundaries))][1]
    if abs(normal[2]) < 1.0 - 1e-7:
        raise ValueError('unsupported non-horizontal 0xe/3528 floor')
    square_xy = [loop[:, :2] for loop in square_loops]
    cap_meshes = {}
    for relative_offset, (boundary, cap_normal) in cap_boundaries.items():
        z = float(boundary[0, 2])
        loops = [boundary]
        loops.extend(np.c_[loop, np.full(len(loop), z)] for loop in square_xy)
        for wall_vertices, _ in circle_meshes.values():
            ring = wall_vertices[:96]
            if abs(float(ring[:, 2].mean()) - z) > 1e-6:
                ring = wall_vertices[96:]
            loops.append(ring)
        cap_meshes[relative_offset] = _triangulate_planar_loops(loops, cap_normal)
    return dict(cap_meshes=cap_meshes, circle_meshes=circle_meshes)


def _point_in_triangle(point, triangle, tolerance=1e-12):
    a, b, c = triangle
    cross2 = lambda first, second: first[0] * second[1] - first[1] * second[0]
    signs = np.array((cross2(b - a, point - a),
                      cross2(c - b, point - b),
                      cross2(a - c, point - c)))
    return bool(np.all(signs >= -tolerance) or np.all(signs <= tolerance))


def _ordered_ear_clip(points):
    """Retain the historical ear order, testing all remaining points in NumPy."""
    orientation = np.sign(_polygon_area(points))
    pending = np.arange(len(points))
    triangles = []
    while len(pending) > 3:
        for position, current in enumerate(pending):
            previous, following = pending[position-1], pending[(position+1) % len(pending)]
            a, b, c = points[[previous, current, following]]
            turn = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            if orientation * turn <= 1e-12:
                continue
            others = pending[(pending != previous) & (pending != current) & (pending != following)]
            samples = points[others]
            signs = np.column_stack(((b[0]-a[0])*(samples[:,1]-a[1])-(b[1]-a[1])*(samples[:,0]-a[0]),
                                     (c[0]-b[0])*(samples[:,1]-b[1])-(c[1]-b[1])*(samples[:,0]-b[0]),
                                     (a[0]-c[0])*(samples[:,1]-c[1])-(a[1]-c[1])*(samples[:,0]-c[0])))
            if np.any(np.all(signs >= -1e-12, axis=1) | np.all(signs <= 1e-12, axis=1)):
                continue
            triangles.append((previous, current, following))
            pending = np.delete(pending, position)
            break
        else:
            raise ValueError('self-intersecting or unsupported VUE profile')
    triangles.append(tuple(pending))
    return np.asarray(triangles, dtype=np.int64)


def _triangulate_simple_polygon(profile, preserve_order=False):
    """Triangulate a simple planar profile, preserving every boundary edge."""
    import mapbox_earcut
    profile = np.asarray(profile, dtype=float)
    if (profile.ndim != 2 or profile.shape[1] != 3 or len(profile) < 3 or
            not np.isfinite(profile).all()):
        raise ValueError('invalid VUE profile')
    relative = profile - profile[0]
    candidates = np.cross(relative[1:-1], relative[2:])
    usable = np.flatnonzero(np.linalg.norm(candidates, axis=1) > 1e-12)
    if not len(usable):
        raise ValueError('degenerate VUE profile')
    normal = candidates[usable[0]]
    drop = int(np.argmax(np.abs(normal)))
    points = np.ascontiguousarray(relative[:, [axis for axis in range(3) if axis != drop]])
    area = _polygon_area(points)
    if abs(area) <= 1e-12:
        raise ValueError('zero-area VUE profile')
    following = np.roll(points, -1, axis=0)
    edges = following - points
    if np.any(np.linalg.norm(edges, axis=1) <= 1e-12):
        raise ValueError('zero-length VUE profile edge')
    # Earcut is deliberately permissive about invalid input. Reject crossing
    # or touching non-neighbour edges before accepting its triangulation.
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for start in range(0, len(points), 128):
        ids = np.arange(start, min(start + 128, len(points)))
        delta = points[None, :, :] - points[ids, None, :]
        first = cross(edges[ids, None, :], delta)
        second = cross(edges[ids, None, :], following[None, :, :] - points[ids, None, :])
        third = cross(edges[None, :, :], -delta)
        fourth = cross(edges[None, :, :], following[ids, None, :] - points[None, :, :])
        overlap = np.all(np.maximum(np.minimum(points[ids, None], following[ids, None]),
                                   np.minimum(points[None, :], following[None, :])) <=
                         np.minimum(np.maximum(points[ids, None], following[ids, None]),
                                    np.maximum(points[None, :], following[None, :])) + 1e-12, axis=2)
        crosses = (((first >= -1e-12) & (second <= 1e-12)) |
                   ((second >= -1e-12) & (first <= 1e-12)))
        crosses &= (((third >= -1e-12) & (fourth <= 1e-12)) |
                    ((fourth >= -1e-12) & (third <= 1e-12)))
        others = np.arange(len(points))[None, :]
        nonadjacent = (others > ids[:, None] + 1) & ~((ids[:, None] == 0) & (others == len(points)-1))
        if np.any(crosses & overlap & nonadjacent):
            raise ValueError('self-intersecting or unsupported VUE profile')
    if len(points) <= 4 or preserve_order:
        # Cap triangulation affects area-weighted corner normals. Preserve
        # the old diagonals for quadrilaterals and sampled sweep end caps.
        faces = _ordered_ear_clip(points)
    else:
        faces = mapbox_earcut.triangulate_float64(points, np.array([len(points)], dtype=np.uint32)).reshape(-1, 3)
    if len(faces) != len(points) - 2:
        faces = _split_planar_edges_at_vertices(points, faces)
    faces = np.asarray(faces, dtype=np.int64).copy()
    signed = cross(points[faces[:, 1]] - points[faces[:, 0]],
                   points[faces[:, 2]] - points[faces[:, 0]])
    if (len(faces) != len(points) - 2 or np.any(abs(signed) <= 1e-16) or
            abs(abs(signed).sum()/2 - abs(area)) > max(1e-12, abs(area)*1e-8)):
        raise ValueError('invalid VUE profile triangulation')
    faces[signed * area < 0] = faces[signed * area < 0, ::-1]
    all_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(all_edges, axis=0, return_counts=True)
    expected = {tuple(sorted((i, (i+1) % len(points)))) for i in range(len(points))}
    if np.any(counts > 2) or {tuple(edge) for edge in unique[counts == 1]} != expected:
        raise ValueError('VUE profile boundary changed during triangulation')
    return faces


def polygon_extrusion_from_vue(data, offset, matrix=None, *, with_normals=False):
    """Decode the eight-edge stair-step profile in an observed 0x1f/720 block."""
    if struct.unpack_from('<2I', data, offset) != (0x1f, 720):
        raise ValueError('unsupported stair profile layout')
    if struct.unpack_from('<I', data, offset + 0x7c)[0] != 8:
        raise ValueError('unsupported stair profile edge count')
    profile = np.array([struct.unpack_from('<3d', data, offset + 0xc8 + 0x40 * i)
                        for i in range(8)], dtype=float)
    vector = float_view(data, offset + 0x50, 3)
    if not np.isfinite(np.r_[profile.ravel(), vector]).all() or np.linalg.norm(vector) <= 1e-10:
        raise ValueError('invalid stair profile payload')
    if np.linalg.norm(profile[0] - profile[-1]) <= 1e-10:
        profile = profile[:-1]
    if len(profile) < 3:
        raise ValueError('open or empty stair profile')
    normal = None
    for i in range(1, len(profile) - 1):
        candidate = np.cross(profile[i] - profile[0], profile[i + 1] - profile[0])
        if np.linalg.norm(candidate) > 1e-12:
            normal = unit(candidate)
            break
    if normal is None or abs(float(normal @ unit(vector))) < 1 - 1e-6:
        raise ValueError('stair profile is not perpendicular to extrusion')
    cap_faces = _triangulate_simple_polygon(profile)
    local_vertices = np.vstack((profile, profile + vector))
    n = len(profile)
    if not with_normals:
        sides=primitive_geometry.ring_faces(2,n)
        faces=np.empty((2*len(cap_faces)+len(sides),3),dtype=np.int64)
        faces[:len(cap_faces)]=cap_faces
        faces[len(cap_faces):2*len(cap_faces)]=cap_faces+n
        faces[2*len(cap_faces):]=sides
    candidates = [local_vertices]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        if not np.array_equal(m,np.eye(4)):
            candidates.append(local_vertices @ m[:3,:3] + m[3,:3])
    source_box = float_view(data, offset + 8, 6)
    vertices = min(candidates, key=lambda candidate: bounds_error(candidate, source_box))
    if bounds_error(vertices, source_box) > 0.005:
        raise ValueError('stair profile bounds mismatch')
    if with_normals:
        return primitive_geometry.prism(vertices[:n],vertices[n:],cap_faces)
    return _orient_generated_surface(vertices, faces)


def cage_hoop_from_vue(data, offset, size, record_start, matrix=None):
    """Decode the observed 0x1f/2192 ladder cage-hoop band from VUE.

    The payload stores one closed chain of native straight/arc profile
    entities plus an extrusion vector at +0x50; sweeping the flat-bar
    outline along that vector yields the vertical cage band with both
    horizontal faces closed.  The VUE bbox pads the band by the placement
    margin of the source bar, so the tolerance follows the existing
    0.1×payload-size rule used for native cylinders.
    """
    if struct.unpack_from('<2I', data, offset) != (0x1f, 2192):
        raise ValueError('unsupported 0x1f/2192 cage hoop layout')
    vector = _profile_vector(data, offset + 0x50)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-10:
        raise ValueError('invalid cage hoop extrusion vector')
    chains = _profile_chain_entities(data, offset, size, record_start, 0x70)
    if len(chains) != 1:
        raise ValueError('unsupported cage hoop chain layout')
    outline, closed = _profile_loop(data, chains[0])
    if not closed:
        raise ValueError('cage hoop outline is not closed')
    cap_faces = _triangulate_simple_polygon(outline)
    local_vertices = np.vstack((outline, outline + vector))
    n = len(outline)
    faces = [tuple(int(index) for index in face) for face in cap_faces]
    faces.extend(tuple(int(index) + n for index in face) for face in cap_faces)
    faces.extend((i, (i + 1) % n, (i + 1) % n + n) for i in range(n))
    faces.extend((i, (i + 1) % n + n, i + n) for i in range(n))
    faces = np.asarray(faces, dtype=np.int64)
    candidates = [local_vertices]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        if not np.array_equal(m,np.eye(4)):
            candidates.append(local_vertices @ m[:3,:3] + m[3,:3])
    source_box = float_view(data, offset + 8, 6)
    tolerance = max(0.005, 0.1 * float(np.linalg.norm(vector)))
    vertices = min(candidates, key=lambda candidate: bounds_error(candidate, source_box))
    if bounds_error(vertices, source_box) > tolerance:
        raise ValueError('cage hoop bounds mismatch')
    return _orient_generated_surface(vertices, faces)


def annulus_from_vue(data, offset, matrix=None):
    """Decode the 0x0e/408 annular tube-end surface from its two VUE circles."""
    if struct.unpack_from('<2I', data, offset) != (0x0e, 408):
        raise ValueError('unsupported annulus layout')
    center = float_view(data, offset + 0xe8, 3)
    axis = float_view(data, offset + 0x100, 3)
    outer_radial = float_view(data, offset + 0x118, 3)
    outer_ratio = struct.unpack_from('<d', data, offset + 0x130)[0]
    center2 = float_view(data, offset + 0x140, 3)
    axis2 = float_view(data, offset + 0x158, 3)
    inner_radial = float_view(data, offset + 0x170, 3)
    inner_ratio = struct.unpack_from('<d', data, offset + 0x188)[0]
    values = np.r_[center, axis, outer_radial, outer_ratio, center2, axis2,
                   inner_radial, inner_ratio]
    if not np.isfinite(values).all():
        raise ValueError('nonfinite annulus payload')
    axis = unit(axis)
    axis2 = unit(axis2)
    outer_radius = np.linalg.norm(outer_radial)
    inner_radius = np.linalg.norm(inner_radial)
    if (outer_radius <= inner_radius or inner_radius <= 1e-10 or
            not (0 < outer_ratio <= 1 and 0 < inner_ratio <= 1) or
            np.linalg.norm(center2 - center) > 1e-7 or
            np.linalg.norm(axis + axis2) > 1e-7 or
            abs(axis @ outer_radial) > 1e-7 or abs(axis @ inner_radial) > 1e-7):
        raise ValueError('invalid annulus circles')
    radial = outer_radial / outer_radius
    tangent = unit(np.cross(axis, radial))
    angles = np.arange(96) * (2 * math.pi / 96)
    outer = center + outer_radius * np.cos(angles)[:, None] * radial + \
        outer_radius * outer_ratio * np.sin(angles)[:, None] * tangent
    inner = center + inner_radius * np.cos(angles)[:, None] * (inner_radial / inner_radius) + \
        inner_radius * inner_ratio * np.sin(angles)[:, None] * unit(np.cross(axis, inner_radial))
    vertices = np.vstack((outer, inner))
    faces = primitive_geometry.ring_faces(2,96).copy()
    source_box = float_view(data, offset + 8, 6)
    candidates = [vertices]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        candidates.append((np.c_[vertices, np.ones(len(vertices))] @ m)[:, :3])
    vertices = min(candidates, key=lambda candidate: bounds_error(candidate, source_box))
    # A few flat end-ring boxes omit the positive side of the outer ellipse
    # and are short by 3.9116 mm; the VUE ellipse fields remain self-consistent
    # and the resulting surface stays within the owning record bounds.
    cardinal = np.array([center + sign * vector for vector in
                         (outer_radial, np.cross(axis, outer_radial) * outer_ratio)
                         for sign in (-1, 1)])
    cardinal_error = min(bounds_error(v, source_box)
                         for v in _world_candidates(cardinal, matrix))
    if bounds_error(vertices, source_box) > 0.005 and cardinal_error > 1e-7:
        raise ValueError('annulus bounds mismatch')
    return _orient_generated_surface(vertices, faces)


def boundary_reference_from_vue(data, offset, record_start, matrix=None):
    """Read a 0x0e/584 closed VUE loop already represented by a surface patch."""
    if struct.unpack_from('<2I', data, offset) != (0x0e, 584):
        raise ValueError('unsupported boundary-reference layout')
    points = np.array([struct.unpack_from('<3d', data, offset + 0x110 + 24 * i)
                       for i in range(9)], dtype=float)
    if not np.isfinite(points).all() or np.linalg.norm(points[0] - points[-1]) > 1e-8:
        raise ValueError('open VUE boundary reference')
    edge = np.diff(points, axis=0)
    normal = None
    for first in edge:
        for second in edge:
            candidate = np.cross(first, second)
            if np.linalg.norm(candidate) > 1e-12:
                normal = unit(candidate)
                break
        if normal is not None:
            break
    if normal is None or np.max(np.abs((points - points[0]) @ normal)) > 1e-7:
        raise ValueError('non-planar VUE boundary reference')
    candidates = [points]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        candidates.append((np.c_[points, np.ones(len(points))] @ m)[:, :3])
    record_box = float_view(data, record_start + 8, 6)
    valid = [candidate for candidate in candidates
             if np.all(candidate.min(0) >= record_box[:3] - 1e-5) and
             np.all(candidate.max(0) <= record_box[3:] + 1e-5)]
    if not valid:
        raise ValueError('VUE boundary reference outside owning record')
    return min(valid, key=lambda candidate: bounds_error(candidate, record_box))


def planar_quad(data, offset, record_start, root_matrix):
    """Observed 0xE/512: one bounded loop of four straight, ordered edges."""
    if offset < 0 or offset+512 > len(data):
        raise ValueError('truncated planar face')
    u = lambda rel: struct.unpack_from('<I', data, offset+rel)[0]
    if (u(0),u(4),u(0x9c),u(0xcc)) != (14,512,1,4):
        raise ValueError('unsupported face loop layout')
    points, directions, lengths = [], [], []
    for i in range(4):
        pointer, kind = struct.unpack_from('<2I',data,offset+0xd8+8*i)
        edge = offset+0xf8+64*i
        if record_start+0x70+pointer != edge or kind != 3:
            raise ValueError('unsupported edge reference or curve type')
        values = float_view(data, edge, 8)
        if not np.isfinite(values).all():
            raise ValueError('invalid straight edge parameters')
        # VUE has two native encodings for a straight curve length.  The
        # usual one stores ``(auxiliary=0, length>0)``; a compact face
        # variant stores ``(auxiliary=-length, length=0)``.  Both fields are
        # part of the curve record, so accept the latter only when the first
        # encoding is unambiguous.
        if values[7] > 1e-10 and abs(values[6]) <= 1e-10:
            length = values[7]
            point = values[:3]
        elif values[7] <= 1e-10 and values[6] < -1e-10:
            length = -values[6]
            # In this encoding the stored point is the end of the line at
            # parameter 0; the ordered-loop decoder needs its other end as
            # the edge start.
            point = values[:3] + values[3:6] * values[6]
        else:
            raise ValueError('invalid straight edge parameters')
        points.append(point); directions.append(values[3:6]); lengths.append(length)
    v = np.array(points); directions=np.array(directions); lengths=np.array(lengths)
    if np.max(np.abs(np.linalg.norm(directions,axis=1)-1))>1e-6:
        raise ValueError('nonunit edge directions')
    if np.max(np.abs(v+directions*lengths[:,None]-np.roll(v,-1,axis=0)))>1e-6:
        raise ValueError('edge loop does not close')
    normal=unit(struct.unpack_from('<3d',data,offset+0x68))
    if np.max(np.abs((v-v[0])@normal))>1e-6:
        raise ValueError('nonplanar boundary')
    turns=np.array([np.dot(np.cross(v[(i+1)%4]-v[i],v[(i+2)%4]-v[(i+1)%4]),normal) for i in range(4)])
    if not (np.all(turns>1e-12) or np.all(turns< -1e-12)):
        raise ValueError('nonconvex or degenerate quadrilateral')
    f=np.array([(0,1,2),(0,2,3)],dtype=np.int64)
    if turns[0]<0:f=f[:,::-1]
    source_box=float_view(data, offset+8, 6)
    candidates=[v,(np.c_[v,np.ones(4)]@root_matrix)[:,:3]]
    chosen=min(candidates,key=lambda points:bounds_error(points,source_box))
    if bounds_error(chosen,source_box)>1e-5:
        # Some stored face boxes omit an extreme corner of an inclined face.
        # Keep the closed, planar boundary when it lies in its owning record.
        record_box=float_view(data, record_start+8, 6)
        if np.any(chosen.min(0)<record_box[:3]-1e-5) or np.any(chosen.max(0)>record_box[3:]+1e-5):
            raise ValueError('face placement outside owning record bounds')
    return chosen,f


def planar_face_456_from_vue(data, offset, record_start, root_matrix):
    """Decode the compact 0x0e/456 planar quadrilateral face.

    This face stores an ordered five-point polyline (the fifth point repeats
    the first) inline at ``+0x110``.  It is the same native planar boundary
    concept as 0x0e/368, but does not use the older boundary-pointer layout.
    """
    if offset < 0 or offset + 456 > len(data):
        raise ValueError('truncated compact planar face')
    if struct.unpack_from('<2I', data, offset) != (0x0e, 456):
        raise ValueError('unsupported compact planar face')
    if (struct.unpack_from('<2I', data, offset + 0xf8) != (5, 7) or
            struct.unpack_from('<I', data, offset + 0x108)[0] != 66050):
        raise ValueError('unsupported compact planar face header')
    points = np.array([
        struct.unpack_from('<3d', data, offset + 0x110 + 24 * i)
        for i in range(5)
    ], dtype=float)
    normal = unit(struct.unpack_from('<3d', data, offset + 0x68))
    if (not np.isfinite(np.r_[points.ravel(), normal]).all() or
            np.linalg.norm(points[0] - points[-1]) > 1e-8):
        raise ValueError('invalid compact planar face loop')
    polygon = points[:4]
    if np.linalg.norm(np.cross(polygon[1] - polygon[0],
                               polygon[2] - polygon[0])) <= 1e-12:
        raise ValueError('degenerate compact planar face')
    if np.max(np.abs((polygon - polygon[0]) @ normal)) > 1e-7:
        raise ValueError('nonplanar compact planar face')
    turns = np.array([
        np.cross(polygon[(i + 1) % 4] - polygon[i],
                 polygon[(i + 2) % 4] - polygon[(i + 1) % 4]) @ normal
        for i in range(4)
    ])
    if not (np.all(turns > 1e-12) or np.all(turns < -1e-12)):
        raise ValueError('nonconvex or degenerate compact planar face')

    source_box = float_view(data, offset + 8, 6)
    candidates = [polygon]
    m = np.asarray(root_matrix).reshape(4, 4)
    candidates.append((np.c_[polygon, np.ones(4)] @ m)[:, :3])
    chosen = min(candidates, key=lambda candidate: bounds_error(candidate,
                                                                 source_box))
    if bounds_error(chosen, source_box) > 1e-5:
        record_box = float_view(data, record_start + 8, 6)
        if (np.any(chosen.min(0) < record_box[:3] - 1e-5) or
                np.any(chosen.max(0) > record_box[3:] + 1e-5)):
            raise ValueError('compact face placement outside owning record')
    chosen_normal = normal
    if chosen is not polygon:
        chosen_normal = unit(normal @ m[:3, :3])
    f = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    if np.cross(chosen[1] - chosen[0], chosen[2] - chosen[0]) @ chosen_normal < 0:
        f = f[:, ::-1]
    return chosen, f


@lru_cache(maxsize=1)
def _unit_sphere_template():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    vertices.setflags(write=False); faces.setflags(write=False)
    return vertices, faces


def sphere_from_vue(data, offset, matrix, *, with_normals=False):
    """Decode the untrimmed 0x1d/224 sphere: centre, frame and radius.

    Its cached box uses the two poles and eight equator samples, not the
    analytic world extrema of a rotated sphere. The reference surfaces
    confirm a full sphere, including all eight octants.
    """
    if struct.unpack_from('<2I', data, offset) != (29, 224):
        raise ValueError('unsupported sphere layout')
    center = _profile_vector(data, offset + 0x50)
    axis = _profile_vector(data, offset + 0x68)
    radial = _profile_vector(data, offset + 0x80)
    radius = struct.unpack_from('<d', data, offset + 0x98)[0]
    if (not np.isfinite(np.r_[center, axis, radial, radius]).all() or
            not 0 < radius < 1e5 or
            abs(np.linalg.norm(axis) - 1) > 1e-6 or
            abs(np.linalg.norm(radial) - 1) > 1e-6 or
            abs(axis @ radial) > 1e-6):
        raise ValueError('invalid sphere frame or radius')
    if (struct.unpack_from('<d', data, offset + 0xa0)[0] != 0 or
            struct.unpack_from('<I', data, offset + 0xa8)[0] != 256 or
            any(data[offset + 0xac:offset + 216])):
        raise ValueError('unsupported trimmed sphere')
    basis = np.vstack((radial, np.cross(axis, radial), axis))
    angles = np.arange(8) * (np.pi / 4)
    poles = center + radius * np.vstack((axis, -axis,
        np.cos(angles)[:, None] * basis[0] + np.sin(angles)[:, None] * basis[1]))
    source_box = float_view(data, offset + 8, 6)
    transforms = [np.eye(4), np.asarray(matrix).reshape(4, 4)]
    def sphere_cache_error(m):
        pole_error = bounds_error((np.c_[poles, np.ones(len(poles))] @ m)[:, :3], source_box)
        wc = (np.r_[center, 1] @ m)[:3]
        half = radius * np.linalg.norm(m[:3, :3], axis=0)
        return min(pole_error, float(np.max(abs(np.r_[wc-half, wc+half]-source_box))))
    chosen = min(transforms, key=sphere_cache_error)
    if sphere_cache_error(chosen) > 1e-5:
        raise ValueError('sphere native pole bounds mismatch')
    unit_vertices, faces = _unit_sphere_template()
    radial = unit_vertices @ basis
    local = radius * radial + center
    if with_normals:
        return primitive_geometry.finish(local, faces, radial, chosen)
    world = (np.c_[local, np.ones(len(local))] @ chosen)[:, :3]
    return world, faces.copy()


def torus_from_vue(data, offset, matrix, segments=None, section_segments=None, *, with_normals=False):
    """Observed full circular torus, 0x1E/232; all parameters are native.

    ``segments`` sets both the major-ring and tube cross-section density (P1
    plan): pass the shared export value.  ``None`` keeps the historical 96x24
    direct-call sampling.
    """
    if struct.unpack_from('<2I', data, offset) != (0x1e,232):
        raise ValueError('unsupported torus layout')
    center=float_view(data, offset+0x50, 3)
    axis=float_view(data, offset+0x68, 3)
    ref=float_view(data, offset+0x80, 3)
    radius,tube=struct.unpack_from('<2d',data,offset+0x98)
    if not np.isfinite(np.r_[center,axis,ref,radius,tube]).all() or not 0<tube<radius:
        raise ValueError('invalid torus parameters')
    if abs(np.linalg.norm(axis)-1)>1e-6 or abs(np.linalg.norm(ref)-1)>1e-6 or abs(axis@ref)>1e-6:
        raise ValueError('invalid torus frame')
    if data[offset+0xa8:offset+0xe0] != struct.pack('<dI',0.,256)+bytes(44):
        raise ValueError('unsupported trimmed torus')
    matrix=np.asarray(matrix).reshape(4,4)
    if not np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6):
        raise ValueError('unsupported nonrigid torus transform')
    world_center=(np.r_[center,1]@matrix)[:3]; world_axis=axis@matrix[:3,:3]
    half=radius*np.sqrt(np.maximum(0,1-world_axis**2))+tube
    box=float_view(data, offset+8, 6)
    if np.max(abs(np.r_[world_center-half,world_center+half]-box))>1e-5:
        raise ValueError('torus analytic bounds mismatch')
    ring_segments = validate_segments(segments) if segments is not None else LEGACY_TORUS_RING_SEGMENTS
    if section_segments is None:
        section_segments = (LEGACY_TORUS_SECTION_SEGMENTS if segments is None
                            else ring_segments)
    else:
        section_segments = validate_segments(section_segments)
    a=np.arange(ring_segments)*2*np.pi/ring_segments
    b=np.arange(section_segments)*2*np.pi/section_segments
    radial=np.cos(a)[:,None]*ref+np.sin(a)[:,None]*np.cross(axis,ref)
    v=(center+(radius+tube*np.cos(b))[None,:,None]*radial[:,None,:]+tube*np.sin(b)[None,:,None]*axis).reshape(-1,3)
    normals = (np.cos(b)[None,:,None]*radial[:,None,:]
               + np.sin(b)[None,:,None]*axis).reshape(-1,3)
    f = primitive_geometry.ring_faces(ring_segments,section_segments,True)
    if with_normals:
        return primitive_geometry.finish(v,f,normals,matrix)
    v=(np.c_[v,np.ones(len(v))]@matrix)[:,:3]
    return _orient_generated_solid(v, f)


def cone_from_vue(data, offset, record_start, matrix, segments=None, *, with_normals=False):
    """Observed 0x17/424 cone surface with a circular base and point apex.

    ``segments`` sets the circumferential density (P1 plan); ``None`` keeps the
    historical 96-segment direct-call sampling.
    """
    if struct.unpack_from('<2I',data,offset)!=(0x17,424):
        raise ValueError('unsupported cone layout')
    if (struct.unpack_from('<I',data,offset+0xb4)[0] != 2 or
            struct.unpack_from('<I',data,offset+0xec)[0] != 1 or
            abs(struct.unpack_from('<d',data,offset+0x98)[0]-1)>1e-8):
        raise ValueError('unsupported cone loop or noncircular base')
    # Boundary curve references resolve to the repeated native circle and line.
    for descriptor,target,kind in [(0x100,0x108,15),(0x158,0x160,3)]:
        pointer,curve_kind=struct.unpack_from('<2I',data,offset+descriptor)
        if record_start+0x70+pointer!=offset+target or curve_kind!=kind:
            raise ValueError('unsupported cone boundary references')
    base=float_view(data, offset+0x50, 3)
    axis=float_view(data, offset+0x68, 3)
    ref=float_view(data, offset+0x80, 3)
    apex=float_view(data, offset+0x160, 3)
    sine,cosine=struct.unpack_from('<2d',data,offset+0xa0)
    radius=np.linalg.norm(ref);height=float((apex-base)@axis)
    values=np.r_[base,axis,ref,apex,sine,cosine]
    if not np.isfinite(values).all() or radius<=0 or height<=0:
        raise ValueError('invalid cone parameters')
    if abs(np.linalg.norm(axis)-1)>1e-6 or abs(ref@axis)>1e-8 or np.linalg.norm(apex-base-height*axis)>1e-7:
        raise ValueError('invalid cone axis or apex')
    if cosine<=0 or sine>=0 or abs(sine*sine+cosine*cosine-1)>1e-6 or abs(radius+height*sine/cosine)>1e-7:
        raise ValueError('cone taper does not reach native apex; possible frustum')
    if data[offset+0x50:offset+0xa0]!=data[offset+0x108:offset+0x158]:
        raise ValueError('cone circular boundary disagrees with surface')
    matrix=np.asarray(matrix).reshape(4,4)
    if not np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6):
        raise ValueError('unsupported nonrigid cone transform')
    world=(np.c_[np.array([base,apex]),np.ones(2)]@matrix)[:,:3]
    wa=axis@matrix[:3,:3];half=radius*np.sqrt(np.maximum(0,1-wa*wa))
    expected=np.r_[np.minimum(world[0]-half,world[1]),np.maximum(world[0]+half,world[1])]
    source=float_view(data, offset+8, 6)
    # Some caches use only the four native meridians, before rotation.
    wu=ref@matrix[:3,:3];wv=np.cross(axis,ref)@matrix[:3,:3]
    cardinal_half=np.maximum(abs(wu),abs(wv))
    cardinal=np.r_[np.minimum(world[0]-cardinal_half,world[1]),
                   np.maximum(world[0]+cardinal_half,world[1])]
    if min(np.max(abs(expected-source)),np.max(abs(cardinal-source)))>1e-5:
        raise ValueError('cone analytic bounds mismatch')
    count = validate_segments(segments) if segments is not None else LEGACY_CONE_SEGMENTS
    angles=np.arange(count)*2*np.pi/count
    ring=base+np.cos(angles)[:,None]*ref+np.sin(angles)[:,None]*np.cross(axis,ref)
    if with_normals:
        radial=(ring-base)/radius
        vertices=np.empty((2*count,3));vertices[:count]=ring;vertices[count:]=apex
        normals=np.empty_like(vertices)
        normals[:count]=radial+(radius/height)*axis
        middle=(angles+np.pi/count)
        normals[count:]=(np.cos(middle)[:,None]*ref+np.sin(middle)[:,None]*np.cross(axis,ref))/radius+(radius/height)*axis
        i=np.arange(count)
        faces=np.column_stack((i,(i+1)%count,count+i))
        return primitive_geometry.finish(vertices,faces,normals,matrix)
    v=(np.c_[np.vstack([ring,apex]),np.ones(count+1)]@matrix)[:,:3]
    f=np.column_stack((np.arange(count),(np.arange(count)+1)%count,np.full(count,count)))
    # The bounded payload is a conical surface; do not invent a base disk.
    return v,f


def _rational_closed_curve_from_vue(data, offset, base, control_count=13,
                                    knot_count=16):
    """Evaluate one closed rational curve embedded in 0x17/1464.

    The curve header stores the control-point and knot-array pointers, but the
    arrays themselves are inline in the tag block.  The caller validates the
    pointers against the enclosing record before using this helper.
    """
    point_end = base + 24 * control_count
    weight_base = point_end
    knot_base = weight_base + 8 * control_count
    points = np.frombuffer(data, dtype='<f8', count=3*control_count,
                           offset=offset + base).reshape(-1, 3)
    weights = np.frombuffer(data, dtype='<f8', count=control_count, offset=offset + weight_base)
    knots = np.frombuffer(data, dtype='<f8', count=knot_count, offset=offset + knot_base)
    if (not np.isfinite(np.r_[points.ravel(), weights, knots]).all() or
            np.any(weights <= 0) or np.any(np.diff(knots) < -1e-12)):
        raise ValueError('invalid 0x17/1464 rational boundary curve')
    degree = knot_count - control_count - 1
    if not (1 <= degree < control_count):
        raise ValueError('unsupported 0x17/1464 boundary curve degree')
    parameters = _surface_parameters(knots, 12)
    rational_basis = _bspline_basis_matrix(parameters, degree, knots, control_count) * weights
    denominator = rational_basis.sum(axis=1)
    if np.any(denominator <= 1e-15):
        raise ValueError('zero 0x17/1464 curve basis denominator')
    samples = (rational_basis[:, :, None] * points).sum(axis=1) / denominator[:, None]
    if np.linalg.norm(samples[0] - samples[-1]) > 1e-7:
        raise ValueError('0x17/1464 boundary curve is not closed')
    return samples


def cylinder_surface_from_vue(data, offset, record_start, matrix=None, *, with_normals=False):
    """Decode the observed 0x17/1464 bounded cylindrical surface.

    This variant carries two inline degree-2 rational closed curves.  They
    are oblique circular sections of the cylinder: after reversing the second
    curve's parameter direction, corresponding points share the same radial
    projection and differ only along the native axis.  The block's six-value
    cache is a narrow local/under-approximate box for this variant, so the
    recovered bounds are intentionally recomputed from the VUE curves.
    """
    if struct.unpack_from('<2I', data, offset) != (0x17, 1464):
        raise ValueError('unsupported 0x17/1464 layout')
    if (struct.unpack_from('<3I', data, offset + 0x120) != (13, 16, 1) or
            struct.unpack_from('<I', data, offset + 0x130)[0] not in (66051, 66563) or
            struct.unpack_from('<3I', data, offset + 0x378) != (13, 16, 1) or
            struct.unpack_from('<I', data, offset + 0x388)[0] not in (66051, 66563)):
        raise ValueError('unsupported 0x17/1464 curve headers')
    # Resolve both curve descriptors through the record-relative pointers.
    # These checks bind the arrays to this block and avoid treating arbitrary
    # doubles in an unknown 0x17 payload as geometry.
    pointer_checks = [
        (0x100, 0x108), (0x108, 0x138), (0x110, 0x2d8),
        (0x358, 0x360), (0x360, 0x390), (0x368, 0x530),
    ]
    for descriptor, target in pointer_checks:
        pointer = struct.unpack_from('<I', data, offset + descriptor)[0]
        if record_start + 0x70 + pointer != offset + target:
            raise ValueError('invalid 0x17/1464 curve pointer')
    first = _rational_closed_curve_from_vue(data, offset, 0x138)
    second = _rational_closed_curve_from_vue(data, offset, 0x390)
    center = float_view(data, offset + 0x50, 3)
    axis = unit(struct.unpack_from('<3d', data, offset + 0x68))
    radial_reference = np.asarray(
        struct.unpack_from('<3d', data, offset + 0x80), dtype=float)
    radius = float(np.linalg.norm(radial_reference))
    if (not np.isfinite(np.r_[center, axis, radial_reference]).all() or
            radius <= 1e-12 or abs(np.linalg.norm(axis) - 1) > 1e-7 or
            abs(axis @ radial_reference) > 1e-7):
        raise ValueError('invalid 0x17/1464 cylinder frame')

    first_open, second_open = first[:-1], second[:-1]
    first_radial = ((first_open - center) -
                    ((first_open - center) @ axis)[:, None] * axis)
    same_radial = ((second_open - center) -
                   ((second_open - center) @ axis)[:, None] * axis)
    # Keep the sampled endpoint at parameter 1.0 when reversing, then drop
    # the duplicate endpoint.  Reversing ``second[:-1]`` would shift the
    # closed curve by one sample and falsely break radial correspondence.
    reverse_open = second[::-1][:-1]
    reverse_radial = ((reverse_open - center) -
                      ((reverse_open - center) @ axis)[:, None] * axis)
    same_error = float(np.max(np.linalg.norm(same_radial - first_radial, axis=1)))
    reverse_error = float(np.max(np.linalg.norm(reverse_radial - first_radial, axis=1)))
    second_open = reverse_open if reverse_error < same_error else second_open
    second_radial = reverse_radial if reverse_error < same_error else same_radial
    if reverse_error >= same_error or reverse_error > 1e-6:
        raise ValueError('0x17/1464 boundaries do not share radial parameterization')
    if (np.max(np.abs(np.linalg.norm(first_radial, axis=1) - radius)) > 1e-6 or
            np.max(np.abs(np.linalg.norm(second_radial, axis=1) - radius)) > 1e-6):
        raise ValueError('0x17/1464 boundary radius disagrees with header')
    displacement = second_open - first_open
    axial_displacement = displacement @ axis
    perpendicular_displacement = displacement - axial_displacement[:, None] * axis
    if (np.max(np.linalg.norm(perpendicular_displacement, axis=1)) > 1e-6 or
            np.min(np.abs(axial_displacement)) <= 1e-8 or
            np.any(axial_displacement * axial_displacement[0] <= 0)):
        raise ValueError('0x17/1464 boundaries are not coaxial')

    native = np.vstack((first_open, second_open))
    count = len(first_open)
    faces = primitive_geometry.ring_faces(2,count)
    m = np.eye(4) if matrix is None else np.asarray(matrix).reshape(4, 4)
    # Current 15300 instances store this inline B-rep in world coordinates;
    # a non-identity root transform is not silently applied twice.
    if not np.allclose(m, np.eye(4), atol=1e-10):
        raise ValueError('0x17/1464 inline curves require an identity root transform')
    if with_normals:
        return primitive_geometry.finish(native,faces,np.vstack((first_radial,second_radial)))
    return _orient_generated_surface(native, faces)


def frustum_from_vue(data, offset, record_start, matrix, *, with_normals=False):
    """0x17/440: conical surface bounded by two coaxial circles."""
    if struct.unpack_from('<2I',data,offset)!=(0x17,440):
        raise ValueError('unsupported frustum layout')
    if struct.unpack_from('<I',data,offset+0xb4)[0]!=2 or struct.unpack_from('<I',data,offset+0xec)[0]!=1:
        raise ValueError('unsupported frustum boundary loops')
    for descriptor,target in [(0x100,0x108),(0x158,0x160)]:
        pointer,kind=struct.unpack_from('<2I',data,offset+descriptor)
        if kind!=15 or record_start+0x70+pointer!=offset+target:
            raise ValueError('frustum needs two native circular boundaries')
    c1=float_view(data, offset+0x50, 3);a1=float_view(data, offset+0x68, 3)
    u1=float_view(data, offset+0x80, 3)
    c2=float_view(data, offset+0x160, 3);a2=float_view(data, offset+0x178, 3)
    u2=float_view(data, offset+0x190, 3)
    sine,cosine=struct.unpack_from('<2d',data,offset+0xa0)
    if not np.isfinite(np.r_[c1,a1,u1,c2,a2,u2,sine,cosine]).all():raise ValueError('nonfinite frustum')
    r1=np.linalg.norm(u1);r2=np.linalg.norm(u2);height=float((c2-c1)@a1)
    if min(r1,r2,height)<=1e-10 or cosine<=0 or abs(sine*sine+cosine*cosine-1)>1e-6:
        raise ValueError('invalid frustum radius or taper')
    if np.linalg.norm(a1+a2)>1e-7 or abs(np.linalg.norm(a1)-1)>1e-7 or abs(a1@u1)>1e-8 or abs(a2@u2)>1e-8 or np.linalg.norm(c2-c1-height*a1)>1e-7:
        raise ValueError('noncoaxial frustum boundaries')
    if abs(r1+height*sine/cosine-r2)>1e-7:
        raise ValueError('frustum taper disagrees with second radius')
    if data[offset+0x50:offset+0xa0]!=data[offset+0x108:offset+0x158] or any(abs(struct.unpack_from('<d',data,offset+k)[0]-1)>1e-8 for k in [0x98,0x1a8]):
        raise ValueError('noncircular or inconsistent frustum boundary')
    matrix=np.asarray(matrix).reshape(4,4)
    if not np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6):raise ValueError('nonrigid frustum transform')
    centers=(np.c_[np.array([c1,c2]),np.ones(2)]@matrix)[:,:3];axis=a1@matrix[:3,:3]
    half=np.sqrt(np.maximum(0,1-axis*axis))[None,:]*np.array([r1,r2])[:,None]
    expected=np.r_[(centers-half).min(0),(centers+half).max(0)]
    radial_world = np.array([u1, u2]) @ matrix[:3,:3]
    tangent_world = np.array([np.cross(a1,u1),np.cross(a2,u2)]) @ matrix[:3,:3]
    extent = np.maximum(abs(radial_world),abs(tangent_world))
    cardinal = np.r_[(centers-extent).min(0),(centers+extent).max(0)]
    source_box = float_view(data, offset+8, 6)
    if min(np.max(abs(expected-source_box)),np.max(abs(cardinal-source_box)))>1e-5:
        raise ValueError('frustum analytic bounds mismatch')
    angle=np.arange(96)*2*np.pi/96;u=u1/r1;w=np.cross(a1,u)
    radial=np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*w
    local=np.vstack([c1+r1*radial,c2+r2*radial])
    f=primitive_geometry.ring_faces(2,96)
    if with_normals:
        normals=np.tile(radial-((r2-r1)/height)*a1,(2,1))
        return primitive_geometry.finish(local,f,normals,matrix)
    v=(np.c_[local,np.ones(192)]@matrix)[:,:3]
    return v,f.copy()


def native_circle(data,offset,matrix):
    center=float_view(data, offset, 3);axis=float_view(data, offset+24, 3)
    radial=float_view(data, offset+48, 3);ratio=struct.unpack_from('<d',data,offset+72)[0]
    if not np.isfinite(np.r_[center,axis,radial,ratio]).all() or abs(np.linalg.norm(axis)-1)>1e-7 or np.linalg.norm(radial)<=1e-10 or abs(axis@radial)>1e-8 or abs(ratio-1)>1e-8:
        raise ValueError('unsupported circular boundary')
    m=np.asarray(matrix).reshape(4,4)
    if not np.allclose(m[:3,:3]@m[:3,:3].T,np.eye(3),atol=1e-6):raise ValueError('nonrigid boundary transform')
    return (np.r_[center,1]@m)[:3],axis@m[:3,:3],radial@m[:3,:3]


def ruled_circles_from_vue(data,offset,record_start,matrix, *, with_normals=False):
    if struct.unpack_from('<2I',data,offset)!=(0x21,280):raise ValueError('unsupported ruled-circle layout')
    for descriptor,target in [(0x60,0x68),(0xb8,0xc0)]:
        ptr,kind=struct.unpack_from('<2I',data,offset+descriptor)
        if kind!=15 or record_start+0x70+ptr!=offset+target:raise ValueError('invalid ruled circle reference')
    c1,a1,u1=native_circle(data,offset+0x68,matrix);c2,a2,u2=native_circle(data,offset+0xc0,matrix)
    r1=np.linalg.norm(u1);r2=np.linalg.norm(u2)
    if np.linalg.norm(a1-a2)>1e-7 or np.linalg.norm(u1/r1-u2/r2)>1e-7 or abs((c2-c1)@a1)<1e-8:
        raise ValueError('unsupported ruled-circle frame')
    half=np.sqrt(np.maximum(0,1-a1*a1))[None,:]*np.array([r1,r2])[:,None];centers=np.array([c1,c2])
    expect=np.r_[(centers-half).min(0),(centers+half).max(0)]
    cardinal_half=np.array([np.maximum(abs(u),abs(np.cross(a,u)))
                            for a,u in [(a1,u1),(a2,u2)]])
    cardinal=np.r_[(centers-cardinal_half).min(0),(centers+cardinal_half).max(0)]
    source=float_view(data, offset+8, 6)
    if min(np.max(abs(expect-source)),np.max(abs(cardinal-source)))>1e-5:raise ValueError('ruled-circle bounds mismatch')
    t=np.arange(96)*2*np.pi/96
    rings=[c+np.cos(t)[:,None]*u+np.sin(t)[:,None]*np.cross(a,u) for c,a,u in [(c1,a1,u1),(c2,a2,u2)]]
    v=np.vstack(rings);f=primitive_geometry.ring_faces(2,96).copy()
    if with_normals:
        tangent=-np.sin(t)[:,None]*(u1/r1)+np.cos(t)[:,None]*np.cross(a1,u1/r1)
        normals=np.cross(tangent,rings[1]-rings[0])
        if (c2-c1)@a1<0:normals=-normals
        return primitive_geometry.finish(v,f,np.tile(normals,(2,1)))
    if (c2-c1)@a1<0:
        f=f[:,::-1]
    return v,f


def boundary_face_from_vue(data,offset,record_start,matrix):
    tag,size=struct.unpack_from('<2I',data,offset)
    if tag!=14 or size not in [312,368]:raise ValueError('unsupported bounded face')
    if struct.unpack_from('<I',data,offset+0x9c)[0]!=1 or struct.unpack_from('<I',data,offset+0xcc)[0]!=1:raise ValueError('unsupported face loops')
    ptr,kind=struct.unpack_from('<2I',data,offset+0xd8)
    if record_start+0x70+ptr!=offset+0xe0:raise ValueError('invalid face boundary reference')
    m=np.asarray(matrix).reshape(4,4);normal=unit(struct.unpack_from('<3d',data,offset+0x68))@m[:3,:3]
    plane=(np.r_[struct.unpack_from('<3d',data,offset+0x50),1]@m)[:3]
    underbounded_circle = False
    if size==368 and kind==4:
        p,n=struct.unpack_from('<2I',data,offset+0xe0)
        if n!=5 or record_start+0x70+p!=offset+0xf0:raise ValueError('unsupported polyline face boundary')
        points=np.array([struct.unpack_from('<3d',data,offset+0xf0+i*24) for i in range(5)])
        if np.linalg.norm(points[0]-points[-1])>1e-8:raise ValueError('open face loop')
        v=(np.c_[points[:4],np.ones(4)]@m)[:,:3];f=np.array([(0,1,2),(0,2,3)])
        edges=np.roll(v,-1,axis=0)-v
        turns=np.array([np.cross(edges[i],edges[(i+1)%4])@normal for i in range(4)])
        if not (np.all(turns>1e-12) or np.all(turns< -1e-12)):raise ValueError('nonconvex face')
    elif size==312 and kind==15:
        center,axis,radial=native_circle(data,offset+0xe0,m)
        if abs(abs(axis@normal)-1)>1e-7:raise ValueError('circle plane mismatch')
        t=np.arange(96)*2*np.pi/96
        ring=center+np.cos(t)[:,None]*radial+np.sin(t)[:,None]*np.cross(axis,radial)
        v=np.vstack([ring,center]);f=np.column_stack((np.arange(96),(np.arange(96)+1)%96,np.full(96,96)))
        source_box=float_view(data, offset+8, 6)
        underbounded_circle=bool(np.all(source_box[:3]-1e-7<=center) and
                                 np.all(center<=source_box[3:]+1e-7))
    else:raise ValueError('unsupported face boundary kind')
    if not np.isfinite(v).all() or np.max(abs((v-plane)@normal))>1e-7:raise ValueError('face outside plane')
    if np.cross(v[f[0,1]]-v[f[0,0]],v[f[0,2]]-v[f[0,0]])@normal<0:f=f[:,::-1]
    if (bounds_error(v,float_view(data, offset+8, 6))>1e-5 and
            not underbounded_circle):
        raise ValueError('bounded face placement mismatch')
    return v,f


def plate_solid_from_vue(data, offset, matrix, record_start=None, *, with_normals=False):
    """Decode the closed native triangle/quadrilateral thin-plate profiles.

    The counted point list at +0x60 uses the same (0,x,y) frame as the
    sibling planar face. Its world bbox supplies the observed thickness.
    """
    tag, size = struct.unpack_from('<2I', data, offset)
    if tag != 4 or size not in (200, 224):
        raise ValueError('unsupported nested plate layout')
    pointer, count = struct.unpack_from('<2I', data, offset + 0x50)
    # The native point list is in the same (0,x,y) frame as its sibling face.
    if record_start is not None and record_start+0x70+pointer != offset+0x60:
        raise ValueError('invalid nested plate point reference')
    if count != (size-104)//24:
        raise ValueError('invalid nested plate point count')
    points = float_view(data, offset+0x60, 3*count).reshape(-1,3)
    if (not np.isfinite(points).all() or np.max(abs(points[:,0])) > 1e-10 or
            np.linalg.norm(points[0]-points[-1]) > 1e-10):
        raise ValueError('invalid nested plate profile')
    m = np.asarray(matrix).reshape(4, 4)
    child_profile = points[:-1]
    face = (np.c_[child_profile, np.ones(len(child_profile))] @ m)[:, :3]
    source_box = float_view(data, offset + 8, 6)
    if (np.max(np.abs(face[:, 2] - face[0, 2])) > 1e-7 or
            np.any(face[:, :2].min(0) < source_box[:2] - 1e-7) or
            np.any(face[:, :2].max(0) > source_box[3:5] + 1e-7)):
        raise ValueError('nested plate placement is not a VUE horizontal plate')
    bottom_z, top_z = source_box[2], source_box[5]
    bottom = face.copy()
    top = face.copy()
    bottom[:, 2] = bottom_z
    top[:, 2] = top_z
    vertices = np.vstack((bottom, top))
    cap_faces = _triangulate_simple_polygon(bottom)
    if with_normals:
        if bounds_error(vertices,source_box)>1e-5:
            raise ValueError('nested plate bbox mismatch')
        return primitive_geometry.prism(bottom,top,cap_faces)
    n = len(bottom)
    faces = [tuple(face_indices) for face_indices in cap_faces]
    faces.extend(tuple(int(index) + n for index in face_indices)
                 for face_indices in cap_faces)
    faces.extend((i, (i + 1) % n, (i + 1) % n + n) for i in range(n))
    faces.extend((i, (i + 1) % n + n, i + n) for i in range(n))
    vertices, faces = _orient_generated_surface(vertices, np.asarray(faces, dtype=np.int64))
    if bounds_error(vertices, source_box) > 1e-5:
        raise ValueError('nested plate bbox mismatch')
    return vertices, faces


def _line_loop_from_vue(data, offset, record_start, descriptor_start,
                        target_start, count=4, closed=True):
    """Read an ordered closed line loop from VUE pointer/curve references.

    The 0x20/464 and 0x1f/464 payloads use the same native straight-curve
    record as the older planar-quad decoder, but keep the four references at
    different offsets.  The curve records themselves contain the complete
    endpoints, direction and length, so no bounding-box fitting is involved.
    """
    points, directions, lengths = [], [], []
    for index in range(count):
        pointer, kind = struct.unpack_from('<2I', data,
                                           offset + descriptor_start + 8 * index)
        target = offset + target_start + 0x40 * index
        if kind != 3 or record_start + 0x70 + pointer != target:
            raise ValueError('invalid native straight-curve reference')
        values = float_view(data, target, 8)
        if not np.isfinite(values).all() or values[6] != 0.0 or values[7] <= 0:
            raise ValueError('invalid native straight-curve parameters')
        point, direction, length = values[:3], values[3:6], values[7]
        if abs(np.linalg.norm(direction) - 1.0) > 1e-6:
            raise ValueError('nonunit native straight-curve direction')
        points.append(point)
        directions.append(direction)
        lengths.append(length)
    points = np.asarray(points)
    directions = np.asarray(directions)
    lengths = np.asarray(lengths)
    ends = points + directions * lengths[:, None]
    expected = np.roll(points, -1, axis=0) if closed else points[1:]
    actual = ends if closed else ends[:-1]
    if np.max(np.abs(actual - expected)) > 1e-7:
        raise ValueError('native straight-curve chain does not connect')
    return points


def _place_candidates(points, source_box, matrix):
    """Choose the native or root-matrix placement matching a VUE bbox."""
    chosen, error = _choose_placement(np.asarray(points, dtype=float), matrix, source_box)
    if error > 0.005:
        raise ValueError('native line-loop bounds mismatch')
    return chosen


def curved_strip_face_from_vue(data, offset, record_start, matrix=None):
    """Decode the observed 0x20/464 swept strip surface from VUE.

    The four native line records are the section at the first sweep angle;
    +0x50/+0x68/+0x80 hold the native revolution centre, axis and sweep.  The
    VUE bbox confirms that this is a partial revolution, rather than a flat
    quadrilateral (the latter only covers the first section).
    """
    if struct.unpack_from('<2I', data, offset) != (0x20, 464):
        raise ValueError('unsupported 0x20/464 strip surface layout')
    center = float_view(data, offset + 0x50, 3)
    axis = float_view(data, offset + 0x68, 3)
    angle = float(struct.unpack_from('<d', data, offset + 0x80)[0])
    if (not np.isfinite(np.r_[center, axis, angle]).all() or
            abs(np.linalg.norm(axis) - 1.0) > 1e-7 or
            not 0 < abs(angle) <= 2 * math.pi + 1e-7):
        raise ValueError('invalid native strip sweep parameters')
    section = _line_loop_from_vue(data, offset, record_start, 0xa8, 0xc8)
    parameters = np.linspace(0.0, angle,
                             max(8, int(math.ceil(abs(angle) / (math.pi / 48)))) + 1)
    axis = unit(axis)
    swept = []
    for parameter in parameters:
        cosine, sine = math.cos(parameter), math.sin(parameter)
        delta = section - center
        rotated = (delta * cosine + np.cross(axis, delta) * sine +
                   axis[None, :] * (delta @ axis)[:, None] * (1.0 - cosine))
        swept.append(center + rotated)
    local = np.asarray(swept).reshape(-1, 3)
    source_box = float_view(data, offset + 8, 6)
    vertices = _place_candidates(local, source_box, matrix)
    section_count = len(section)
    faces = []
    for sweep_index in range(len(parameters) - 1):
        for section_index in range(section_count):
            next_section = (section_index + 1) % section_count
            a = sweep_index * section_count + section_index
            b = sweep_index * section_count + next_section
            c = (sweep_index + 1) * section_count + next_section
            d = (sweep_index + 1) * section_count + section_index
            faces.extend(((a, b, c), (a, c, d)))
    mesh = trimesh.Trimesh(vertices, np.asarray(faces, dtype=np.int64), process=False)
    mesh.fix_normals()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def curved_strip_solid_from_vue(data, offset, record_start, matrix=None):
    """Decode the observed 0x1f/464 VUE profile extrusion.

    Four native straight curves describe the rectangular/oblique profile and
    the first three doubles at +0x50 are the extrusion vector.  Both the
    profile and vector are transformed together when the root matrix is the
    valid placement candidate.
    """
    if struct.unpack_from('<2I', data, offset) != (0x1f, 464):
        raise ValueError('unsupported 0x1f/464 strip solid layout')
    vector = float_view(data, offset + 0x50, 3)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-10:
        raise ValueError('invalid native strip extrusion vector')
    line_points = _line_loop_from_vue(data, offset, record_start, 0x88, 0xa8,
                                      count=3, closed=False)
    pointer, kind = struct.unpack_from('<2I', data, offset + 0xa0)
    arc_offset = offset + 0x168
    if kind != 16 or record_start + 0x70 + pointer != arc_offset:
        raise ValueError('invalid native curved strip profile reference')
    arc_center = float_view(data, arc_offset, 3)
    arc_normal = float_view(data, arc_offset + 0x18, 3)
    arc_major = float_view(data, arc_offset + 0x30, 3)
    ratio, start, end = struct.unpack_from('<3d', data, arc_offset + 0x48)
    if (not np.isfinite(np.r_[arc_center, arc_normal, arc_major, ratio, start, end]).all() or
            abs(np.linalg.norm(arc_normal) - 1.0) > 1e-7 or
            np.linalg.norm(arc_major) <= 1e-10 or not 0 < ratio <= 1 or
            not abs(end - start) <= 2 * math.pi + 1e-7 or
            abs(arc_normal @ arc_major) > 1e-7):
        raise ValueError('invalid native curved strip profile arc')
    arc_minor = np.cross(arc_normal, arc_major) * ratio
    arc_parameters = np.linspace(start, end,
                                 max(8, int(math.ceil(abs(end - start) /
                                                      (math.pi / 48)))) + 1)
    arc_points = (arc_center[None, :] +
                  np.cos(arc_parameters)[:, None] * arc_major[None, :] +
                  np.sin(arc_parameters)[:, None] * arc_minor[None, :])
    third_line = float_view(data, offset + 0x128, 8)
    third_line_end = third_line[:3] + third_line[3:6] * third_line[7]
    if (np.linalg.norm(third_line_end - arc_points[0]) > 1e-7 or
            np.linalg.norm(arc_points[-1] - line_points[0]) > 1e-7):
        raise ValueError('native curved strip profile arc does not close')
    profile = np.vstack((line_points, arc_points[:-1]))
    source_box = float_view(data, offset + 8, 6)
    candidates = [(profile, vector)]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        candidates.append(((np.c_[profile, np.ones(len(profile))] @ m)[:, :3],
                           vector @ m[:3, :3]))
    profile, vector = min(
        candidates,
        key=lambda candidate: bounds_error(
            np.vstack((candidate[0], candidate[0] + candidate[1])), source_box))
    vertices = np.vstack((profile, profile + vector))
    if bounds_error(vertices, source_box) > 0.005 + 1e-8:
        raise ValueError('native strip solid bounds mismatch')
    normal = None
    for index in range(1, len(profile) - 1):
        candidate = np.cross(profile[index] - profile[0],
                             profile[index + 1] - profile[0])
        if np.linalg.norm(candidate) > 1e-12:
            normal = unit(candidate)
            break
    if normal is None or abs(float(normal @ unit(vector))) < 1 - 1e-6:
        raise ValueError('native strip profile is not perpendicular to extrusion')
    cap_faces = _triangulate_simple_polygon(profile)
    n = len(profile)
    faces = [tuple(face) for face in cap_faces]
    faces.extend(tuple(int(index) + n for index in face) for face in cap_faces)
    faces.extend((i, (i + 1) % n, (i + 1) % n + n) for i in range(n))
    faces.extend((i, (i + 1) % n + n, i + n) for i in range(n))
    return _orient_generated_surface(vertices, np.asarray(faces, dtype=np.int64))


def internal_head_join_disks(disks, heads, cylinders):
    """Identify paired native caps internal to a decoded head/cylinder union.

    The circumferential ring length is derived from the head's own vertex
    layout (``ELLIPSOID_MERIDIAN_SAMPLES`` meridian rings plus one pole) so it
    follows the requested ``segments`` instead of a hard-coded 96.
    """
    internal=[]
    for head_entry in heads:
        # Entries are (vertices, faces[, sampling_error]) meshes.
        head = head_entry[0]
        segments, remainder = divmod(len(head) - 1, ELLIPSOID_MERIDIAN_SAMPLES)
        if segments < 3 or remainder:
            raise ValueError('unsupported ellipsoid head vertex layout')
        center=head[:segments].mean(0);axis=unit(head[-1]-center)
        radius=np.linalg.norm(head[0]-center)
        matches=[d for d in disks if np.linalg.norm(d[-1]-center)<1e-7
                 and abs(np.linalg.norm(d[0]-d[-1])-radius)<1e-7
                 and np.max(abs((d-center)@axis))<1e-7]
        signs=[np.cross(d[0]-center,d[1]-center)@axis for d in matches]
        if not any(x>0 for x in signs) or not any(x<0 for x in signs):continue
        for cylinder_entry in cylinders:
            v = cylinder_entry[0]
            delta=v-center;z=delta@axis;rad=delta-z[:,None]*axis
            rr=np.linalg.norm(rad,axis=1)
            if abs(z.max())<1e-7 and z.min() < -1e-7 and np.all((rr<1e-7)|(abs(rr-radius)<1e-7)):
                internal.extend(matches);break
    return internal


def native_reference(data,offset,matrix):
    tag,size=struct.unpack_from('<2I',data,offset);m=np.asarray(matrix).reshape(4,4)
    if (tag,size)==(15,168):
        c,a,u=native_circle(data,offset+0x50,m);points=np.array([c+u,c-u,c+np.cross(a,u),c-np.cross(a,u)])
    elif (tag,size)==(40,112):
        points=np.array([(np.r_[struct.unpack_from('<3d',data,offset+0x50),1]@m)[:3]])
    elif (tag,size)==(3,152):
        p=float_view(data, offset+0x50, 3);a=float_view(data, offset+0x68, 3);lo,hi=struct.unpack_from('<2d',data,offset+0x80)
        if abs(np.linalg.norm(a)-1)>1e-7 or not hi>lo:raise ValueError('invalid line parameters')
        local=np.array([p+lo*a,p+(lo+hi)/2*a,p+hi*a]);points=(np.c_[local,np.ones(3)]@m)[:,:3]
    else:raise ValueError('unknown reference')
    box=float_view(data, offset+8, 6)
    if not np.isfinite(points).all() or np.any(points.min(0)<box[:3]-1e-5) or np.any(points.max(0)>box[3:]+1e-5):raise ValueError('reference placement mismatch')
    return points


class SurfaceQuery:
    """Lazy surface bounds and bounded batches of exact point/triangle queries.

    Sampling error remains local to each mesh. Scene queries keep vertices in
    their original local frame, avoiding a second world-coordinate array.
    """
    def __init__(self, vertices, faces, sampling_error=0.0, *, origin=None, bounds=None):
        self.vertices, self.faces = vertices, faces
        self.sampling_error = float(sampling_error)
        self.origin = origin
        if bounds is None:
            self.low, self.high = vertices.min(0), vertices.max(0)
            if origin is not None:
                self.low, self.high = self.low + origin, self.high + origin
        else:
            self.low, self.high = bounds
        self.triangle_low = self.triangle_high = None

    def _prepare_bounds(self):
        # Keep only the two triangle bounds, not a permanent Fx3x3 copy of
        # every triangle. Chunk the temporary gather to control peak memory.
        self.triangle_low = np.empty((len(self.faces),3),dtype=float)
        self.triangle_high = np.empty_like(self.triangle_low)
        for start in range(0,len(self.faces),8192):
            stop=min(start+8192,len(self.faces))
            triangles=self.vertices[self.faces[start:stop]]
            self.triangle_low[start:stop]=triangles.min(axis=1)
            self.triangle_high[start:stop]=triangles.max(axis=1)

    def contains_points(self, points, tolerance=1e-7):
        points=np.asarray(points,dtype=float)
        tolerance=float(tolerance)+self.sampling_error
        covered=np.zeros(len(points),dtype=bool)
        selected=np.flatnonzero(np.all(points>=self.low-tolerance,axis=1) &
                                np.all(points<=self.high+tolerance,axis=1))
        if not len(selected) or not len(self.faces):
            return covered
        if self.triangle_low is None:
            self._prepare_bounds()
        # Several samples on one reference can share a single closest-point
        # call. Bound the point/face pair mask for large curved surfaces.
        batch=max(1,32768//len(self.faces))
        low=self.triangle_low-tolerance
        high=self.triangle_high+tolerance
        for start in range(0,len(selected),batch):
            ids=selected[start:start+batch]
            local=points[ids] if self.origin is None else points[ids]-self.origin
            keep=(np.all(local[:,None,:]>=low[None,:,:],axis=2) &
                  np.all(local[:,None,:]<=high[None,:,:],axis=2))
            point_ids,face_ids=np.nonzero(keep)
            if not len(face_ids):
                continue
            samples=local[point_ids]
            triangles=self.vertices[self.faces[face_ids]]
            closest=trimesh.triangles.closest_point(triangles,samples)
            delta=closest-samples
            hits=np.einsum('ij,ij->i',delta,delta)<=tolerance*tolerance
            covered[ids[point_ids[hits]]]=True
        return covered

    def contains(self, point, tolerance=1e-7):
        return bool(self.contains_points(np.asarray(point)[None,:],tolerance)[0])


def references_on_surfaces(points, meshes):
    """Cover every reference sample, querying meshes lazily and in batches."""
    points=np.asarray(points,dtype=float)
    if not len(points):
        return True
    pending=np.arange(len(points))
    for mesh in meshes:
        query=mesh if isinstance(mesh,SurfaceQuery) else SurfaceQuery(*mesh)
        covered=query.contains_points(points[pending])
        pending=pending[~covered]
        if not len(pending):
            return True
    return False


# --- explicit per-mesh sampling error bounds (0928 P1 review fix) -----------
# Each parameterized generator knows the native radius/frame it tessellates,
# so its chord error against the analytic surface is computed from those
# parameters and carried with the mesh into SurfaceQuery.  Reference coverage
# then no longer follows the requested output segments, and the 1e-7 base
# tolerance is never globally inflated.  Families that keep fixed legacy
# sampling (sphere/NURBS/native polygons/...) stay at bound 0.0 exactly as
# before: their meshes do not change with segments.


def arc_tube_sampling_error(radius, tube, angle, segments):
    """Chord error of a sampled swept tube (path chording + section chording).

    A surface point stays within ``radius + tube`` of the arc centre, so the
    centreline path step ``|angle|/n`` contributes
    ``(radius+tube)*(1-cos(step/2))`` and the tube ring contributes one full
    turn chord error at ``tube``.
    """
    n = arc_path_segments(angle, segments)
    step = abs(float(angle)) / n
    path = (float(radius) + float(tube)) * (1.0 - math.cos(step / 2.0))
    return path + full_turn_chord_error(tube, segments)


def torus_sampling_error(data, offset, segments):
    """Native 0x1e/232 radii bound both chording directions of the torus."""
    radius, tube = struct.unpack_from('<2d', data, offset + 0x98)
    return (full_turn_chord_error(radius + tube, segments)
            + full_turn_chord_error(tube, segments))


def cone_sampling_error(data, offset, segments):
    """Native 0x17/424 base radius bounds the cone ring chording."""
    ref = float_view(data, offset + 0x80, 3)
    return full_turn_chord_error(float(np.linalg.norm(ref)), segments)


def revolved_profile_416_sampling_error(data, offset, segments):
    """Native 0x20/416 profile radii bound the revolution ring chording."""
    radii = [abs(struct.unpack_from('<d', data, offset + 0xa0 + 24 * i + 8)[0])
             for i in range(10)]
    return full_turn_chord_error(max(radii), segments)


def _meridian_chord_error(a, b, rings, density=32):
    """Max gap between a quarter-ellipse meridian and its inscribed polyline.

    Evaluated against the native ellipse (radial ``a cos(t)``, axial
    ``-b sin(t)`` over ``[1.5*pi, 2*pi]``), matching the head generator's
    meridian layout including the welded pole.
    """
    ends = np.linspace(1.5 * math.pi, 2.0 * math.pi, rings + 1)
    poly = np.column_stack((a * np.cos(ends), -b * np.sin(ends)))
    t = np.linspace(1.5 * math.pi, 2.0 * math.pi, rings * density + 1)
    curve = np.column_stack((a * np.cos(t), -b * np.sin(t)))
    start, vector = poly[:-1], poly[1:] - poly[:-1]
    length2 = np.einsum('ij,ij->i', vector, vector)
    rel = curve[:, None, :] - start[None, :, :]
    along = np.clip(np.einsum('ijk,jk->ij', rel, vector) / length2[None, :],
                    0.0, 1.0)
    closest = start[None, :, :] + along[:, :, None] * vector[None, :, :]
    distance = np.linalg.norm(curve[:, None, :] - closest, axis=2)
    return float(distance.min(axis=1).max())


def ellipsoid_head_sampling_error(data, offset, segments):
    """Head tessellation bound: circumferential ring chords + meridian chords.

    The circumferential term follows the requested ``segments`` (radial
    distance from the axis is at most the major semi-axis); the meridian stays
    at ``ELLIPSOID_MERIDIAN_SAMPLES`` rings and contributes its own
    tier-independent polyline gap, so every native-surface reference point is
    covered at every tier.
    """
    major = float_view(data, offset + 0xc8, 3)
    ratio = struct.unpack_from('<d', data, offset + 0xe0)[0]
    a = float(np.linalg.norm(major))
    b = a * float(ratio)
    return (full_turn_chord_error(a, segments)
            + _meridian_chord_error(a, b, ELLIPSOID_MERIDIAN_SAMPLES))


def ellipsoid_head_from_vue(data,offset,record_start,matrix, segments=None, *, with_normals=False):
    """Observed 0x20/256: quarter ellipse revolved through 2*pi.

    ``segments`` sets the circumferential density; the meridian sampling stays
    at ``ELLIPSOID_MERIDIAN_SAMPLES`` in P1.  ``None`` keeps the historical
    96-segment direct-call sampling.
    """
    if struct.unpack_from('<2I',data,offset)!=(32,256):raise ValueError('unsupported head layout')
    axis_point=float_view(data, offset+0x50, 3);axis=float_view(data, offset+0x68, 3)
    sweep=struct.unpack_from('<d',data,offset+0x80)[0]
    ptr,kind=struct.unpack_from('<2I',data,offset+0x90)
    if kind!=16 or record_start+0x70+ptr!=offset+0x98 or abs(sweep-2*np.pi)>1e-8:raise ValueError('unsupported head sweep or curve reference')
    center=float_view(data, offset+0x98, 3);normal=float_view(data, offset+0xb0, 3);major=float_view(data, offset+0xc8, 3)
    ratio,start,end=struct.unpack_from('<3d',data,offset+0xe0);a=np.linalg.norm(major)
    if not np.isfinite(np.r_[axis_point,axis,center,normal,major,ratio,start,end]).all() or not (a>0 and 0<ratio<=1):raise ValueError('invalid ellipse parameters')
    if abs(np.linalg.norm(axis)-1)>1e-7 or abs(np.linalg.norm(normal)-1)>1e-7 or abs(major@normal)>1e-8 or abs(major@axis)>1e-8 or np.linalg.norm(center-axis_point)>1e-8:raise ValueError('unsupported ellipse revolution frame')
    minor=np.cross(normal,major)*ratio
    if np.linalg.norm(minor+axis*a*ratio)>1e-8 or abs(start-1.5*np.pi)>1e-8 or abs(end-2*np.pi)>1e-8:raise ValueError('unsupported ellipse arc quadrant')
    m=np.asarray(matrix).reshape(4,4)
    if not np.allclose(m[:3,:3]@m[:3,:3].T,np.eye(3),atol=1e-6):raise ValueError('nonrigid ellipsoid transform')
    wc=(np.r_[center,1]@m)[:3];wa=axis@m[:3,:3];b=a*ratio
    # Exact support function of a half ellipsoid, including inclined axes.
    radial=a*np.sqrt(np.maximum(0,1-wa*wa));full=np.sqrt(radial*radial+(b*wa)**2)
    lo=wc-np.where(wa<0,full,radial);hi=wc+np.where(wa>0,full,radial)
    # Rotated instances can cache only the four native meridians.
    wu=major@m[:3,:3];wv=np.cross(axis,major)@m[:3,:3]
    cardinal_radial=np.maximum(abs(wu),abs(wv))
    cardinal_full=np.sqrt(cardinal_radial**2+(b*wa)**2)
    cardinal=np.r_[wc-np.where(wa<0,cardinal_full,cardinal_radial),
                   wc+np.where(wa>0,cardinal_full,cardinal_radial)]
    source_box=float_view(data, offset+8, 6)
    if min(np.max(abs(np.r_[lo,hi]-source_box)),np.max(abs(cardinal-source_box)))>1e-5:
        raise ValueError('ellipsoid head bounds mismatch')
    count = validate_segments(segments) if segments is not None else LEGACY_ELLIPSOID_SEGMENTS
    rings = ELLIPSOID_MERIDIAN_SAMPLES
    meridian=np.linspace(end,start,rings+1)[:-1];phi=np.arange(count)*2*np.pi/count
    u=major/a;vaxis=np.cross(axis,u)
    radial_frame=np.cos(phi)[:,None]*u+np.sin(phi)[:,None]*vaxis
    local=(center+(-b*np.sin(meridian))[:,None,None]*axis+a*np.cos(meridian)[:,None,None]*radial_frame).reshape(-1,3)
    local=np.vstack([local,center+b*axis])
    side=primitive_geometry.ring_faces(rings,count)
    f=np.empty((len(side)+count,3),dtype=np.int64);f[:len(side)]=side
    i=np.arange(count)
    f[len(side):]=np.column_stack(((rings-1)*count+i,(rings-1)*count+(i+1)%count,np.full(count,len(local)-1)))
    if with_normals:
        delta=local-center;axial=(delta@axis)[:,None]*axis
        gradient=(delta-axial)/(a*a)+axial/(b*b)
        return primitive_geometry.finish(local,f,gradient,m)
    v=(np.c_[local,np.ones(len(local))]@m)[:,:3]
    # Orient by the analytic ellipsoid gradient, not by open-surface volume.
    tri=v[f];delta=tri.mean(1)-wc;z=delta@wa;rad=delta-z[:,None]*wa;gradient=rad/(a*a)+z[:,None]*wa/(b*b)
    geometric=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);reverse=np.sum(geometric*gradient,axis=1)<0;f[reverse]=f[reverse,::-1]
    return v,f


def revolved_profile_416_from_vue(data, offset, record_start, matrix, segments=None, *, with_normals=False):
    """Decode the VUE 0x20/416 full-revolution profile surface.

    This family is not a collection of independent cylinders.  It stores a
    ten-point meridian in the canonical world-aligned ``(x=0, radius, z)``
    frame and revolves it through the native ``2*pi`` sweep.  The display
    group's translation supplies the profile origin; the group's rotation is
    the placement convention for neighboring primitive blocks and is not
    applied to this already world-aligned profile.  The native bbox is used
    as an independent placement check.
    """
    if offset < 0 or offset + 416 > len(data):
        raise ValueError('truncated 0x20/416 revolved profile')
    if struct.unpack_from('<2I', data, offset) != (0x20, 416):
        raise ValueError('unsupported 0x20/416 revolved profile')
    if (struct.unpack_from('<I', data, offset + 0x38)[0] != 7 or
            abs(struct.unpack_from('<d', data, offset + 0x70)[0] - 1.0) > 1e-10 or
            abs(struct.unpack_from('<d', data, offset + 0x80)[0] - 2 * math.pi) > 1e-8 or
            struct.unpack_from('<2I', data, offset + 0x198) != (2, 1)):
        raise ValueError('unsupported 0x20/416 revolution header')
    pointer, kind = struct.unpack_from('<2I', data, offset + 0x88)
    if (kind != 16711936 or
            record_start + 0x70 + pointer != offset + 0x90):
        raise ValueError('invalid 0x20/416 profile descriptor')
    pointer, count = struct.unpack_from('<2I', data, offset + 0x90)
    if (count != 4 or record_start + 0x70 + pointer != offset + 0x98):
        raise ValueError('invalid 0x20/416 profile descriptor chain')
    profile_count = struct.unpack_from('<2I', data, offset + 0x98)[1]
    if profile_count != 10:
        raise ValueError('unsupported 0x20/416 profile point count')
    profile = np.array([
        struct.unpack_from('<3d', data, offset + 0xa0 + 24 * i)
        for i in range(profile_count)
    ], dtype=float)
    if (not np.isfinite(profile).all() or
            np.max(np.abs(profile[:, 0])) > 1e-10 or
            np.min(profile[:, 1]) < -1e-10 or
            np.any(np.diff(profile[:, 2]) < -1e-10) or
            profile[0, 1] > 1e-10 or profile[-1, 1] > 1e-10 or
            profile[-1, 2] <= profile[0, 2] + 1e-10):
        raise ValueError('invalid 0x20/416 meridian profile')

    m = np.asarray(matrix).reshape(4, 4)
    origin = np.asarray(m[3, :3], dtype=float)
    radius = float(np.max(profile[:, 1]))
    source_box = float_view(data, offset + 8, 6)
    expected_box = np.r_[origin[:2] - radius, origin[2] + profile[0, 2],
                         origin[:2] + radius, origin[2] + profile[-1, 2]]
    if (not np.isfinite(np.r_[origin, radius, expected_box]).all() or
            radius <= 1e-10 or np.max(np.abs(expected_box - source_box)) > 1e-5):
        raise ValueError('0x20/416 revolved profile bounds mismatch')

    segments = validate_segments(segments) if segments is not None else LEGACY_REVOLVED_PROFILE_SEGMENTS
    if with_normals:
        return primitive_geometry.linear_revolution(profile[:,1:3],origin,segments,HARD_EDGE_ANGLE_DEG)
    angle = np.arange(segments, dtype=float) * 2 * math.pi / segments
    vertices = []
    rings = []
    for _, radial, axial in profile:
        if abs(radial) <= 1e-10:
            rings.append([len(vertices)])
            vertices.append(origin + np.array([0.0, 0.0, axial]))
        else:
            start = len(vertices)
            ring = list(range(start, start + segments))
            rings.append(ring)
            vertices.extend(origin + np.column_stack((
                radial * np.cos(angle), radial * np.sin(angle),
                np.full(segments, axial))))
    faces = []
    for first, second in zip(rings, rings[1:]):
        if len(first) == 1 and len(second) == segments:
            pole = first[0]
            faces.extend((pole, second[i], second[(i + 1) % segments])
                         for i in range(segments))
        elif len(first) == segments and len(second) == 1:
            pole = second[0]
            faces.extend((first[i], pole, first[(i + 1) % segments])
                         for i in range(segments))
        elif len(first) == len(second) == segments:
            faces.extend((first[i], second[i], second[(i + 1) % segments])
                         for i in range(segments))
            faces.extend((first[i], second[(i + 1) % segments],
                          first[(i + 1) % segments])
                         for i in range(segments))
        else:
            raise ValueError('invalid 0x20/416 profile ring transition')
    mesh = trimesh.Trimesh(np.asarray(vertices, dtype=float),
                           np.asarray(faces, dtype=np.int64), process=False)
    if not mesh.is_watertight or bounds_error(mesh.vertices, source_box) > 1e-5:
        raise ValueError('0x20/416 revolved profile mesh mismatch')
    return _orient_generated_surface(np.asarray(mesh.vertices), np.asarray(mesh.faces))


# Display width of a virtual Weld marker, not a physical pipe/weld dimension.
WELD_MARKER_WIDTH_DIAMETER_RATIO = 0.005
WELD_MARKER_MAX_WIDTH_M = 0.002


def weld_marker_radius(type_name, ring_radius, source_radius):
    """Only native Weld records receive a narrow visualization cross-section."""
    if type_name != 'Weld':
        return source_radius
    if not np.isfinite([ring_radius, source_radius]).all() or min(ring_radius, source_radius) <= 0:
        raise ValueError('invalid weld marker radii')
    return min(source_radius, ring_radius * WELD_MARKER_WIDTH_DIAMETER_RATIO,
               WELD_MARKER_MAX_WIDTH_M / 2)


@lru_cache(maxsize=128)
def _tube_faces(n, m, closed):
    """Shared sweep topology, in the same face order as the scalar generator."""
    ring = np.arange(n)[:, None] * m
    section = np.arange(m)[None, :]
    a = np.broadcast_to(ring + section, (n, m)).ravel()
    b = (ring + (section+1) % m).ravel()
    c, d = b + m, a + m
    faces = np.stack((np.column_stack((a, b, c)), np.column_stack((a, c, d))), axis=1).reshape(-1, 3)
    if closed:
        faces[faces >= n*m] -= n*m
    else:
        section = np.arange(m)
        start = (n+1)*m
        first = np.column_stack((np.full(m, start), (section+1) % m, section))
        last = np.column_stack((np.full(m, start+1), n*m+section, n*m+(section+1) % m))
        faces = np.vstack((faces, np.stack((first, last), axis=1).reshape(-1, 3)))
    faces.setflags(write=False)
    return faces


def arc_tube(radius, tube, angle, matrix, segments=None, *, with_normals=False):
    # Existing 255 envelope hypothesis, generalized for preview only.
    angle = normalize_sweep(angle)
    if segments is None:
        n, m = max(4, int(math.ceil(abs(angle) / (math.pi / 36)))), LEGACY_ARC_SECTION_SEGMENTS
    else:
        segments = validate_segments(segments)
        n, m = arc_path_segments(angle, segments), segments
    t = np.linspace(0, angle, n + 1)
    q = np.arange(m) * (2 * math.pi / m)
    centers = np.c_[radius*np.sin(t), radius*(1-np.cos(t)), t*0]
    radial = np.c_[np.sin(t), -np.cos(t), t*0]
    if with_normals:
        closed=is_full_turn(angle)
        use=slice(None,-1) if closed else slice(None)
        tangent=np.sign(angle)*np.c_[np.cos(t),np.sin(t),t*0]
        return primitive_geometry.tube(centers[use],radial[use],
            np.tile([0.,0.,1.],(len(centers[use]),1)),tangent[use],tube,m,closed,matrix)
    v = (centers[:, None, :] + tube * (np.cos(q)[None, :, None] * radial[:, None, :]
         + np.sin(q)[None, :, None] * np.array([0, 0, 1]))).reshape(-1, 3)
    f = _tube_faces(n, m, False)
    v = np.vstack([v, centers[0], centers[-1]])
    v = np.c_[v, np.ones(len(v))] @ matrix
    return _orient_generated_solid(v[:, :3], f)


def arc_tube_radius_from_vue(data, offset):
    """Read the orientation-independent tube-radius vector in 0x20/240."""
    if struct.unpack_from('<2I', data, offset) != (0x20, 240):
        raise ValueError('unsupported precise arc-tube layout')
    radius_vector = float_view(data, offset + 0xc8, 3)
    if not np.isfinite(radius_vector).all():
        raise ValueError('nonfinite precise arc-tube radius vector')
    tube = float(np.linalg.norm(radius_vector))
    if tube <= 1e-12:
        raise ValueError('precise arc-tube radius vector is zero')
    return tube


def arc_tube_precise(center, start_point, end_axis, plane_normal, tube_radius, angle, matrix,
                     segments=None, *, with_normals=False):
    """0x20/240 payload variant carrying complete arc parameters.

    Observed layout: arc centre @0x50, arc-plane normal @0x68, sweep angle
    @0x80, first end-cap centre @0x98, first end pipe axis @0xb0 and the
    tube radius @0xc8 or @0xd0.  The centre line is
    ``centre + R(cos(t)*u1 + sin(t)*u2)`` with ``u1 = unit(start_point-centre)``
    and ``u2 = end_axis``; ``R = |start_point - centre|``.  Every parameter is
    native, so unlike ``arc_tube`` this is not a bbox-constrained hypothesis.
    """
    center = np.asarray(center, dtype=float)
    start_point = np.asarray(start_point, dtype=float)
    end_axis = np.asarray(end_axis, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)
    if not np.isfinite(np.r_[center, start_point, end_axis, plane_normal, angle]).all():
        raise ValueError('nonfinite arc parameters')
    if not 0 < abs(angle) <= 2 * math.pi + 1e-6:
        raise ValueError('unusable arc sweep')
    # 0928 P2 fix: one shared closure rule.  Sweeps within the closure
    # tolerance of +/-2*pi become the signed exact full turn BEFORE counting
    # path divisions, so 16/24/32 never degrade into 17/25/33 rings and the
    # closed ring keeps its full-turn symmetry.
    angle = normalize_sweep(angle)
    u1 = start_point - center
    radius = float(np.linalg.norm(u1))
    if radius <= 1e-12 or tube_radius <= 0:
        raise ValueError('degenerate arc radius or tube radius')
    u1 = u1 / radius
    if (abs(np.linalg.norm(end_axis) - 1) > 1e-6 or
            abs(np.linalg.norm(plane_normal) - 1) > 1e-6):
        raise ValueError('nonunit arc frame vectors')
    if (abs(plane_normal @ u1) > 1e-6 or abs(end_axis @ u1) > 1e-6 or
            abs(plane_normal @ end_axis) > 1e-6):
        raise ValueError('arc frame is not orthogonal')
    if segments is None:
        n = max(4, int(math.ceil(abs(angle) / (math.pi / 36))))
        m = LEGACY_ARC_SECTION_SEGMENTS
    else:
        segments = validate_segments(segments)
        n = arc_path_segments(angle, segments)
        m = segments
    t = np.linspace(0.0, angle, n + 1)
    centreline = center[None, :] + radius * (
        np.cos(t)[:, None] * u1[None, :] + np.sin(t)[:, None] * end_axis[None, :])
    tangent = -np.sin(t)[:, None] * u1[None, :] + np.cos(t)[:, None] * end_axis[None, :]
    w1 = np.repeat(plane_normal[None, :], n + 1, axis=0)
    w2 = np.cross(tangent, w1)
    lengths = np.linalg.norm(w2, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('degenerate arc section frame')
    w2 = w2 / lengths[:, None]
    if with_normals:
        closed=is_full_turn(angle)
        use=slice(None,-1) if closed else slice(None)
        return primitive_geometry.tube(centreline[use],w1[use],w2[use],
            np.sign(angle)*tangent[use],tube_radius,m,closed,matrix)
    q = np.arange(m) * (2 * math.pi / m)
    v = (centreline[:, None, :] + tube_radius * (
        np.cos(q)[None, :, None] * w1[:, None, :]
        + np.sin(q)[None, :, None] * w2[:, None, :])).reshape(-1, 3)
    closed = is_full_turn(angle)
    f = _tube_faces(n, m, closed)
    if closed:
        # A full sweep has no end disks; weld the final ring to the first.
        v=v[:n*m]
    else:
        v = np.vstack([v, centreline[0], centreline[-1]])
    v = np.c_[v, np.ones(len(v))] @ np.asarray(matrix).reshape(4, 4)
    return _orient_generated_solid(v[:, :3], f)


def weld_ring_line(center, start_point, end_axis, plane_normal, angle, matrix, segments):
    """A recognised weld ring as a polyline of its native circle.

    The native frame of the 0x20/240 payload gives the centre, centreline
    radius ``R = |start_point - centre|``, plane normal and sweep angle.  The
    ring is emitted directly as line segments — no tube cross-section, no
    normals and no split: a closed full turn has ``N`` unique vertices and
    ``N`` edges, a partial weld keeps its actual arc endpoints and direction.
    """
    center = np.asarray(center, dtype=float)
    start_point = np.asarray(start_point, dtype=float)
    end_axis = np.asarray(end_axis, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)
    if not np.isfinite(np.r_[center, start_point, end_axis, plane_normal, angle]).all():
        raise ValueError('nonfinite weld ring parameters')
    angle = normalize_sweep(angle)
    if not 0 < abs(angle) <= 2 * math.pi + 1e-8:
        raise ValueError('unusable weld ring sweep')
    u1 = start_point - center
    radius = float(np.linalg.norm(u1))
    if radius <= 1e-12:
        raise ValueError('degenerate weld ring radius')
    u1 = u1 / radius
    if abs(np.linalg.norm(end_axis) - 1) > 1e-6 or abs(np.linalg.norm(plane_normal) - 1) > 1e-6:
        raise ValueError('nonunit weld ring frame vectors')
    if (abs(plane_normal @ u1) > 1e-6 or abs(end_axis @ u1) > 1e-6
            or abs(plane_normal @ end_axis) > 1e-6):
        raise ValueError('weld ring frame is not orthogonal')
    segments = validate_segments(segments)
    if is_full_turn(angle):
        count = segments
        t = np.arange(count) * (angle / count)
        edges = np.column_stack((np.arange(count), (np.arange(count) + 1) % count))
    else:
        count = arc_path_segments(angle, segments)
        t = np.linspace(0.0, angle, count + 1)
        edges = np.column_stack((np.arange(count), np.arange(1, count + 1)))
    points = center[None, :] + radius * (
        np.cos(t)[:, None] * u1[None, :] + np.sin(t)[:, None] * end_axis[None, :])
    points = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).reshape(4, 4)
    return points[:, :3], edges.astype(np.int64)


PROFILE_ENTITY_BYTES = {3: 64, 16: 96}


def _profile_vector(data, offset):
    return float_view(data, offset, 3)


def _profile_entity(data, offset, kind):
    """Decode one VUE profile entity: kind 3 straight edge, kind 16 ellipse arc.

    Both layouts were read from the 15300 sample and cross-checked against the
    reference FBX: a straight edge stores point (+0x00), direction (+0x18) and
    the two signed parameters (+0x30, +0x38) that bound the segment, so
    p(t) = point + direction * t for t in [start, end]; an arc stores centre
    (+0x00), plane normal (+0x18), major semi-axis vector (+0x30) and the
    ratio/start/end triple (+0x48).  The arc is
    p(t) = centre + a cos(t) u + b sin(t) v with u = major/|major|,
    v = normal x u and b = |normal x major| * ratio.
    """
    if kind == 3:
        point = _profile_vector(data, offset)
        direction = _profile_vector(data, offset + 24)
        start, end = struct.unpack_from('<2d', data, offset + 48)
        if (not np.isfinite(np.r_[point, direction, start, end]).all() or
                abs(np.linalg.norm(direction) - 1.0) > 1e-6 or
                not 1e-9 < abs(end - start) <= 1e5):
            raise ValueError('invalid VUE profile edge')
        return dict(kind=3, point=point, direction=direction,
                    start=float(start), end=float(end))
    if kind == 16:
        center = _profile_vector(data, offset)
        normal = _profile_vector(data, offset + 24)
        major = _profile_vector(data, offset + 48)
        ratio, start, end = struct.unpack_from('<3d', data, offset + 72)
        minor = np.cross(normal, major)
        major_radius = float(np.linalg.norm(major))
        minor_length = float(np.linalg.norm(minor))
        if (not np.isfinite(np.r_[center, normal, major, ratio, start, end]).all() or
                major_radius <= 1e-12 or minor_length <= 1e-12 or
                not 0.0 < ratio <= 1.0 + 1e-9 or end <= start or
                end - start > 2 * math.pi + 1e-9 or major_radius > 1e5):
            raise ValueError('invalid VUE profile arc')
        return dict(kind=16, center=center, major=major, major_radius=major_radius,
                    minor_dir=minor / minor_length, minor_radius=minor_length * ratio,
                    start=float(start), end=float(end))
    raise ValueError('unsupported VUE profile entity')


def _profile_points(entity, arc_samples=None):
    """Sample one profile entity, keeping straight edges to their endpoints."""
    if entity['kind'] == 3:
        return np.vstack((entity['point'] + entity['direction'] * entity['start'],
                          entity['point'] + entity['direction'] * entity['end']))
    if arc_samples is None:
        arc_samples = max(4, int(math.ceil((entity['end'] - entity['start']) /
                                           (math.pi / 12.0))))
    angles = np.linspace(entity['start'], entity['end'], arc_samples + 1)
    axis_u = entity['major'] / entity['major_radius']
    return (entity['center'][None, :] +
            (entity['major_radius'] * np.cos(angles))[:, None] * axis_u[None, :] +
            (entity['minor_radius'] * np.sin(angles))[:, None] * entity['minor_dir'][None, :])


def _profile_chain_entities(data, offset, size, record_start, scan_start):
    """Collect ordered profile entity chains from a VUE descriptor slot region.

    Slots are 8-byte (pointer, kind) pairs whose pointer is relative to
    record_start + 0x70.  Entity data follows the slot region, so a chain is
    accepted only when consecutive slot targets are exactly the observed
    64-byte line or 96-byte arc apart and when no slot of the chain falls
    inside another chain's data.
    """
    end = offset + size
    slots = []
    for position in range(offset + scan_start, end - 7, 8):
        pointer, kind = struct.unpack_from('<2I', data, position)
        if kind not in PROFILE_ENTITY_BYTES:
            continue
        target = record_start + 0x70 + pointer
        if offset + 0x20 <= target <= end - 8:
            slots.append((target, kind, position))
    slots.sort()
    chains, current = [], []
    for target, kind, position in slots:
        if (current and current[-1][2] < position and
                target == current[-1][0] + PROFILE_ENTITY_BYTES[current[-1][1]]):
            current.append((target, kind, position))
        else:
            if len(current) >= 2:
                chains.append(current)
            current = [(target, kind, position)]
    if len(current) >= 2:
        chains.append(current)
    spans = [(chain[0][0], chain[-1][0] + PROFILE_ENTITY_BYTES[chain[-1][1]])
             for chain in chains]
    usable = []
    for index, chain in enumerate(chains):
        low, high = spans[index]
        if high > end - 8:
            continue
        if any(spans[other][0] <= position < spans[other][1]
               for other in range(len(chains)) if other != index
               for _, _, position in chain):
            continue
        usable.append(chain)
    return usable


def _profile_loop(data, chain, allow_open=False):
    """Sample a profile chain into a closed polygon of unique points.

    Returns the polygon and whether the chain's own endpoints coincided before
    sampling the duplicated closing point away.
    """
    points = []
    for index, (target, kind, _) in enumerate(chain):
        entity = _profile_entity(data, target, kind)
        piece = _profile_points(entity)
        points.extend(piece[:-1] if index + 1 < len(chain) else piece)
    points = np.asarray(points, dtype=float)
    if not np.isfinite(points).all() or len(points) < 3:
        raise ValueError('unusable VUE profile loop')
    closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1e-8)
    if closed:
        points = points[:-1]
    elif not allow_open:
        raise ValueError('open VUE profile loop')
    if len(points) < 3:
        raise ValueError('unusable VUE profile loop')
    return points, closed


def _orient_triangles(vertices, faces, references):
    """Flip individual triangles whose winding disagrees with an outward ref."""
    triangles = vertices[faces]
    geometric = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
    flip = np.einsum('ij,ij->i', geometric, references) < 0
    faces = np.asarray(faces).copy()
    faces[flip] = faces[flip, ::-1]
    return faces


def _orient_closed_solid(vertices, faces):
    """Make a closed mesh consistently outward-facing using its signed volume."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if mesh.is_watertight and mesh.volume < 0:
        faces = np.asarray(faces)[:, ::-1]
    return faces


def _world_candidates(local_vertices, matrix):
    candidates = [local_vertices]
    if matrix is not None:
        m = np.asarray(matrix).reshape(4, 4)
        if not np.array_equal(m,np.eye(4)):
            candidates.append(local_vertices @ m[:3,:3] + m[3,:3])
    return candidates


def _choose_placement(local_vertices, matrix, source_box):
    """Return (world, bounds_error) with each candidate's box evaluated once.

    Equivalent to
    ``world = min(_world_candidates(v, m), key=lambda p: bounds_error(p, box))``
    followed by a ``bounds_error(world, box)`` recheck, but reuses the winner's
    error instead of recomputing it.  ``min`` keeps the first minimal element,
    which ``moved_error < best_error`` reproduces exactly.
    """
    best = local_vertices
    best_error = bounds_error(best, source_box)
    if matrix is not None and best_error != 0.0:
        m=np.asarray(matrix).reshape(4,4)
        if np.array_equal(m,np.eye(4)):
            return best,best_error
        moved = local_vertices @ m[:3,:3] + m[3,:3]
        moved_error = bounds_error(moved, source_box)
        if moved_error < best_error:
            best, best_error = moved, moved_error
    return best, best_error


def profile_extrusion_from_vue(data, offset, size, record_start, matrix=None):
    """Decode the observed 0x1f/640 and 0x1f/848 rounded-profile extrusions.

    The block stores one closed rounded profile (straight edges and corner
    arcs) plus the extrusion vector at +0x50.  The +0x6c flag bit 0x10000
    marks blocks with native solid caps. Other blocks remain sheets here;
    export assembles their 0x0e/1432 end faces and closes standalone sheets
    through close_profile_shells without changing the native flag meaning.
    """
    tag, declared = struct.unpack_from('<2I', data, offset)
    if tag != 0x1f or declared != size or size not in (640, 848):
        raise ValueError('unsupported profile extrusion layout')
    extrusion = _profile_vector(data, offset + 0x50)
    if not np.isfinite(extrusion).all() or not 1e-10 < np.linalg.norm(extrusion) <= 1e4:
        raise ValueError('invalid profile extrusion vector')
    chains = _profile_chain_entities(data, offset, size, record_start, 0x68)
    if len(chains) != 1:
        raise ValueError('unsupported profile extrusion entity list')
    profile, closed = _profile_loop(data, chains[0])
    if len(profile) < 3 or not closed:
        raise ValueError('unusable profile extrusion loop')
    direction = unit(extrusion)
    span = float(np.max(profile.max(0) - profile.min(0)))
    if span <= 1e-9 or span > 1e4:
        raise ValueError('unusable profile extrusion size')
    solid = bool(struct.unpack_from('<I', data, offset + 0x6c)[0] & 0x10000)
    count = len(profile)
    centre = profile.mean(axis=0)
    vertices = np.vstack((profile, profile + extrusion))
    faces, references = [], []
    for index in range(count):
        following = (index + 1) % count
        middle = (profile[index] + profile[following]) / 2 + extrusion / 2 - centre
        middle = middle - float(middle @ direction) * direction
        for triangle in ((index, following, count + following),
                         (index, count + following, count + index)):
            faces.append(triangle)
            references.append(middle)
    if solid:
        cap = _triangulate_simple_polygon(profile)
        for triangle in cap:
            faces.append(tuple(int(index) for index in triangle))
            references.append(-direction)
        for triangle in cap:
            faces.append(tuple(int(index) + count for index in triangle))
            references.append(direction)
    faces = np.asarray(faces, dtype=np.int64)
    references = np.asarray(references, dtype=float)
    lengths = np.linalg.norm(references, axis=1)
    usable = lengths > 1e-12
    references[usable] /= lengths[usable, None]
    faces = _orient_triangles(vertices, faces, references)
    source_box = float_view(data, offset + 8, 6)
    world, error = _choose_placement(vertices, matrix, source_box)
    if error > 0.005:
        raise ValueError('profile extrusion bounds mismatch')
    return world, faces


def close_profile_shells(parts):
    """Join native yoke walls/caps topologically and orient each closed shell.

    Parts are (vertices, faces, is_extrusion). A lone closed-profile sheet gets
    planar end caps; paired hole walls keep their native annular end faces.
    Returned arrays preserve the original part ownership and coordinates.
    """
    if not parts:
        return [], []
    result=[np.asarray(f,dtype=np.int64).copy() for _,f,_ in parts]

    def joined():
        origin=parts[0][0][0]
        mesh=trimesh.util.concatenate([trimesh.Trimesh(p-origin,f,process=False)
                                      for (p,_,_),f in zip(parts,result)])
        mesh.merge_vertices(digits_vertex=8)
        owners=np.repeat(np.arange(len(parts)),[len(f) for f in result])
        components=trimesh.graph.connected_components(mesh.face_adjacency,
                            nodes=np.arange(len(mesh.faces)),min_len=1)
        return mesh,owners,components

    mesh,owners,components=joined()
    capped=set()
    for component in components:
        ids=np.unique(owners[component])
        if len(ids)!=1:
            continue
        index=int(ids[0]);points,_,is_extrusion=parts[index]
        if not is_extrusion or len(component)!=len(result[index]):
            continue
        sub=mesh.submesh([component],append=True,repair=False)
        if sub.is_watertight:
            continue
        count=len(points)//2
        if len(points)!=2*count or len(result[index])!=2*count:
            raise ValueError('unexpected open profile shell topology')
        vector=points[count:]-points[:count]
        if np.max(abs(vector-vector[0]))>1e-7:
            raise ValueError('unmatched profile extrusion rims')
        cap=_triangulate_simple_polygon(points[:count]-points[0])
        result[index]=np.vstack((result[index],cap,cap+count))
        capped.add(index)
    mesh,owners,components=joined()
    original_normals=mesh.face_normals.copy()
    report=[]
    for component in components:
        sub=mesh.submesh([component],append=True,repair=False)
        if not sub.is_watertight:
            raise ValueError('profile shell remains open after native cap assembly')
        sub.fix_normals(multibody=True)
        if (not sub.is_winding_consistent or sub.volume<=0 or
                np.any(sub.area_faces<=1e-16)):
            raise ValueError('invalid closed profile shell')
        flip=np.einsum('ij,ij->i',original_normals[component],sub.face_normals)<0
        selected=component[flip]
        start=0
        for index,faces in enumerate(result):
            local=selected[(selected>=start)&(selected<start+len(faces))]-start
            faces[local]=faces[local,::-1]
            start+=len(faces)
        report.append(dict(watertight=True,winding_consistent=True,
                           volume_m3=float(sub.volume),boundary_edges=0,
                           capped_parts=[int(i) for i in sorted(capped.intersection(owners[component]))]))
    return result,report


def revolved_profile_from_vue(data, offset, size, record_start, matrix=None):
    """Decode the observed 0x20/352 and 0x20/392 surfaces of revolution.

    Both blocks store an axis point (+0x50), axis direction (+0x68) and a 2*pi
    sweep (+0x80) followed by a meridian chain of straight edges and arcs.
    Revolving the native meridian reproduces the block's own bounding box, so
    no fitting or reference geometry is involved.
    """
    tag, declared = struct.unpack_from('<2I', data, offset)
    if tag != 0x20 or declared != size or size not in (352, 392):
        raise ValueError('unsupported revolution layout')
    axis_point = _profile_vector(data, offset + 0x50)
    axis = unit(_profile_vector(data, offset + 0x68))
    sweep = struct.unpack_from('<d', data, offset + 0x80)[0]
    if (not np.isfinite(np.r_[axis_point, sweep]).all() or
            not 1e-6 < sweep <= 2 * math.pi + 1e-9):
        raise ValueError('unsupported revolution sweep')
    chains = _profile_chain_entities(data, offset, size, record_start, 0x88)
    if len(chains) != 1:
        raise ValueError('unsupported revolution meridian')
    meridian, meridian_closed = _profile_loop(data, chains[0], allow_open=True)
    if len(meridian) < 3:
        raise ValueError('unusable revolution meridian')
    relative = meridian - axis_point
    height = relative @ axis
    radial = relative - height[:, None] * axis
    radius = np.linalg.norm(radial, axis=1)
    scale = float(max(radius.max(), np.abs(height).max()))
    if scale <= 1e-9 or scale > 1e4:
        raise ValueError('unusable revolution size')
    # A meridian that starts and ends on the axis is closed by the axis itself;
    # both endpoints then become single pole vertices of the revolution.  The
    # 0x20/352 sample stores its axis point 10 um off the meridian plane, so
    # the pole test is scaled to the block's own size.
    pole = radius <= max(1e-9, scale * 1e-4)
    closed = meridian_closed
    if not closed and not (pole[0] and pole[-1]):
        raise ValueError('open revolution meridian')
    sweep = normalize_sweep(sweep)
    full = is_full_turn(sweep)
    segments = 96 if full else max(8, int(math.ceil(sweep / (2 * math.pi) * 96)))
    angles = (np.arange(segments) * (2 * math.pi / segments) if full
              else np.linspace(0.0, sweep, segments + 1))
    direction_radial = np.zeros_like(radial)
    direction_radial[~pole] = radial[~pole] / radius[~pole, None]
    ring_index, vertices = [], []
    for index in range(len(meridian)):
        if pole[index]:
            ring_index.append(('pole', len(vertices)))
            vertices.append(axis_point + height[index] * axis)
            continue
        base = len(vertices)
        tangent = np.cross(axis, direction_radial[index])
        for angle in angles:
            vertices.append(axis_point + height[index] * axis +
                            radius[index] * (math.cos(angle) * direction_radial[index] +
                                             math.sin(angle) * tangent))
        ring_index.append(('ring', base))
    vertices = np.asarray(vertices, dtype=float)
    faces = []
    last = len(meridian) if closed else len(meridian) - 1
    for index in range(last):
        first = ring_index[index]
        second = ring_index[(index + 1) % len(meridian)]
        if first[0] == 'pole' and second[0] == 'pole':
            continue
        if first[0] == 'pole' or second[0] == 'pole':
            pole_index = first[1] if first[0] == 'pole' else second[1]
            base = second[1] if first[0] == 'pole' else first[1]
            for step in range(segments):
                following = (step + 1) % segments if full else step + 1
                if first[0] == 'pole':
                    faces.append((pole_index, base + step, base + following))
                else:
                    faces.append((base + step, pole_index, base + following))
            continue
        first_base, second_base = first[1], second[1]
        for step in range(segments):
            following = (step + 1) % segments if full else step + 1
            faces.extend(((first_base + step, second_base + step, second_base + following),
                          (first_base + step, second_base + following, first_base + following)))
    faces = np.asarray(faces, dtype=np.int64)
    faces = _orient_closed_solid(vertices, faces)
    source_box = float_view(data, offset + 8, 6)
    world, error = _choose_placement(vertices, matrix, source_box)
    if error > 0.005:
        raise ValueError('revolution bounds mismatch')
    return world, faces


def planar_face_with_holes_from_vue(data, offset, size, record_start, matrix=None):
    """Decode the observed 0x0e/1432 planar face bounded by two loops.

    The block keeps an outer rounded profile and an inner hole loop in the
    plane given by the stored unit normal (+0x68).  The annular region between
    them is triangulated with the existing boundary vertices, so the caps
    share complete edges with their independently decoded extrusion walls.
    The hole stays open; complete-shell orientation is resolved at export.
    """
    tag, declared = struct.unpack_from('<2I', data, offset)
    if tag != 0x0e or declared != 1432 or size != 1432:
        raise ValueError('unsupported planar face layout')
    normal = _profile_vector(data, offset + 0x68)
    if not np.isfinite(normal).all() or abs(np.linalg.norm(normal) - 1.0) > 1e-6:
        raise ValueError('invalid planar face normal')
    chains = _profile_chain_entities(data, offset, size, record_start, 0x98)
    if len(chains) != 2:
        raise ValueError('unsupported planar face loop count')
    loops = []
    for chain in chains:
        points, closed = _profile_loop(data, chain)
        if not closed:
            raise ValueError('open planar face loop')
        loops.append(points)
    vertices, faces = _annulus_between_loops(loops[0], loops[1], normal)
    source_box = float_view(data, offset + 8, 6)
    world, error = _choose_placement(vertices, matrix, source_box)
    reference = normal if matrix is None else normal @ np.asarray(matrix).reshape(4, 4)[:3, :3]
    faces = _orient_triangles(world, faces, np.tile(reference, (len(faces), 1)))
    if error > 0.005:
        raise ValueError('planar face bounds mismatch')
    return world, faces


def native_arc_from_vue(data, offset, matrix):
    center=_profile_vector(data,offset+0x50)
    normal=_profile_vector(data,offset+0x68)
    angle=struct.unpack_from('<d',data,offset+0x80)[0]
    start=_profile_vector(data,offset+0x98)
    axis=_profile_vector(data,offset+0xb0)
    return arc_tube_precise(center,start,axis,normal,arc_tube_radius_from_vue(data,offset),angle,matrix)


def _native_polyline(data, descriptor, record_start, block_end):
    pointer, kind = struct.unpack_from('<2I', data, descriptor)
    base = record_start+0x70+pointer
    if kind != 4 or base != descriptor+8:
        raise ValueError('invalid native polyline descriptor')
    pointer, count = struct.unpack_from('<2I', data, base)
    start = record_start+0x70+pointer
    if start != base+16 or not 3 <= count <= 10000 or start+24*count > block_end:
        raise ValueError('invalid native polyline points')
    points = float_view(data,start,3*count).reshape(-1,3)
    if not np.isfinite(points).all() or np.linalg.norm(points[0]-points[-1])>1e-8:
        raise ValueError('open native polyline')
    points = points[:-1]
    while len(points)>1 and np.linalg.norm(points[-1]-points[0])<1e-8:
        points=points[:-1]
    return points


def ruled_polylines_from_vue(data, offset, record_start, matrix):
    """0x21/392 and /440: closed ruled solids between native quadrilaterals.

    The payload stores the two end loops and the four ruled side strips.  The
    original decoder emitted only those side strips, which made every such
    component an open shell.  Both loops are planar quadrilaterals in the
    observed variants, so retain their native vertices and add two triangles
    per end.  This also keeps the cap/side edges coincident for downstream
    welding.
    """
    tag, size = struct.unpack_from('<2I', data, offset)
    if tag != 33 or size not in (392, 440):
        raise ValueError('unsupported ruled polyline layout')
    pointers = struct.unpack_from('<2I', data, offset+0x50)
    first_descriptor, second_descriptor = [record_start+0x70+p for p in pointers]
    if first_descriptor != offset+0x60 or second_descriptor != offset+{392:0xf0,440:0x108}[size]:
        raise ValueError('invalid ruled boundary pointers')
    first=_native_polyline(data,first_descriptor,record_start,offset+size-8)
    second=_native_polyline(data,second_descriptor,record_start,offset+size-8)
    if len(first)!=4 or len(second)!=4:
        raise ValueError('unsupported ruled boundary count')
    first_cap = _triangulate_simple_polygon(first)
    second_cap = _triangulate_simple_polygon(second)
    v=np.vstack((first,second));f=[]
    for i in range(4):
        j=(i+1)%4;f.extend(((i,j,j+4),(i,j+4,i+4)))
    f.extend(tuple(int(index) for index in triangle)
             for triangle in first_cap)
    f.extend(tuple(int(index + 4) for index in triangle[::-1])
             for triangle in second_cap)
    f=np.asarray(f,dtype=np.int64)
    v=(np.c_[v,np.ones(len(v))]@np.asarray(matrix).reshape(4,4))[:,:3]
    if bounds_error(v,struct.unpack_from('<6d',data,offset+8)) > (0.005+1e-8 if size==392 else 1e-5):
        raise ValueError('ruled polyline bounds mismatch')
    mesh=trimesh.Trimesh(v,f,process=False)
    if not mesh.is_watertight:
        raise ValueError('ruled polyline end caps do not close the solid')
    return _orient_generated_surface(np.asarray(mesh.vertices), np.asarray(mesh.faces))


def revolved_native_boundary_from_vue(data, offset, record_start, matrix):
    """Revolve the bounded arc/polyline in /256, /320 and /344 payloads.

    Keep the signed radial vector (including spindle-torus profiles). Axis
    endpoints are welded poles; an open surface gets no invented end disks.
    """
    tag, size = struct.unpack_from('<2I', data, offset)
    if tag != 32 or size not in (256, 320, 344) or offset + size > len(data):
        raise ValueError('unsupported bounded revolution')
    base = record_start + 0x70
    if base + struct.unpack_from('<I', data, offset+0x88)[0] != offset+0x90:
        raise ValueError('invalid bounded revolution pointer')
    pointer, kind = struct.unpack_from('<2I', data, offset+0x90)
    if base+pointer != offset+0x98:
        raise ValueError('invalid bounded revolution descriptor')
    if size == 256:
        if kind != 16:
            raise ValueError('invalid bounded revolution arc')
        arc_data = bytearray(data[offset:offset+size])
        start, end = struct.unpack_from('<2d', arc_data, 0xe8)
        if end < start:
            struct.pack_into('<d', arc_data, 0xf0, end+2*np.pi)
        entity = _profile_entity(arc_data, 0x98, kind)
        profile = _profile_points(entity, 128)
    else:
        pointer, count = struct.unpack_from('<2I', data, offset+0x98)
        if kind != 4 or base+pointer != offset+0xa8 or count != (6 if size == 320 else 7):
            raise ValueError('invalid bounded revolution polyline')
        profile = float_view(data, offset+0xa8, 3*count).reshape(-1,3)
    center = _profile_vector(data, offset+0x50)
    axis = _profile_vector(data, offset+0x68)
    sweep = struct.unpack_from('<d', data, offset+0x80)[0]
    if (not np.isfinite(np.r_[profile.ravel(), center, axis, sweep]).all() or
            abs(np.linalg.norm(axis)-1) > 1e-7 or not 0 < sweep <= 2*np.pi+1e-8):
        raise ValueError('invalid bounded revolution frame')
    closed = np.linalg.norm(profile[0]-profile[-1]) < 1e-8
    if closed:
        profile = profile[:-1]
    delta = profile-center
    axial = (delta@axis)[:,None]*axis
    radial = delta-axial
    sweep = normalize_sweep(sweep)
    full = is_full_turn(sweep)
    steps = max(8, int(np.ceil(sweep/(np.pi/96))))
    angles = np.arange(steps)*sweep/steps if full else np.linspace(0,sweep,steps+1)
    points = np.vstack([center+axial+radial*np.cos(t)+np.cross(axis,radial)*np.sin(t) for t in angles])
    faces=[]; n=len(profile)
    for j in range(steps):
        following=(j+1)%len(angles)
        for i in range(n if closed else n-1):
            k=(i+1)%n
            a,b,c,d=j*n+i,j*n+k,following*n+k,following*n+i
            faces.extend(((a,b,c),(a,c,d)))
    # Work in the local frame when welding poles to avoid large-coordinate loss.
    mesh=trimesh.Trimesh(points,faces,process=False)
    mesh.merge_vertices(digits_vertex=10)
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-10))
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    world=(np.c_[mesh.vertices,np.ones(len(mesh.vertices))]@np.asarray(matrix).reshape(4,4))[:,:3]
    source=float_view(data, offset+8, 6)
    error = bounds_error(world,source)
    if size == 256:
        # This family caches eight intervals along the trimmed section arc.
        coarse = _profile_points(entity, 8)-center
        ca = (coarse@axis)[:,None]*axis
        cr = coarse-ca
        cache = np.vstack([center+ca+cr*np.cos(t)+np.cross(axis,cr)*np.sin(t)
                           for t in angles])
        cache = (np.c_[cache,np.ones(len(cache))]@np.asarray(matrix).reshape(4,4))[:,:3]
        error = min(error,bounds_error(cache,source))
    if error>1e-4:
        raise ValueError('bounded revolution bounds mismatch')
    return world,np.asarray(mesh.faces)


def swept_native_profile_from_vue(data, offset, record_start, matrix):
    """Revolve a native polyline or B-spline around its stored axis."""
    tag,size=struct.unpack_from('<2I',data,offset)
    if tag!=32 or size not in (296,768):
        raise ValueError('unsupported native sweep layout')
    if record_start+0x70+struct.unpack_from('<I',data,offset+0x88)[0]!=offset+0x90:
        raise ValueError('invalid sweep profile pointer')
    if size==296:
        profile=_native_polyline(data,offset+0x90,record_start,offset+size-8)
    else:
        pointer,kind=struct.unpack_from('<2I',data,offset+0x90)
        if kind!=27 or record_start+0x70+pointer!=offset+0x98:
            raise ValueError('invalid sweep spline descriptor')
        pointers=struct.unpack_from('<3I',data,offset+0x98)
        if [record_start+0x70+p for p in pointers]!=[offset+0xc8,offset+0x200,offset+0x268]:
            raise ValueError('invalid sweep spline array pointers')
        if struct.unpack_from('<2I',data,offset+0xb0)!=(13,18):
            raise ValueError('unsupported sweep spline dimensions')
        profile=_rational_closed_curve_from_vue(data,offset,0xc8,13,18)[:-1]
    center=_profile_vector(data,offset+0x50)
    axis=unit(_profile_vector(data,offset+0x68))
    sweep=struct.unpack_from('<d',data,offset+0x80)[0]
    if not 0<sweep<=2*np.pi+1e-8:
        raise ValueError('invalid native sweep angle')
    sweep = normalize_sweep(sweep)
    full = is_full_turn(sweep)
    steps=max(12,int(np.ceil(sweep/(np.pi/72))))
    # A complete sweep must reuse its first ring at the seam.  The previous
    # linspace(..., steps+1) form created a duplicate final ring and left an
    # open seam in every 0x20/296 annular/revolved profile.
    angles=(np.arange(steps)*(2*np.pi/steps) if full else
            np.linspace(0,sweep,steps+1))
    ring_count=len(angles)
    delta=profile-center;axial=(delta@axis)[:,None]*axis;radial=delta-axial
    v=np.vstack([center+axial+radial*np.cos(t)+np.cross(axis,radial)*np.sin(t) for t in angles])
    n=len(profile);f=[]
    for j in range(steps):
        for i in range(n):
            k=(i+1)%n
            following=(j+1)%ring_count if full else j+1
            a=j*n+i;b=j*n+k;c=following*n+k;d=following*n+i
            f.extend(((a,b,c),(a,c,d)))
    if not full:
        # A partial revolution of a closed profile has two planar end faces.
        # Without them the side surface is still open at both sweep angles.
        cap = _triangulate_simple_polygon(profile, preserve_order=True)
        f.extend(tuple(int(index) for index in triangle)
                 for triangle in cap)
        last = (ring_count - 1) * n
        f.extend(tuple(int(last + index) for index in triangle[::-1])
                 for triangle in cap)
    v=(np.c_[v,np.ones(len(v))]@np.asarray(matrix).reshape(4,4))[:,:3]
    return _orient_generated_surface(v, np.asarray(f, dtype=np.int64))


def planar_polygon_from_vue(data, offset, record_start, matrix=None):
    """Decode native three- or six-line planar boundaries in any orientation."""
    tag, size = struct.unpack_from('<2I', data, offset)
    if tag != 14 or size not in (440, 656):
        raise ValueError('unsupported polygon face layout')
    chains = _profile_chain_entities(data, offset, size, record_start, 0x98)
    if len(chains) != 1 or len(chains[0]) != {440: 3, 656: 6}[size]:
        raise ValueError('invalid polygon boundary count')
    points, _ = _profile_loop(data, chains[0])
    normal = unit(_profile_vector(data, offset + 0x68))
    if np.max(abs((points-points[0]) @ normal)) > 1e-7:
        raise ValueError('nonplanar polygon boundary')
    faces = _triangulate_simple_polygon(points-points[0])
    faces = _orient_triangles(points, faces, np.tile(normal, (len(faces),1)))
    source_box = float_view(data, offset+8, 6)
    world, error = _choose_placement(points, matrix, source_box)
    if error > 1e-5:
        raise ValueError('polygon bounds mismatch')
    return world,faces


def _loop_area(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    centre = points.mean(axis=0)
    return float(np.linalg.norm(np.cross(points - centre, np.roll(points, -1, axis=0) - centre).sum(axis=0)) / 2.0)


def _annulus_between_loops(first, second, normal):
    """Triangulate the planar region between two nested simple loops."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 3 or len(second) < 3:
        raise ValueError('degenerate planar face loop')
    normal = unit(normal)
    if _loop_area(first) < _loop_area(second):
        first, second = second, first
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    axis_u = unit(np.cross(normal, helper))
    axis_v = np.cross(normal, axis_u)
    origin = first.mean(axis=0)

    def flatten(points):
        relative = points - origin
        return np.c_[relative @ axis_u, relative @ axis_v]

    outer = flatten(first)
    inner = flatten(second)
    if _polygon_area(outer) < 0:
        outer = outer[::-1]
    if _polygon_area(inner) < 0:
        inner = inner[::-1]
    if (np.max(np.abs((first-origin)@normal))>1e-6 or
            np.max(np.abs((second-origin)@normal))>1e-6):
        raise ValueError('planar face loops are not coplanar')
    # Keep the original boundary vertices: projecting new points onto an edge
    # creates T-junctions against the independently decoded extrusion wall.
    import mapbox_earcut
    flat=np.ascontiguousarray(np.vstack((outer,inner)),dtype=np.float64)
    faces=mapbox_earcut.triangulate_float64(
        flat,np.array([len(outer),len(flat)],dtype=np.uint32)).reshape(-1,3)
    edge1=flat[faces[:,1]]-flat[faces[:,0]];edge2=flat[faces[:,2]]-flat[faces[:,0]]
    signed=edge1[:,0]*edge2[:,1]-edge1[:,1]*edge2[:,0]
    expected=abs(_polygon_area(outer))-abs(_polygon_area(inner))
    if (expected<=0 or np.any(abs(signed)<1e-16) or
            abs(abs(signed).sum()/2-expected)>max(1e-12,expected*1e-8)):
        raise ValueError('invalid annular triangulation')
    vertices=origin+flat[:,0:1]*axis_u+flat[:,1:2]*axis_v
    return vertices,np.asarray(faces,dtype=np.int64)


def bounds_error(vertices, source_box):
    return float(np.max(np.abs(np.r_[vertices.min(0), vertices.max(0)] - source_box)))


def box_error(box, source_box):
    """max|·| between two ``[lo..., hi...]`` boxes (sampling-free comparison)."""
    return float(np.max(np.abs(np.asarray(box, dtype=float)
                               - np.asarray(source_box, dtype=float))))


def cylinder_bounds(point, axis, radius, direction, matrix=None):
    """Exact AABB of the native finite cylinder, before any sampling.

    The solid is the convex hull of its two end disks.  The ring basis spans
    the plane perpendicular to the axis, so the extent along coordinate axis
    ``k`` is ``radius*sqrt(|r_k|^2-(z.r_k)^2)`` for placement-matrix column
    ``r_k`` (``radius*sqrt(1-z_k^2)`` unplaced) — exact for rigid and non-rigid
    placements alike.  Placement, anchoring and tentative decisions compare
    THIS box with the VUE source box so they separate sampling error from real
    placement error and never follow the requested segments (plan §2.2).
    """
    point = np.asarray(point, dtype=float)
    a = np.asarray(axis, dtype=float)
    z = unit(a if float(np.linalg.norm(a)) > 1e-12
             else np.asarray(direction, dtype=float))
    ends = np.array([point, point + a])
    if matrix is not None:
        m = np.asarray(matrix, dtype=float).reshape(4, 4)
        ends = (np.c_[ends, np.ones(2)] @ m)[:, :3]
        extent = np.array([
            float(radius) * math.sqrt(max(
                0.0, float(m[:3, k] @ m[:3, k] - (z @ m[:3, k]) ** 2)))
            for k in range(3)])
    else:
        extent = float(radius) * np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.r_[np.minimum(ends[0], ends[1]) - extent,
                 np.maximum(ends[0], ends[1]) + extent]


def swept_tube_bounds(points, w1, w2, tube, matrix=None, grid_error=None):
    """AABB of a ``tube``-radius section swept along sampled centre points.

    ``w1``/``w2`` are the per-sample orthonormal section vectors; the support
    of the swept section along coordinate axis ``k`` is
    ``tube*sqrt(w1_k^2+w2_k^2)``.  The dense support sampling used by the
    callers is a FIXED validation sampling — deliberately independent of the
    user's ``segments`` (plan §2.2: validation sampling stays separate from
    output sampling) — with grid error far below the placement thresholds.
    """
    points = np.asarray(points, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    if matrix is not None:
        m = np.asarray(matrix, dtype=float).reshape(4, 4)
        points = (np.c_[points, np.ones(len(points))] @ m)[:, :3]
        w1 = w1 @ m[:3, :3]
        w2 = w2 @ m[:3, :3]
    extent = float(tube) * np.sqrt(w1 * w1 + w2 * w2)
    low = (points - extent).min(0)
    high = (points + extent).max(0)
    return np.r_[low, high]


# Fallback support sampling for non-orthonormal/spindle arc frames. Error is
# bounded by (radius+tube)*(1-cos(pi/GRID)) ~ 1e-6*(radius+tube), orders of
# magnitude below the 5mm placement thresholds for any realistic elbow.
ARC_BOUNDS_GRID = 2048


def _circular_tube_bounds(center, u, v, normal, radius, tube, angle, matrix):
    """Exact AABB from endpoints and coordinate extrema of a circular arc.

    For a matrix column, write r(t)=a*cos(t)+b*sin(t), n=normal.column.
    Tube support is center + R*r(t) +/- tube*sqrt(r(t)**2+n**2).
    When R >= tube both expressions are monotone in r, so only r's extrema
    and the two signed arc endpoints are needed. This also supports affine
    placements. Non-orthonormal frames/spindle arcs retain the grid fallback.
    """
    frame=np.asarray([u,v,normal],dtype=float)
    if (radius<tube or not np.isfinite(frame).all() or
            not np.allclose(frame@frame.T,np.eye(3),rtol=0,atol=1e-10)):
        return None
    m=np.eye(4) if matrix is None else np.asarray(matrix,dtype=float).reshape(4,4)
    a,b,n=frame@m[:3,:3]
    c=np.asarray(center)@m[:3,:3]+m[3,:3]
    end=a*math.cos(angle)+b*math.sin(angle)
    low=np.minimum(a,end);high=np.maximum(a,end)
    amplitude=np.hypot(a,b)
    phase=np.arctan2(b,a)
    start,stop=min(0.0,angle),max(0.0,angle)
    maximum=phase+2*np.pi*np.ceil((start-phase)/(2*np.pi))
    minimum=phase+np.pi+2*np.pi*np.ceil((start-phase-np.pi)/(2*np.pi))
    high=np.where(maximum<=stop,amplitude,high)
    low=np.where(minimum<=stop,-amplitude,low)
    return np.r_[c+radius*low-tube*np.hypot(low,n),
                 c+radius*high+tube*np.hypot(high,n)]


def arc_tube_precise_bounds(center, start_point, end_axis, plane_normal,
                            tube_radius, angle, matrix, samples=ARC_BOUNDS_GRID):
    """Exact world AABB of the native 0x20/240 swept tube (no output sampling).

    Use closed-form arc extrema for the ordinary orthonormal frame, retaining
    the fixed support grid for other frames. Neither path uses export segments.
    """
    center = np.asarray(center, dtype=float)
    start_point = np.asarray(start_point, dtype=float)
    u1 = start_point - center
    radius = float(np.linalg.norm(u1))
    if radius <= 1e-12 or float(tube_radius) <= 0:
        raise ValueError('degenerate arc radius or tube radius')
    u1 = u1 / radius
    u2 = np.asarray(end_axis, dtype=float)
    w1 = np.asarray(plane_normal, dtype=float)
    angle = normalize_sweep(angle)
    exact=_circular_tube_bounds(center,u1,u2,w1,radius,float(tube_radius),angle,matrix)
    if exact is not None:
        return exact
    t = np.linspace(0.0, angle, samples)
    cosine, sine = np.cos(t), np.sin(t)
    points = center[None, :] + radius * (
        cosine[:, None] * u1[None, :] + sine[:, None] * u2[None, :])
    tangent = (-sine[:, None] * u1[None, :] + cosine[:, None] * u2[None, :])
    w2 = np.cross(tangent, w1[None, :])
    lengths = np.linalg.norm(w2, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('degenerate arc section frame')
    w2 = w2 / lengths[:, None]
    w1 = np.repeat(w1[None, :], samples, axis=0)
    return swept_tube_bounds(points, w1, w2, tube_radius, matrix)


def arc_tube_bounds(radius, tube, angle, matrix, samples=ARC_BOUNDS_GRID):
    """Exact world AABB of the bbox-hypothesis swept tube (``arc_tube`` shape).

    Mirrors ``arc_tube``'s local centreline ``[R sin t, R(1-cos t), 0]`` and
    section frame (``[sin t, -cos t, 0]``, ``z``), using analytic extrema
    where valid and the fixed dense grid otherwise.
    """
    radius = float(radius)
    angle = normalize_sweep(angle)
    exact=_circular_tube_bounds([0,radius,0],[0,-1,0],[1,0,0],[0,0,1],
                                radius,float(tube),angle,matrix)
    if exact is not None:
        return exact
    t = np.linspace(0.0, angle, samples)
    cosine, sine = np.cos(t), np.sin(t)
    points = np.column_stack((radius * sine, radius * (1.0 - cosine),
                              np.zeros(samples)))
    w1 = np.column_stack((sine, -cosine, np.zeros(samples)))
    w2 = np.column_stack((np.zeros(samples), np.zeros(samples), np.ones(samples)))
    return swept_tube_bounds(points, w1, w2, tube, matrix)


def bbox_cage(box):
    """Return distinct box corners and edges for a diagnostic line primitive."""
    box = np.asarray(box, dtype=float)
    low, high = box[:3], box[3:]
    if not np.all(np.isfinite(box)) or np.any(high < low):
        raise ValueError('invalid placeholder bounding box')
    corners = np.asarray([[high[axis] if index & (1 << axis) else low[axis]
                           for axis in range(3)] for index in range(8)])
    vertices, inverse = np.unique(corners, axis=0, return_inverse=True)
    edges, seen = [], set()
    for index in range(8):
        for axis in range(3):
            if high[axis]-low[axis] <= 1e-12:
                continue
            other = index ^ (1 << axis)
            if index >= other:
                continue
            pair = tuple(sorted((int(inverse[index]), int(inverse[other]))))
            if pair[0] != pair[1] and pair not in seen:
                edges.append(pair)
                seen.add(pair)
    if not edges:
        raise ValueError('zero-size placeholder bounding box')
    return vertices, np.asarray(edges, dtype=np.int64)


def write_fbx_ascii(path, pieces):
    """Small ASCII FBX 7.4 writer, retaining metre units and source Z-up axes."""
    def arr(stream, name, values):
        stream.write(f'\t\t{name}: *{len(values)} {{\n\t\t\ta: ')
        stream.write(','.join(str(x) for x in values))
        stream.write('\n\t\t}\n')
    with path.open('w', encoding='utf-8', newline='\n') as out:
        out.write('; FBX 7.4.0 project file\nFBXHeaderExtension: {\n FBXHeaderVersion: 1003\n FBXVersion: 7400\n}\n')
        out.write('GlobalSettings: {\n Version: 1000\n Properties70: {\n')
        for key, value in [('UpAxis', 2), ('UpAxisSign', 1), ('FrontAxis', 1),
                           ('FrontAxisSign', 1), ('CoordAxis', 0), ('CoordAxisSign', -1)]:
            out.write(f' P: "{key}", "int", "Integer", "",{value}\n')
        out.write(' P: "UnitScaleFactor", "double", "Number", "",100\n P: "OriginalUnitScaleFactor", "double", "Number", "",100\n }\n}\n')
        out.write('Documents: {\n Count: 1\n Document: 1, "", "Scene" {\n RootNode: 0\n }\n}\n')
        out.write('Objects: {\n')
        connections, groups = [], {}
        next_id = 1000
        def null_node(key, name, parent):
            nonlocal next_id
            if key not in groups:
                ident = next_id
                next_id += 1
                groups[key] = ident
                out.write(f' Model: {ident}, "Model::{name}", "Null" {{\n Version: 232\n }}\n')
                connections.append((ident, parent))
            return groups[key]
        for item in pieces:
            parent = null_node(('discipline', item['discipline']), item['discipline'], 0)
            parent = null_node(('object', item['record_offset']), item['moniker'], parent)
            gid, mid, matid = next_id, next_id+1, next_id+2
            next_id += 3
            name = item['name']
            origin = item['origin']
            out.write(f' Geometry: {gid}, "Geometry::{name}", "Mesh" {{\n')
            vertices = item['export_vertices']
            faces = item['export_faces']
            normals = item['export_normals']
            arr(out, 'Vertices', vertices.reshape(-1))
            # FBX marks the last index of each polygon negative; keep the
            # arithmetic in signed integers (uint32 input would wrap).
            indices = faces.astype(np.int64, copy=True)
            indices[:, 2] = -indices[:, 2] - 1
            arr(out, 'PolygonVertexIndex', indices.reshape(-1))
            polygon_normals = normals[faces].reshape(-1)
            out.write('  GeometryVersion: 124\n  LayerElementNormal: 0 {\n   Version: 101\n   Name: ""\n   MappingInformationType: "ByPolygonVertex"\n   ReferenceInformationType: "Direct"\n')
            arr(out, 'Normals', polygon_normals)
            out.write('  }\n  LayerElementMaterial: 0 {\n   Version: 101\n   Name: ""\n   MappingInformationType: "AllSame"\n   ReferenceInformationType: "IndexToDirect"\n   Materials: *1 { a: 0 }\n  }\n  Layer: 0 {\n   Version: 100\n   LayerElement: { Type: "LayerElementNormal"\n    TypedIndex: 0 }\n   LayerElement: { Type: "LayerElementMaterial"\n    TypedIndex: 0 }\n  }\n }\n')
            out.write(f' Model: {mid}, "Model::{name}", "Mesh" {{\n Version: 232\n Properties70: {{\n')
            out.write(f' P: "Lcl Translation", "Lcl Translation", "", "A",{origin[0]},{origin[1]},{origin[2]}\n')
            out.write(' P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0\n P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1\n }\n Shading: T\n Culling: "CullingOff"\n }\n')
            r, g, b, alpha = item['color']
            # Match the reference Navisworks FBX material convention observed
            # in 15300.fbx: Phong, fixed grey ambient 0.2, grey specular 0.7,
            # near-zero shininess, emissive off; VUE opacity kept as-is.
            out.write(f' Material: {matid}, "Material::{name}", "" {{\n Version: 102\n ShadingModel: "Phong"\n Properties70: {{\n P: "DiffuseColor", "Color", "", "A",{r},{g},{b}\n P: "AmbientColor", "Color", "", "A",0.2,0.2,0.2\n P: "EmissiveColor", "Color", "", "A",0,0,0\n P: "SpecularColor", "Color", "", "A",0.7,0.7,0.7\n P: "Shininess", "Number", "", "A",0.001\n P: "TransparencyFactor", "Number", "", "A",{1-alpha}\n P: "Opacity", "Number", "", "A",{alpha}\n }}\n }}\n')
            connections.extend([(gid, mid), (matid, mid), (mid, parent)])
        out.write('}\nConnections: {\n')
        for child, parent in connections:
            out.write(f' C: "OO",{child},{parent}\n')
        out.write('}\n')


def read_fbx_mesh_translations(path, names):
    """Read named mesh translations and their D-property byte locations."""
    wanted=set(names)
    patches={}
    with Path(path).open('rb') as stream:
        if stream.read(23)!=b'Kaydara FBX Binary  \x00\x1a\x00':
            raise ValueError('not a binary FBX')
        version=struct.unpack('<I',stream.read(4))[0]
        header='<QQQB' if version>=7500 else '<IIIB'
        header_size=struct.calcsize(header)
        length=stream.seek(0,2);stream.seek(27)

        def read_node():
            position=stream.tell()
            raw=stream.read(header_size)
            if len(raw)!=header_size:raise ValueError('truncated FBX node')
            end,count,property_bytes,name_size=struct.unpack(header,raw)
            if not end:return None
            if end<=position or end>length:raise ValueError('invalid FBX node bounds')
            name=stream.read(name_size).decode('utf-8')
            props_start=stream.tell()
            children=props_start+property_bytes
            if children>end:raise ValueError('invalid FBX properties bounds')
            return end,count,name,props_start,children

        def read_properties(node):
            end,count,name,start,children=node
            stream.seek(start);values=[];locations=[]
            for _ in range(count):
                tag=stream.read(1);location=stream.tell();locations.append((tag,location))
                if tag==b'S':
                    size=struct.unpack('<I',stream.read(4))[0]
                    values.append(stream.read(size).decode('utf-8'))
                elif tag in (b'L',b'I',b'D',b'F',b'C',b'Y'):
                    fmt={b'L':'q',b'I':'i',b'D':'d',b'F':'f',b'C':'?',b'Y':'h'}[tag]
                    values.append(struct.unpack('<'+fmt,stream.read(struct.calcsize(fmt)))[0])
                else:raise ValueError('unsupported FBX model property')
                if stream.tell()>children:raise ValueError('overrun FBX properties')
            if stream.tell()!=children:raise ValueError('invalid FBX property length')
            return values,locations

        while True:
            top=read_node()
            if top is None:break
            if top[2]=='Objects':
                stream.seek(top[4])
                while stream.tell()<top[0]-header_size:
                    model=read_node()
                    if model is None:break
                    if model[2]=='Model':
                        values,_=read_properties(model)
                        name=values[1].split('\x00',1)[0]
                        if name.startswith('Model::'):name=name[7:]
                        if name in wanted:
                            stream.seek(model[4])
                            while stream.tell()<model[0]-header_size:
                                child=read_node()
                                if child is None:break
                                if child[2]=='Properties70':
                                    stream.seek(child[4])
                                    while stream.tell()<child[0]-header_size:
                                        prop=read_node()
                                        if prop is None:break
                                        if prop[2]=='P':
                                            values,locations=read_properties(prop)
                                            if values and values[0]=='Lcl Translation':
                                                if len(values)!=7 or any(t!=b'D' for t,_ in locations[4:]) or name in patches:
                                                    raise ValueError('ambiguous FBX translation property')
                                                patches[name]=([pos for _,pos in locations[4:]],np.asarray(values[4:],dtype=float))
                                        stream.seek(prop[0])
                                stream.seek(child[0])
                    stream.seek(model[0])
            stream.seek(top[0])
    if set(patches)!=set(wanted):raise ValueError('missing FBX mesh translations')
    return patches


def restore_fbx_translations(path, pieces):
    """Restore VUE double translations lost by Assimp's float32 matrices.

    Binary FBX stores these properties as doubles. Patch only the three
    existing D values after validating every expected mesh node; lengths,
    topology, materials and all node offsets remain unchanged.
    """
    wanted={item['name']:np.asarray(item['origin'],dtype=float) for item in pieces}
    if len(wanted)!=len(pieces) or any(v.shape!=(3,) or not np.isfinite(v).all() for v in wanted.values()):
        raise ValueError('invalid FBX translation source')
    patches=read_fbx_mesh_translations(path,wanted)
    with Path(path).open('r+b') as stream:
        for name,(locations,stored) in patches.items():
            for position,value in zip(locations,wanted[name]):
                stream.seek(position);stream.write(struct.pack('<d',float(value)))
    return len(patches)


def write_fbx(path, pieces):
    # The minimal ASCII intermediate is readable by Assimp, but rejected by
    # Navisworks. Always publish a full binary FBX. Assimp's ASCII importer
    # drops SpecularColor/Shininess/ShadingModel, so the final binary FBX is
    # produced by vue_fbx_phong_material.py, which re-injects the reference
    # Navisworks Phong convention. The helper exits without releasing the
    # Assimp scene (heap corruption otherwise), hence the subprocess.
    import subprocess
    import sys
    intermediate = path.with_suffix('.intermediate.fbx')
    try:
        write_fbx_ascii(intermediate, pieces)
        helper = Path(__file__).with_name('vue_fbx_phong_material.py')
        subprocess.run(helper_command('vue_fbx_phong_material', intermediate, path),
                       check=True, timeout=3600)
        restore_fbx_translations(path,pieces)
        with path.open('rb') as stream:
            if not stream.read(23).startswith(b'Kaydara FBX Binary'):
                raise ValueError('FBX binary export failed')
    finally:
        if intermediate.exists():
            intermediate.unlink()


def build_export_tree(solid_pieces, line_pieces):
    """Build the glTF (tree, binary) directly from split pieces.

    Replaces trimesh's scene assembly/export for solid meshes and diagnostic
    line cages.  The tree keeps the historical node layout (world ->
    VUE_Z_UP_METRES -> discipline -> owner -> mesh node with translation),
    which merge_gltf_tree, apply_business_tree and finalize_glb_from_tree all
    rely on, while the geometry buffer is written with a single pass.
    """
    views, accessors, meshes, materials = [], [], [], []
    out = bytearray()
    reused = {}

    def add(array, component, type_, target, normalized=False, bounds=False):
        array = np.ascontiguousarray(array)
        # Content-addressed reuse: identical arrays (shared instances of the
        # same primitive/material data) map to one accessor + bufferView, as
        # trimesh's exporter did.
        raw = array.tobytes()
        key = (component, type_, target, normalized, bounds, array.shape,
               hashlib.blake2b(raw, digest_size=16).digest())
        if key in reused:
            return reused[key]
        padding = (-len(out)) % 4
        if padding:
            out.extend(b'\x00' * padding)
        offset = len(out)
        out.extend(raw)
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(raw),
                          target=target))
        accessor = dict(bufferView=len(views) - 1, byteOffset=0,
                        componentType=component, count=len(array), type=type_)
        if normalized:
            accessor['normalized'] = True
        if bounds:
            accessor['min'] = np.asarray(array.min(axis=0), dtype=float).tolist()
            accessor['max'] = np.asarray(array.max(axis=0), dtype=float).tolist()
        accessors.append(accessor)
        reused[key] = len(accessors) - 1
        return len(accessors) - 1

    mesh_nodes = defaultdict(list)  # (discipline, owner) -> node dicts
    node_items = []  # (discipline, owner, node_dict)

    def append_solid(item):
        positions = np.asarray(item['export_vertices'], dtype='<f4')
        normals = np.asarray(item['export_normals'], dtype='<f4')
        indices = np.asarray(item['export_faces'], dtype=np.int64).reshape(-1)
        use_small = len(positions) <= 65536
        index_array = indices.astype('<u2' if use_small else '<u4')
        color = [float(channel) for channel in item['color']]
        material = dict(name=item['name'], pbrMetallicRoughness=dict(
            baseColorFactor=color, metallicFactor=0.0, roughnessFactor=1.0),
            doubleSided=True)
        if item['color'][3] < 1:
            material['alphaMode'] = 'BLEND'
        materials.append(material)
        meshes.append(dict(name=item['name'], extras=dict(
            moniker=item['moniker'], record_offset=item['record_offset'],
            preview=True), primitives=[dict(
                attributes=dict(POSITION=add(positions, 5126, 'VEC3', 34962, bounds=True),
                                NORMAL=add(normals, 5126, 'VEC3', 34962)),
                indices=add(index_array.reshape(-1, 1), 5123 if use_small else 5125,
                            'SCALAR', 34963),
                material=len(materials) - 1, mode=4)]))
        node_items.append((item['discipline'], item['name'], item,
                           dict(name=item['name'], mesh=len(meshes) - 1,
                                translation=[float(v) for v in item['origin']])))

    def append_line(item):
        positions = np.asarray(item['local_vertices'], dtype='<f4')
        edges = np.asarray(item['faces'], dtype=np.int64).reshape(-1)
        color = np.rint(line_color_for(item) * 255).astype(np.uint8)
        colors = np.tile(color, (len(positions), 1))
        meshes.append(dict(name=item['name'], extras=dict(
            moniker=item['moniker'], record_offset=item['record_offset'],
            preview=True), primitives=[dict(
                attributes=dict(POSITION=add(positions, 5126, 'VEC3', 34962, bounds=True),
                                COLOR_0=add(colors, 5121, 'VEC4', 34962, normalized=True)),
                indices=add(edges.astype('<u4').reshape(-1, 1), 5125, 'SCALAR', 34963),
                mode=1)]))
        node_items.append((item['discipline'], item['name'], item,
                           dict(name=item['name'], mesh=len(meshes) - 1,
                                translation=[float(v) for v in item['origin']])))

    for item in solid_pieces:
        append_solid(item)
    for item in line_pieces:
        append_line(item)

    # Node hierarchy: world -> VUE_Z_UP_METRES(matrix) -> discipline -> owner.
    root_children, discipline_nodes, owner_nodes = [], {}, {}
    nodes = [dict(name='world'),
             dict(name='VUE_Z_UP_METRES',
                  matrix=VUE_TO_EXPORT_MATRIX.T.flatten().tolist())]
    nodes[0]['children'] = [1]
    for discipline, name, item, node in node_items:
        owner = item['moniker'] + '__R' + str(item['record_offset'])
        if discipline not in discipline_nodes:
            discipline_nodes[discipline] = len(nodes)
            nodes.append(dict(name=discipline, children=[]))
            nodes[1].setdefault('children', []).append(discipline_nodes[discipline])
        key = (discipline, owner)
        if key not in owner_nodes:
            owner_nodes[key] = len(nodes)
            nodes.append(dict(name=owner, children=[]))
            nodes[discipline_nodes[discipline]]['children'].append(owner_nodes[key])
        nodes[owner_nodes[key]]['children'].append(len(nodes))
        nodes.append(node)
    tree = dict(scene=0, scenes=[dict(nodes=[0])],
                asset=dict(version='2.0', generator='vue_preview_export'),
                nodes=nodes, meshes=meshes, materials=materials,
                accessors=accessors, bufferViews=views,
                buffers=[dict(byteLength=len(out))])
    return tree, bytes(out)


def is_line_piece(item):
    return item.get('primitive_type') == 'lines' or item.get('placeholder', False)


def needs_normal_split(item):
    return not is_line_piece(item) and item.get('normals') is None


def line_color_for(item):
    return np.sqrt(np.clip(np.asarray(item['color'], dtype=float), 0, 1))


def write_models(base, pieces, options=None, business=None, progress=None):
    from contextlib import ExitStack
    import subprocess
    import tempfile
    from vue_mesh_merge import prepare_moniker_groups, merge_gltf_tree, merge_fbx
    settings = options or ExportOptions()
    if not settings.unknown_bbox_wireframe:
        pieces = [piece for piece in pieces if not piece.get('placeholder', False)]
    if not pieces:
        raise ValueError('No geometry remains to export after applying export options.')
    formats = settings.formats
    # GLB multiplies vertex and material colors; FBX viewers can use either.
    line_colors = {piece['name']: np.sqrt(np.clip(np.asarray(piece['color'],dtype=float),0,1))
                   for piece in pieces if is_line_piece(piece)}
    line_names = set(line_colors)
    progress = progress if isinstance(progress, ExportProgress) else ExportProgress(progress, base.stem)
    specs = []
    if settings.merge_moniker:
        progress('按 Moniker 对齐几何原点')
        pieces, specs = prepare_moniker_groups(pieces)
    if len({piece['name'] for piece in pieces}) != len(pieces):
        raise ValueError('duplicate geometry node names would overwrite scene nodes')
    export_pieces = []
    line_data = []
    with ExitStack() as stack:
        out = stack.enter_context(base.with_suffix('.obj').open('w', encoding='utf-8')) if 'obj' in formats else None
        if out:
            out.write('# VUE preview; original world coordinates, metres, Z up. No recentering.\n')
        vertex_offset = normal_offset = 1
        obj_owners = set()
        # Batch only surfaces without generator normals. Analytic primitives
        # already separate their hard rims and bypass this work entirely.
        # 0928 review fix: the CLI scratch budget (workload limit) propagates
        # into the split batch scheduler instead of being a post-hoc check.
        split_batch_budget = 64 * 1024 * 1024
        scratch_budget_mb = getattr(options, 'scratch_budget_mb', None)
        if scratch_budget_mb is not None:
            split_batch_budget = min(split_batch_budget, scratch_budget_mb * 2 ** 20)
        pending_splits = iter(split_hard_edges_many(
            [(item['local_vertices'], item['faces'])
             for item in pieces if needs_normal_split(item)],
            budget_bytes=split_batch_budget,
            progress=progress, downcast=True))
        for piece_index, item in enumerate(pieces):
            progress.count('处理网格法线与材质', piece_index, len(pieces))
            is_line = is_line_piece(item)
            if is_line:
                export_vertices, export_edges = item['local_vertices'], item['faces']
            else:
                if item.get('normals') is None:
                    export_vertices, export_faces, export_normals = next(pending_splits)
                else:
                    export_vertices = np.asarray(item['local_vertices'], dtype='<f4')
                    export_faces = np.asarray(item['faces'], dtype='<u4')
                    export_normals = np.asarray(item['normals'], dtype='<f4')
                # B3: publish the downcast arrays as the piece geometry so the
                # double-precision intermediates are not retained for the rest
                # of the export. origin stays float64 (world placement).
                item['local_vertices'] = export_vertices
                item['faces'] = export_faces
                export_pieces.append(dict(item, export_vertices=export_vertices,
                                          export_faces=export_faces, export_normals=export_normals))
            name, origin = item['name'], item['origin']
            if out:
                obj_name = item['moniker']+'__merged' if settings.merge_moniker else name
                lines = []
                if obj_name not in obj_owners:
                    lines.append(f'o {obj_name}'); obj_owners.add(obj_name)
                lines.extend('v %.9f %.9f %.9f' % tuple(point)
                             for point in export_vertices + origin)
                if is_line:
                    for edge in export_edges:
                        a,b = (int(index)+vertex_offset for index in edge)
                        lines.append(f'l {a} {b}')
                else:
                    lines.extend('vn %.9f %.9f %.9f' % tuple(normal)
                                 for normal in export_normals)
                    for face in export_faces:
                        points = [int(index)+vertex_offset for index in face]
                        normals = [int(index)+normal_offset for index in face]
                        lines.append('f ' + ' '.join(f'{point}//{normal}'
                                                   for point,normal in zip(points,normals)))
                    normal_offset += len(export_normals)
                vertex_offset += len(export_vertices)
                out.write('\n'.join(lines) + '\n')
            if not any(fmt in formats for fmt in ('glb','fbx')):
                continue
            if is_line:
                line_data.append(item)
            # solid pieces were already collected into export_pieces above.
        progress.count('处理网格法线与材质', len(pieces), len(pieces))
        def color_line_materials(tree):
            for mesh in tree['meshes']:
                color = line_colors.get(mesh.get('name'))
                if color is None:
                    continue
                material = dict(name=mesh['name'], pbrMetallicRoughness=dict(
                    baseColorFactor=color.tolist(), metallicFactor=0, roughnessFactor=1))
                if color[3] < 1:
                    material['alphaMode'] = 'BLEND'
                material_id = len(tree.setdefault('materials',[]))
                tree['materials'].append(material)
                for primitive in mesh['primitives']:
                    if primitive.get('mode') != 1:
                        raise ValueError('Expected line primitive')
                    primitive['material'] = material_id
        # Build the glTF tree and geometry buffer ONCE and share it between the
        # published GLB and the FBX-intermediate GLB.  Previously each
        # scene.export call re-ran trimesh's expensive buffer assembly and the
        # GLB was serialized twice (export + finalize's read_glb round-trip).
        from vue_glb_finalize import (encode_glb_trimesh,
                                      finalize_glb_from_tree)
        need_fbx_intermediate = 'fbx' in formats and (settings.fbx_mode == 'glb' or line_names)
        glb_tree = glb_binary = fbx_intermediate = None
        if 'glb' in formats or need_fbx_intermediate:
            if 'glb' in formats:
                progress('编码 GLB 几何缓冲区')
            glb_tree, glb_binary = build_export_tree(export_pieces, line_data)
            color_line_materials(glb_tree)
            if need_fbx_intermediate:
                fbx_intermediate = encode_glb_trimesh(glb_tree, glb_binary)
        if 'glb' in formats:
            # B5 (0928 review fix): output sizes are read from the FINAL
            # encoded GLB chunks after finalize — no recursive whole-tree
            # estimate (which counted characters, not UTF-8 bytes, and cost
            # ~1.1s on 15300) and no extra traversal just for logging.

            def post_rest(tree):
                if specs:
                    progress('合并元件节点')
                    merge_gltf_tree(tree, specs)
                if business is not None:
                    progress('组织 MDB2 业务树')
                    from vue_business import apply_business_tree
                    apply_business_tree(tree, business)
            post_rest(glb_tree)
            progress('合并同材质网格并整理 GLB' if settings.merge_moniker else '整理 GLB 坐标与元数据')
            published = finalize_glb_from_tree(glb_tree, glb_binary, settings.glb_up_axis, settings.merge_moniker)
            progress(f'写出 GLB: {base.with_suffix(".glb")}')
            base.with_suffix('.glb').write_bytes(published)
            json_len = struct.unpack_from('<I', published, 12)[0]
            bin_len = struct.unpack_from('<I', published, 20 + json_len)[0]
            progress('产物尺寸: GLB JSON chunk %.1f MB, BIN chunk %.1f MB'
                     % (json_len / 2 ** 20, bin_len / 2 ** 20))
        if 'fbx' in formats:
            progress('转换 FBX')
            fbx_path = base.parent / (base.name+'_compatible.fbx')
            def via_glb():
                # FBX-only exports never publish the intermediate GLB. Before
                # material-aware FBX merging, each part must retain its identity.
                with tempfile.TemporaryDirectory(prefix='vue_fbx_') as directory:
                    source = Path(directory)/'intermediate.glb'
                    if fbx_intermediate is not None:
                        payload = fbx_intermediate
                    else:
                        # write_fbx fallback for formats without an FBX
                        # intermediate: assemble one from the shared tree now.
                        fbx_tree, fbx_binary = build_export_tree(export_pieces, line_data)
                        color_line_materials(fbx_tree)
                        payload = encode_glb_trimesh(fbx_tree, fbx_binary)
                    source.write_bytes(payload)
                    args = [] if settings.reports else ['--no-report']
                    subprocess.run(helper_command('vue_fbx_from_exported_glb', source, fbx_path, *args), check=True, timeout=3600)
            if settings.fbx_mode == 'glb' or line_names:
                via_glb()
            else:
                try:
                    write_fbx(fbx_path, export_pieces)
                except subprocess.CalledProcessError:
                    via_glb()
            if specs:
                # FBX line Geometry has two-index segments and no face normals.
                # Keep those models separate while merging each Moniker's solid parts.
                fbx_specs = [dict(spec, parts=[name for name in spec['parts'] if name not in line_names])
                             for spec in specs]
                fbx_specs = [spec for spec in fbx_specs if spec['parts']]
                if fbx_specs:
                    merge_fbx(fbx_path, fbx_specs)
                translations = read_fbx_mesh_translations(fbx_path, [s['name'] for s in fbx_specs])
                for spec in fbx_specs:
                    if not np.array_equal(translations[spec['name']][1], spec['origin']):
                        raise ValueError('merged FBX origin mismatch')
                if settings.reports:
                    (fbx_path.with_suffix('.verification.json')).write_text(json.dumps(dict(
                        merged_meshes=len(fbx_specs), unmerged_line_meshes=len(line_names),
                        max_translation_error_m=0.0, material_slots_preserved=True),
                        indent=2), encoding='utf-8')
    if settings.reports and specs:
        (base.parent/'mesh_merge.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2),encoding='utf-8')


def export_sample(path, destination, options=None, progress=None):
    settings = options or ExportOptions()
    progress = progress if isinstance(progress, ExportProgress) else ExportProgress(progress, path.stem)
    progress('读取 VUE 与 Moniker 索引')
    parser = VuePipeParser(path)
    data = parser.data
    progress('解析记录与场景表')
    _, records, errors = read_records(data, indexes=(parser.tokens, parser.ref_by_tok))
    native = {r['record_offset']: r for r in read_scene_tables(data, records)['records']}
    business = None
    business_future = None
    business_pool = None
    if 'glb' in settings.formats and settings.mdb2 is not None:
        from vue_business import resolve_database, load_business
        database = resolve_database(path, settings.mdb2)
        if database is not None:
            progress(f'读取 MDB2 属性: {database}')
            # Parse the MDB2 in a background process so it overlaps with the
            # geometry build below; joined just before write_models.
            try:
                from concurrent.futures import ProcessPoolExecutor
                business_pool = ProcessPoolExecutor(max_workers=1)
                business_future = business_pool.submit(load_business, database, records, native, path.stem)
            except Exception:
                business = load_business(database, records, native, path.stem)
                progress(f'MDB2 属性加载完成: {len(business["objects"])} 个对象')
        else:
            print('No same-stem MDB2 found; exporting VUE geometry only.', flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    input_sha256 = hashlib.sha256(data).hexdigest()
    if settings.reports:
        material_report = dict(input_sha256=input_sha256,
            source='VUE only; record+0x3c absolute material pointer; no reference FBX input',
            fields={'diffuse_rgb':'material+0x30 f32[3], duplicate +0x58',
                    'opacity':'material+0x54 f32, duplicate +0x64; 1 opaque, 0.5 half opacity'},
            records=list(native.values()))
        (destination/'materials.json').write_text(json.dumps(material_report, ensure_ascii=False, indent=2), encoding='utf-8')
    pieces, rows, gaps, counts = [], [], [], Counter()
    reference_rows = []
    profile_shell_rows = []
    # P1: actual per-family segment counts observed during this export, for the
    # log/manifest (requested N lives in settings.segments).
    segment_usage = defaultdict(set)
    for record_index, record in enumerate(records):
        progress.count('构建几何', record_index, len(records))
        rs, moniker = record['record_offset'], canonical(record['moniker'])
        meta = native[rs]
        collected = defaultdict(list)
        pending_references = []
        native_disks = []
        needs_record_placeholder = False
        blocks = parser.scan_cylinders(rs, record['record_end'])
        allowed = set()
        walks = {}
        root_matrices={group['index']:read_matrix_pair(data,group)[0]
                       for group in record['display_groups']}
        for group in record['display_groups']:
            elements, status = walk_group_elements(data, group, rs)
            walks[group['index']] = elements
            for e in elements:
                if e['kind'] == 'nested_node':
                    allowed.update(e['inner_d0_markers'])
                    if not e.get('inner_blocks_complete', False):
                        needs_record_placeholder = True
                        gaps.append(dict(moniker=moniker, record_offset=rs, group=group['index'],
                                         offset=e['off_rel_record'],
                                         reason=e['inner_walk_status']))
                elif (e['type_raw'], e['size_raw']) == (0x1f, 0xd0):
                    allowed.add(e['off_rel_record'])
            if status not in ('exact_end', 'exact_end_with_node'):
                gaps.append(dict(moniker=moniker, record_offset=rs, group=group['index'], reason=status))
        blocks = [b for b in blocks if b['off'] in allowed]
        for b in blocks:
            group = next(g for g in record['display_groups'] if g['header_start'] <= rs+b['off'] < g['end'])
            b.update(group_index=group['index'], display_code_raw=group['display_code_raw'])
        parsed = dict(rec_start=rs, display_groups=record['display_groups'], blocks=blocks,
                      body_blocks=[b for b in blocks if b['display_code_raw'] != 69],
                      insulation_blocks=[b for b in blocks if b['display_code_raw'] == 69])
        blocks = worldize_vue_blocks(parser, parsed, walks=walks)['blocks']

        def signature_at(offset):
            """Read a tag signature from a VUE-relative block offset.

            Record-level placeholders use offset zero, so they intentionally
            have no block signature.  For every actual tag offset this keeps
            the name suffix sourced from VUE bytes, including unsupported
            blocks that fall through to a diagnostic bbox.
            """
            if not isinstance(offset, (int, np.integer)) or int(offset) <= 0:
                return None
            offset = int(offset)
            try:
                type_raw, size_raw = struct.unpack_from('<2I', data, rs + offset)
            except struct.error:
                return None
            if not (0 < type_raw <= 0xffff and 8 <= size_raw <= 0x1000000):
                return None
            return type_raw, size_raw

        active_node = None

        def add(v, f, group, kind, offset, error=0.0, tentative=False, placement='decoded_fields', part='', sampling=0.0, normals=None):
            if active_node is not None:
                part = '__node_%s' % active_node + part
            if not np.all(np.isfinite(v)) or len(v) == 0 or len(f) == 0:
                raise ValueError('nonfinite or empty geometry')
            if np.max(np.abs(v)) > 1e9:
                raise ValueError('unusable coordinate magnitude')
            if normals is None and kind in {
                    'polyline_face','circular_face','planar_quad','native_planar_face',
                    'annular_surface','planar_polygon','planar_face_with_holes',
                    'planar_face_with_hole'}:
                normals = primitive_geometry.planar_normals(v,f)
            if normals is None and kind == 'cylindrical_hole_surface':
                # Floor contexts retain the established two-ring geometry layout.
                rings=v.reshape(2,-1,3)
                normals=-(rings-rings.mean(axis=1)[:,None,:]).reshape(-1,3)
                normals/=np.linalg.norm(normals,axis=1)[:,None]
            signature = signature_at(offset)
            # Keep optional generator normals alongside topology and sampling error.
            # Unknown/complex surfaces retain the common split path.
            collected[(group, tentative, part, signature)].append((v, f, float(sampling), normals))
            rows.append(dict(moniker=moniker, record_offset=rs, group=group, kind=kind,
                             offset=offset, vue_signature=vue_signature_label(signature),
                             bbox_error_m=error, tentative=tentative, placement=placement, mesh_part=part))
            counts[kind] += 1

        recovered_d0 = set()
        for b in blocks:
            try:
                if not math.isfinite(b['radius']) or b['radius'] <= 0:
                    raise ValueError('invalid radius')
                v, f, normals = cylinder(b, settings.segments, with_normals=True)
                # 0928 P1 review fix: placement choice, bbox-centre anchoring
                # and the tentative flag are decided from the ANALYTIC native
                # bounds of the cylinder, never from the sampled mesh bbox.
                # A coarse 16-segment chord mesh must not move a valid block
                # across these thresholds (plan §2.2: separate sampling error
                # from placement error).  ``err`` remains the measured sampled
                # bbox error and is only recorded in geometry.csv.
                native_box = cylinder_bounds(b['point'], b['axis'], b['radius'], b['dirv'])
                native_err = box_error(native_box, b['bbox'])
                err = bounds_error(v, b['bbox'])
                # For direct blocks also try the existing root placement rule.
                if b.get('coordinate_space') == 'world':
                    matrix = root_matrices[b['group_index']]
                    placed_native = cylinder_bounds(b['point'], b['axis'], b['radius'], b['dirv'], matrix)
                    placed_err = box_error(placed_native, b['bbox'])
                    if placed_err + 1e-8 < native_err:
                        placed = (np.c_[v, np.ones(len(v))] @ matrix)[:, :3]
                        linear = matrix[:3, :3]
                        try:
                            normals = normals @ np.linalg.inv(linear).T
                            normals /= np.linalg.norm(normals, axis=1)[:, None]
                            if np.linalg.det(linear) < 0:
                                f = f[:, ::-1].copy()
                        except np.linalg.LinAlgError:
                            normals = None  # Degenerate transforms retain the split path.
                        v, err = placed, bounds_error(placed, b['bbox'])
                        native_box, native_err = placed_native, placed_err
                anchored = False
                if native_err > 0.05:
                    box = np.asarray(b['bbox'])
                    shift = (box[:3]+box[3:]-native_box[:3]-native_box[3:])/2
                    shifted = v+shift
                    shifted_err = box_error(native_box+np.r_[shift, shift], box)
                    if shifted_err < native_err:
                        v, err, anchored = shifted, bounds_error(shifted, box), True
                        native_err = shifted_err
                add(v, f, b['display_code_raw'], 'disk' if b['height'] < 1e-10 else 'cylinder',
                    b['off'], err, anchored or native_err > max(0.005, b['radius']*0.1),
                    'bbox_center_preview' if anchored else 'decoded_fields',
                    sampling=full_turn_chord_error(b['radius'], settings.segments), normals=normals,
                    part='__node_%s__cylinders' % b['transform_node_offset']
                    if b.get('transform_node_offset') is not None else '__cylinders')
                recovered_d0.add(b['off'])
                segment_usage['cylinder'].add((settings.segments,))
            except (ValueError, FloatingPointError) as exc:
                gaps.append(dict(moniker=moniker, record_offset=rs, offset=b['off'], reason=str(exc)))
        for group in record['display_groups']:
            elements = walks[group['index']]
            root = root_matrices[group['index']]
            node = next((e for e in elements if e['kind'] == 'nested_node'), None)
            floor_contexts = {}
            scopes = [(None, elements, root)] + [
                (e['off_rel_record'], e['inner_blocks'],
                 np.asarray(e['matrix_a_rowmajor']).reshape(4, 4))
                for e in elements if e['kind'] == 'nested_node']
            for scope, children, matrix in scopes:
                if any(e.get('kind') == 'tag_block' and
                       (e['type_raw'], e['size_raw']) == (0x0e, 3528)
                       for e in children):
                    try:
                        floor_contexts[scope] = floor_shell_context_from_vue(
                            data, children, rs, matrix)
                    except (ValueError, struct.error, FloatingPointError):
                        # Unsupported layouts retain per-block diagnostics.
                        pass
            # Every leaf uses the same decoder regardless of its owning scope.
            leaves = []
            for element in elements:
                if element['kind'] == 'nested_node':
                    leaves.extend((child, element) for child in element['inner_blocks'])
                else:
                    leaves.append((element, None))
            group_root = root
            for e, owner in leaves:
                active_node = owner['off_rel_record'] if owner else None
                floor_context = floor_contexts.get(active_node)
                root = (np.asarray(owner['matrix_a_rowmajor']).reshape(4, 4)
                        if owner else group_root)
                if e['kind'] != 'tag_block' or (e['type_raw'], e['size_raw']) == (0x1f, 0xd0):
                    continue
                off = rs + e['off_rel_record']
                signature = (e['type_raw'], e['size_raw'])
                try:
                    rawbox = float_view(data, off+8, 6)
                    if signature in [(15,168),(40,112),(3,152)]:
                        points=native_reference(data,off,root)
                        pending_references.append((e['off_rel_record'],group['display_code_raw'],points));continue
                    if signature == (29, 224):
                        v, f, normals = sphere_from_vue(data, off, root, with_normals=True)
                        add(v, f, group['display_code_raw'], 'sphere', e['off_rel_record'],
                            bounds_error(v, rawbox), placement='vue_native_sphere', part='__spheres', normals=normals)
                    elif signature in ((4, 200), (4, 224)):
                        v, f, normals = plate_solid_from_vue(data, off, root, rs, with_normals=True)
                        add(v, f, group['display_code_raw'], 'plate_solid', e['off_rel_record'],
                            bounds_error(v, rawbox), part='__plates', normals=normals)
                    elif signature == (30, 232):
                        v, f, normals = torus_from_vue(data, off, root, settings.segments, with_normals=True)
                        segment_usage['torus'].add((settings.segments, settings.segments))
                        add(v, f, group['display_code_raw'], 'torus', e['off_rel_record'],
                            bounds_error(v, rawbox), part='__torus',
                            sampling=torus_sampling_error(data, off, settings.segments), normals=normals)
                    elif signature in ((33, 392), (33, 440)):
                        v, f = ruled_polylines_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'ruled_polylines', e['off_rel_record'],
                            bounds_error(v, rawbox), part='__ruled_polylines')
                    elif signature == (32, 296):
                        v, f = swept_native_profile_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'revolved_polyline', e['off_rel_record'],
                            bounds_error(v, rawbox), part='__revolved_polylines')
                    elif signature == (32,256):
                        try:
                            v,f,normals=ellipsoid_head_from_vue(data,off,rs,root, settings.segments, with_normals=True)
                            segment_usage['ellipsoid_head'].add(
                                (settings.segments, ELLIPSOID_MERIDIAN_SAMPLES))
                            kind='ellipsoid_head'
                            surface_error=ellipsoid_head_sampling_error(data, off, settings.segments)
                        except ValueError:
                            v,f=revolved_native_boundary_from_vue(data,off,rs,root)
                            kind='revolved_arc_surface'
                            normals=None
                            surface_error=0.0
                        add(v,f,group['display_code_raw'],kind,e['off_rel_record'],bounds_error(v,rawbox),False,
                            sampling=surface_error,part='__'+kind,normals=normals)
                    elif signature in ((32,320),(32,344)):
                        v,f=revolved_native_boundary_from_vue(data,off,rs,root)
                        add(v,f,group['display_code_raw'],'revolved_native_polyline',e['off_rel_record'],bounds_error(v,rawbox),False,part='__revolved_native_polyline')
                    elif signature == (0x20, 768):
                        v, f = swept_native_profile_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'revolved_spline_surface',
                            e['off_rel_record'], bounds_error(v,rawbox), False,
                            placement='vue_axis_spline_revolution',part='__revolved_spline_surfaces')
                    elif signature == (0x20, 416):
                        v, f, normals = revolved_profile_416_from_vue(data, off, rs, root, settings.segments, with_normals=True)
                        segment_usage['revolved_profile_416'].add((settings.segments, 10))
                        add(v, f, group['display_code_raw'],
                            'revolved_profile_surface', e['off_rel_record'],
                            bounds_error(v, rawbox), False,
                            placement='vue_full_revolution_profile',
                            sampling=revolved_profile_416_sampling_error(
                                data, off, settings.segments),
                            part='__revolved_profile_surfaces', normals=normals)
                    elif signature == (33,280):
                        v,f,normals=ruled_circles_from_vue(data,off,rs,root,with_normals=True)
                        add(v,f,group['display_code_raw'],'eccentric_reducer',e['off_rel_record'],bounds_error(v,rawbox),False,part='__ruled_circles',normals=normals)
                    elif signature in [(14,368),(14,312)]:
                        try:
                            v,f=boundary_face_from_vue(data,off,rs,root)
                        except ValueError:
                            if signature[1] != 368:
                                raise
                            v,f=planar_loops_from_vue(data,off,rs,root)
                        if signature[1]==312:native_disks.append(v)
                        add(v,f,group['display_code_raw'],'polyline_face' if signature[1]==368 else 'circular_face',e['off_rel_record'],bounds_error(v,rawbox),False,part='__boundary_faces')
                    elif signature == (0xe, 512):
                        v,f=planar_quad(data,off,rs,root)
                        add(v,f,group['display_code_raw'],'planar_quad',e['off_rel_record'],
                            bounds_error(v,rawbox),False,part='__planar_faces')
                    elif signature == (0x0e, 3528):
                        if floor_context is None or e['off_rel_record'] not in floor_context['cap_meshes']:
                            raise ValueError('unsupported 0xe/3528 floor layout')
                        v, f = floor_context['cap_meshes'][e['off_rel_record']]
                        add(v, f, group['display_code_raw'], 'planar_face_with_holes',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            placement='vue_floor_caps_with_holes', part='__floor_faces')
                    elif signature == (0xe, 456):
                        v, f = planar_face_456_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'planar_quad',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            part='__planar_faces')
                    elif signature == (0x0e, 408):
                        v, f = annulus_from_vue(data, off, root)
                        add(v, f, group['display_code_raw'], 'annular_surface', e['off_rel_record'],
                            bounds_error(v, rawbox), False,
                            placement='vue_annulus_boundaries', part='__annular_surfaces')
                    elif signature == (0x0e, 584):
                        if (struct.unpack_from('<I', data, off + 0x9c)[0] == 1 and
                                struct.unpack_from('<I', data, off + 0xcc)[0] == 5):
                            v, f = planar_loops_from_vue(data, off, rs, root)
                            add(v, f, group['display_code_raw'], 'native_planar_face',
                                e['off_rel_record'], bounds_error(v, rawbox), False,
                                part='__native_planar_faces')
                        else:
                            boundary_reference_from_vue(data, off, rs, root)
                            rows.append(dict(moniker=moniker, record_offset=rs,
                                             group=group['display_code_raw'],
                                             kind='native_boundary_reference',
                                             offset=e['off_rel_record'],
                                             vue_signature=vue_signature_label(signature),
                                             bbox_error_m=0.0,
                                             tentative=False, placement='vue_boundary_reference',
                                             mesh_part='__boundary_references'))
                            counts['native_boundary_reference'] += 1
                    elif signature in ((0xe, 440), (0xe, 656)):
                        v, f = planar_polygon_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'planar_polygon',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            part='__planar_faces')
                    elif signature in [(0x18, 752), (0x18, 760), (0x18, 2128), (0x18, 2464), (0x18,2288), (0x18,2488)]:
                        v, f = nurbs_surface_from_vue(data, off, root)
                        kind = {
                            (0x18, 752): 'stair_surface_patch',
                            (0x18, 760): 'cylinder_surface',
                            (0x18, 2128): 'handrail_surface',
                            (0x18, 2464): 'handrail_surface',
                            (0x18, 2288): 'handrail_surface',
                            (0x18, 2488): 'handrail_surface',
                        }[signature]
                        part = {
                            (0x18, 752): '__stair_surface_patches',
                            (0x18, 760): '__cylinder_surfaces',
                            (0x18, 2128): '__handrail_surfaces',
                            (0x18, 2464): '__handrail_surfaces',
                            (0x18, 2288): '__handrail_surfaces',
                            (0x18, 2488): '__handrail_surfaces',
                        }[signature]
                        add(v, f, group['display_code_raw'], kind, e['off_rel_record'],
                            bounds_error(v, rawbox), False,
                            placement='vue_nurbs_surface', part=part)
                    elif signature in [(0x1f, 640), (0x1f, 848)]:
                        v, f = profile_extrusion_from_vue(data, off, signature[1], rs, root)
                        solid = bool(struct.unpack_from('<I', data, off + 0x6c)[0] & 0x10000)
                        add(v, f, group['display_code_raw'],
                            'profile_extrusion_solid' if solid else 'profile_extrusion_surface',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            placement='vue_profile_extrusion',
                            part='__profile_extrusion_solids' if solid else '__profile_extrusion_surfaces')
                    elif signature in [(0x20, 352), (0x20, 392)]:
                        v, f = revolved_profile_from_vue(data, off, signature[1], rs, root)
                        add(v, f, group['display_code_raw'], 'revolved_profile_solid',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            placement='vue_meridian_revolution', part='__revolved_solids')
                    elif signature == (0xe, 1432):
                        v, f = planar_face_with_holes_from_vue(data, off, signature[1], rs, root)
                        add(v, f, group['display_code_raw'], 'planar_face_with_hole',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            placement='vue_annular_face', part='__annular_faces')
                    elif signature == (0x17,440):
                        if floor_context is not None and e['off_rel_record'] in floor_context['circle_meshes']:
                            v, f = floor_context['circle_meshes'][e['off_rel_record']]
                            add(v, f, group['display_code_raw'], 'cylindrical_hole_surface',
                                e['off_rel_record'], bounds_error(v, rawbox), False,
                                placement='vue_inline_circular_hole_wall', part='__floor_hole_walls')
                        else:
                            try:
                                v,f,normals=frustum_from_vue(data,off,rs,root,with_normals=True)
                                kind,part='frustum_surface','__frusta'
                            except ValueError:
                                if not _inline_cylinder_440_layout(data,off,rs):
                                    raise
                                v,f,normals=cylindrical_surface_440_from_vue(data,off,rs,root,with_normals=True)
                                kind,part='cylindrical_hole_surface','__floor_hole_walls'
                            add(v,f,group['display_code_raw'],kind,e['off_rel_record'],
                                bounds_error(v,rawbox),False,part=part,normals=normals)
                    elif signature == (0x17,1464):
                        v,f,normals=cylinder_surface_from_vue(data,off,rs,root,with_normals=True)
                        add(v,f,group['display_code_raw'],'cylinder_surface_nurbs',
                            e['off_rel_record'],bounds_error(v,rawbox),False,
                            placement='vue_two_rational_boundary_curves',
                            part='__cylinder_surface_nurbs',normals=normals)
                    elif signature == (0x17,424):
                        v,f,normals=cone_from_vue(data,off,rs,root, settings.segments, with_normals=True)
                        segment_usage['cone'].add((settings.segments,))
                        add(v,f,group['display_code_raw'],'cone_surface',e['off_rel_record'],
                            bounds_error(v,rawbox),False,
                            sampling=cone_sampling_error(data, off, settings.segments),
                            part='__cones',normals=normals)
                    elif signature == (0x1f, 264):
                        view = tag_block_views(data, e, rs)['extrusion_candidate']
                        v, f, normals = extrusion(view['profile_points'], view['extrusion_vector'], root, with_normals=True)
                        add(v, f, group['display_code_raw'], 'box', e['off_rel_record'],
                            bounds_error(v, rawbox), True, part='__boxes', normals=normals)
                    elif signature == (0x1f, 464):
                        v, f = curved_strip_solid_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'curved_strip_solid',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            placement='vue_profile_extrusion', part='__curved_strip_solids')
                    elif signature == (0x1f, 720):
                        v, f, normals = polygon_extrusion_from_vue(data, off, root, with_normals=True)
                        add(v, f, group['display_code_raw'], 'stair_step', e['off_rel_record'],
                            bounds_error(v, rawbox), False,
                            placement='vue_profile_extrusion', part='__stair_steps', normals=normals)
                    elif signature == (0x1f, 2192):
                        v, f = cage_hoop_from_vue(data, off, signature[1], rs, root)
                        add(v, f, group['display_code_raw'], 'cage_hoop', e['off_rel_record'],
                            bounds_error(v, rawbox), False,
                            placement='vue_profile_extrusion', part='__cage_hoops')
                    elif signature == (0x20, 464):
                        v, f = curved_strip_face_from_vue(data, off, rs, root)
                        add(v, f, group['display_code_raw'], 'curved_strip_surface',
                            e['off_rel_record'], bounds_error(v, rawbox), False,
                            placement='vue_line_boundary', part='__curved_strip_surfaces')
                    elif signature == (0x20, 240):
                        transforms = [root]
                        if node and owner is None:
                            transforms.append(np.array(node['matrix_a_rowmajor']).reshape(4, 4))
                        center = float_view(data, off+0x50, 3)
                        plane_normal = float_view(data, off+0x68, 3)
                        angle = struct.unpack_from('<d', data, off+0x80)[0]
                        start_point = float_view(data, off+0x98, 3)
                        end_axis = float_view(data, off+0xb0, 3)
                        tube = arc_tube_radius_from_vue(data, off)
                        try:
                            # The 0x20/240 payload carries the complete native
                            # arc frame.  Some exporters cache a smaller box
                            # here (the 90-degree elbow is the clearest case),
                            # so the cache is useful for choosing between the
                            # root and nested placements but is not a rejection
                            # criterion for an otherwise valid native arc.
                            # 0928 P1 review fix: pick the root/nested
                            # placement by the ANALYTIC swept-tube bounds, not
                            # by the sampled mesh bbox, so the choice cannot
                            # change with the requested segments.
                            bounds = [arc_tube_precise_bounds(
                                center, start_point, end_axis, plane_normal,
                                tube, angle, m) for m in transforms]
                            chosen = min(range(len(transforms)),
                                         key=lambda i: box_error(bounds[i], rawbox))
                            ring_radius = float(np.linalg.norm(start_point-center))
                            if meta['type_name'] == 'Weld':
                                # Recognised weld rings export as the circular
                                # LINE of their native centreline: the tube
                                # surface, its normals and the split are never
                                # generated.
                                lv, le = weld_ring_line(center, start_point, end_axis,
                                                        plane_normal, angle,
                                                        transforms[chosen], settings.segments)
                                add(lv, le, group['display_code_raw'], 'weld_line',
                                    e['off_rel_record'], bounds_error(lv, rawbox), False,
                                    placement='weld_ring_line', part='__weld_line')
                                rows[-1].update(virtual_geometry=True, native_type='Weld',
                                                ring_radius_m=ring_radius)
                            else:
                                v, f, normals = arc_tube_precise(center, start_point, end_axis,
                                                        plane_normal, tube, angle,
                                                        transforms[chosen], settings.segments, with_normals=True)
                                segment_usage['arc_tube'].add(
                                    (arc_path_segments(angle, settings.segments), settings.segments))
                                add(v, f, group['display_code_raw'], 'arc_tube', e['off_rel_record'],
                                    bounds_error(v, rawbox), False,
                                    placement='vue_native_arc_parameters',
                                    sampling=arc_tube_sampling_error(
                                        ring_radius, tube, angle, settings.segments),
                                    part='__arc_tube', normals=normals)
                        except (ValueError, FloatingPointError):
                            if meta['type_name'] == 'Weld':
                                raise
                            # Fall back to the bbox-constrained envelope hypothesis.
                            radius = abs(struct.unpack_from('<d', data, off+0x50)[0])
                            if tube <= 0:
                                raise ValueError('arc payload has zero tube radius; likely complete-sweep or other variant')
                            if not (0 < radius < 10000 and 0 < tube < 1000
                                    and 0 < abs(angle) <= 2*math.pi+1e-6):
                                raise ValueError('unusable arc parameters')
                            # Avoid world coordinates misread as radius in other payload variants.
                            scale = max(float(np.linalg.norm(rawbox[3:]-rawbox[:3])), 0.001)
                            if radius > scale*4 or tube > scale:
                                raise ValueError('arc hypothesis conflicts with bbox; placeholder only')
                            # 0928 P1 review fix: placement and acceptance of
                            # the hypothesis use its ANALYTIC swept-tube
                            # bounds.  The output sampling must not decide
                            # whether a legal elbow is exported — a coarse
                            # 16-segment mesh bbox missed the rotated extremes
                            # (17.9mm > 10mm) while 24/32 passed the very same
                            # native placement.
                            bounds = [arc_tube_bounds(radius, tube, angle, m)
                                      for m in transforms]
                            chosen = min(range(len(transforms)),
                                         key=lambda i: box_error(bounds[i], rawbox))
                            v, f, normals = arc_tube(radius,tube,angle,transforms[chosen],settings.segments,with_normals=True)
                            if box_error(bounds[chosen], rawbox) > max(0.005, tube*.1):
                                raise ValueError('arc candidate placement unverified; placeholder only')
                            add(v, f, group['display_code_raw'], 'arc_tube_candidate', e['off_rel_record'],
                                bounds_error(v, rawbox), True,
                                sampling=arc_tube_sampling_error(radius, tube, angle, settings.segments),
                                part='__arc_tube', normals=normals)
                            segment_usage['arc_tube'].add(
                                (arc_path_segments(angle, settings.segments), settings.segments))
                    elif signature[0] == 14:
                        v,f=planar_loops_from_vue(data,off,rs,root)
                        add(v,f,group['display_code_raw'],'native_planar_face',e['off_rel_record'],
                            bounds_error(v,rawbox),part='__native_planar_faces')
                    elif signature[0] == 31:
                        v,f=native_profile_extrusion_from_vue(data,off,rs,root)
                        solid=bool(struct.unpack_from('<I',data,off+0x6c)[0] & 0x10000)
                        add(v,f,group['display_code_raw'],
                            'native_profile_solid' if solid else 'native_profile_surface',e['off_rel_record'],
                            bounds_error(v,rawbox),part='__native_profile_solids' if solid else '__native_profile_surfaces')
                    else:
                        raise ValueError('unsupported payload 0x%x/%d' % signature)
                except (ValueError, struct.error, FloatingPointError) as exc:
                    gaps.append(dict(moniker=moniker, record_offset=rs, group=group['index'],
                                     offset=e['off_rel_record'], reason=str(exc)))
                    try:
                        rawbox = struct.unpack_from('<6d', data, off+8)
                        v, f = bbox_cage(rawbox)
                        add(v, f, group['display_code_raw'], 'unparsed_block_bbox',
                            e['off_rel_record'], tentative='placeholder')
                    except (ValueError, struct.error):
                        needs_record_placeholder = True
        active_node = None
        # Native walls and annular faces must form one outward-facing shell.
        profile_entries=[(key,index,v,f) for key,items in collected.items()
                         if key[3] in ((31,640),(31,848),(14,1432),(31,288),(31,496),(14,728))
                         and key[1]!='placeholder'
                         for index,(v,f,_sampling,_normals) in enumerate(items)]
        if profile_entries:
            try:
                fixed,shells=close_profile_shells([(v,f,key[3][0]==31)
                                      for key,index,v,f in profile_entries])
                for (key,index,v,f),new_faces in zip(profile_entries,fixed):
                    old=collected[key][index]
                    normals=(primitive_geometry.planar_normals(v,new_faces)
                             if old[3] is not None else None)
                    collected[key][index]=(v,new_faces,old[2],normals)
                profile_shell_rows.extend(dict(moniker=moniker,record_offset=rs,**shell)
                                          for shell in shells)
            except ValueError as exc:
                gaps.append(dict(moniker=moniker,record_offset=rs,reason=str(exc)))
        # Native circular faces replace coincident procedural cylinder caps.
        # Keep cylinder side walls and avoid duplicate coplanar surfaces.
        for key in list(collected):
            if not key[2].endswith('__cylinders') or not native_disks:continue
            kept=[]
            for v,f,sampling,normals in collected[key]:
                tri=v[f];remove=np.zeros(len(f),dtype=bool)
                for disk in native_disks:
                    center=disk[-1];radius=np.linalg.norm(disk[0]-center)
                    normal=unit(np.cross(disk[0]-center,disk[1]-center))
                    delta=tri-center
                    coplanar=np.max(abs(delta@normal),axis=1)<1e-8
                    inside=np.max(np.linalg.norm(delta,axis=2),axis=1)<=radius+1e-8
                    remove|=coplanar&inside
                if np.any(~remove):kept.append((v,f[~remove],sampling,normals))
            if kept:collected[key]=kept
            else:del collected[key]
        # Opposite native disks at a proven head/cylinder join are internal caps.
        internal=internal_head_join_disks(native_disks,
            [mesh for (g,t,p,s),items in collected.items() if p.endswith('__ellipsoid_head') for mesh in items],
            [mesh for (g,t,p,s),items in collected.items() if p.endswith('__cylinders') for mesh in items])
        for key in list(collected):
            if not key[2].endswith('__boundary_faces'):continue
            kept=[(v,f,sampling,normals) for v,f,sampling,normals in collected[key]
                  if not any(v is disk for disk in internal)]
            if kept:collected[key]=kept
            else:del collected[key]
        real_meshes=[mesh[:3] for (g,t,p,s),items in collected.items() if t!='placeholder' for mesh in items if mesh[1].shape[1] == 3]
        # References to decoded internal caps remain understood after export suppression.
        # The 96-sample native circular face keeps its fixed legacy sampling.
        real_meshes += [(d,np.array([(i,(i+1)%96,96) for i in range(96)]),0.0) for d in internal]
        if pending_references:
            real_meshes = [SurfaceQuery(*mesh) for mesh in real_meshes]
        for offset,group,points in pending_references:
            covered=references_on_surfaces(points,real_meshes)
            reference_rows.append(dict(moniker=moniker,record_offset=rs,group=group,offset=offset,signature=struct.unpack_from('<2I',data,rs+offset),world_samples=points.tolist(),covered_by_decoded_surfaces=covered))
            if not covered:
                rawbox=float_view(data, rs+offset+8, 6);v,f=bbox_cage(rawbox)
                add(v,f,group,'unparsed_reference_bbox',offset,tentative='placeholder')
                gaps.append(dict(moniker=moniker,record_offset=rs,offset=offset,reason='reference not covered by decoded surfaces'))
        if allowed - recovered_d0:
            needs_record_placeholder = True
        if needs_record_placeholder or not collected:
            try:
                v, f = bbox_cage(record['bbox_world'])
                add(v, f, -1, 'unparsed_record_bbox', 0, tentative='placeholder')
            except ValueError as exc:
                gaps.append(dict(moniker=moniker, record_offset=rs, reason=str(exc)))
        for (group, tentative, part, signature), meshes in collected.items():
            vertex_count = sum(len(mesh[0]) for mesh in meshes)
            face_count = sum(len(mesh[1]) for mesh in meshes)
            v = np.empty((vertex_count, 3), dtype=float)
            f = np.empty((face_count, meshes[0][1].shape[1]), dtype=np.uint32)
            normals = (np.empty((vertex_count, 3), dtype=np.float32)
                       if all(mesh[3] is not None for mesh in meshes) else None)
            vo = fo = 0
            for vertices, faces, _sampling, generated_normals in meshes:
                nv, nf = len(vertices), len(faces)
                v[vo:vo + nv] = vertices
                np.add(faces, vo, out=f[fo:fo + nf], casting='unsafe')
                if normals is not None:
                    normals[vo:vo + nv] = generated_normals
                vo += nv
                fo += nf
            low,high=v.min(0),v.max(0)
            origin = (low+high)/2
            color = piece_color(meta, group)
            if part.endswith('__weld_line'):
                color = [0.0, 0.0, 0.0, color[3]]
            name = f'{moniker}__R{rs}__G{group}' + part + vue_signature_suffix(signature)
            if tentative is True:
                name += '__candidate'
            if tentative == 'placeholder':
                color = [1.0, 0.10, 0.65, 1.0]
                name = f'{moniker}__R{rs}__G{group}__UNPARSED_BBOX' + part + vue_signature_suffix(signature)
            pieces.append(dict(name=name, moniker=moniker, record_offset=rs, discipline=meta['discipline_name'],
                               group=group, placeholder=tentative == 'placeholder',
                               primitive_type='lines' if f.shape[1] == 2 else 'triangles',
                               local_vertices=v-origin, faces=f, origin=origin, color=color, normals=normals,
                               bounds=np.array([low,high]),
                               sampling_error=max([mesh[2] for mesh in meshes], default=0.0)))
            # 0928 review fix: check the scratch workload estimate periodically
            # DURING the geometry loop (early rejection), not once afterwards.
            if (settings is not None and getattr(settings, 'scratch_budget_mb', None) is not None
                    and len(pieces) % 256 == 0):
                demand_so_far = sum(split_scratch_estimate(len(p['local_vertices']), len(p['faces']))
                                    for p in pieces if needs_normal_split(p))
                if demand_so_far > settings.scratch_budget_mb * 2 ** 20:
                    raise SystemExit('scratch budget exceeded: geometry split scratch estimate reached '
                                     '%.0f MiB after %d pieces, over --scratch-budget-mb %d '
                                     '(workload budget, not a process memory cap); raise the budget '
                                     'or narrow the export'
                                     % (demand_so_far / 2 ** 20, len(pieces), settings.scratch_budget_mb))
    # Point/line references may lie on a different object's decoded face.
    # Resolve geometrically without merging object ownership or adding solids.
    progress.count('构建几何', len(records), len(records))
    progress('检查参考边界覆盖')
    solid_pieces=[piece for piece in pieces if not is_line_piece(piece)]
    lows=np.array([piece['bounds'][0] for piece in solid_pieces]).reshape(-1,3)
    highs=np.array([piece['bounds'][1] for piece in solid_pieces]).reshape(-1,3)
    sampling=np.array([piece.get('sampling_error', 0.0) for piece in solid_pieces])
    lows-=sampling[:,None]
    highs+=sampling[:,None]
    resolved=set()
    surface_queries = OrderedDict()

    def scene_queries(indices):
        for index in indices:
            if index not in surface_queries:
                piece=solid_pieces[index]
                surface_queries[index]=SurfaceQuery(piece['local_vertices'],piece['faces'],
                    piece.get('sampling_error',0.0),origin=piece['origin'],bounds=piece['bounds'])
            surface_queries.move_to_end(index)
            query=surface_queries[index]
            if len(surface_queries)>32:
                surface_queries.popitem(last=False)
            yield query

    for reference in reference_rows:
        if reference['covered_by_decoded_surfaces'] or not solid_pieces:
            continue
        points=np.asarray(reference['world_samples'])
        candidates=np.flatnonzero(np.all(lows<=points.max(0)+1e-7,axis=1)&
                                  np.all(highs>=points.min(0)-1e-7,axis=1))
        if references_on_surfaces(points,scene_queries(candidates)):
            reference['covered_by_decoded_surfaces']=True
            reference['coverage_scope']='decoded scene surfaces; ownership unchanged'
            resolved.add((reference['record_offset'],reference['offset']))
    if resolved:
        removed=[row for row in rows if row['kind']=='unparsed_reference_bbox'
                 and (row['record_offset'],row['offset']) in resolved]
        rows=[row for row in rows if row not in removed]
        gaps=[gap for gap in gaps if not (gap['reason']=='reference not covered by decoded surfaces'
              and (gap['record_offset'],gap.get('offset')) in resolved)]
        counts['unparsed_reference_bbox']-=len(removed)
        for row in removed:
            key=(row['record_offset'],row['group'],row['vue_signature'])
            if any((other['record_offset'],other['group'],other['vue_signature'])==key
                   and other['tentative']=='placeholder' for other in rows):
                continue
            suffix=vue_signature_suffix(tuple(struct.unpack_from('<2I',data,row['record_offset']+row['offset'])))
            name=f"{row['moniker']}__R{row['record_offset']}__G{row['group']}__UNPARSED_BBOX"+suffix
            pieces=[piece for piece in pieces if piece['name']!=name]
    published_pieces = (pieces if settings.unknown_bbox_wireframe else
                        [piece for piece in pieces if not piece['placeholder']])
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / path.stem
    print(f'{path.stem}: writing {len(published_pieces)} meshes; decoded {sum(counts.values())} primitives', flush=True)
    if business_future is not None:
        from vue_business import load_business
        try:
            business = business_future.result()
        except Exception:
            business = load_business(database, records, native, path.stem)
        business_pool.shutdown()
        progress(f'MDB2 属性加载完成: {len(business["objects"])} 个对象')
    if settings.reports:
        (destination/'profile_shells.json').write_text(json.dumps(profile_shell_rows,indent=2),encoding='utf-8')
    if settings.scratch_budget_mb is not None:
        # Workload/scheduling budget (0928 review fix): summed geometry-split
        # scratch ESTIMATE for pieces needing split — NOT a process memory cap
        # (resident data + in-flight batch + output encode/subprocess are not
        # summed here and never were). Early periodic checks run inside the
        # geometry loop; this final check covers tail pieces.
        demand = sum(split_scratch_estimate(len(p['local_vertices']), len(p['faces']))
                     for p in published_pieces if needs_normal_split(p))
        limit = settings.scratch_budget_mb * 2 ** 20
        if demand > limit:
            raise SystemExit('scratch budget exceeded: summed geometry split scratch estimate %.0f MiB '
                             'exceeds --scratch-budget-mb %d (workload budget, not a process memory cap); '
                             'raise the budget or narrow the export'
                             % (demand / 2 ** 20, settings.scratch_budget_mb))
    if progress.callback is not None:
        write_models(base, published_pieces, settings, business, progress=progress)
    elif business is not None:
        write_models(base, published_pieces, settings, business)
    elif options is None:
        write_models(base, published_pieces)
    else:
        write_models(base, published_pieces, settings)
    if settings.reports:
        for suffix, items in [('geometry.csv', rows), ('gaps.csv', gaps)]:
            keys = list(dict.fromkeys(k for row in items for k in row))
            with (destination / suffix).open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(items)
    included = {p['record_offset'] for p in pieces if not p['placeholder']}
    summary = dict(sample=path.stem, input_sha256=input_sha256,
                   records=len(records), objects_with_geometry=len(included),
                   objects_without_geometry=len(records)-len(included), primitives=dict(counts),
                   tentative_primitives=sum(r['tentative'] is True for r in rows),
                   placeholder_objects=len({p['record_offset'] for p in published_pieces if p['placeholder']}),
                   triangles=sum(len(p['faces']) for p in published_pieces if not is_line_piece(p)),
                   line_segments=sum(len(p['faces']) for p in published_pieces if is_line_piece(p)),
                   header_errors=errors,
                   objects_without_geometry_ids=[canonical(r['moniker']) for r in records if r['record_offset'] not in included],
                   scope='Partial visual preview; includes unvalidated placement and arc hypotheses; no claim of complete objects.',
                   placement='Unresolved cylinder origins may use VUE block bbox centre for preview; marked candidate and listed in geometry.csv.',
                   mesh_name_signature_suffix='__0xTYPE/SIZE appended to every block-derived mesh node name; record-level placeholders have no block signature.',
                   coordinates='FBX/OBJ: original E,N,EL metres, Z-up. GLB: root basis conversion according to glb_up_axis; y maps (E,N,EL) to (E,EL,-N). No recentering.',
                   appearance='VUE material RGB +0x30 and opacity +0x54 (duplicates +0x58/+0x64); display group 69 insulation shells render translucent at the configured INSULATION_OPACITY=0.5 while keeping the object RGB; no candidate color override, no brightening or emissive fill. When enabled, magenta line-segment __UNPARSED_BBOX is a diagnostic placeholder, not source geometry.')
    summary.update(glb_up_axis=settings.glb_up_axis, output_formats=list(settings.formats), reports=settings.reports,
                   merge_moniker=settings.merge_moniker,
                   unknown_bbox_wireframe=settings.unknown_bbox_wireframe,
                   segments=settings.segments,
                   segmentation=dict(
                       requested_segments=settings.segments,
                       policy_version=SEGMENTATION_POLICY_VERSION,
                       alignment=SEGMENTS_ALIGNMENT,
                       valid_range=[SEGMENTS_MINIMUM, SEGMENTS_MAXIMUM],
                       observed={kind: sorted(values)
                                 for kind, values in sorted(segment_usage.items())},
                       retained_legacy=['sphere icosphere', 'native circular faces',
                                        'NURBS/ruled/revolved native surfaces',
                                        'profile extrusions and other polygons']),
                   exported_meshes=len({p['moniker'] for p in published_pieces}) if settings.merge_moniker else len(published_pieces))
    if business is not None:
        summary['business'] = {k: business['report'][k] for k in ('matched', 'object_nodes', 'directory_nodes', 'geometry_nodes', 'objects_with_geometry')}
        summary['business'].update(vue_only=len(business['report']['vue_only']), mdb_only=len(business['report']['mdb_only']))
        summary['mesh_name_signature_suffix'] = 'GLB business names and numbered geometry children; Moniker on object extras.'
        if settings.reports:
            (destination/'business.json').write_text(json.dumps(business['report'], ensure_ascii=False, indent=2), encoding='utf-8')
    if settings.merge_moniker and business is None:
        summary['mesh_name_signature_suffix'] = 'Merged nodes use <Moniker>__merged; original block-derived names are retained in GLB extras and optional mesh_merge.json.'
    if settings.reports:
        (destination/'references.json').write_text(json.dumps(reference_rows,indent=2),encoding='utf-8')
        (destination/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('objects_without_geometry_ids', 'header_errors')}, ensure_ascii=False), flush=True)
    progress(f'完成: {destination}')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('samples', nargs='*', default=['063E', '15300'])
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/model_preview'))
    add_export_arguments(ap)
    args = ap.parse_args()
    if args.mdb2 not in (None, 'auto') and len(args.samples) != 1:
        ap.error('--mdb2 with an explicit path requires exactly one VUE input.')
    for sample in args.samples:
        source = Path(sample).with_suffix('.vue')
        export_sample(source, args.output_dir/source.stem, options_from_args(args), progress=console_progress)
