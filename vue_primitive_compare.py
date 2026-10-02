"""Compare verified VUE cylinder fields with independent FBX fit evidence.

This is the first bridge between the VUE straight-pipe decoder and the FBX
primitive reference.  VUE values are read directly from the ``0x1F/0xD0``
block through :class:`vue_pipe_parser.VuePipeParser`:

* ``|axis @ +0x50|`` is the finite cylinder length;
* ``point @ +0x78`` plus ``axis`` locates one endpoint;
* ``|refvec @ +0xA8|`` is the cylinder radius.

The FBX fit is used only as an independent comparison reference.  It never
supplies values to the VUE-side generated geometry.  Two generated meshes are
written for each object: a complete finite cylinder with caps and a side-only
variant, because the current Navisworks/Assimp D0 FBX reference contains the
side surface but no end caps.
"""

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

import fbx_primitive_fit as fpf
from vue_block_sequence_probe import walk_group_elements
from vue_pipe_parser import VuePipeParser


ROOT = Path(__file__).resolve().parent
VERSION = "0.2.0"
DEFAULT_RUN_ID = "compare1"
DEFAULT_FBX_RUN_ID = "final3"
DEFAULT_MONIKERS = [
    "0027-80012-301858369261339043",  # 063E linkage 430
    "0027-80012-301858369032094224",  # 063E linkage 429
]
PARAMETER_TOL_M = fpf.DEFAULT_TOL_M
AXIS_ANGLE_TOL_DEG = 0.5


def canonical_moniker(value):
    return value.strip().replace("@a=", "", 1).replace("!!", "-", 1).replace("##", "-", 1)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_moniker_for(parser, canonical):
    for raw in parser.tokens.values():
        if canonical_moniker(raw) == canonical:
            return raw
    return None


def as_list(value):
    return [float(x) for x in np.asarray(value, dtype=float)]


def unit(value):
    value = np.asarray(value, dtype=float)
    length = float(np.linalg.norm(value))
    if length <= 1.0e-15:
        return None
    return value / length


def bbox_from_vertices(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) == 0:
        return None
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return {
        "min": as_list(low),
        "max": as_list(high),
        "spans": as_list(high - low),
    }


def bbox_from_six(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": as_list(values[:3]),
        "max": as_list(values[3:]),
        "spans": as_list(values[3:] - values[:3]),
    }


def bbox_error(first, second):
    if first is None or second is None:
        return None
    values = list(zip(first["min"] + first["max"], second["min"] + second["max"]))
    return float(max(abs(a - b) for a, b in values))


def bbox_center(bbox):
    if bbox is None:
        return None
    return (np.asarray(bbox["min"], dtype=float)
            + np.asarray(bbox["max"], dtype=float)) / 2.0


def transform_row_vector(matrix, vector, translate=False):
    """Apply an observed row-vector node matrix to a point or direction."""
    matrix = np.asarray(matrix, dtype=float).reshape((4, 4))
    vector = np.asarray(vector, dtype=float)
    transformed = vector @ matrix[:3, :3]
    if translate:
        transformed = transformed + matrix[3, :3]
    return transformed


def block_geometry(block):
    """Return finite-cylinder geometry from a decoded (worldized) D0 block."""
    axis = np.asarray(block["axis"], dtype=float)
    point = np.asarray(block["point"], dtype=float)
    length = float(block["height"])
    p2 = point + axis
    return {
        "axis": as_list(unit(axis)),
        "center_m": as_list((point + p2) / 2.0),
        "p1_m": as_list(point),
        "p2_m": as_list(p2),
        "radius_m": float(block["radius"]),
        "length_m": length,
    }


def serialize_vue_block(block):
    geometry = block_geometry(block)
    return {
        "offset_rel_record": int(block["off"]),
        "bbox_world": bbox_from_six(block["bbox"]),
        "axis_m": as_list(block["axis"]),
        "axis_length_m": float(block["height"]),
        "point_m": as_list(block["point"]),
        "dirv": as_list(block["dirv"]),
        "refvec_m": as_list(block["refvec"]),
        "radius_m": float(block["radius"]),
        "center_m": geometry["center_m"],
        "p1_m": geometry["p1_m"],
        "p2_m": geometry["p2_m"],
        "axis_line": geometry["axis"],
        "coordinate_space": block.get("coordinate_space", "world"),
        "transform_node_offset_rel_record": block.get("transform_node_offset"),
        "raw_axis_m": as_list(block.get("raw_axis", block["axis"])),
        "raw_point_m": as_list(block.get("raw_point", block["point"])),
        "raw_refvec_m": as_list(block.get("raw_refvec", block["refvec"])),
        "raw_local_origin_m": as_list(block.get("raw_local_origin", (0.0, 0.0, 0.0))),
        "display_code_raw": block.get("display_code_raw"),
        "group_index": block.get("group_index"),
        "field_offsets_rel_block": {
            "axis": "0x50",
            "point": "0x78",
            "refvec": "0xA8",
        },
    }


def worldize_vue_blocks(parser, parsed, walks=None):
    """Transform D0 fields found inside an observed nested geometry node.

    The D0 bbox is already stored in world coordinates in these records, while
    point/axis/refvec are local to the node.  The node's matrix A is an
    observed row-vector affine transform.  Plain pipe D0 blocks have no node
    entry and are retained as world-coordinate fields.
    """
    matrix_by_offset = {}
    node_offset_by_d0 = {}
    local_origin_by_d0 = {}
    node_offsets = []
    for group in parsed["display_groups"]:
        if walks is None:
            elements, _ = walk_group_elements(parser.data, group, parsed["rec_start"])
        else:
            elements = walks[group['index']]
        for element in elements:
            if element.get("kind") != "nested_node":
                continue
            node_offsets.append(int(element["off_rel_record"]))
            matrix = element["matrix_a_rowmajor"]
            for child_index, block_offset in enumerate(element.get("inner_d0_markers", [])):
                matrix_by_offset[int(block_offset)] = matrix
                node_offset_by_d0[int(block_offset)] = int(element["off_rel_record"])
                # Each sibling now has its own native transform. The former
                # +0x160 origin workaround compensated for a merged node span
                # and must not be added to an already correctly placed child.
                local_origin_by_d0[int(block_offset)] = (0.0, 0.0, 0.0)

    by_offset = {}
    for original in parsed["blocks"]:
        block = dict(original)
        matrix = matrix_by_offset.get(int(original["off"]))
        if matrix is None:
            block["coordinate_space"] = "world"
        else:
            local_origin = np.asarray(
                local_origin_by_d0.get(int(original["off"]), (0.0, 0.0, 0.0)),
                dtype=float,
            )
            block["raw_axis"] = tuple(original["axis"])
            block["raw_point"] = tuple(original["point"])
            block["raw_refvec"] = tuple(original["refvec"])
            block["axis"] = tuple(transform_row_vector(matrix, original["axis"]))
            block["point"] = tuple(transform_row_vector(
                matrix, np.asarray(original["point"], dtype=float) + local_origin,
                translate=True))
            block["refvec"] = tuple(transform_row_vector(matrix, original["refvec"]))
            block["dirv"] = tuple(transform_row_vector(matrix, original["dirv"]))
            block["coordinate_space"] = "nested_local_transformed_to_world"
            block["transform_node_offset"] = node_offset_by_d0.get(int(original["off"]))
            block["raw_local_origin"] = tuple(local_origin)
        by_offset[int(block["off"])] = block

    worldized = dict(parsed)
    worldized["blocks"] = list(by_offset.values())
    worldized["body_blocks"] = [by_offset[int(block["off"])]
                                for block in parsed["body_blocks"]]
    worldized["insulation_blocks"] = [by_offset[int(block["off"])]
                                      for block in parsed["insulation_blocks"]]
    worldized["coordinate_transform"] = {
        "status": "nested_node_matrix_applied" if matrix_by_offset else "world_fields",
        "node_count": len(node_offsets),
        "transformed_d0_count": len(matrix_by_offset),
        "local_origins_by_d0": {
            str(offset): as_list(origin)
            for offset, origin in local_origin_by_d0.items()
        },
        "matrix_convention": "row_vector_point_times_matrix_A",
        "source": "nested node +0x18 4x4 row-major matrix A",
        "secondary_origin_source": "native sibling node matrix; no inferred secondary offset",
    }
    return worldized


def vue_cylinder_evidence(parser, canonical):
    raw = raw_moniker_for(parser, canonical)
    if raw is None:
        return {"status": "not_in_vue", "moniker": canonical}
    parsed = parser.parse_pipe(raw)
    if parsed is None:
        return {"status": "not_a_supported_straight_pipe", "moniker": canonical}
    parsed = worldize_vue_blocks(parser, parsed)

    main_segments = VuePipeParser.merge_main_pipe(parsed["body_blocks"])[2]
    if not main_segments:
        return {"status": "no_main_cylinder", "moniker": canonical}
    p1_raw, p2_raw = VuePipeParser.pipe_endpoints(main_segments)
    p1 = np.asarray(p1_raw, dtype=float)
    p2 = np.asarray(p2_raw, dtype=float)
    axis_line = unit(p2 - p1)
    main_length = float(sum(block["height"] for block in main_segments))
    main_radius = float(main_segments[0]["radius"])
    if axis_line is None or main_length <= 0 or main_radius <= 0:
        return {"status": "invalid_decoded_cylinder", "moniker": canonical}
    return {
        "status": "found",
        "moniker": canonical,
        "raw_moniker": raw.strip(),
        "record_offset": int(parsed["rec_start"]),
        "record_end": int(parsed["rec_end"]),
        "record_bbox": bbox_from_six(parser.record_bbox(parsed["rec_start"])),
        "length_m": main_length,
        "radius_m": main_radius,
        "p1_m": as_list(p1),
        "p2_m": as_list(p2),
        "center_m": as_list((p1 + p2) / 2.0),
        "axis": as_list(axis_line),
        "main_segment_count": len(main_segments),
        "main_blocks": [serialize_vue_block(block) for block in main_segments],
        "secondary_blocks": [
            serialize_vue_block(block)
            for block in parsed["body_blocks"]
            if block["off"] not in {main["off"] for main in main_segments}
        ],
        "all_body_block_count": len(parsed["body_blocks"]),
        "all_block_count": len(parsed["blocks"]),
        "coordinate_transform": parsed["coordinate_transform"],
        "display_groups": [
            {
                "index": int(group["index"]),
                "display_code_raw": int(group["display_code_raw"]),
                "bbox_world": bbox_from_six(group["bbox_world"]),
            }
            for group in parsed["display_groups"]
        ],
        "insulation_status": parsed["insulation_status"],
        "source": {
            "axis_field": "0x1F/0xD0 block +0x50, float64[3]",
            "point_field": "0x1F/0xD0 block +0x78, float64[3]",
            "refvec_field": "0x1F/0xD0 block +0xA8, float64[3]",
            "decoder": "VuePipeParser.parse_pipe + observed nested-node matrix A",
        },
    }


def load_fbx_fit_records(path):
    if not path.exists():
        return {}, {"status": "missing", "path": str(path)}
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            record = item.get("record", {})
            key = canonical_moniker(record.get("moniker", ""))
            if key:
                records[key] = item
    return records, {"status": "found", "path": str(path), "sha256": sha256(path)}


def fbx_cylinder_candidates(fit_item):
    candidates = []
    record = fit_item.get("record", {})
    for mesh in record.get("meshes", []):
        if mesh.get("is_insulation"):
            continue
        for component in mesh.get("components", []):
            for primitive_index, primitive in enumerate(component.get("primitives", [])):
                if primitive.get("kind") != "cylinder":
                    continue
                if primitive.get("subtype") != "cylinder":
                    continue
                candidate = dict(primitive)
                candidate.update({
                    "mesh_index": int(mesh["mesh_index"]),
                    "component_index": int(component["component_index"]),
                    "primitive_index": primitive_index,
                    "component_status": component.get("status"),
                })
                candidates.append(candidate)
    return candidates


def select_fbx_candidate(vue, candidates):
    if not candidates:
        return None
    vr = float(vue["radius_m"])
    vl = float(vue["length_m"])

    def score(candidate):
        radius_scale = max(vr, 1.0e-6)
        length_scale = max(vl, 1.0e-6)
        return (abs(float(candidate["radius_m"]) - vr) / radius_scale
                + abs(float(candidate["length_m"]) - vl) / length_scale)

    selected = min(candidates, key=score)
    return {"score": float(score(selected)), "candidate": selected}


def select_fbx_radius_candidate(block, candidates):
    """Select a radius-near FBX fit without treating its length as equivalent."""
    if not candidates:
        return None
    radius = float(block["radius"])

    def score(candidate):
        return abs(float(candidate["radius_m"]) - radius) / max(radius, 1.0e-6)

    selected = min(candidates, key=score)
    return {"score": float(score(selected)), "candidate": selected}


def compare_parameters(vue, selected):
    candidate = selected["candidate"]
    vue_axis = unit(vue["axis"])
    fbx_axis = unit(candidate["axis"])
    axis_alignment = abs(float(np.dot(vue_axis, fbx_axis)))
    axis_angle_deg = math.degrees(math.acos(np.clip(axis_alignment, -1.0, 1.0)))
    vue_center = np.asarray(vue["center_m"], dtype=float)
    fbx_center = np.asarray(candidate["center_m"], dtype=float)
    radius_delta = float(candidate["radius_m"] - vue["radius_m"])
    length_delta = float(candidate["length_m"] - vue["length_m"])
    center_distance = float(np.linalg.norm(fbx_center - vue_center))
    vue_endpoint = np.asarray(vue["p1_m"], dtype=float)
    half_length = float(candidate["length_m"]) / 2.0
    fbx_end_a = fbx_center - fbx_axis * half_length
    fbx_end_b = fbx_center + fbx_axis * half_length
    endpoint_distance = float(min(np.linalg.norm(vue_endpoint - fbx_end_a),
                                  np.linalg.norm(vue_endpoint - fbx_end_b)))
    status = "confirmed" if (
        abs(radius_delta) <= PARAMETER_TOL_M
        and abs(length_delta) <= PARAMETER_TOL_M
        and center_distance <= 2.0 * PARAMETER_TOL_M
        and axis_angle_deg <= AXIS_ANGLE_TOL_DEG
    ) else "needs_review"
    return {
        "status": status,
        "fbx_mesh_index": selected["candidate"]["mesh_index"],
        "fbx_component_index": selected["candidate"]["component_index"],
        "fbx_primitive_index": selected["candidate"]["primitive_index"],
        "candidate_score": selected["score"],
        "radius_delta_m": radius_delta,
        "length_delta_m": length_delta,
        "center_distance_m": center_distance,
        "axis_alignment_abs_dot": axis_alignment,
        "axis_angle_deg": axis_angle_deg,
        "endpoint_distance_m": endpoint_distance,
        "fbx_candidate": candidate,
    }


def build_cylinder_primitive(vue, with_caps):
    return {
        "kind": "cylinder",
        "axis": np.asarray(vue["axis"], dtype=float),
        "center": np.asarray(vue["center_m"], dtype=float),
        "radius_m": float(vue["radius_m"]),
        "length_m": float(vue["length_m"]),
        "has_caps": bool(with_caps),
        "face_ids": [],
    }


def secondary_block_hypotheses(sample, vue, block, comparison):
    """Record thin/near-zero D0 evidence without promoting a partial match."""
    common = {
        "sample": sample,
        "moniker": vue["moniker"],
        "record_offset": vue["record_offset"],
        "block_offset_rel_record": block["offset_rel_record"],
        "fbx_mesh_index": comparison.get("fbx_mesh_index"),
        "fbx_component_index": comparison.get("fbx_component_index"),
        "fbx_primitive_index": comparison.get("fbx_primitive_index"),
        "evidence": "secondary VUE D0 block vs radius-near FBX fit; length is not promoted when the FBX component is partial",
    }
    rows = []

    def add(field, offset, vue_value, fbx_value, delta, comparison_kind, status):
        row = dict(common)
        row.update({
            "vue_field": field,
            "vue_field_offset": offset,
            "vue_value": json.dumps(vue_value, ensure_ascii=False),
            "fbx_value": json.dumps(fbx_value, ensure_ascii=False),
            "delta_or_error": float(delta),
            "comparison": comparison_kind,
            "status": status,
        })
        rows.append(row)

    status = comparison.get("status", "reference_missing")
    radius_status = (
        "confirmed"
        if comparison.get("fbx_radius_m") is not None
        and abs(comparison.get("radius_delta_m", float("inf"))) <= PARAMETER_TOL_M
        else status
    )
    add("secondary_axis_length", "+0x50", block["axis_length_m"],
        comparison.get("fbx_length_m"), comparison.get("length_delta_m", 0.0),
        "preserved_near_zero_length_not_filled", status)
    add("secondary_refvec_norm_radius", "+0xA8", block["radius_m"],
        comparison.get("fbx_radius_m"), comparison.get("radius_delta_m", 0.0),
        "radius_delta_m", radius_status)
    return rows


def compare_secondary_block(sample, vue, block, candidates, output_dir, index):
    """Compare a non-main D0 block while preserving its exact finite length."""
    geometry = block_geometry({
        "axis": block["axis_m"],
        "point": block["p1_m"],
        "height": block["axis_length_m"],
        "radius": block["radius_m"],
    })
    selection = select_fbx_radius_candidate(
        {"radius": block["radius_m"]}, candidates)
    generated_full = write_obj(
        output_dir / f"vue_generated_{vue['moniker']}_block_{block['offset_rel_record']}_full.obj",
        build_cylinder_primitive(geometry, True),
        f"{vue['moniker']}_vue_block_{index}_full",
    )
    generated_side = write_obj(
        output_dir / f"vue_generated_{vue['moniker']}_block_{block['offset_rel_record']}_side.obj",
        build_cylinder_primitive(geometry, False),
        f"{vue['moniker']}_vue_block_{index}_side",
    )
    result = {
        "index": int(index),
        "status": "reference_missing",
        "vue": block,
        "generated": {
            "full": generated_full,
            "side_only": generated_side,
            "bbox_vs_vue_d0": bbox_error(
                generated_full["bbox"] if generated_full else None,
                block["bbox_world"]),
            "center_vs_vue_d0_bbox_center_m": float(np.linalg.norm(
                np.asarray(geometry["center_m"], dtype=float)
                - bbox_center(block["bbox_world"]))),
        },
        "fbx": {"status": "no_radius_near_candidate"},
        "notes": [
            "The VUE D0 length is retained exactly, including the observed 2e-5 m near-zero value.",
            "A radius-near partial FBX component does not justify replacing this D0 with a longer cylinder.",
        ],
    }
    if selection is None:
        return result, secondary_block_hypotheses(
            sample, vue, block, {"status": "reference_missing"})

    candidate = selection["candidate"]
    radius_delta = float(candidate["radius_m"] - block["radius_m"])
    length_delta = float(candidate["length_m"] - block["axis_length_m"])
    radius_match = abs(radius_delta) <= PARAMETER_TOL_M
    result["status"] = "partial" if radius_match else "needs_review"
    result["fbx"] = {
        "status": "radius_near_candidate",
        "candidate_score": selection["score"],
        "fbx_mesh_index": candidate["mesh_index"],
        "fbx_component_index": candidate["component_index"],
        "fbx_primitive_index": candidate["primitive_index"],
        "component_status": candidate.get("component_status"),
        "radius_m": candidate["radius_m"],
        "length_m": candidate["length_m"],
        "radius_delta_m": radius_delta,
        "length_delta_m": length_delta,
        "fit_explained_ratio": candidate.get("explained_ratio"),
    }
    comparison = {
        "status": result["status"],
        "fbx_mesh_index": candidate["mesh_index"],
        "fbx_component_index": candidate["component_index"],
        "fbx_primitive_index": candidate["primitive_index"],
        "fbx_radius_m": candidate["radius_m"],
        "fbx_length_m": candidate["length_m"],
        "radius_delta_m": radius_delta,
        "length_delta_m": length_delta,
    }
    if radius_match:
        result["notes"].append(
            "The FBX radius is a partial-component hint only; its longer fitted length is kept as unresolved residual evidence.")
    return result, secondary_block_hypotheses(sample, vue, block, comparison)


def combine_fbx_meshes(meshes):
    selected = [mesh for mesh in meshes if not mesh.get("is_insulation")]
    if not selected:
        return None, None
    vertices = []
    faces = []
    offset = 0
    for mesh in selected:
        mesh_vertices = np.asarray(mesh["vertices"], dtype=float)
        mesh_faces = np.asarray(mesh["faces"], dtype=np.int64)
        if len(mesh_vertices) == 0 or len(mesh_faces) == 0:
            continue
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + offset)
        offset += len(mesh_vertices)
    if not vertices:
        return None, None
    return np.vstack(vertices), np.vstack(faces)


def compare_against_fbx(fbx_vertices, fbx_faces, vue, with_caps, rng):
    if fbx_vertices is None or fbx_faces is None:
        return {"status": "reference_missing"}
    primitive = build_cylinder_primitive(vue, with_caps)
    return fpf.compare_surfaces(fbx_vertices, fbx_faces, [primitive], rng)


def write_obj(path, primitive, name):
    vertices, faces = fpf.rebuild_mesh([primitive])
    if vertices is None or faces is None:
        return None
    lines = ["# VUE-generated research cylinder; not a production VUE export", f"o {name}"]
    lines.extend(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}" for v in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)), "bbox": bbox_from_vertices(vertices)}


def hypothesis_rows(sample, vue, parameter_comparison):
    candidate = parameter_comparison["fbx_candidate"]
    block = vue["main_blocks"][0]
    moniker = vue["moniker"]
    record_offset = vue["record_offset"]
    block_offset = block["offset_rel_record"]
    common = {
        "sample": sample,
        "moniker": moniker,
        "record_offset": record_offset,
        "block_offset_rel_record": block_offset,
        "fbx_mesh_index": parameter_comparison["fbx_mesh_index"],
        "fbx_component_index": parameter_comparison["fbx_component_index"],
        "fbx_primitive_index": parameter_comparison["fbx_primitive_index"],
        "evidence": "VUE 0x1F/0xD0 fields vs independent FBX cylinder fit",
    }
    rows = []

    def add(field, offset, vue_value, fbx_value, delta, comparison):
        row = dict(common)
        row.update({
            "vue_field": field,
            "vue_field_offset": offset,
            "vue_value": json.dumps(vue_value, ensure_ascii=False),
            "fbx_value": json.dumps(fbx_value, ensure_ascii=False),
            "delta_or_error": float(delta),
            "comparison": comparison,
            "status": parameter_comparison["status"] if comparison != "axis_sign_agnostic" else "confirmed",
        })
        rows.append(row)

    add("axis_length", "+0x50", block["axis_length_m"], candidate["length_m"],
        parameter_comparison["length_delta_m"], "absolute_delta_m")
    add("refvec_norm_radius", "+0xA8", block["radius_m"], candidate["radius_m"],
        parameter_comparison["radius_delta_m"], "absolute_delta_m")
    add("center_from_point_plus_half_axis", "+0x78 + 0.5*+0x50",
        vue["center_m"], candidate["center_m"], parameter_comparison["center_distance_m"],
        "euclidean_distance_m")
    add("axis_direction_line", "+0x50",
        vue["axis"], candidate["axis"], parameter_comparison["axis_angle_deg"],
        "angle_deg_sign_agnostic")
    add("point_endpoint_line_match", "+0x78",
        vue["p1_m"], "fbx_candidate_endpoint_nearest", parameter_comparison["endpoint_distance_m"],
        "euclidean_distance_m_sign_agnostic")
    return rows


def compare_object(sample, parser, canonical, fit_item, meshes, output_dir, rng):
    vue = vue_cylinder_evidence(parser, canonical)
    if vue["status"] != "found":
        return {"moniker": canonical, "status": vue["status"], "vue": vue,
                "fbx": {"status": "not_compared"}, "hypotheses": []}

    candidates = fbx_cylinder_candidates(fit_item) if fit_item else []
    selected = select_fbx_candidate(vue, candidates)
    if selected is None:
        return {"moniker": canonical, "status": "fbx_fit_missing",
                "vue": vue, "fbx": {"status": "no_cylinder_candidate"},
                "hypotheses": []}
    parameter_comparison = compare_parameters(vue, selected)
    fbx_vertices, fbx_faces = combine_fbx_meshes(meshes)
    side_primitive = build_cylinder_primitive(vue, False)
    full_primitive = build_cylinder_primitive(vue, True)
    side_result = compare_against_fbx(fbx_vertices, fbx_faces, vue, False, rng)
    full_result = compare_against_fbx(fbx_vertices, fbx_faces, vue, True, rng)
    generated = {
        "full": write_obj(output_dir / f"vue_generated_{canonical}_full.obj",
                           full_primitive, f"{canonical}_vue_full"),
        "side_only": write_obj(output_dir / f"vue_generated_{canonical}_side.obj",
                                side_primitive, f"{canonical}_vue_side"),
    }
    secondary_results = []
    secondary_hypotheses = []
    for index, block in enumerate(vue["secondary_blocks"], start=1):
        secondary_result, block_hypotheses = compare_secondary_block(
            sample, vue, block, candidates, output_dir, index)
        secondary_results.append(secondary_result)
        secondary_hypotheses.extend(block_hypotheses)
    status = "confirmed" if parameter_comparison["status"] == "confirmed" else "needs_review"
    if any(item["status"] != "confirmed" for item in secondary_results):
        status = "partial" if status == "confirmed" else status
    return {
        "moniker": canonical,
        "status": status,
        "vue": vue,
        "fbx": {
            "status": "found" if meshes else "no_fbx_mesh",
            "mesh_count": len(meshes),
            "fit_candidate_count": len(candidates),
            "selected": parameter_comparison,
            "merged_bbox": bbox_from_vertices(fbx_vertices) if fbx_vertices is not None else None,
            "secondary_block_count": len(secondary_results),
        },
        "generated": {
            "full_cylinder": generated["full"],
            "side_only_cylinder": generated["side_only"],
            "full_vs_fbx": full_result,
            "side_only_vs_fbx": side_result,
            "full_cylinder_bbox_vs_vue_record": bbox_error(
                generated["full"]["bbox"] if generated["full"] else None,
                vue["record_bbox"]),
            "full_cylinder_bbox_vs_vue_main_d0": bbox_error(
                generated["full"]["bbox"] if generated["full"] else None,
                vue["main_blocks"][0]["bbox_world"]),
        },
        "secondary_blocks": secondary_results,
        "hypotheses": (hypothesis_rows(sample, vue, parameter_comparison)
                        + secondary_hypotheses),
        "notes": [
            "The full VUE-generated cylinder includes caps as a finite primitive.",
            "The side-only comparison isolates the currently visible FBX D0 surface.",
            "A full-cylinder surface mismatch can be caused by FBX missing end caps; it is not a VUE field rejection by itself.",
            "Non-main D0 blocks are generated at their decoded length; a radius-near but longer partial FBX fit is not used to thicken them.",
            "FBX values remain comparison evidence and are never written into the VUE geometry model.",
        ],
    }


def load_linkage_context(path):
    if path is None or not path.exists():
        return {}
    context = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            context[canonical_moniker(row.get("display_key", row.get("moniker", "")))] = {
                "linkage_index": row.get("linkage_index"),
            }
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", nargs="?", default="063E",
                        help="sample stem or VUE path, e.g. 063E or 063E.vue")
    parser.add_argument("--moniker", action="append", dest="monikers",
                        help="canonical or raw moniker; may be repeated")
    parser.add_argument("--fbx-fit", type=Path,
                        help="fbx_primitive_fits.jsonl; defaults to the latest configured run")
    parser.add_argument("--fbx-run-id", default=DEFAULT_FBX_RUN_ID)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pipe-jsonl", type=Path,
                        help="optional existing pipe_geometry.jsonl for linkage context")
    args = parser.parse_args()

    vue_path = Path(args.sample)
    if vue_path.suffix.lower() != ".vue":
        vue_path = ROOT / f"{args.sample}.vue"
    fbx_path = vue_path.with_suffix(".fbx")
    sample = vue_path.stem
    output_dir = args.output_dir or ROOT / "outputs" / "vue_primitive_compare" / args.run_id / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.fbx_fit or ROOT / "outputs" / "primitive_research" / args.fbx_run_id / sample / "fbx_primitive_fits.jsonl"
    default_pipe_jsonl = ROOT / "outputs" / sample / "pipe_geometry.jsonl"
    pipe_jsonl = args.pipe_jsonl or (default_pipe_jsonl if default_pipe_jsonl.exists() else None)
    linkage = load_linkage_context(pipe_jsonl)

    wanted = [canonical_moniker(value) for value in (args.monikers or DEFAULT_MONIKERS)]
    parser_obj = VuePipeParser(vue_path)
    fit_records, fit_meta = load_fbx_fit_records(fit_path)
    meshes_by_moniker = fpf.collect_meshes(fbx_path, wanted) if fbx_path.exists() else {key: [] for key in wanted}
    rng = np.random.default_rng(fpf.SEED)
    results = []
    hypotheses = []
    for canonical in wanted:
        result = compare_object(sample, parser_obj, canonical,
                                fit_records.get(canonical),
                                meshes_by_moniker.get(canonical, []),
                                output_dir, rng)
        if canonical in linkage:
            result["linkage_index"] = linkage[canonical]["linkage_index"]
        results.append(result)
        hypotheses.extend(result.get("hypotheses", []))

    result_path = output_dir / "vue_primitive_compare.jsonl"
    with result_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps({
                "script_version": VERSION,
                "sample": sample,
                "vue_sha256": sha256(vue_path),
                "fbx_sha256": sha256(fbx_path) if fbx_path.exists() else None,
                "fbx_fit": fit_meta,
                "candidate": "verified 0x1F/0xD0 finite cylinder; world fields or nested-node-local fields after matrix A",
                "record": result,
            }, ensure_ascii=False) + "\n")

    hypothesis_path = output_dir / "field_hypotheses.csv"
    fields = [
        "sample", "moniker", "record_offset", "block_offset_rel_record",
        "fbx_mesh_index", "fbx_component_index", "fbx_primitive_index",
        "vue_field", "vue_field_offset", "vue_value", "fbx_value",
        "delta_or_error", "comparison", "status", "evidence",
    ]
    with hypothesis_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hypotheses)

    print(json.dumps({
        "sample": sample,
        "output_dir": str(output_dir),
        "compare_jsonl": str(result_path),
        "field_hypotheses": str(hypothesis_path),
        "objects": len(results),
        "status_counts": {
            status: sum(1 for result in results if result["status"] == status)
            for status in sorted({result["status"] for result in results})
        },
        "fbx_fit": fit_meta,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
