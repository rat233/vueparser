# -*- coding: utf-8 -*-
"""063E 显示组内顶层元素序列探测（阶段 B 证据脚本）。

已观察规则（2026-09-07，证据见 block_sequence_probe.json）：
组子块区自组头 +0x1E0 起，顶层元素为两类交替，当前样本中嵌套节点均位于组尾：
1. 标记块：u32 (type,size)，size 含标记本身；0x1F/0xD0 圆柱、0x1F/0x108 轮廓
   拉伸、0x20/0xF0 弯头参数、0x1D/0xE0 等已见；末块可在 tag+0xC8 结束。
2. 嵌套节点：头 (A,0,0,ptr1,ptr2,0) + 4x4 行主序矩阵 A + 4x4 矩阵 B
   （B 旋转 = A 旋转转置）+ 世界点等；ptr 基于 R+0x70 指向组内；
   D0 圆柱等子几何位于节点区间内部，是“局部参数+世界 bbox 并存”的来源。
   2026-09-09 修正：节点可串联；+0x10 后继定位 R+0x78+ptr2，子块只属于自己的节点。
本脚本在全样本上验证该规则、与 D0 标记交叉核对，并对代表阀门/弯头导出
字段级视图。仅记录证据，不据此声明几何已解码。
"""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

from vue_record_parser import WORKSPACE, read_records

BLOCK_AREA_START = 0x1E0
CYL_TYPE, CYL_SIZE = 0x1F, 0xD0
D0_FIELDS = {
    "bbox": (0x08, "6d"), "seq": (0x38, "I"), "axis": (0x50, "3d"),
    "point": (0x78, "3d"), "dirv": (0x90, "3d"), "refvec": (0xA8, "3d"),
}
# 代表对象记录偏移，来自 outputs/063E/component_examples.json；非通用定位规则。
TARGET_OFFSETS = {343068: "245", 813908: "247", 794324: "255", 996612: "338",
                  727316: "431"}


def walk_group_elements(data, group, record_start):
    """Return tagged blocks and bounded sibling transform nodes.

    Node +0x10 links to the next node's 8-byte prefix relative to R+0x70.
    Thus the next matrix-bearing header is R+0x78+link. Never assign the
    first node's matrix to D0 children belonging to subsequent siblings.
    """
    return _walk_element_region(data, group, record_start,
                                group['header_start'] + BLOCK_AREA_START,
                                group['end'])


def _walk_element_region(data, group, record_start, start, end, parent=None):
    """Walk one bounded scope; node children use the same framing as roots.

    Nodes remain in preorder for existing audit consumers. ``inner_blocks``
    and ``inner_d0_markers`` contain only directly owned geometry, so a parent
    can never apply its transform to a descendant's cylinders.
    """
    elements, pos = [], start
    while pos < end:
        if pos + 24 > end:
            return elements, f"tail_short_R+{pos-record_start:#x}"
        a, b, c = struct.unpack_from("<3I", data, pos)
        if b == 0 and c == 0 and a > 0x100:
            head = struct.unpack_from("<6I", data, pos)
            next_link = head[4]
            node_end = record_start + 0x78 + next_link if next_link else end
            expected_span = a + 4 if next_link else a - 4
            if (not pos + 0x180 <= node_end <= end or
                    node_end - pos != expected_span or
                    (next_link and node_end + 0x180 > end)):
                return elements, f"invalid_node_link_R+{pos-record_start:#x}"
            matrix_a = struct.unpack_from("<16d", data, pos + 0x18)
            matrix_b = struct.unpack_from("<16d", data, pos + 0x98)
            if (not all(math.isfinite(v) for v in matrix_a + matrix_b) or
                    max(abs(sum(matrix_a[i*4+k] * matrix_b[k*4+j]
                                for k in range(4)) - (i == j))
                        for i in range(4) for j in range(4)) > 1e-6):
                return elements, f"invalid_node_matrix_R+{pos-record_start:#x}"
            node = {
                "kind": "nested_node",
                "off_rel_record": pos - record_start,
                "off_rel_group": pos - group["header_start"],
                "head_u32": head,
                "span_to_group_end": end - pos,
                "node_end_rel_record": node_end - record_start,
                "node_span": node_end - pos,
                "next_node_offset_rel_record": node_end - record_start if next_link else None,
                "parent_node_offset": parent,
            }
            node["matrix_a_rowmajor"] = matrix_a
            node["matrix_b_rowmajor"] = matrix_b
            node["world_points"] = (struct.unpack_from("<3d", data, pos + 0x118),
                                    struct.unpack_from("<3d", data, pos + 0x130))
            children, child_status = _walk_element_region(
                data, group, record_start, pos + 0x180, node_end,
                pos - record_start)
            node['inner_blocks'] = [c for c in children if c['kind'] == 'tag_block']
            node['inner_d0_markers'] = [
                c['off_rel_record'] for c in node['inner_blocks']
                if (c['type_raw'], c['size_raw']) == (CYL_TYPE, CYL_SIZE)]
            node['inner_blocks_complete'] = child_status in ('exact_end', 'exact_end_with_node')
            node['inner_walk_status'] = child_status
            elements.append(node)
            elements.extend(c for c in children if c['kind'] == 'nested_node')
            if not next_link:
                return elements, "exact_end_with_node"
            pos = node_end
            continue
        if b < 8 or b % 8:
            return elements, f"fail_R+{pos-record_start:#x}"
        framing = "declared"
        if pos + b > end:
            if pos + b - 8 != end:
                return elements, f"fail_R+{pos-record_start:#x}"
            framing = "tail_no_framing"
        elements.append({
            "kind": "tag_block", "type_raw": a, "size_raw": b,
            "off_rel_record": pos - record_start,
            "off_rel_group": pos - group["header_start"],
            "framing": framing,
        })
        pos += b - 8 if framing == "tail_no_framing" else b
    return elements, "exact_end"


def scan_d0_markers(data, start, end):
    marker = struct.pack("<II", CYL_TYPE, CYL_SIZE)
    hits, pos = [], start
    while True:
        idx = data.find(marker, pos, end)
        if idx < 0:
            return hits
        hits.append(idx)
        pos = idx + 4


def tag_block_views(data, element, record_start):
    off, size = element["off_rel_record"], element["size_raw"]
    body = data[record_start + off: record_start + off + size]
    view = {
        "type_raw": element["type_raw"], "size_raw": size,
        "off_rel_record": off,
        "hex": body.hex(),
        "u32": list(struct.unpack_from(f"<{size // 4}I", body)) if size % 4 == 0 else None,
        "f64": list(struct.unpack_from(f"<{size // 8}d", body)) if size % 8 == 0 else None,
    }
    if element["type_raw"] == CYL_TYPE and size == CYL_SIZE:
        view["decoded"] = {
            name: struct.unpack_from("<" + fmt, body, rel)
            for name, (rel, fmt) in D0_FIELDS.items()
        }
    elif element["type_raw"] == CYL_TYPE and size >= 0x108:
        # 0x1F/0x108 候选轮廓拉伸：seq、声明拉伸长度、自引用指针、闭合点列
        view["extrusion_candidate"] = {
            "seq": struct.unpack_from("<I", body, 0x38)[0],
            "declared_length": struct.unpack_from("<d", body, 0x50)[0],  # legacy X component
            "extrusion_vector": struct.unpack_from("<3d", body, 0x50),
            "self_refs_u32": struct.unpack_from("<3I", body, 0x68),
            "profile_points": [struct.unpack_from("<3d", body, 0x88 + 24 * i)
                               for i in range((size - 0x88 - 8) // 24)],
            "tail_u32": struct.unpack_from("<I", body, size - 8)[0],
        }
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vue", nargs="?", type=Path, default=WORKSPACE / "063E.vue")
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    out_dir = args.output_dir or WORKSPACE / "outputs" / args.vue.stem
    data = args.vue.read_bytes()

    _, records, errors = read_records(data)
    by_offset = {r["record_offset"]: r for r in records}

    status_counter, d0_total, d0_on_start = Counter(), 0, 0
    type_counter, node_count = Counter(), 0
    group_rows = []
    for record in records:
        rs = record["record_offset"]
        for group in record["display_groups"]:
            elements, status = walk_group_elements(data, group, rs)
            status_counter[status] += 1
            starts = set()
            for el in elements:
                if el["kind"] == "tag_block":
                    starts.add(rs + el["off_rel_record"])
                    type_counter[f"0x{el['type_raw']:x}/{el['size_raw']}"] += 1
                else:
                    node_count += 1
                    starts.update(rs + o for o in el["inner_d0_markers"])
            markers = scan_d0_markers(data, group["header_start"], group["end"])
            d0_total += len(markers)
            d0_on_start += sum(1 for m in markers if m in starts)
            group_rows.append({
                "record_offset": rs, "group_index": group["index"],
                "display_code_raw": group["display_code_raw"],
                "n_elements": len(elements),
                "n_tag_blocks": sum(1 for e in elements if e["kind"] == "tag_block"),
                "n_nodes": sum(1 for e in elements if e["kind"] == "nested_node"),
                "walk_status": status,
            })

    detailed = []
    for offset, label in sorted(TARGET_OFFSETS.items()):
        record = by_offset.get(offset)
        if record is None:
            detailed.append({"linkage_index": label,
                             "record_offset": offset, "error": "record not indexed"})
            continue
        entry = {"linkage_index": label, "record_offset": offset,
                 "moniker": record["moniker"], "groups": []}
        for group in record["display_groups"]:
            elements, status = walk_group_elements(data, group, offset)
            g_entry = {
                "group_index": group["index"],
                "display_code_raw": group["display_code_raw"],
                "bbox_world": group["bbox_world"],
                "walk_status": status, "elements": [],
            }
            for element in elements:
                if element["kind"] == "tag_block":
                    g_entry["elements"].append(tag_block_views(data, element, offset))
                else:
                    g_entry["elements"].append(element)
            entry["groups"].append(g_entry)
        detailed.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "file": str(args.vue.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "records": len(records), "header_errors": errors,
        "groups_total": len(group_rows),
        "walk_status": dict(status_counter),
        "nested_nodes_total": node_count,
        "d0_markers_total": d0_total,
        "d0_markers_at_element_start": d0_on_start,
        "tag_type_size_histogram": dict(type_counter.most_common()),
        "block_area_start": hex(BLOCK_AREA_START),
        "scope": ("Candidate element-framing rule for this 06.02.00.17 sample; "
                  "nested nodes observed only as trailing elements so far."),
    }
    (out_dir / "block_walk_summary.json").write_text(
        json.dumps({**summary, "groups": group_rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "block_sequence_probe.json").write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
