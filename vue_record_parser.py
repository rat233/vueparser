"""Persistent, bounded header and display-group reader for the two VUE samples.

Known fields are decoded; undecoded header bytes and display codes remain raw.
This is a research reader, not a claim of universal VUE layout compatibility.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

RECORD_MAGIC = b"\x12\x00\x00\x00"
WORKSPACE = Path(__file__).resolve().parent


def build_moniker_indexes(data):
    tokens = {}
    pos = 0
    while True:
        start = data.find(b" @a=", pos)
        if start < 0:
            break
        end = data.find(b"\0", start)
        if end < 0:
            break
        tokens[start] = data[start:end].decode("ascii", "replace").strip()
        pos = end + 1
    references = {}
    for offset in range(0, len(data) - 3, 4):
        value = struct.unpack_from("<I", data, offset)[0]
        if value in tokens:
            references.setdefault(value, []).append(offset)
    return tokens, references


def parse_record_header(data, start):
    if start < 0 or start + 0xC0 > len(data) or data[start:start+4] != RECORD_MAGIC:
        raise ValueError("Not a complete supported object header")
    u32 = lambda off: struct.unpack_from("<I", data, start + off)[0]
    payload_start, payload_size = u32(0x70), u32(0x6C)
    if payload_start != start + 0xB0 or u32(0x44) != start + 0x90:
        raise ValueError("Unsupported payload/header pointer layout")
    end = payload_start + payload_size
    if payload_size < 16 or end > len(data):
        raise ValueError("Payload extent outside file or too short")
    if u32(4) not in (0, end):
        raise ValueError("Non-null successor conflicts with payload extent")
    bbox = struct.unpack_from("<6d", data, start + 8)
    if not all(math.isfinite(v) for v in bbox) or any(bbox[i] > bbox[i+3] for i in range(3)):
        raise ValueError("Invalid object bounding box")
    return {
        "record_offset": start, "next_ptr_raw": u32(4),
        "record_end": end, "payload_start": payload_start, "payload_size": payload_size,
        "extension_ptr": u32(0x44), "field_50_raw": u32(0x50),
        "moniker_token_offset": u32(0x8C), "bbox_world": bbox,
        "header_raw_hex": data[start:start+0xB0].hex(),
        "payload_prefix_raw_hex": data[payload_start:payload_start+16].hex(),
        "core_region_raw_hex": data[start+0x38:start+0x8C].hex(),
        "undecoded_ranges": [
            {"relative_start": lo, "relative_end": hi, "raw_hex": data[start+lo:start+hi].hex()}
            for lo, hi in ((0x38, 0x44), (0x48, 0x6C), (0x74, 0x8C), (0x90, 0xB0))
        ],
        "boundary_evidence": "payload_ptr_70_plus_size_6c; nonnull_next_consistency",
    }


def parse_display_groups(data, header):
    start, end = header["record_offset"], header["record_end"]
    u32 = lambda absolute: struct.unpack_from("<I", data, absolute)[0]
    heads = []
    if u32(start+0xBC) == 10:
        layout = "direct_root"
        heads = [start+0xB0]
        if u32(heads[0]+8) != 0:
            raise ValueError("Unsupported linked direct root")
    else:
        layout = "root_list"
        first, last, storage_tail, count = struct.unpack_from("<4I", data, start+0xB0)
        if not 1 <= count <= 10000:
            raise ValueError("Unsupported root count")
        ptr, visited = first, set()
        while ptr:
            if ptr in visited or len(heads) >= count:
                raise ValueError("Cyclic/oversized root list")
            visited.add(ptr)
            head = start + 0x70 + ptr
            if head < start+0xC0 or head+88 > end:
                raise ValueError("Root pointer outside payload")
            if heads and head <= heads[-1]:
                raise ValueError("Non-monotonic root ranges not supported")
            heads.append(head)
            ptr = u32(head+8)
        if len(heads) != count or heads[-1] != start+0x70+last:
            raise ValueError("Root count/tail mismatch")
    groups = []
    for index, head in enumerate(heads):
        stop = heads[index+1] if index+1 < len(heads) else end
        if head+88 > stop or u32(head+12) != 10:
            raise ValueError("Unsupported display-group layout")
        bbox = struct.unpack_from("<6d", data, head+40)
        if not all(math.isfinite(v) for v in bbox):
            raise ValueError("Non-finite display-group bbox")
        code = u32(head+32)
        groups.append({
            "index": index, "header_start": head, "tag_offset": head+12,
            "end": stop, "type_raw": 10, "size_raw": u32(head+16),
            "display_code_raw": code, "bbox_world": bbox,
            "role_candidate": {64: "primary", 69: "insulation"}.get(code, "unclassified"),
            "header_raw_hex": data[head:head+88].hex(),
            "opacity": None,
        })
    return layout, groups


def annotate_field_50(records):
    """Resolve +0x50 to the next record START in a discipline list.

    The directory and material discipline index establish the list semantics
    (vue_scene_tables.py). An adjacent start also equals this record's end;
    that numeric coincidence is not a chain-block-end interpretation.
    """
    starts = {r['record_offset'] for r in records}
    for record in records:
        value = record['field_50_raw']
        record['field_50_interpretation'] = ('discipline_chain_tail' if value == 0 else
            'next_record_start' if value in starts else 'unmatched')
        record['next_in_discipline_offset'] = value if value in starts else None
    return records


def read_records(data, indexes=None):
    """Read records, optionally reusing Moniker indexes built from this data."""
    tokens, references = build_moniker_indexes(data) if indexes is None else indexes
    candidates = {}
    for token, refs in references.items():
        for ref in refs:
            start = ref-0x8C
            if start >= 0 and data[start:start+4] == RECORD_MAGIC:
                candidates[start] = token
    records, errors = [], []
    for start, token in sorted(candidates.items()):
        try:
            record = parse_record_header(data, start)
            record["moniker"] = tokens[token]
            record["layout"], record["display_groups"] = parse_display_groups(data, record)
            records.append(record)
        except ValueError as error:
            errors.append({"record_offset": start, "moniker": tokens[token], "error": str(error)})
    annotate_field_50(records)
    return tokens, records, errors


def export_headers(vue_path, output_dir):
    data = vue_path.read_bytes()
    tokens, records, errors = read_records(data)
    starts = sorted(r["record_offset"] for r in records)
    adjacent_matches = sum(
        r["record_end"] == (starts[i+1] if i+1 < len(starts) else len(data))
        for i, r in enumerate(records)
    )
    from collections import Counter
    summary = {
        "file": str(vue_path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "tokens": len(tokens), "records": len(records),
        "header_errors": errors, "null_successors": sum(r["next_ptr_raw"] == 0 for r in records),
        "ends_match_adjacent_record_or_eof": adjacent_matches,
        "layouts": dict(Counter(r["layout"] for r in records)),
        "display_codes": dict(Counter(g["display_code_raw"] for r in records for g in r["display_groups"])),
        "scope": "Core layouts only; native table semantics in vue_scene_tables.py; display overrides remain undecoded",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False)+"\n")
    (output_dir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vue", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    destination = args.output_dir or WORKSPACE/"outputs"/args.vue.stem/"headers"
    print(json.dumps(export_headers(args.vue, destination), ensure_ascii=False, indent=2))
