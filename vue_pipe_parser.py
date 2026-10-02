# -*- coding: utf-8 -*-
"""
vue_pipe_parser.py — 两个 SmartPlant3D 06.02.00.17 样本的直管研究解析器。

对象物理终点由 +0x70 载荷指针和 +0x6c 大小确定，空 next_ptr 也可解析。
0x1F/0xD0 圆柱以 refvec 的模表示半径；该布局不能推广到所有几何编码。
先按显示组拆分管体与保温，再提取主管、附件及保温对应关系。
已验证直管参数采用世界坐标 (E,N,EL)、米；其他元件可能使用局部坐标。

用法：
    parser = VuePipeParser("063E.vue")
    result = parser.parse_pipe("@a=0027!!80012##302702794249865165",
                               {"Insulation Thickness": "30.00 mm"})

2026-09-06：063E 的 90 条直管可读，其中 25 条保温关联通过几何/MDB 校核；
15300 的 907 条候选有 798 条得到解析结果，不等于 798 条均已通过独立验证。
批量属性和分层结果请用 vue_sample_audit.py，详见 VUE_063E_INSULATION_RESEARCH.md。
"""
import math
import mmap
import re
import struct
from pathlib import Path
from vue_record_parser import build_moniker_indexes, parse_record_header, parse_display_groups

REC_MAGIC = b"\x12\x00\x00\x00"
CYL_MARKER = b"\x1f\x00\x00\x00\xd0\x00\x00\x00"
MONIKER_REF_OFF = 0x8C


def _vlen(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


class VuePipeParser:
    def __init__(self, vue_path):
        self.path = Path(vue_path)
        # B4 stage 1: map the input read-only instead of slurping it into RAM.
        # An mmap supports every bytes interface used downstream (find,
        # slicing, struct.unpack_from, np.frombuffer, hashlib), so records can
        # be decoded straight from the file-backed pages; the writable-copy
        # sites already copy explicitly (e.g. bytearray slices).
        with self.path.open('rb') as stream:
            if stream.seek(0, 2):
                self._mmap = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                self.data = self._mmap
            else:
                self._mmap = None
                self.data = b''
        self._build_indexes()

    # ---------- 底层索引 ----------
    def _build_indexes(self):
        self.tokens, self.ref_by_tok = build_moniker_indexes(self.data)

    def moniker_to_display_key(self, moniker):
        return (moniker.strip().replace("@a=", "", 1)
                .replace("!!", "-", 1).replace("##", "-", 1))

    # ---------- 记录级 ----------
    def find_record(self, moniker):
        """moniker → (rec_start, next_ptr); 找不到返回 None"""
        # 建立 token 文本→偏移映射 (懒加载)
        if not hasattr(self, "_tok_by_text"):
            self._tok_by_text = {v: k for k, v in self.tokens.items()}
        tok = self._tok_by_text.get(moniker.strip())
        if tok is None:
            return None
        refs = self.ref_by_tok.get(tok)
        if not refs:
            return None
        # token 也会被记录外的其他结构引用，首个引用不一定属于对象头。
        for ref in refs:
            rs = ref - MONIKER_REF_OFF
            if rs < 0 or self.data[rs:rs + 4] != REC_MAGIC:
                continue
            nxt = struct.unpack_from("<I", self.data, rs + 4)[0]
            if nxt == 0 or rs + MONIKER_REF_OFF + 4 <= nxt <= len(self.data):
                return rs, nxt
        return None

    def record_bbox(self, rec_start):
        return struct.unpack_from("<6d", self.data, rec_start + 8)

    # ---------- 圆柱子块 ----------
    def scan_cylinders(self, rec_start, next_ptr):
        """Scan observed 0x1F/0xD0 candidates within a trusted physical record end.

        This is not a general nested-geometry traversal. The readable fields end
        at tag+0xC8; the raw 0xD0 size includes framing not fully decoded yet.
        """
        data = self.data
        blocks = []
        pos = rec_start + 0xA0
        while True:
            idx = data.find(CYL_MARKER, pos, next_ptr)
            if idx < 0:
                break
            if idx + 0xC8 <= next_ptr:
                b = dict(
                    off=idx - rec_start,
                    bbox=struct.unpack_from("<6d", data, idx + 8),
                    seq=struct.unpack_from("<I", data, idx + 0x38)[0],
                    axis=struct.unpack_from("<3d", data, idx + 0x50),
                    point=struct.unpack_from("<3d", data, idx + 0x78),
                    dirv=struct.unpack_from("<3d", data, idx + 0x90),
                    refvec=struct.unpack_from("<3d", data, idx + 0xA8),
                )
                b["height"] = _vlen(b["axis"])
                b["radius"] = _vlen(b["refvec"])
                blocks.append(b)
            pos = idx + 8
        return blocks

    # ---------- 直管语义 ----------
    @staticmethod
    def merge_main_pipe(blocks):
        """同半径柱段并集 → (总长, 半径, 段列表)"""
        cands = [b for b in blocks if b["height"] > 1e-6 and b["radius"] > 1e-6]
        groups = []
        for b in sorted(cands, key=lambda b: -b["radius"]):
            for g in groups:
                if abs(g["r"] - b["radius"]) < 0.002:
                    g["segs"].append(b)
                    break
            else:
                groups.append(dict(r=b["radius"], segs=[b]))
        for g in groups:
            g["total"] = sum(s["height"] for s in g["segs"])
        if not groups:
            return 0.0, 0.0, []
        main = max(groups, key=lambda g: g["total"])
        return main["total"], main["r"], main["segs"]

    @staticmethod
    def classify_attachments(blocks, main_segs, main_r):
        """附加块分类: 法兰/密封面/其他"""
        flanges, gaskets, others = [], [], []
        main_ids = {id(b) for b in main_segs}
        for b in blocks:
            if id(b) in main_ids:
                continue
            if b["height"] < 1e-6 and b["radius"] > main_r:
                gaskets.append(b)
            elif b["radius"] > main_r + 0.002:
                flanges.append(b)
            else:
                others.append(b)
        return flanges, gaskets, others

    @staticmethod
    def pipe_endpoints(segs):
        """同轴段 → 中心线两端点"""
        ends = []
        for s in segs:
            a, v = s["point"], s["axis"]
            ends.append(tuple(a))
            ends.append((a[0] + v[0], a[1] + v[1], a[2] + v[2]))
        if not ends:
            return None, None
        # 沿轴向排序取两端
        pts = sorted(ends)
        return pts[0], pts[-1]

    # ---------- 顶层 API ----------
    def parse_pipe(self, moniker, properties=None):
        """Decode observed straight-pipe geometry and retain separate insulation.

        Body/shell roles use the validated two-sample display-group layout.
        Non-pipe local geometry and unsupported type/size pairs remain outside
        this straight-pipe decoder.
        """
        loc = self.find_record(moniker)
        if loc is None:
            return None
        rs, nxt = loc
        try:
            header = parse_record_header(self.data, rs)
            _, display_groups = parse_display_groups(self.data, header)
            end = header["record_end"]
        except ValueError:
            return None
        blocks = self.scan_cylinders(rs, end)
        if not blocks:
            return None
        insulation_blocks = []
        body_blocks = []
        for block in blocks:
            group = next((g for g in display_groups
                          if g["header_start"] <= rs + block["off"] < g["end"]), None)
            block["display_code_raw"] = group["display_code_raw"] if group else None
            block["group_index"] = group["index"] if group else None
            if group and group["display_code_raw"] == 69:
                insulation_blocks.append(block)
            else:
                body_blocks.append(block)
        plen, prad, segs = self.merge_main_pipe(body_blocks)
        if not segs:
            return None
        flanges, gaskets, others = self.classify_attachments(body_blocks, segs, prad)
        p1, p2 = self.pipe_endpoints(segs)
        matches = []
        for shell in insulation_blocks:
            sa = shell["point"]
            sb = tuple(sa[i] + shell["axis"][i] for i in range(3))
            for body in segs:
                ba = body["point"]
                bb = tuple(ba[i] + body["axis"][i] for i in range(3))
                distance = lambda a, b: _vlen(tuple(a[i] - b[i] for i in range(3)))
                endpoint_error = min(max(distance(sa, ba), distance(sb, bb)),
                                     max(distance(sa, bb), distance(sb, ba)))
                delta = shell["radius"] - body["radius"]
                if endpoint_error < 1e-7 and delta > 0:
                    matches.append({"body_offset": body["off"], "shell_offset": shell["off"],
                                    "thickness_m": delta, "endpoint_error_m": endpoint_error})
                    break
        paired = bool(insulation_blocks) and len(matches) == len(insulation_blocks)
        status = "none"
        thickness = None
        if insulation_blocks:
            status = "paired_geometry_only" if paired else "unverified_shell_geometry"
            if paired:
                thickness = matches[0]["thickness_m"]
                if properties is not None:
                    raw = properties.get("Insulation Thickness", "").strip()
                    matched = re.fullmatch(r"([0-9.]+)\s*(mm|in)", raw)
                    if matched:
                        expected = float(matched[1]) * (0.001 if matched[2] == "mm" else 0.0254)
                        status = ("matched_mdb_and_geometry"
                                  if all(abs(m["thickness_m"]-expected) < 1e-7 for m in matches)
                                  else "mdb_geometry_mismatch")
        bbox_errors = []
        for body in body_blocks:
            height = body["height"]
            if height <= 1e-12:
                continue
            a = body["point"]
            b = tuple(a[i]+body["axis"][i] for i in range(3))
            ext = [body["radius"] * math.sqrt(max(0.0, 1.0-(body["axis"][i]/height)**2))
                   for i in range(3)]
            calculated = [min(a[i], b[i])-ext[i] for i in range(3)]
            calculated += [max(a[i], b[i])+ext[i] for i in range(3)]
            bbox_errors.append(max(abs(x-y) for x, y in zip(calculated, body["bbox"])))
        return dict(
            moniker=moniker.strip(), display_key=self.moniker_to_display_key(moniker),
            rec_start=rs, rec_end=end, next_ptr_raw=nxt,
            length=plen, radius=prad, p1=p1, p2=p2,
            n_cyl_blocks=len(blocks), n_main_segs=len(segs),
            n_flanges=len(flanges), n_gaskets=len(gaskets),
            blocks=blocks, body_blocks=body_blocks, insulation_blocks=insulation_blocks,
            display_groups=display_groups, insulation_status=status,
            insulation_thickness_m=thickness, insulation_matches=matches,
            body_bbox_error_m=max(bbox_errors, default=0.0),
            insulation_opacity=None,
        )


# ---------- CLI ----------
if __name__ == "__main__":
    import csv
    import argparse

    workspace = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Export the observed straight-pipe parameters.")
    ap.add_argument("vue", nargs="?", type=Path, default=workspace/"15300.vue")
    ap.add_argument("mdb_csv", nargs="?", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    vue = args.vue
    mdb_csv = args.mdb_csv or vue.with_name(vue.stem+"_mdb_export")/"elements.csv"

    parser = VuePipeParser(vue)

    pipes = []
    with open(mdb_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("file_name") == "Piping"
                    and row.get("Name", "").strip() == "Pipe"):
                pipes.append(row)

    out_rows = []
    for p in pipes:
        r = parser.parse_pipe(p["moniker"], properties=p)
        if r is None:
            continue
        out_rows.append([
            p["linkage_index"], r["display_key"], p.get("NPD", ""),
            f"{r['length']:.4f}", f"{r['radius']:.6f}",
            f"{r['p1'][0]:.4f}", f"{r['p1'][1]:.4f}", f"{r['p1'][2]:.4f}",
            f"{r['p2'][0]:.4f}", f"{r['p2'][1]:.4f}", f"{r['p2'][2]:.4f}",
            r["n_cyl_blocks"], r["n_main_segs"], r["n_flanges"], r["n_gaskets"],
        ])

    out = args.output or workspace/"outputs"/vue.stem/"pipe_parser.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["linkage_index", "display_key", "npd", "length_m", "radius_m",
                    "p1_e", "p1_n", "p1_el", "p2_e", "p2_n", "p2_el",
                    "n_cyl_blocks", "n_main_segs", "n_flanges", "n_gaskets"])
        w.writerows(out_rows)
    print(f"解析 {len(out_rows)} 根直管 → {out.resolve()}")
