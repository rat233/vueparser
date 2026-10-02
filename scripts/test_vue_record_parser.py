"""Behavioral checks for VUE extent recovery and insulation separation."""
import struct
import tempfile
import unittest
from pathlib import Path
from vue_record_parser import parse_record_header, parse_display_groups, annotate_field_50
from vue_pipe_parser import VuePipeParser


class HeaderAndInsulationTests(unittest.TestCase):
    def fixture(self, insulated=False):
        start = 64
        span = 1552 if insulated else 872
        token = 1900
        data = bytearray(2048)
        moniker = b" @a=0027!!80012##987654321\0"
        data[token:token+len(moniker)] = moniker
        struct.pack_into("<II", data, start, 0x12, 0)
        struct.pack_into("<6d", data, start+8, -0.05, -0.05, 0, 0.05, 0.05, 1)
        for off, value in [(0x44, start+0x90), (0x50, 0xFFFFFFFF),
                           (0x6C, span-0xB0), (0x70, start+0xB0), (0x8C, token)]:
            struct.pack_into("<I", data, start+off, value)
        struct.pack_into("<4I", data, start+0xB0, 80, 760 if insulated else 80,
                         span-184, 2 if insulated else 1)
        for index in range(2 if insulated else 1):
            head = start+0xC0+680*index
            cylinder = start+0x2A0+680*index
            radius = 0.02+0.03*index
            struct.pack_into("<9I", data, head, 168, 168, 760 if insulated and index == 0 else 0,
                             10, 676, 1, 1, 0, 64 if index == 0 else 69)
            struct.pack_into("<6d", data, head+40, -radius, -radius, 0, radius, radius, 1)
            struct.pack_into("<II", data, cylinder, 0x1F, 0xD0)
            struct.pack_into("<6d", data, cylinder+8, -radius, -radius, 0, radius, radius, 1)
            struct.pack_into("<3d", data, cylinder+0x50, 0, 0, 1)
            struct.pack_into("<3d", data, cylinder+0x90, 0, 0, 1)
            struct.pack_into("<3d", data, cylinder+0xA8, radius, 0, 0)
        return data, start, span

    def test_null_successor_uses_payload_extent_not_field_50(self):
        data, start, span = self.fixture()
        header = parse_record_header(data, start)
        self.assertEqual(header["next_ptr_raw"], 0)
        self.assertEqual(header["record_end"], start+span)
        self.assertEqual(header["field_50_raw"], 0xFFFFFFFF)

    def test_out_of_file_payload_is_rejected(self):
        data, start, _ = self.fixture()
        struct.pack_into("<I", data, start+0x6C, 100000)
        with self.assertRaises(ValueError):
            parse_record_header(data, start)

    def test_cyclic_group_list_is_rejected(self):
        data, start, _ = self.fixture(insulated=True)
        struct.pack_into("<I", data, start+0xC0+8, 80)
        with self.assertRaises(ValueError):
            parse_display_groups(data, parse_record_header(data, start))

    def test_equal_length_outer_shell_is_not_selected_as_pipe(self):
        data, start, _ = self.fixture(insulated=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"insulated.vue"
            path.write_bytes(data)
            parsed = VuePipeParser(path).parse_pipe(
                "@a=0027!!80012##987654321", {"Insulation Thickness": "30.00 mm"})
        self.assertAlmostEqual(parsed["radius"], 0.02)
        self.assertAlmostEqual(parsed["insulation_blocks"][0]["radius"], 0.05)
        self.assertEqual(parsed["n_flanges"], 0)
        self.assertEqual(parsed["insulation_status"], "matched_mdb_and_geometry")
        self.assertAlmostEqual(parsed["length"], 1.0)


    def test_field_50_resolves_record_starts_even_when_equal_to_ends(self):
        records = [
            {"record_offset": 100, "record_end": 500, "field_50_raw": 500},
            {"record_offset": 500, "record_end": 900, "field_50_raw": 1300},
            {"record_offset": 900, "record_end": 1300, "field_50_raw": 1300},
            {"record_offset": 1300, "record_end": 1700, "field_50_raw": 0},
        ]
        annotate_field_50(records)
        self.assertEqual(records[0]["field_50_interpretation"], "next_record_start")
        self.assertEqual(records[1]["field_50_interpretation"], "next_record_start")
        self.assertEqual(records[2]["field_50_interpretation"], "next_record_start")
        self.assertEqual(records[3]["field_50_interpretation"], "discipline_chain_tail")

    def test_field_50_unmatched_value_stays_raw(self):
        records = [{"record_offset": 100, "record_end": 500, "field_50_raw": 0xFFFFFFFF}]
        annotate_field_50(records)
        self.assertEqual(records[0]["field_50_interpretation"], "unmatched")


if __name__ == "__main__":
    unittest.main()
