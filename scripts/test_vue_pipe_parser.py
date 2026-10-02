"""Regression checks for moniker references preceding the actual object record."""

import struct
import tempfile
import unittest
from pathlib import Path

from vue_pipe_parser import VuePipeParser


class RecordReferenceTests(unittest.TestCase):
    moniker = "@a=0027!!80012##123456789"

    def make_parser(self, early_reference, valid_record=True, terminal=False):
        data = bytearray(2048)
        token, record, cylinder = 1536, 384, 1056
        text = (" " + self.moniker + "\0").encode("ascii")
        data[token:token + len(text)] = text
        struct.pack_into("<I", data, early_reference, token)
        if valid_record:
            struct.pack_into("<II", data, record, 0x12, 0 if terminal else cylinder + 208)
            struct.pack_into("<I", data, record + 0x8C, token)
            struct.pack_into("<6d", data, record+8, 0.9, 1.9, 3, 1.1, 2.1, 3.75)
            struct.pack_into("<I", data, record+0x44, record+0x90)
            struct.pack_into("<I", data, record+0x6C, cylinder+208-record-0xB0)
            struct.pack_into("<I", data, record+0x70, record+0xB0)
            struct.pack_into("<II", data, record+0xBC, 10, 704)
            struct.pack_into("<I", data, record+0xB0+32, 64)
            struct.pack_into("<II", data, cylinder, 0x1F, 0xD0)
            struct.pack_into("<3d", data, cylinder + 0x50, 0, 0, 0.75)
            struct.pack_into("<3d", data, cylinder + 0x78, 1, 2, 3)
            struct.pack_into("<3d", data, cylinder + 0x90, 0, 0, 1)
            struct.pack_into("<3d", data, cylinder + 0xA8, 0.1, 0, 0)
        if early_reference == 172:
            # A marker match with an out-of-file link must not hide a later record.
            struct.pack_into("<II", data, 32, 0x12, 5000)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "references.vue"
        path.write_bytes(data)
        return VuePipeParser(path)

    def test_earlier_nonrecord_reference_does_not_hide_pipe(self):
        parser = self.make_parser(8)
        result = parser.parse_pipe(self.moniker)
        self.assertEqual(result["rec_start"], 384)
        self.assertAlmostEqual(result["length"], 0.75)
        self.assertEqual(result["p2"], (1.0, 2.0, 3.75))

    def test_invalid_link_does_not_hide_later_record(self):
        parser = self.make_parser(172)
        self.assertEqual(parser.find_record(self.moniker), (384, 1264))

    def test_token_without_valid_record_remains_unresolved(self):
        parser = self.make_parser(172, valid_record=False)
        self.assertIsNone(parser.find_record(self.moniker))

    def test_null_successor_does_not_make_record_disappear(self):
        parser = self.make_parser(8, terminal=True)
        self.assertEqual(parser.find_record(self.moniker), (384, 0))


if __name__ == "__main__":
    unittest.main()
