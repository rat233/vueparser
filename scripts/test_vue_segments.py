"""Configurable segment counts and shared primitive/export behavior."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

import vue_preview_export as v
from vue_export_options import (DEFAULT_SEGMENTS, SEGMENTS_ALIGNMENT,
                                ExportOptions, add_export_arguments,
                                arc_path_segments, options_from_args)


class SegmentPolicyTests(unittest.TestCase):
    def test_default_is_sixteen(self):
        self.assertEqual(DEFAULT_SEGMENTS, 16)
        self.assertEqual(ExportOptions().segments, 16)

    def test_valid_values_are_accepted(self):
        for value in (16, 24, 32, 40, 256):
            with self.subTest(value=value):
                self.assertEqual(ExportOptions(segments=value).segments, value)

    def test_invalid_values_are_rejected_early(self):
        for value in (0, -16, 8, 15, 18, 20, 257, 512, True, 16.5, '16', None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ExportOptions(segments=value)

    def test_arc_path_scales_with_sweep(self):
        self.assertEqual(arc_path_segments(2 * np.pi, 16), 16)
        self.assertEqual(arc_path_segments(np.pi / 2, 16), 4)
        self.assertEqual(arc_path_segments(0.05, 16), 3)  # rvmparser fixed-mode minimum
        self.assertEqual(arc_path_segments(np.pi / 2, 32), 8)

    def test_cli_surface(self):
        parser = argparse.ArgumentParser()
        add_export_arguments(parser)
        self.assertEqual(options_from_args(parser.parse_args([])).segments, 16)
        self.assertEqual(options_from_args(parser.parse_args(['--segments', '24'])).segments, 24)
        self.assertEqual(options_from_args(parser.parse_args(['--segments', '32'])).segments, 32)
        for bad in ('8', '18', '300', 'abc', '16.5'):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(['--segments', bad])
                self.assertEqual(error.exception.code, 2)


class PureGeneratorSegmentTests(unittest.TestCase):
    def test_cylinder_segment_count(self):
        block = {'point': np.zeros(3), 'axis': np.array([0.0, 0.0, 1.0]),
                 'dirv': np.array([0.0, 0.0, 1.0]), 'radius': 1.0}
        vertices, faces = v.cylinder(block, segments=16)
        self.assertEqual(len(vertices), 2 * 16 + 2)
        self.assertEqual(len(faces), 4 * 16)
        _, faces32 = v.cylinder(block, segments=32)
        self.assertEqual(len(faces32), 4 * 32)

    def test_arc_tube_precise_is_closed_across_segment_tiers(self):
        for segments in (16, 24, 32):
            with self.subTest(segments=segments):
                vertices, faces = v.arc_tube_precise(
                    [0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1], 0.2, np.pi / 2,
                    np.eye(4), segments=segments)
                mesh = trimesh.Trimesh(vertices, faces, process=False)
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_winding_consistent)
                self.assertGreater(mesh.volume, 0)
                # A quarter arc keeps its own endpoint count, not a full circle.
                self.assertEqual(len(faces) // 2 // segments - 1,
                                 arc_path_segments(np.pi / 2, segments))


class RealSampleSegmentTests(unittest.TestCase):
    """Parameterized primitive counts against the 15300 sample when present."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / '15300.vue'
        if not path.exists():
            raise unittest.SkipTest('real VUE sample unavailable')
        from vue_record_parser import read_records
        cls.data = path.read_bytes()
        _, records, _ = read_records(cls.data)
        cls.records = {r['record_offset']: r for r in records}

    def _root(self, record_offset):
        from vue_root_transform_audit import read_matrix_pair
        record = self.records[record_offset]
        return read_matrix_pair(self.data, record['display_groups'][0])[0]

    def test_ellipsoid_head_circumferential_count(self):
        from vue_block_sequence_probe import walk_group_elements
        record_start = 5768316
        elements, _ = walk_group_elements(
            self.data, self.records[record_start]['display_groups'][0], record_start)
        node = next(e for e in elements if e['off_rel_record'] == 3840)
        matrix = np.array(node['matrix_a_rowmajor']).reshape(4, 4)
        for segments in (16, 24, 32):
            with self.subTest(segments=segments):
                vertices, faces = v.ellipsoid_head_from_vue(
                    self.data, record_start + 4816, record_start, matrix, segments=segments)
                self.assertEqual(len(vertices), v.ELLIPSOID_MERIDIAN_SAMPLES * segments + 1)
                self.assertEqual(len(faces),
                                 (v.ELLIPSOID_MERIDIAN_SAMPLES - 1) * segments * 2 + segments)
                mesh = trimesh.Trimesh(vertices - vertices.mean(0), faces, process=False)
                self.assertTrue(mesh.is_winding_consistent)

    def test_torus_ring_and_section_use_requested_count(self):
        from vue_block_sequence_probe import walk_group_elements
        record_start, node_offset, torus_offset = 6806132, 2480, 4112
        elements, _ = walk_group_elements(
            self.data, self.records[record_start]['display_groups'][0], record_start)
        node = next(e for e in elements if e['off_rel_record'] == node_offset)
        for segments in (16, 24, 32):
            with self.subTest(segments=segments):
                vertices, faces = v.torus_from_vue(
                    self.data, record_start + torus_offset,
                    node['matrix_a_rowmajor'], segments=segments)
                self.assertEqual(len(vertices), segments * segments)
                mesh = trimesh.Trimesh(vertices - vertices.mean(0), faces, process=False)
                self.assertTrue(mesh.is_watertight)
                self.assertEqual(mesh.euler_number, 0)

    def test_cone_vertex_and_pole_count(self):
        record_start = 8335260
        root = self._root(record_start)
        for segments in (16, 24, 32):
            with self.subTest(segments=segments):
                vertices, faces = v.cone_from_vue(
                    self.data, record_start + 672, record_start, root, segments=segments)
                self.assertEqual(len(vertices), segments + 1)
                self.assertEqual(len(faces), segments)

    def test_revolved_profile_416_watertight_at_tiers(self):
        record_start = 2991020
        root = self._root(record_start)
        for segments in (16, 24, 32):
            with self.subTest(segments=segments):
                vertices, faces = v.revolved_profile_416_from_vue(
                    self.data, record_start + 1520, record_start, root, segments=segments)
                self.assertEqual(len(faces), 16 * segments)
                mesh = trimesh.Trimesh(vertices, faces, process=False)
                self.assertTrue(mesh.is_watertight)

    def test_internal_cap_join_follows_head_segments(self):
        from vue_block_sequence_probe import walk_group_elements
        from vue_root_transform_audit import read_matrix_pair
        record_start = 2229660
        record = self.records[record_start]
        group = record['display_groups'][0]
        root = read_matrix_pair(self.data, group)[0]
        elements, _ = walk_group_elements(self.data, group, record_start)
        node = next(e for e in elements if e['kind'] == 'nested_node')
        for segments in (16, 96):
            with self.subTest(segments=segments):
                head = v.ellipsoid_head_from_vue(
                    self.data, record_start + 1264, record_start, root, segments=segments)
                disks = [v.boundary_face_from_vue(self.data, record_start + offset, record_start, m)[0]
                         for offset, m in ((672, root), (2672, node['matrix_a_rowmajor']),
                                           (2984, node['matrix_a_rowmajor']))]
                body = v.cylinder(dict(point=np.array([2480.3, 1549.5, 6]),
                                       axis=np.array([0, 0, 5]), radius=2.4), segments=segments)
                internal = v.internal_head_join_disks(disks, [head], [body])
                self.assertEqual(len(internal), 2)


class ExportWiringTests(unittest.TestCase):
    """export_sample must pass ExportOptions.segments to the generators."""

    def test_summary_records_requested_segments(self):
        source = Path('063E.vue')
        if not source.exists():
            self.skipTest('real VUE sample unavailable')
        with tempfile.TemporaryDirectory() as directory:
            summary = v.export_sample(
                source, Path(directory) / 's16',
                ExportOptions(format='glb', reports=False, mdb2=None))
            self.assertEqual(summary['segments'], 16)
            self.assertEqual(summary['segmentation']['requested_segments'], 16)
            self.assertEqual(summary['segmentation']['alignment'], SEGMENTS_ALIGNMENT)
            self.assertIn('cylinder', summary['segmentation']['observed'])
            lower = summary['triangles']
            higher = v.export_sample(
                source, Path(directory) / 's32',
                ExportOptions(format='glb', reports=False, mdb2=None, segments=32))
            self.assertEqual(higher['segmentation']['observed']['cylinder'], [(32,)])
            self.assertGreater(higher['triangles'], lower)


if __name__ == '__main__':
    unittest.main()
