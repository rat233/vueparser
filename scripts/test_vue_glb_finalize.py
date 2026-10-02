"""Regression for GLB Y-up presentation and component/material consolidation."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from vue_glb_finalize import Z_TO_Y, read_glb, finalize_glb, node_matrix
from vue_export_options import ExportOptions, add_export_arguments, options_from_args
from vue_preview_export import write_models
from scripts.test_vue_export_options import pieces
from scripts.test_vue_business import fixture


class GlbPresentationTests(unittest.TestCase):
    def test_known_basis_and_idempotent_conversion(self):
        np.testing.assert_array_equal(Z_TO_Y[:3,:3]@np.array([0,0,1]),[0,1,0])
        self.assertEqual(np.linalg.det(Z_TO_Y[:3,:3]),1)
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'test'
            write_models(base,pieces(),ExportOptions('glb',False,False,glb_up_axis='y'))
            blob=base.with_suffix('.glb').read_bytes()
            once,binary=read_glb(blob)
            twice,binary2=read_glb(finalize_glb(blob,'y'))
            self.assertEqual(once,twice);self.assertEqual(binary,binary2)
            restored,_=read_glb(finalize_glb(blob,'z'))
            root=restored['scenes'][0]['nodes'][0]
            np.testing.assert_array_equal(node_matrix(restored['nodes'][root]),np.eye(4))
            scene=trimesh.load(base.with_suffix('.glb'),force='scene')
            expected=np.vstack([p['local_vertices']+p['origin'] for p in pieces()])@Z_TO_Y[:3,:3].T
            np.testing.assert_allclose(scene.bounds,[expected.min(0),expected.max(0)],atol=1e-7,rtol=0)

    def test_same_material_parts_merge_but_transparency_stays_separate(self):
        source=pieces()
        source.append(dict(source[0],name='A__extra',record_offset=100,origin=source[0]['origin']+np.array([2.,0,0])))
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'merged'
            write_models(base,source,ExportOptions('glb',False,True,glb_up_axis='y'),fixture())
            tree,_=read_glb(base.with_suffix('.glb').read_bytes())
            owners={n['extras']['moniker']:n for n in tree['nodes'] if 'moniker' in n.get('extras',{})}
            self.assertEqual(len(tree['meshes']),2)
            self.assertEqual(len(tree['meshes'][owners['A']['mesh']]['primitives']),2)
            self.assertNotIn('children',owners['A'])
            self.assertNotIn('mesh',owners['C'])
            self.assertEqual(sum('mesh' in n for n in tree['nodes']),2)
            alpha=[m['pbrMetallicRoughness']['baseColorFactor'][3] for m in tree['materials'] if m.get('alphaMode')=='BLEND']
            self.assertEqual(len(alpha),1)
            # 保留源材质的透明度。
            self.assertAlmostEqual(alpha[0],0.5)
            scene=trimesh.load(base.with_suffix('.glb'),force='scene')
            self.assertEqual(sum(len(g.faces) for g in scene.geometry.values()),48)
            expected=np.vstack([p['local_vertices']+p['origin'] for p in source])@Z_TO_Y[:3,:3].T
            np.testing.assert_allclose(scene.bounds,[expected.min(0),expected.max(0)],atol=1e-7,rtol=0)


    def test_command_line_defaults_and_merge_alias(self):
        parser=argparse.ArgumentParser();add_export_arguments(parser)
        defaults=options_from_args(parser.parse_args([]))
        self.assertFalse(defaults.reports)
        self.assertFalse(defaults.merge_moniker)
        self.assertTrue(defaults.unknown_bbox_wireframe)
        opts=options_from_args(parser.parse_args(['--merge-components','true']))
        self.assertTrue(opts.merge_moniker);self.assertEqual(opts.glb_up_axis,'y')
        opts=options_from_args(parser.parse_args(['--merge-components','false','--glb-up-axis','z']))
        self.assertFalse(opts.merge_moniker);self.assertEqual(opts.glb_up_axis,'z')
        opts=options_from_args(parser.parse_args(['--reports','true','--merge-moniker','true',
                                                 '--unknown-bbox-wireframe','false']))
        self.assertTrue(opts.reports)
        self.assertTrue(opts.merge_moniker)
        self.assertFalse(opts.unknown_bbox_wireframe)
        for name in ('--reports','--merge-moniker','--unknown-bbox-wireframe'):
            with self.subTest(name=name),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args([name,'yes'])
                self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
