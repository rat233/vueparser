import csv,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import vue_preview_export as v
from vue_record_parser import read_records

class WeldDisplayTests(unittest.TestCase):
    def test_target_export_is_virtual_and_retains_ring_center_and_radius(self):
        data=Path('15300.vue').read_bytes();header,records,errors=read_records(data)
        selected=[r for r in records if r['record_offset'] in (2651348,1996484)]
        tables=v.read_scene_tables(data,records)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(v,'read_records',return_value=(header,selected,errors)),patch.object(v,'read_scene_tables',return_value=tables),patch.object(v,'write_models') as writer,patch.object(v,'arc_tube_precise',side_effect=AssertionError('weld generated tube')),patch('builtins.print'):
                v.export_sample(Path('15300.vue'),Path(folder),v.ExportOptions(unknown_bbox_wireframe=False))
            rows=list(csv.DictReader((Path(folder)/'geometry.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(row['kind'],'weld_line')
                self.assertEqual(row['virtual_geometry'],'True')
                self.assertEqual(row['placement'],'weld_ring_line')
            item=next(p for p in writer.call_args.args[1] if p['record_offset']==2651348)
            self.assertEqual(item['color'][:3], [0, 0, 0])
            # Exercise the default Python export used for the published samples.
            from vue_glb_finalize import read_glb, accessor_array, node_matrix
            base=Path(folder)/'black_weld'
            v.write_models(base,writer.call_args.args[1],v.ExportOptions(format='glb'))
            tree,binary=read_glb(base.with_suffix('.glb').read_bytes())
            self.assertEqual(tree['extras']['vue_coordinates']['up_axis'],'y')
            root=tree['scenes'][0]['nodes'][0]
            np.testing.assert_allclose(node_matrix(tree['nodes'][root])[:3,:3] @ [0,0,1], [0,1,0])
            for mesh in tree['meshes']:
                for primitive in mesh['primitives']:
                    self.assertEqual(primitive['mode'],1)
                    tint=tree['materials'][primitive['material']]['pbrMetallicRoughness']['baseColorFactor']
                    self.assertEqual(tint[:3],[0,0,0])
                    colors=accessor_array(tree,binary,primitive['attributes']['COLOR_0'])
                    self.assertTrue(np.all(colors[:,:3]==0))
            world=item['local_vertices']+item['origin']
            center=np.array(struct.unpack_from('<3d',data,2651348+672+0x50))
            delta=world-center
            np.testing.assert_allclose((world.max(0)+world.min(0))/2,center,atol=1e-9)
            self.assertFalse(item['placeholder'])
            self.assertEqual(item['primitive_type'], 'lines')
            self.assertEqual(item['faces'].shape, (16, 2))
            self.assertEqual(len(world), 16)
            self.assertAlmostEqual(np.ptp(delta[:,2]),0,places=8)
            radial=np.linalg.norm(delta[:,:2],axis=1)
            np.testing.assert_allclose(radial, .4065, atol=1e-9)

if __name__=='__main__':unittest.main()
