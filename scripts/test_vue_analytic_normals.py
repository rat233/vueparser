"""P2/P3 cylinder normals and the merged export path, using existing samples."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import vue_preview_export as v
from vue_glb_finalize import read_glb, accessor_array
from vue_record_parser import read_records


class CylinderAnalyticNormalsTests(unittest.TestCase):
    def test_wall_and_caps_have_native_normals(self):
        block=dict(point=np.array([1., 2., 3.]), axis=np.array([0., 0., 2.]),
                   dirv=np.array([0., 0., 1.]), radius=.5)
        vertices, faces, normals=v.cylinder(block, 16, with_normals=True)
        self.assertEqual(vertices.shape, (66, 3))
        self.assertEqual(faces.shape, (64, 3))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        np.testing.assert_allclose(normals[:16], (vertices[:16]-block['point'])/.5, atol=1e-14)
        np.testing.assert_allclose(normals[32:48], np.tile([0,0,-1], (16,1)))
        np.testing.assert_allclose(normals[48:64], np.tile([0,0,1], (16,1)))
        triangles=vertices[faces]
        geometric=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        self.assertTrue(np.all(np.einsum('ij,ij->i', geometric,normals[faces].mean(1))>0))
        block['axis']=np.zeros(3)
        vertices,faces,normals=v.cylinder(block,16,with_normals=True)
        self.assertEqual(vertices.shape,(17,3))
        self.assertEqual(faces.shape,(16,3))
        np.testing.assert_allclose(normals,np.tile([0,0,1],(17,1)))

    def test_real_cylinders_skip_split_and_survive_merged_export(self):
        data=Path('15300.vue').read_bytes()
        header,records,errors=read_records(data)
        selected=[r for r in records if r['record_offset'] in (2229660,5785204,2651348)]
        self.assertEqual(len(selected),3)
        tables=v.read_scene_tables(data,records)
        settings=v.ExportOptions(format='all',reports=False,merge_moniker=True,
                                 unknown_bbox_wireframe=False)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(v,'read_records',return_value=(header,selected,errors)),patch.object(v,'read_scene_tables',return_value=tables),patch.object(v,'write_models') as writer,patch('builtins.print'):
                summary=v.export_sample(Path('15300.vue'),Path(folder),settings)
            pieces=writer.call_args.args[1]
            analytic=[p for p in pieces if p.get('normals') is not None]
            self.assertTrue(analytic)
            for p in analytic:
                normals=p['normals']
                self.assertEqual(normals.shape,p['local_vertices'].shape)
                np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-6)
                triangles=p['local_vertices'][p['faces']]
                geometric=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                self.assertTrue(np.all(np.einsum('ij,ij->i',geometric,normals[p['faces']].mean(1))>=-1e-12))
            fallback=sum(v.needs_normal_split(p) for p in pieces)
            base=Path(folder)/'merged'
            with patch.object(v,'split_hard_edges_many',wraps=v.split_hard_edges_many) as splitter:
                v.write_models(base,pieces,settings)
            self.assertEqual(len(splitter.call_args.args[0]),fallback)
            self.assertLess(fallback,len([p for p in pieces if not v.is_line_piece(p)]))
            tree,binary=read_glb(base.with_suffix('.glb').read_bytes())
            self.assertEqual(len(tree['meshes']),len({p['moniker'] for p in pieces}))
            self.assertEqual(tree['extras']['vue_coordinates']['up_axis'],'y')
            triangles=0
            for mesh in tree['meshes']:
                for primitive in mesh['primitives']:
                    if primitive.get('mode',4)!=4:
                        continue
                    normals=accessor_array(tree,binary,primitive['attributes']['NORMAL'])
                    np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-5)
                    triangles+=tree['accessors'][primitive['indices']]['count']//3
            self.assertEqual(triangles,summary['triangles'])
            obj=base.with_suffix('.obj').read_text(encoding='utf-8')
            self.assertEqual(sum(row.startswith('o ') for row in obj.splitlines()),len(tree['meshes']))
            self.assertTrue(base.with_name('merged_compatible.fbx').exists())


class BasicPrimitiveNormalsTests(unittest.TestCase):
    def assert_normals(self, vertices, faces, normals):
        self.assertEqual(vertices.shape,normals.shape)
        self.assertTrue(np.isfinite(normals).all())
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-6)
        triangle=vertices[faces]
        cross=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
        self.assertTrue(np.all(np.einsum('ij,ij->i',cross,normals[faces].mean(1))>=-1e-10))

    def test_box_has_six_flat_faces(self):
        profile=np.array([[0,0,0],[0,2,0],[0,2,3],[0,0,3],[0,0,0]],dtype=float)
        vertices,faces,normals=v.extrusion(profile,[4,0,0],np.eye(4),with_normals=True)
        self.assert_normals(vertices,faces,normals)
        self.assertEqual(len(vertices),24)
        self.assertEqual(len(faces),12)
        self.assertEqual(len(np.unique(normals,axis=0)),6)
        self.assertTrue(np.allclose(normals[faces][:,0],normals[faces][:,1]))
        self.assertTrue(np.allclose(normals[faces][:,1],normals[faces][:,2]))

    def test_round_sweeps_keep_caps_and_signed_endpoints(self):
        for angle in (np.pi/2,-np.pi/2,2*np.pi):
            vertices,faces,normals=v.arc_tube_precise(
                [0,0,0],[2,0,0],[0,1,0],[0,0,1],.2,angle,np.eye(4),16,with_normals=True)
            self.assert_normals(vertices,faces,normals)
            import trimesh
            mesh=trimesh.Trimesh(vertices,faces,process=True)
            self.assertTrue(mesh.is_watertight)
            self.assertGreater(mesh.volume,0)
            if abs(angle)<2*np.pi:
                np.testing.assert_allclose(vertices[-2],[2,0,0],atol=1e-14)
                np.testing.assert_allclose(vertices[-1],[2*np.cos(angle),2*np.sin(angle),0],atol=1e-14)
                self.assertEqual(len(faces),160)
            else:
                self.assertEqual(len(vertices),256)
                self.assertEqual(len(faces),512)

    def test_real_basic_families_and_merged_glb(self):
        suffixes=('__cylinders','__spheres','__plates','__torus','__ellipsoid_head',
                  '__revolved_profile_surfaces','__ruled_circles','__boundary_faces',
                  '__planar_faces','__annular_surfaces','__native_planar_faces',
                  '__annular_faces','__frusta','__cones','__boxes','__arc_tube',
                  '__floor_hole_walls','__cylinder_surface_nurbs','__stair_steps')
        seen=set()
        original=v.write_models
        for sample in ('063E','15300'):
            with self.subTest(sample=sample),tempfile.TemporaryDirectory() as folder:
                totals={}
                def inspect_and_write(base,pieces,settings,*args,**kwargs):
                    self.assertTrue(settings.merge_moniker)
                    expected=len({p['moniker'] for p in pieces})
                    analytic=0
                    for piece in pieces:
                        family=next((x for x in suffixes if x+'__0x' in piece['name']),None)
                        if family is not None:
                            self.assertIsNotNone(piece.get('normals'),piece['name'])
                            seen.add(family)
                        if piece.get('normals') is not None:
                            self.assert_normals(piece['local_vertices'],piece['faces'],piece['normals'])
                            analytic+=1
                    original(base,pieces,settings,*args,**kwargs)
                    tree,binary=read_glb(base.with_suffix('.glb').read_bytes())
                    self.assertEqual(len(tree['meshes']),expected)
                    self.assertEqual(tree['extras']['vue_coordinates']['up_axis'],'y')
                    totals.update(analytic=analytic,total=len(pieces),merged=expected)
                    for mesh in tree['meshes']:
                        for primitive in mesh['primitives']:
                            if primitive.get('mode',4)==4:
                                n=accessor_array(tree,binary,primitive['attributes']['NORMAL'])
                                self.assertTrue(np.isfinite(n).all())
                            else:
                                self.assertNotIn('NORMAL',primitive['attributes'])
                                color=tree['materials'][primitive['material']]['pbrMetallicRoughness']['baseColorFactor']
                                self.assertEqual(color[:3],[0,0,0])
                with patch.object(v,'write_models',side_effect=inspect_and_write),patch('builtins.print'):
                    summary=v.export_sample(Path(sample+'.vue'),Path(folder),v.ExportOptions(
                        format='glb',reports=False,merge_moniker=True,unknown_bbox_wireframe=False))
                self.assertEqual(summary['objects_without_geometry'],0)
                print(sample,totals,'triangles',summary['triangles'],'weld_edges',summary['line_segments'])
        self.assertEqual(seen,set(suffixes))


if __name__=='__main__':
    unittest.main()
