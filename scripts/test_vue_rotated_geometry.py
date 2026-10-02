"""Regression checks for rotated native geometry (no reference FBX input)."""
import struct
import unittest
from pathlib import Path
import numpy as np
import trimesh
import vue_preview_export as v
from vue_record_parser import read_records
from vue_root_transform_audit import read_matrix_pair
from vue_block_sequence_probe import walk_group_elements

class RotatedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=Path('063E.vue').read_bytes()
        _,records,_=read_records(cls.data)
        cls.records={r['record_offset']:r for r in records}

    def test_export_keeps_nested_nodes_and_resolves_target_gaps(self):
        import tempfile, csv
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(v,'write_models') as writer, patch('builtins.print'):
                v.export_sample(Path('063E.vue'),Path(directory))
            pieces=writer.call_args.args[1]
            names=[p['name'] for p in pieces]
            self.assertEqual(len(names),len(set(names)))
            gaps=list(csv.DictReader((Path(directory)/'gaps.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertFalse([r for r in gaps if int(r['record_offset']) in (1007964,1048132,1087444,794324)])
            self.assertFalse([r for r in gaps if int(r['record_offset']) == 900996 and r['offset'] == '672'])
            rows=list(csv.DictReader((Path(directory)/'geometry.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertTrue(any(r['record_offset'] == '900996' and r['offset'] == '672' and
                                r['kind'] == 'arc_tube' and r['tentative'] == 'False'
                                for r in rows))
            equipment=[r for r in rows if r['record_offset']=='348836']
            self.assertTrue(any(r['kind']=='frustum_surface' and r['offset']=='1280' for r in equipment))
            self.assertTrue(any(r['kind']=='ellipsoid_head' and r['offset']=='3880' for r in equipment))
            self.assertFalse(any(r['kind']=='unparsed_record_bbox' for r in equipment))
            shells=[r for r in rows if r['group']=='69' and r['offset']=='2568']
            self.assertEqual(len(shells),23)
            self.assertTrue(all(r['kind'] in ('arc_tube','revolved_spline_surface') and r['tentative']=='False' for r in shells))

    def test_equipment_frustum_and_rotated_head(self):
        rs=348836
        elements,_=walk_group_elements(self.data,self.records[rs]['display_groups'][0],rs)
        for node,rel,fn in ((896,1280,v.frustum_from_vue),(2904,3880,v.ellipsoid_head_from_vue)):
            matrix=next(e['matrix_a_rowmajor'] for e in elements if e['off_rel_record']==node)
            points,faces=fn(self.data,rs+rel,rs,matrix)
            mesh=trimesh.Trimesh(points-points[0],faces,process=False)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertGreater(mesh.area_faces.min(),1e-10)
            self.assertAlmostEqual(points[:,0].max()-points[:,0].min(),1,delta=.001)
            corrupt=bytearray(self.data)
            struct.pack_into('<d',corrupt,rs+rel+8,struct.unpack_from('<d',corrupt,rs+rel+8)[0]-1)
            with self.assertRaisesRegex(ValueError,'bounds mismatch'):
                fn(corrupt,rs+rel,rs,matrix)

    def test_rotated_annulus_accepts_native_cardinal_box_only(self):
        off=1007964+672
        points,faces=v.annulus_from_vue(self.data,off,np.eye(4))
        outer=np.linalg.norm(v._profile_vector(self.data,off+0x118))
        inner=np.linalg.norm(v._profile_vector(self.data,off+0x170))
        area=trimesh.Trimesh(points,faces,process=False).area
        self.assertAlmostEqual(area/(np.pi*(outer**2-inner**2)),1,delta=.001)
        changed=bytearray(self.data)
        struct.pack_into('<d',changed,off+8,struct.unpack_from('<d',changed,off+8)[0]-1)
        with self.assertRaisesRegex(ValueError,'bounds mismatch'):
            v.annulus_from_vue(changed,off,np.eye(4))

    def test_large_rotated_insulation_has_cardinal_not_analytic_cache(self):
        rs=83140;off=rs+2568
        matrix=read_matrix_pair(self.data,self.records[rs]['display_groups'][1])[0]
        points,faces=v.native_arc_from_vue(self.data,off,matrix)
        box=struct.unpack_from('<6d',self.data,off+8)
        self.assertGreater(v.bounds_error(points,box),.08)
        cardinal=points[:-2].reshape(-1,24,3)[:,::6].reshape(-1,3)
        self.assertLess(v.bounds_error(cardinal,box),1e-7)
        mesh=trimesh.Trimesh(points,faces,process=False)
        self.assertTrue(mesh.is_watertight)

    def test_closed_concave_plate_is_watertight(self):
        rs=1048132;points=[];faces=[];count=0
        for rel in (656,1168,1824,2480,2992,3504,4016,4528):
            fn=v.planar_polygon_from_vue if rel in (1168,1824) else v.planar_quad
            p,f=fn(self.data,rs+rel,rs,np.eye(4))
            points.extend(p);faces.extend(f+count);count+=len(p)
        mesh=trimesh.Trimesh(points,faces,process=True)
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(abs(mesh.volume),9.4062514188*.032,places=7)

    def test_new_rail_surfaces_validate_array_pointers(self):
        for rel in (21024,23720,26208,28696,30984,33272):
            off=1007964+rel
            points,faces=v.nurbs_surface_from_vue(self.data,off,np.eye(4))
            self.assertGreater(len(faces),1000)
            self.assertTrue(np.isfinite(points).all())
            corrupt=bytearray(self.data);struct.pack_into('<I',corrupt,off+0x50,120)
            with self.assertRaisesRegex(ValueError,'array pointers'):
                v.nurbs_surface_from_vue(corrupt,off,np.eye(4))

    def test_nested_ring_is_watertight_without_end_disks(self):
        rs=1087444
        es,_=walk_group_elements(self.data,self.records[rs]['display_groups'][0],rs)
        matrix=next(e['matrix_a_rowmajor'] for e in es if e['off_rel_record']==3008)
        points,faces=v.native_arc_from_vue(self.data,rs+4456,matrix)
        mesh=trimesh.Trimesh(points,faces,process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(mesh.euler_number,0)
        self.assertAlmostEqual(abs(mesh.volume)/(2*np.pi**2*.315*.035**2),1,delta=.02)
        for rel in (3808,5112,5552,5992):
            p,f=v.ruled_polylines_from_vue(self.data,rs+rel,rs,matrix)
            self.assertLess(v.bounds_error(p,struct.unpack_from('<6d',self.data,rs+rel+8)),1e-7)

    def test_ruled_polylines_and_partial_sweep_are_closed(self):
        rs=996612
        es,_=walk_group_elements(self.data,self.records[rs]['display_groups'][0],rs)
        matrix=next(e['matrix_a_rowmajor'] for e in es if e['off_rel_record']==3008)
        for rel in (3808,5112,5552,5992):
            p,f=v.ruled_polylines_from_vue(self.data,rs+rel,rs,matrix)
            mesh=trimesh.Trimesh(p,f,process=False)
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertGreater(mesh.volume,0)
        p,f=v.swept_native_profile_from_vue(self.data,rs+6432,rs,matrix)
        mesh=trimesh.Trimesh(p,f,process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume,0)

    def test_insulation_revolves_about_native_axis(self):
        rs=794324;g=self.records[rs]['display_groups'][1];m=read_matrix_pair(self.data,g)[0]
        points,faces=v.swept_native_profile_from_vue(self.data,rs+2568,rs,m)
        local=(np.c_[points,np.ones(len(points))]@np.linalg.inv(m))[:,:3]
        center=v._profile_vector(self.data,rs+2568+0x50)
        axis=v.unit(v._profile_vector(self.data,rs+2568+0x68))
        axial=(local-center)@axis
        self.assertLessEqual(abs(axial).max(),.0467+1e-8)
        self.assertGreater(abs(axial).max(),.046)
        corrupt=bytearray(self.data);struct.pack_into('<I',corrupt,rs+2568+0x98,0)
        with self.assertRaisesRegex(ValueError,'array pointers'):
            v.swept_native_profile_from_vue(corrupt,rs+2568,rs,m)

if __name__=='__main__':unittest.main()
