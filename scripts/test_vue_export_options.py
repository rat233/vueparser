"""Output selection, optional reports, and material-preserving Moniker merging."""
import csv
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
import vue_preview_export as v
from vue_export_options import ExportOptions
from vue_mesh_merge import read_fbx,child,value


def gltf_json(path):
    data=path.read_bytes();size=struct.unpack_from('<I',data,12)[0]
    return json.loads(data[20:20+size])


def pieces():
    box=trimesh.creation.box(extents=(.1,.2,.3))
    return [dict(name='A__part'+str(i),moniker='A',record_offset=10+i,discipline='Piping',
        origin=np.array([9000.+i,2400.,30.]),local_vertices=np.array(box.vertices),faces=np.array(box.faces),
        color=[1-i,i,0,1-i*.5]) for i in range(2)] + [dict(name='B__part',moniker='B',record_offset=12,
        discipline='Piping',origin=np.array([9005.,2400.,30.]),local_vertices=np.array(box.vertices),
        faces=np.array(box.faces),color=[0,0,1,1])]


class ExportOptionsTests(unittest.TestCase):
    def test_glb_only_never_starts_fbx_and_leaves_no_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'test'
            with patch('subprocess.run',side_effect=AssertionError('unexpected helper')):
                v.write_models(base,pieces(),ExportOptions('glb',False,False,'glb'))
            self.assertEqual({p.name for p in Path(directory).iterdir()},{'test.glb'})
            self.assertEqual(len(gltf_json(base.with_suffix('.glb'))['meshes']),3)

    def test_obj_only_and_merge_object_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'test'
            with patch('subprocess.run',side_effect=AssertionError('unexpected helper')):
                v.write_models(base,pieces(),ExportOptions('obj',False,True))
            self.assertEqual({p.name for p in Path(directory).iterdir()},{'test.obj'})
            text=base.with_suffix('.obj').read_text()
            self.assertEqual([line for line in text.splitlines() if line.startswith('o ')],['o A__merged','o B__merged'])
            self.assertEqual(sum(line.startswith('f ') for line in text.splitlines()),36)

    def test_glb_merge_retains_materials_world_positions_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'test';source=pieces()
            v.write_models(base,source,ExportOptions('glb',True,True))
            tree=gltf_json(base.with_suffix('.glb'))
            self.assertEqual(len(tree['meshes']),2)
            self.assertEqual(len([n for n in tree['nodes'] if 'mesh' in n]),2)
            mesh=next(m for m in tree['meshes'] if m['name']=='A__merged')
            self.assertEqual(len(mesh['primitives']),2)
            self.assertEqual(mesh['extras']['record_offsets'],[10,11])
            colors=[tree['materials'][p['material']]['pbrMetallicRoughness']['baseColorFactor'] for p in mesh['primitives']]
            np.testing.assert_allclose(colors,[[1,0,0,1],[0,1,0,.5]],atol=.003)
            scene=trimesh.load(base.with_suffix('.glb'),force='scene')
            expected=np.vstack([p['local_vertices']+p['origin'] for p in source])[:,[0,2,1]] * [1,1,-1]
            np.testing.assert_allclose(scene.bounds,np.vstack((expected.min(0),expected.max(0))),atol=1e-7,rtol=0)
            self.assertEqual(sum(len(m.faces) for m in scene.geometry.values()),36)
            self.assertTrue((Path(directory)/'mesh_merge.json').exists())
            np.testing.assert_array_equal(source[1]['origin'],[9001.,2400.,30.])

    def test_fbx_only_merges_geometry_and_preserves_material_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'test'
            v.write_models(base,pieces(),ExportOptions('fbx',False,True,'glb'))
            target=Path(directory)/'test_compatible.fbx'
            self.assertEqual({p.name for p in Path(directory).iterdir()},{target.name})
            _,roots,_=read_fbx(target)
            objects=next(n for n in roots if n['name']=='Objects')['children']
            geometry=[n for n in objects if n['name']=='Geometry']
            self.assertEqual(len(geometry),2)
            g=next(g for g in geometry if value(g['props'][1]).startswith(b'A__merged'))
            self.assertEqual(len(value(child(g,'PolygonVertexIndex')['props'][0])),72)
            materials=value(child(child(g,'LayerElementMaterial'),'Materials')['props'][0])
            np.testing.assert_array_equal(materials,[0]*12+[1]*12)
            xyz=value(child(g,'Vertices')['props'][0]).reshape(-1,3)
            np.testing.assert_allclose([xyz[:,0].min(),xyz[:,0].max()],[-.05,1.05],atol=1e-7)
            translations=v.read_fbx_mesh_translations(target,['A__merged','B__merged'])
            np.testing.assert_array_equal(translations['A__merged'][1],[9000,2400,30])

    def test_auto_fbx_merge_and_unmerged_report_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'auto'
            v.write_models(base,pieces(),ExportOptions('fbx',False,True,'auto'))
            _,roots,_=read_fbx(base.with_name('auto_compatible.fbx'))
            self.assertEqual(sum(n['name']=='Geometry' for n in next(r for r in roots if r['name']=='Objects')['children']),2)
            v.write_models(Path(directory)/'all',pieces(),ExportOptions('all',True,False,'glb'))
            names={p.name for p in Path(directory).iterdir()}
            self.assertEqual(names,{'auto_compatible.fbx','all.glb','all.obj','all_compatible.fbx','all_compatible.verification.json'})

    def test_real_record_reports_opt_in_and_no_mesh_loss(self):
        from vue_record_parser import read_records
        data=Path('15300.vue').read_bytes();header,records,errors=read_records(data)
        tables=v.read_scene_tables(data,records)
        selected=[r for r in records if r['record_offset'] in (2651348,5785204)]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(v,'read_records',return_value=(header,selected,errors)),patch.object(v,'read_scene_tables',return_value=tables),patch('builtins.print'):
                off=Path(directory)/'off';on=Path(directory)/'on'
                a=v.export_sample(Path('15300.vue'),off,ExportOptions('glb',False,True))
                b=v.export_sample(Path('15300.vue'),on,ExportOptions('glb',True,True))
                fbx=Path(directory)/'fbx'
                v.export_sample(Path('15300.vue'),fbx,ExportOptions('fbx',False,True,'glb'))
            self.assertEqual({p.name for p in off.iterdir()},{'15300.glb'})
            self.assertEqual(a['triangles'],b['triangles'])
            self.assertEqual(a['exported_meshes'],2)
            self.assertEqual({p.name for p in on.iterdir()}, {'15300.glb','geometry.csv','gaps.csv','materials.json','references.json','profile_shells.json','summary.json','mesh_merge.json'})
            rows=list(csv.DictReader((on/'geometry.csv').read_text(encoding='utf-8-sig').splitlines()))
            self.assertTrue(any(row['kind']=='weld_line' for row in rows))
            self.assertEqual({p.name for p in fbx.iterdir()},{'15300_compatible.fbx'})
            _,roots,_=read_fbx(fbx/'15300_compatible.fbx')
            geoms=[n for n in next(r for r in roots if r['name']=='Objects')['children'] if n['name']=='Geometry']
            self.assertEqual(len(geoms),2)
            indices=[value(child(g,'PolygonVertexIndex')['props'][0]) for g in geoms]
            sizes=[np.diff(np.r_[-1,np.flatnonzero(idx<0)]) for idx in indices]
            self.assertEqual(sum(np.count_nonzero(size==3) for size in sizes),a['triangles'])
            self.assertEqual(sum(np.count_nonzero(size==2) for size in sizes),a['line_segments'])

    def test_unknown_bbox_wireframe_switch_keeps_gap_diagnostics(self):
        from vue_record_parser import read_records
        corners,edges=v.bbox_cage([1,2,3,4,5,6])
        self.assertEqual(corners.shape,(8,3))
        self.assertEqual(edges.shape,(12,2))
        np.testing.assert_array_equal([corners.min(0),corners.max(0)],[[1,2,3],[4,5,6]])
        source=Path('063E.vue');data=source.read_bytes()
        header,records,errors=read_records(data)
        tables=v.read_scene_tables(data,records)
        selected=[r for r in records if r['record_offset']==642908]
        self.assertEqual(len(selected),1)
        with tempfile.TemporaryDirectory() as directory:
            results=[]
            for enabled in (True,False):
                target=Path(directory)/('on' if enabled else 'off')
                options=ExportOptions(format='all',reports=True,
                                      unknown_bbox_wireframe=enabled)
                with patch.object(v,'read_records',return_value=(header,selected,errors)), \
                     patch.object(v,'read_scene_tables',return_value=tables),redirect_stdout(io.StringIO()):
                    summary=v.export_sample(source,target,options)
                tree=gltf_json(target/'063E.glb')
                names=[node.get('name','') for node in tree['nodes']]
                rows=list(csv.DictReader((target/'geometry.csv').read_text(encoding='utf-8-sig').splitlines()))
                results.append((summary,names,rows))
            on,off=results
            self.assertTrue(any('__UNPARSED_BBOX' in name for name in on[1]))
            self.assertFalse(any('__UNPARSED_BBOX' in name for name in off[1]))
            self.assertGreater(on[0]['placeholder_objects'],0)
            self.assertEqual(off[0]['placeholder_objects'],0)
            self.assertEqual(off[0]['triangles'],on[0]['triangles'])
            self.assertGreater(on[0]['line_segments'],0)
            self.assertEqual(off[0]['line_segments'],0)
            self.assertLess(off[0]['exported_meshes'],on[0]['exported_meshes'])
            self.assertEqual([r['kind'] for r in on[2] if r['kind'].startswith('unparsed_')],
                             [r['kind'] for r in off[2] if r['kind'].startswith('unparsed_')])
            self.assertFalse(off[0]['unknown_bbox_wireframe'])
            on_dir=Path(directory)/'on'
            self.assertIn('\nl ',(on_dir/'063E.obj').read_text(encoding='utf-8'))
            on_tree=gltf_json(on_dir/'063E.glb')
            self.assertTrue(any(primitive.get('mode')==1 for mesh in on_tree['meshes']
                                for primitive in mesh['primitives']))
            _,roots,_=read_fbx(on_dir/'063E_compatible.fbx')
            objects=next(root for root in roots if root['name']=='Objects')['children']
            self.assertTrue(any(np.all(np.diff(np.r_[-1,np.flatnonzero(
                                value(child(node,'PolygonVertexIndex')['props'][0])<0)])==2)
                                for node in objects if node['name']=='Geometry'))
            off_dir=Path(directory)/'off'
            self.assertNotIn('__UNPARSED_BBOX',(off_dir/'063E.obj').read_text(encoding='utf-8'))
            _,roots,_=read_fbx(off_dir/'063E_compatible.fbx')
            objects=next(root for root in roots if root['name']=='Objects')['children']
            fb_names=[value(node['props'][1]) for node in objects if node['name']=='Model']
            self.assertFalse(any(b'__UNPARSED_BBOX' in name for name in fb_names))

    def test_unknown_only_input_reports_no_exportable_geometry(self):
        source=pieces()[0]
        source['placeholder']=True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'No geometry remains'):
                v.write_models(Path(directory)/'unknown',[source],
                               ExportOptions(format='all',unknown_bbox_wireframe=False))
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_unknown_only_line_exports_with_merge_and_auto_fbx(self):
        vertices,edges=v.bbox_cage([0,0,0,1,2,3])
        item=dict(name='A__R1__G64__UNPARSED_BBOX',moniker='A',record_offset=1,
                  discipline='Equipment',origin=np.array([9000.,2400.,30.]),
                  local_vertices=vertices,faces=edges,color=[1.,.1,.65,1.],placeholder=True)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'unknown'
            v.write_models(base,[item],ExportOptions(format='all',reports=True,
                           merge_moniker=True,fbx_mode='auto'))
            tree=gltf_json(base.with_suffix('.glb'))
            self.assertEqual([p['mode'] for p in tree['meshes'][0]['primitives']],[1])
            from vue_glb_finalize import read_glb, accessor_array
            tree,binary=read_glb(base.with_suffix('.glb').read_bytes())
            primitive=tree['meshes'][0]['primitives'][0]
            tint=np.asarray(tree['materials'][primitive['material']]['pbrMetallicRoughness']['baseColorFactor'])
            vertex_color=accessor_array(tree,binary,primitive['attributes']['COLOR_0'])[0]/255
            np.testing.assert_allclose(tint*vertex_color,[1,.1,.65,1],atol=.01)
            obj=base.with_suffix('.obj').read_text(encoding='utf-8')
            self.assertEqual(sum(line.startswith('l ') for line in obj.splitlines()),12)
            self.assertFalse(any(line.startswith('f ') for line in obj.splitlines()))
            _,roots,_=read_fbx(base.with_name('unknown_compatible.fbx'))
            objects=next(root for root in roots if root['name']=='Objects')['children']
            geometry=[node for node in objects if node['name']=='Geometry']
            self.assertEqual(len(geometry),1)
            indices=value(child(geometry[0],'PolygonVertexIndex')['props'][0])
            self.assertTrue(np.all(np.diff(np.r_[-1,np.flatnonzero(indices<0)])==2))
            verification=json.loads((Path(directory)/'unknown_compatible.verification.json').read_text())
            self.assertEqual(verification['unmerged_line_meshes'],1)
            self.assertEqual(verification['merged_meshes'],0)

    def test_obj_line_before_mesh_keeps_separate_normal_indices(self):
        vertices,edges=v.bbox_cage([0,0,0,1,1,1])
        line=dict(name='A__UNPARSED_BBOX',moniker='A',record_offset=1,
                  discipline='Equipment',origin=np.array([9000.,2400.,30.]),
                  local_vertices=vertices,faces=edges,color=[1.,.1,.65,1.],placeholder=True)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'mixed'
            v.write_models(base,[line,pieces()[0]],ExportOptions(format='obj'))
            lines=base.with_suffix('.obj').read_text(encoding='utf-8').splitlines()
            self.assertEqual(sum(row.startswith('l ') for row in lines),12)
            face=next(row for row in lines if row.startswith('f '))
            self.assertEqual([int(point)-int(normal) for point,normal in
                              (token.split('//') for token in face.split()[1:])],[8,8,8])

    def test_merge_keeps_line_and_triangle_primitives(self):
        vertices,edges=v.bbox_cage([0,0,0,1,1,1])
        line=dict(name='A__UNPARSED_BBOX',moniker='A',record_offset=10,
                  discipline='Piping',origin=np.array([9000.,2400.,30.]),
                  local_vertices=vertices,faces=edges,color=[1.,.1,.65,1.],placeholder=True)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'mixed'
            v.write_models(base,[pieces()[0],line],ExportOptions(format='all',reports=True,
                                                                  merge_moniker=True))
            tree=gltf_json(base.with_suffix('.glb'))
            self.assertEqual(len(tree['meshes']),1)
            self.assertEqual({p['mode'] for p in tree['meshes'][0]['primitives']},{1,4})
            obj=base.with_suffix('.obj').read_text(encoding='utf-8')
            self.assertEqual([row for row in obj.splitlines() if row.startswith('o ')],['o A__merged'])
            self.assertIn('\nl ',obj)
            self.assertIn('\nf ',obj)
            _,roots,_=read_fbx(base.with_name('mixed_compatible.fbx'))
            objects=next(root for root in roots if root['name']=='Objects')['children']
            names=[value(node['props'][1]) for node in objects if node['name']=='Model']
            self.assertTrue(any(name.startswith(b'A__merged\x00') for name in names))
            self.assertTrue(any(name.startswith(b'A__UNPARSED_BBOX\x00') for name in names))
            verification=json.loads((Path(directory)/'mixed_compatible.verification.json').read_text())
            self.assertEqual((verification['merged_meshes'],verification['unmerged_line_meshes']),(1,1))

    def test_cli_forwards_options_and_default_reports_are_off(self):
        import vue_cli
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'test.vue';source.write_bytes(b'')
            with patch('sys.argv',['VueParser',str(source),'--format','fbx','--merge-moniker','true','--reports','true']),patch.object(v,'export_sample') as export:
                self.assertEqual(vue_cli.main(),0)
                self.assertEqual(export.call_args.args[2],ExportOptions('fbx',True,True,'glb','auto','y'))
            with patch('sys.argv',['VueParser',str(source)]),patch.object(v,'export_sample') as export:
                self.assertEqual(vue_cli.main(),0)
                self.assertEqual(export.call_args.args[2],ExportOptions('all',False,False,'glb','auto','y'))
            with patch('sys.argv',['VueParser',str(source),'--unknown-bbox-wireframe','false',
                                   '--reports','false','--merge-moniker','false']),patch.object(v,'export_sample') as export:
                self.assertEqual(vue_cli.main(),0)
                self.assertEqual(export.call_args.args[2],
                                 ExportOptions('all',False,False,'glb','auto','y',False))
            for option in ('--reports','--merge-moniker','--unknown-bbox-wireframe'):
                with self.subTest(option=option),patch('sys.argv',['VueParser',str(source),option]), \
                     patch('sys.stderr',new_callable=io.StringIO):
                    with self.assertRaises(SystemExit) as error:
                        vue_cli.main()
                    self.assertEqual(error.exception.code,2)

if __name__=='__main__':unittest.main()
