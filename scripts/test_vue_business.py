"""Business identity, fallback naming and geometry-preserving export regression."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import trimesh

from vue_business import apply_business_tree, load_business, resolve_database
from vue_export_options import ExportOptions
from scripts.test_vue_export_options import pieces, gltf_json
from vue_preview_export import write_models


def fixture():
    return dict(source='sample', report={}, mdb_ids={'A', 'B', 'C'}, objects={
        'A': dict(in_vue=True, properties={'System Path': 'Site\\Pipe', 'Name': 'Valve', 'BASF Quality Index': '0', 'Flag': False}),
        'B': dict(in_vue=True, properties={'System Path': 'Site\\Pipe', 'Name': 'Valve'}),
        'C': dict(in_vue=False, properties={'Name': '', 'Type': 'Port'}),
        'D': dict(in_vue=True, properties={})})


class BusinessTests(unittest.TestCase):
    def test_insulated_equipment_uses_one_root_and_preserves_geometry(self):
        for merge in (False, True):
            with self.subTest(merge=merge), tempfile.TemporaryDirectory() as d:
                source = pieces()
                source[1] = dict(source[1], moniker='B')
                source[2] = dict(source[2], moniker='C')
                business = dict(source='sample', report={}, mdb_ids={'A', 'B', 'C', 'D'}, objects={
                    'A': dict(in_vue=True, properties={'System Path': 'Site\\Equipment', 'Name': 'Vessel'}),
                    'B': dict(in_vue=True, properties={'System Path': 'Site\\Equipment', 'Name': 'Vessel'}),
                    'C': dict(in_vue=True, properties={'System Path': 'Site\\Equipment\\Vessel', 'Name': 'Nozzle'}),
                    'D': dict(in_vue=False, properties={'System Path': 'Site\\Equipment\\Vessel\\Supports', 'Name': 'Foot'})})
                base = Path(d) / 'equipment'
                write_models(base, source, ExportOptions('glb', False, merge), business)
                tree = gltf_json(base.with_suffix('.glb'))
                nodes = tree['nodes']
                owners = {n['extras']['moniker']:i for i,n in enumerate(nodes)
                          if 'moniker' in n.get('extras', {})}
                parents = {c:i for i,n in enumerate(nodes) for c in n.get('children', [])}
                roots = [i for i,n in enumerate(nodes) if n.get('name') == 'Vessel']
                self.assertEqual(len(roots), 1)
                root = roots[0]
                self.assertEqual({parents[owners[i]] for i in ('A', 'B', 'C')}, {root})
                self.assertEqual(parents[parents[owners['D']]], root)
                self.assertEqual(len(owners), 4)
                self.assertEqual(business['report']['shared_object_groups'], 1)
                self.assertEqual(business['report']['directory_nodes'], 4)
                scene = trimesh.load(base.with_suffix('.glb'), force='scene')
                expected = np.vstack([p['local_vertices'] + p['origin'] for p in source])[:,[0,2,1]] * [1,1,-1]
                np.testing.assert_allclose(scene.bounds, [expected.min(0), expected.max(0)], atol=1e-7, rtol=0)
                self.assertEqual(sum(len(m.faces) for m in scene.geometry.values()), 36)
                self.assertTrue(any(m.get('alphaMode') == 'BLEND' for m in tree['materials']))

    def test_unique_equipment_owns_nested_attachment_directories(self):
        business = dict(source='sample', report={}, mdb_ids={'Z', 'A'}, objects={
            'Z': dict(in_vue=True, properties={'System Path': 'Site', 'Name': 'Vessel'}),
            'A': dict(in_vue=True, properties={'System Path': 'Site\\Vessel\\Supports', 'Name': 'Foot'})})
        source = [dict(p, moniker='Z' if p['moniker'] == 'A' else 'A') for p in pieces()]
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / 'equipment'
            write_models(base, source, ExportOptions('glb', False, True), business)
            nodes = gltf_json(base.with_suffix('.glb'))['nodes']
            vessel = next(n for n in nodes if n.get('name') == 'Vessel')
            self.assertEqual(len([n for n in nodes if n.get('name') == 'Vessel']), 1)
            self.assertEqual(nodes[vessel['children'][0]]['name'], 'Supports')
            self.assertIn('mesh', vessel)
            scene = trimesh.load(base.with_suffix('.glb'), force='scene')
            expected = np.vstack([p['local_vertices'] + p['origin'] for p in source])[:,[0,2,1]] * [1,1,-1]
            np.testing.assert_allclose(scene.bounds, [expected.min(0), expected.max(0)], atol=1e-7, rtol=0)

    def test_tree_and_geometry_in_both_merge_modes(self):
        for merge in (False, True):
            with self.subTest(merge=merge), tempfile.TemporaryDirectory() as d:
                source=pieces(); business=fixture(); base=Path(d)/'result'
                write_models(base,source,ExportOptions('glb',False,merge),business)
                tree=gltf_json(base.with_suffix('.glb'))
                owners={n['extras']['moniker']:n for n in tree['nodes'] if 'moniker' in n.get('extras',{})}
                self.assertEqual(set(owners), {'A','B','C','D'})
                self.assertNotEqual(owners['A']['name'],owners['B']['name'])
                self.assertEqual(owners['A']['extras']['properties']['Name'],'Valve')
                self.assertIs(owners['A']['extras']['properties']['Flag'],False)
                self.assertEqual(owners['A']['extras']['properties']['BASF Quality Index'],'0')
                self.assertEqual(owners['C']['extras']['status'],'mdb_only')
                self.assertEqual(owners['D']['extras']['status'],'vue_only')
                self.assertNotIn('children',owners['C'])
                self.assertTrue(all('extras' not in m for m in tree['meshes']))
                scene=trimesh.load(base.with_suffix('.glb'),force='scene')
                expected=np.vstack([p['local_vertices']+p['origin'] for p in source])[:,[0,2,1]] * [1,1,-1]
                np.testing.assert_allclose(scene.bounds,[expected.min(0),expected.max(0)],atol=1e-7,rtol=0)
                self.assertEqual(sum(len(m.faces) for m in scene.geometry.values()),36)

    def test_direct_database_join_samples(self):
        from vue_record_parser import read_records
        for sample,expected in [('063E',(550,0,0)),('15300',(3868,60,722))]:
            records=read_records(Path(sample+'.vue').read_bytes())[1]
            data=load_business(Path(sample+'.mdb2'),records,{},sample)
            self.assertEqual((data['report']['matched'],len(data['report']['vue_only']),len(data['report']['mdb_only'])),expected)
            for obj in data['objects'].values():
                self.assertNotIn('linkage_index',obj['properties'])
                self.assertNotIn('link_key_1',obj['properties'])

    def test_discovery_missing_explicit_database_fails(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'sample.vue'
            self.assertIsNone(resolve_database(source,'auto'))
            self.assertIsNone(resolve_database(source,None))
            with self.assertRaises(ValueError):resolve_database(source,str(source.with_suffix('.mdb2')))

    def test_cli_auto_metadata_with_real_selected_geometry(self):
        import vue_preview_export as v
        from vue_record_parser import read_records
        data=Path('063E.vue').read_bytes()
        header,records,errors=read_records(data)
        native=v.read_scene_tables(data,records)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(v,'read_records',return_value=(header,records[:1],errors)), patch.object(v,'read_scene_tables',return_value=native), patch('builtins.print'):
                summary=v.export_sample(Path('063E.vue'),Path(d),ExportOptions('glb',False,False,'glb','auto'))
            self.assertEqual({p.name for p in Path(d).iterdir()},{'063E.glb'})
            self.assertEqual(summary['business']['object_nodes'],550)
            tree=gltf_json(Path(d)/'063E.glb')
            self.assertEqual(sum('moniker' in n.get('extras',{}) for n in tree['nodes']),550)
            self.assertTrue(any('mesh' in n for n in tree['nodes']))

    def test_empty_numeric_cache_is_not_a_business_zero(self):
        report={k:0 for k in ('duplicate_linkage_indexes','duplicate_label_name_indexes','duplicate_label_value_indexes',
            'missing_label_linkage_foreign_keys','missing_label_name_foreign_keys','missing_label_value_foreign_keys')}
        rows=[{'moniker':'A','Quality Index':0,'Enabled':False,'Empty':''}]
        labels=[dict(moniker='A',label_value='',label_value_numeric=0.0,extended_label=False,label_name='Empty')]
        with patch('vue_mdb_export.read_business_database',return_value=(iter([(rows[0],labels)]),['Quality Index','Enabled','Empty'],report)):
            data=load_business(Path('sample.mdb2'),[],{},'sample')
        self.assertEqual(data['objects']['A']['properties'],{'Quality Index':0,'Enabled':False})
        self.assertNotIn('label_details',data['objects']['A'])

    def test_conflicting_rows_and_extended_labels_preserved(self):
        report={k:0 for k in ('duplicate_linkage_indexes','duplicate_label_name_indexes','duplicate_label_value_indexes',
            'missing_label_linkage_foreign_keys','missing_label_name_foreign_keys','missing_label_value_foreign_keys')}
        rows=[{'moniker':'A','Name':'One'},{'moniker':'A','Name':'Two'}]
        labels=[dict(moniker='A',label_value='',label_value_numeric=12,extended_label='detail',label_name='X')]
        with patch('vue_mdb_export.read_business_database',return_value=(iter([(rows[0],labels),(rows[1],[])]),['Name'],report)):
            data=load_business(Path('sample.mdb2'),[],{},'sample')
        self.assertEqual(data['objects']['A']['properties']['Name'],['One','Two'])
        self.assertEqual(data['objects']['A']['label_details'],[{'numeric':12,'extended':'detail','label':'X'}])


if __name__=='__main__':unittest.main()
