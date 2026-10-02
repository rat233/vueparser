"""Precision regression for final FBX transforms, without Assimp or reference data."""
import struct,tempfile,unittest
from pathlib import Path
import numpy as np
from vue_preview_export import read_fbx_mesh_translations,restore_fbx_translations

class FbxTranslationTests(unittest.TestCase):
    def fixture(self,version):
        fmt='<QQQB' if version>=7500 else '<IIIB';size=struct.calcsize(fmt)
        def prop(value):
            if isinstance(value,str):
                b=value.encode();return b'S'+struct.pack('<I',len(b))+b
            if isinstance(value,int):return b'L'+struct.pack('<q',value)
            return b'D'+struct.pack('<d',value)
        def encode(tree,start):
            name,values,children=tree;name=name.encode();payload=b''.join(map(prop,values))
            position=start+size+len(name)+len(payload);body=b''
            for child in children:
                packed=encode(child,position);body+=packed;position+=len(packed)
            body+=bytes(size);end=start+size+len(name)+len(payload)+len(body)
            return struct.pack(fmt,end,len(values),len(payload),len(name))+name+payload+body
        tree=('Objects',[],[('Model',[123,'part\x00\x01Model','Mesh'],[
            ('Properties70',[],[('P',['Lcl Translation','Lcl Translation','','A',2446.5,1551.5,6.25],[])])])])
        return b'Kaydara FBX Binary  \x00\x1a\x00'+struct.pack('<I',version)+encode(tree,27)+bytes(size)+b'footer preserved'

    def test_restore_both_binary_node_versions_without_other_changes(self):
        for version in (7400,7500):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'test.fbx';before=self.fixture(version);path.write_bytes(before)
                positions,old=read_fbx_mesh_translations(path,['part'])['part']
                expected=np.array([2446.56301234567,1551.56337890123,6.2484599876543])
                self.assertEqual(restore_fbx_translations(path,[dict(name='part',origin=expected)]),1)
                np.testing.assert_array_equal(read_fbx_mesh_translations(path,['part'])['part'][1],expected)
                after=bytearray(path.read_bytes())
                for position in positions:after[position:position+8]=before[position:position+8]
                self.assertEqual(bytes(after),before)

    def test_missing_mesh_rejects_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.fbx';before=self.fixture(7500);path.write_bytes(before)
            with self.assertRaisesRegex(ValueError,'missing FBX mesh translations'):
                restore_fbx_translations(path,[dict(name='part',origin=np.zeros(3)),dict(name='missing',origin=np.ones(3))])
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
