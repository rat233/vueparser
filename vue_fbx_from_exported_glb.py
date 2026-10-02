"""Recover a compatible FBX from an existing VUE GLB and restore double origins.

Used when Assimp cannot import the large ASCII intermediate. Geometry is the
already exported GLB, never the independent reference FBX.
"""
import argparse,json,struct,subprocess,sys
from pathlib import Path
import numpy as np
import trimesh
from vue_runtime import helper_command
from vue_preview_export import restore_fbx_translations,read_fbx_mesh_translations


def restore_vue_global_settings(target):
    """Patch only scalar properties inside the bounded GlobalSettings node."""
    settings={'UpAxis':2,'UpAxisSign':1,'FrontAxis':1,'FrontAxisSign':1,
              'CoordAxis':0,'CoordAxisSign':-1,'UnitScaleFactor':100.,
              'OriginalUnitScaleFactor':100.}
    with Path(target).open('r+b') as stream:
        if stream.read(23)!=b'Kaydara FBX Binary  \x00\x1a\x00':
            raise ValueError('not a binary FBX')
        version=struct.unpack('<I',stream.read(4))[0]
        fmt='<QQQB' if version>=7500 else '<IIIB';length=struct.calcsize(fmt)
        while True:
            start=stream.tell();raw=stream.read(length)
            if len(raw)!=length:raise ValueError('missing FBX GlobalSettings')
            end,count,props,namesize=struct.unpack(fmt,raw)
            if not end:raise ValueError('missing FBX GlobalSettings')
            name=stream.read(namesize)
            if name==b'GlobalSettings':break
            stream.seek(end)
        stream.seek(start);block=stream.read(end-start);patches=[]
        def string(value):
            value=value.encode('ascii');return b'S'+struct.pack('<I',len(value))+value
        for key,value in settings.items():
            numeric=isinstance(value,float)
            marker=(string(key)+string('double' if numeric else 'int')+
                    string('Number' if numeric else 'Integer')+string('')+
                    (b'D' if numeric else b'I'))
            if block.count(marker)!=1:raise ValueError('ambiguous FBX global '+key)
            position=start+block.index(marker)+len(marker)
            patches.append((position,struct.pack('<d' if numeric else '<i',value)))
        for position,value in patches:
            stream.seek(position);stream.write(value)
    return settings


def verify_and_restore(source,target,report=True):
    scene=trimesh.load(source,force='scene')
    pieces=[]
    for name in scene.graph.nodes_geometry:
        transform,_=scene.graph[name]
        if not np.allclose(transform[:3,:3],np.eye(3),atol=1e-12):
            raise ValueError('non-translation node requires separate FBX transform audit')
        pieces.append(dict(name=name,origin=np.asarray(transform[:3,3])))
    globals_restored=restore_vue_global_settings(target)
    restored=restore_fbx_translations(target,pieces)
    values=read_fbx_mesh_translations(target,[p['name'] for p in pieces])
    error=max(float(np.max(abs(values[p['name']][1]-p['origin']))) for p in pieces)
    if error!=0:raise AssertionError('FBX/GLB translation mismatch')
    result=dict(source=str(source),target=str(target),source_meshes=len(pieces),global_settings=globals_restored,restored_mesh_translations=restored,max_translation_error_m=error,note='Fallback imports exported GLB; local vertices retain GLB float32 precision. No reference FBX used.')
    if report:
        target.with_suffix('.verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('target',type=Path);parser.add_argument('--verify-existing',action='store_true');parser.add_argument('--no-report',action='store_true');args=parser.parse_args()
    if not args.verify_existing:
        subprocess.run(helper_command('vue_fbx_phong_material', args.source, args.target),check=True,timeout=3600)
    verify_and_restore(args.source,args.target,report=not args.no_report)

if __name__=='__main__':main()
