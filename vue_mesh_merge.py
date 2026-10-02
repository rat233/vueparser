"""Merge exported object parts while retaining material slots and local origins."""
from collections import defaultdict
from pathlib import Path
import struct
import zlib
import numpy as np


def prepare_moniker_groups(pieces):
    groups=defaultdict(list)
    for piece in pieces:groups[piece['moniker']].append(piece)
    result=[];specs=[]
    for moniker,parts in groups.items():
        origin=np.asarray(parts[0]['origin'],dtype=float)
        name=moniker+'__merged'
        specs.append(dict(name=name,moniker=moniker,parts=[p['name'] for p in parts],
                          record_offsets=sorted({p['record_offset'] for p in parts}),origin=origin.tolist()))
        for p in parts:
            result.append(dict(p,origin=origin,local_vertices=p['local_vertices']+(p['origin']-origin)))
    return result,specs


def merge_gltf_tree(tree,specs):
    nodes=tree['nodes'];lookup={n.get('name'):i for i,n in enumerate(nodes) if 'mesh' in n}
    removed=set();meshes=[]
    for spec in specs:
        ids=[lookup[name] for name in spec['parts']]
        primitive=[]
        for i in ids:primitive.extend(tree['meshes'][nodes[i]['mesh']]['primitives'])
        # A glTF mesh can have multiple material primitives but just one node.
        meshes.append(dict(name=spec['name'],primitives=primitive,
                           extras=dict(moniker=spec['moniker'],record_offsets=spec['record_offsets'],source_parts=spec['parts'])))
        nodes[ids[0]]['mesh']=len(meshes)-1;nodes[ids[0]]['name']=spec['name']
        removed.update(ids[1:])
    remap={old:new for new,old in enumerate(i for i in range(len(nodes)) if i not in removed)}
    for n in nodes:
        if 'children' in n:n['children']=[remap[i] for i in n['children'] if i not in removed]
    for scene in tree['scenes']:scene['nodes']=[remap[i] for i in scene['nodes'] if i not in removed]
    tree['nodes']=[n for i,n in enumerate(nodes) if i not in removed]
    tree['meshes']=meshes


# Minimal binary FBX tree codec. Unchanged properties (including compressed
# arrays) remain encoded verbatim; only merged geometry/links are rebuilt.
def _property(data,pos):
    start=pos;tag=data[pos:pos+1];pos+=1
    sizes={b'Y':2,b'C':1,b'I':4,b'F':4,b'D':8,b'L':8}
    if tag in sizes:pos+=sizes[tag]
    elif tag in (b'S',b'R'):pos+=4+struct.unpack_from('<I',data,pos)[0]
    elif tag in (b'f',b'd',b'l',b'i',b'b',b'c'):
        pos+=12+struct.unpack_from('<I',data,pos+8)[0]
    else:raise ValueError('unsupported FBX property '+repr(tag))
    if pos>len(data):raise ValueError('truncated FBX property')
    return data[start:pos],pos


def value(raw):
    tag=raw[:1]
    if tag==b'S':return raw[5:]
    if tag in (b'I',b'L'):return struct.unpack('<'+('i' if tag==b'I' else 'q'),raw[1:])[0]
    if tag in (b'd',b'i'):
        count,encoding,size=struct.unpack_from('<3I',raw,1)
        payload=raw[13:]
        if encoding==1:payload=zlib.decompress(payload)
        elif encoding!=0:raise ValueError('unsupported FBX array encoding')
        dtype='<f8' if tag==b'd' else '<i4'
        array=np.frombuffer(payload,dtype=dtype)
        if len(array)!=count:raise ValueError('invalid FBX array size')
        return array
    raise ValueError('unsupported FBX value')


def prop(value_,tag=b'S'):
    if tag==b'S':
        data=value_.encode() if isinstance(value_,str) else value_
        return tag+struct.pack('<I',len(data))+data
    if tag in (b'I',b'L'):return tag+struct.pack('<'+('i' if tag==b'I' else 'q'),int(value_))
    data=np.asarray(value_,dtype='<f8' if tag==b'd' else '<i4').ravel()
    blob=zlib.compress(data.tobytes())
    return tag+struct.pack('<3I',len(data),1,len(blob))+blob


def node(name,props=(),children=()):return dict(name=name,props=list(props),children=list(children),terminated=bool(children))

def child(n,name):return next(c for c in n['children'] if c['name']==name)


def read_fbx(path):
    data=Path(path).read_bytes()
    if data[:23]!=b'Kaydara FBX Binary  \x00\x1a\x00':raise ValueError('not binary FBX')
    version=struct.unpack_from('<I',data,23)[0];fmt='<QQQB' if version>=7500 else '<IIIB';hs=struct.calcsize(fmt)
    def read(pos):
        end,count,ps,ns=struct.unpack_from(fmt,data,pos)
        if end==0:return None,pos+hs
        if not pos<end<=len(data):raise ValueError('invalid FBX node range')
        name=data[pos+hs:pos+hs+ns].decode();pos+=hs+ns;properties=[];pe=pos+ps
        for _ in range(count):p,pos=_property(data,pos);properties.append(p)
        if pos!=pe:raise ValueError('invalid FBX property span')
        children=[];terminated=False
        while pos<end:
            c,pos=read(pos)
            if c is None:terminated=True;break
            children.append(c)
        if pos!=end:raise ValueError('invalid FBX child span')
        return dict(name=name,props=properties,children=children,terminated=terminated),end
    nodes=[];pos=27
    while True:
        n,pos=read(pos)
        if n is None:break
        nodes.append(n)
    return data[:27],nodes,data[pos:]


def write_fbx(path,header,nodes,footer):
    version=struct.unpack_from('<I',header,23)[0];fmt='<QQQB' if version>=7500 else '<IIIB';hs=struct.calcsize(fmt)
    with Path(path).open('wb') as stream:
        stream.write(header)
        def write(n):
            start=stream.tell();name=n['name'].encode();stream.write(bytes(hs));stream.write(name)
            ps=sum(map(len,n['props']))
            for p in n['props']:stream.write(p)
            for c in n['children']:write(c)
            if n['terminated'] or n['children']:stream.write(bytes(hs))
            end=stream.tell();stream.seek(start);stream.write(struct.pack(fmt,end,len(n['props']),ps,len(name)));stream.seek(end)
        for n in nodes:write(n)
        stream.write(bytes(hs));stream.write(footer)


def merge_fbx(path,specs):
    header,roots,footer=read_fbx(path)
    objects=next(n for n in roots if n['name']=='Objects');connections=next(n for n in roots if n['name']=='Connections')
    by_id={value(n['props'][0]):n for n in objects['children']}
    models={value(n['props'][1]).split(b'\x00')[0].decode().removeprefix('Model::'):ident
            for ident,n in by_id.items() if n['name']=='Model'}
    links=[(value(n['props'][1]),value(n['props'][2])) for n in connections['children'] if value(n['props'][0])==b'OO']
    removed=set();affected=set();newlinks=[]
    for spec in specs:
        ids=[models[name] for name in spec['parts']];model_id=ids[0]
        vertices=[];indices=[];normals=[];materials=[];face_materials=[];geometries=[];shift=0
        for model in ids:
            gs=[a for a,b in links if b==model and by_id.get(a,{}).get('name')=='Geometry']
            ms=[a for a,b in links if b==model and by_id.get(a,{}).get('name')=='Material']
            if len(gs)!=1 or len(ms)!=1:raise ValueError('expected one source geometry/material per part')
            g=by_id[gs[0]];geometries.append(gs[0]);materials.append(ms[0])
            xyz=value(child(g,'Vertices')['props'][0]);idx=value(child(g,'PolygonVertexIndex')['props'][0])
            if len(idx)%3 or not np.all(idx.reshape(-1,3)[:,2]<0):raise ValueError('expected triangles')
            layer=child(g,'LayerElementNormal');normal=value(child(layer,'Normals')['props'][0]).reshape(-1,3)
            mapping=value(child(layer,'MappingInformationType')['props'][0]);reference=value(child(layer,'ReferenceInformationType')['props'][0])
            normal_ids=np.arange(len(idx)) if mapping==b'ByPolygonVertex' else np.where(idx<0,-idx-1,idx)
            if mapping not in (b'ByPolygonVertex',b'ByVertice',b'ByVertex'):raise ValueError('unsupported normal mapping')
            if reference==b'IndexToDirect':normal_ids=value(child(layer,'NormalsIndex')['props'][0])[normal_ids]
            elif reference!=b'Direct':raise ValueError('unsupported normal reference')
            vertices.append(xyz);indices.append(np.where(idx<0,idx-shift,idx+shift));normals.append(normal[normal_ids].ravel())
            face_materials.extend([len(materials)-1]*(len(idx)//3));shift+=len(xyz)//3
        keep=by_id[geometries[0]]
        keep['props'][1]=prop(spec['name']+'\x00\x01Geometry')
        child(keep,'Vertices')['props']=[prop(np.concatenate(vertices),b'd')]
        child(keep,'PolygonVertexIndex')['props']=[prop(np.concatenate(indices),b'i')]
        nl=child(keep,'LayerElementNormal')
        child(nl,'Normals')['props']=[prop(np.concatenate(normals),b'd')]
        child(nl,'MappingInformationType')['props']=[prop('ByPolygonVertex')]
        child(nl,'ReferenceInformationType')['props']=[prop('Direct')]
        nl['children']=[c for c in nl['children'] if c['name']!='NormalsIndex']
        ml=child(keep,'LayerElementMaterial')
        child(ml,'MappingInformationType')['props']=[prop('ByPolygon')]
        child(ml,'ReferenceInformationType')['props']=[prop('IndexToDirect')]
        child(ml,'Materials')['props']=[prop(face_materials,b'i')]
        by_id[model_id]['props'][1]=prop(spec['name']+'\x00\x01Model')
        removed.update(ids[1:]);removed.update(geometries[1:]);affected.update(ids);affected.update(geometries)
        parent=next(b for a,b in links if a==model_id)
        newlinks.extend([(geometries[0],model_id),(model_id,parent)])
        newlinks.extend((m,model_id) for m in materials)
    objects['children']=[n for n in objects['children'] if value(n['props'][0]) not in removed]
    connections['children']=[n for n in connections['children'] if not any(value(p) in affected for p in n['props'][1:3])]
    connections['children'].extend(node('C',[prop('OO'),prop(a,b'L'),prop(b,b'L')]) for a,b in newlinks)
    definitions=next(n for n in roots if n['name']=='Definitions')
    for d in definitions['children']:
        if d['name']=='ObjectType':
            kind=value(d['props'][0]).decode()
            child(d,'Count')['props']=[prop(sum(n['name']==kind for n in objects['children']),b'I')]
    child(definitions,'Count')['props']=[prop(len(objects['children']),b'I')]
    target=Path(path);temporary=target.with_suffix('.merge-tmp.fbx')
    try:
        write_fbx(temporary,header,roots,footer)
        # Validate the rewritten absolute node offsets before replacing output.
        read_fbx(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
