"""Publish VUE GLB with explicit axes and material-preserving component meshes."""
from collections import defaultdict
import json
import hashlib
import struct
import numpy as np

Z_TO_Y = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=float)
DTYPES = {5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}
WIDTHS = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}


def read_glb(data):
    magic,version,total=struct.unpack_from('<4sII',data)
    if magic!=b'glTF' or version!=2 or total!=len(data):
        raise ValueError('Invalid GLB header')
    size,kind=struct.unpack_from('<I4s',data,12)
    if kind!=b'JSON':raise ValueError('Missing GLB JSON')
    tree=json.loads(data[20:20+size]); offset=20+size
    if offset==len(data):return tree,b''
    size,kind=struct.unpack_from('<I4s',data,offset)
    if kind!=b'BIN\x00' or offset+8+size!=len(data):raise ValueError('Unsupported GLB chunks')
    return tree,data[offset+8:]


def encode_glb(tree,binary):
    payload=json.dumps(tree,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    payload+=b' '*((-len(payload))%4)
    binary+=b'\x00'*((-len(binary))%4)
    return (struct.pack('<4sII',b'glTF',2,28+len(payload)+len(binary))+
            struct.pack('<I4s',len(payload),b'JSON')+payload+
            struct.pack('<I4s',len(binary),b'BIN\x00')+binary)


def node_matrix(node):
    if 'rotation' in node or 'scale' in node:
        raise ValueError('Expected VUE matrix or translation nodes')
    result=np.array(node.get('matrix',np.eye(4).T.flatten()),dtype=float).reshape(4,4).T.copy()
    if 'translation' in node:result[:3,3]=node['translation']
    return result


def orient_tree(tree,up_axis):
    if up_axis not in ('y','z'):raise ValueError('Expected y or z up axis')
    extras=tree.setdefault('extras',{})
    old=extras.get('vue_coordinates',{}).get('up_axis','z')
    previous=Z_TO_Y if old=='y' else np.eye(4)
    target=Z_TO_Y if up_axis=='y' else np.eye(4)
    delta=target@previous.T
    roots={r for scene in tree['scenes'] for r in scene.get('nodes',[])}
    for index in roots:
        node=tree['nodes'][index]
        if up_axis=='y' and node.get('name')=='world':
            node['name']='VUE_SCENE'  # Avoid importers treating a transformed root as their fixed world frame.
        node['matrix']=(delta@node_matrix(node)).T.flatten().tolist()
        node.pop('translation',None)
    extras['vue_coordinates']=dict(source='E,N,EL; Z-up',up_axis=up_axis,unit='metre',
        mapping='(E, EL, -N)' if up_axis=='y' else '(E, N, EL)')


def accessor_array(tree,binary,index):
    a=tree['accessors'][index]
    if 'sparse' in a:raise ValueError('Sparse accessors are not emitted by VUE exporter')
    view=tree['bufferViews'][a['bufferView']]
    if view.get('buffer',0)!=0:raise ValueError('Expected one internal buffer')
    dtype=np.dtype(DTYPES[a['componentType']]); width=WIDTHS[a['type']]
    offset=view.get('byteOffset',0)+a.get('byteOffset',0)
    stride=view.get('byteStride',dtype.itemsize*width)
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,offset=offset,
                      strides=(stride,dtype.itemsize))


def compact_component_primitives(tree,binary):
    """Concatenate equal-material primitives within each already merged mesh."""
    if tree.get('images') or tree.get('animations') or tree.get('skins'):
        raise ValueError('Only static VUE geometry buffers are supported')
    materials=[]; material_ids={}; remap={}
    for index,material in enumerate(tree.get('materials',[])):
        semantic={k:v for k,v in material.items() if k!='name'}
        key=json.dumps(semantic,sort_keys=True,separators=(',',':'))
        if key not in material_ids:
            material_ids[key]=len(materials)
            materials.append(dict(semantic,name='Material_'+str(len(materials)+1)))
        remap[index]=material_ids[key]
    output=bytearray(); views=[]; accessors=[]; array_cache={}
    def add(array,template,semantic):
        raw=np.ascontiguousarray(array).tobytes()
        key=(semantic,template['componentType'],template['type'],template.get('normalized',False),len(array),hashlib.sha256(raw).digest())
        if key in array_cache:return array_cache[key]
        output.extend(b'\x00'*((-len(output))%4))
        offset=len(output);output.extend(raw)
        views.append(dict(buffer=0,byteOffset=offset,byteLength=len(raw),target=34963 if semantic=='indices' else 34962))
        a={k:v for k,v in template.items() if k in ('componentType','type','normalized')}
        a.update(bufferView=len(views)-1,count=len(array))
        if semantic=='POSITION' and len(array):
            a.update(min=array.min(axis=0).tolist(),max=array.max(axis=0).tolist())
        accessors.append(a);array_cache[key]=len(accessors)-1;return len(accessors)-1
    for mesh in tree['meshes']:
        groups=defaultdict(list)
        for primitive in mesh['primitives']:
            mode=primitive.get('mode',4)
            if set(primitive)-{'attributes','indices','material','mode'} or mode not in (1,4):
                raise ValueError('Unsupported VUE primitive layout')
            layout=tuple(sorted((key,tree['accessors'][value]['componentType'],
                tree['accessors'][value]['type'],tree['accessors'][value].get('normalized',False))
                for key,value in primitive['attributes'].items()))
            groups[(mode,remap.get(primitive.get('material')),layout)].append(primitive)
        primitives=[]
        for (mode,material,layout),parts in groups.items():
            attributes={}
            for semantic,_,_,_ in layout:
                template=tree['accessors'][parts[0]['attributes'][semantic]]
                arrays=[accessor_array(tree,binary,p['attributes'][semantic]) for p in parts]
                attributes[semantic]=add(np.concatenate(arrays),template,semantic)
            indices=[]; offset=0
            for p in parts:
                count=tree['accessors'][p['attributes']['POSITION']]['count']
                idx=(accessor_array(tree,binary,p['indices']).reshape(-1) if 'indices' in p else np.arange(count))
                if len(idx)%(2 if mode==1 else 3) or (len(idx) and int(idx.max())>=count):
                    raise ValueError('Invalid VUE primitive indices')
                if offset+count>2**32:raise ValueError('Component exceeds uint32 vertex range')
                indices.append(idx.astype('<u4')+offset);offset+=count
            small=offset<=65536
            combined=np.concatenate(indices).astype('<u2' if small else '<u4').reshape(-1,1)
            primitive=dict(attributes=attributes,indices=add(combined,
                dict(componentType=5123 if small else 5125,type='SCALAR'),'indices'),mode=mode)
            if material is not None:primitive['material']=material
            primitives.append(primitive)
        mesh['primitives']=primitives
    tree['materials']=materials
    tree['accessors']=accessors;tree['bufferViews']=views
    tree['buffers']=[{'byteLength':len(output)}]
    return bytes(output)


def collapse_component_nodes(tree):
    nodes=tree['nodes']; removed=set()
    for owner in nodes:
        if 'moniker' not in owner.get('extras',{}):continue
        children=owner.get('children',[])
        if not children:continue
        # Sub-objects (no mesh of their own here) stay below the owner; only
        # the single merged geometry node collapses into it.
        meshes=[c for c in children if 'mesh' in nodes[c]]
        others=[c for c in children if 'mesh' not in nodes[c]]
        if not meshes:
            continue
        if len(meshes)!=1:
            raise ValueError('Component geometry must be merged before collapsing its node')
        child=nodes[meshes[0]]
        if child.get('children'):raise ValueError('Unexpected nested geometry')
        absorbed=node_matrix(child)
        owner['matrix']=(node_matrix(owner)@absorbed).T.flatten().tolist()
        owner.pop('translation',None);owner['mesh']=child['mesh']
        tree['meshes'][child['mesh']]['name']=owner['name']
        removed.add(meshes[0])
        if others:
            # Retained sub-objects must undo the absorbed geometry transform,
            # otherwise their world positions pick up the owner's origin twice.
            undo=np.linalg.inv(absorbed)
            for other in others:
                nodes[other]['matrix']=(undo@node_matrix(nodes[other])).T.flatten().tolist()
                nodes[other].pop('translation',None)
            owner['children']=others
        else:owner.pop('children',None)
    remap={old:new for new,old in enumerate(i for i in range(len(nodes)) if i not in removed)}
    for node in nodes:
        if 'children' in node:node['children']=[remap[i] for i in node['children'] if i not in removed]
    for scene in tree['scenes']:scene['nodes']=[remap[i] for i in scene['nodes']]
    tree['nodes']=[n for i,n in enumerate(nodes) if i not in removed]


def finalize_glb_from_tree(tree,binary,up_axis='y',merge_components=False):
    """Finalize an already-built (tree, binary) pair without a GLB round-trip.

    Byte-identical to ``finalize_glb(encode_glb_trimesh(tree, binary))``: the
    tree is mutated in place (merge/collapse/orient) then encoded once.
    """
    if merge_components:
        binary=compact_component_primitives(tree,binary)
        collapse_component_nodes(tree)
    orient_tree(tree,up_axis)
    tree.setdefault('extras',{})['vue_export']=dict(merged_components=merge_components,
        primitives='one per distinct mode, material and attribute layout' if merge_components else 'original parts')
    return encode_glb(tree,binary)


def finalize_glb(data,up_axis='y',merge_components=False):
    tree,binary=read_glb(data)
    return finalize_glb_from_tree(tree,binary,up_axis,merge_components)


def build_glb_tree(scene,include_normals=True):
    """Build the glTF tree and geometry buffer once, shared by GLB and FBX.

    Mirrors trimesh ``export_glb``'s structure/buffer assembly so the result is
    the exact (tree, buffer_data) that ``read_glb(scene.export(...))`` yields,
    letting callers avoid re-running the expensive buffer build per output.
    """
    from trimesh.exchange.gltf import _create_gltf_structure,_build_views
    tree,buffer_items=_create_gltf_structure(
        scene=scene,unitize_normals=True,include_normals=include_normals,
        buffer_postprocessor=None,extension_webp=False)
    views=_build_views(buffer_items)
    buffer_data=b''.join(buffer_items.values())
    if len(buffer_data)>0:
        tree['buffers']=[{'byteLength':len(buffer_data)}]
        tree['bufferViews']=views
    return tree,buffer_data


def encode_glb_trimesh(tree,buffer_data):
    """Encode (tree, buffer_data) exactly as trimesh ``export_glb`` does.

    Used for the FBX-intermediate GLB, whose bytes must match the historical
    ``scene.export(file_type='glb', ...)`` output. Note this JSON-serializes
    with trimesh's ``util.jsonify`` (ensure_ascii=True), unlike ``encode_glb``.
    """
    from trimesh.util import jsonify
    from trimesh.exchange.gltf import _magic
    content=jsonify(tree,separators=(',',':'))
    content+=(4-((len(content)+20)%4))*' '
    content=content.encode('utf-8')
    # GLB chunk lengths must be 4-byte aligned; buffers.byteLength may stay
    # up to 3 bytes shorter than the padded BIN chunk.
    padded=buffer_data+b'\x00'*((-len(buffer_data))%4)
    header=np.array([_magic['gltf'],2,len(content)+len(padded)+28,
                     len(content),_magic['json']],dtype='<u4').tobytes()
    bin_header=np.array([len(padded),0x004E4942],dtype='<u4').tobytes()
    return b''.join([header,content,bin_header,padded])
