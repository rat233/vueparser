"""Two-pass FBX converter that restores the reference Phong material rules.

Assimp's ASCII FBX importer silently drops ``SpecularColor``,
``Shininess`` and the Phong shading model, so the first-pass binary FBX
produced from our ASCII intermediate carries only diffuse/ambient/opacity.
This helper reloads that first-pass binary FBX, injects the reference
Navisworks convention observed in 15300.fbx (Phong, specular 0.7,
shininess 0.001) into every material via ctypes, and exports the final
binary FBX.

The loaded scene is deliberately never released: ``aiReleaseImport``
would run C++ destructors over the ctypes-allocated properties and
corrupt the heap.  ``os._exit`` lets the OS reclaim everything instead.
"""
import ctypes
import os
import struct
import sys
from ctypes import POINTER

import pyassimp
from pyassimp import structs

PTI_Float, PTI_Integer = 1, 4

KEEP_ALIVE = []


def make_prop(key, ptype, payload):
    prop = structs.MaterialProperty()
    key_bytes = key.encode()
    prop.mKey.length = len(key_bytes)
    prop.mKey.data = key_bytes
    prop.mSemantic = 0
    prop.mIndex = 0
    prop.mType = ptype
    buf = ctypes.create_string_buffer(payload, len(payload))
    prop.mData = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    prop.mDataLength = len(payload)
    KEEP_ALIVE.extend([prop, buf])
    return prop


def inject_phong(material_ptr):
    mat = material_ptr.contents
    n = mat.mNumProperties
    props = [
        make_prop('$mat.shadingmodel', PTI_Integer, struct.pack('<i', 3)),
        make_prop('$clr.specular', PTI_Float, struct.pack('<3f', 0.7, 0.7, 0.7)),
        make_prop('$mat.shininess', PTI_Float, struct.pack('<f', 0.001)),
    ]
    arr = (POINTER(structs.MaterialProperty) * (n + len(props)))()
    for i in range(n):
        arr[i] = mat.mProperties[i]
    for i, prop in enumerate(props):
        arr[n + i] = ctypes.pointer(prop)
    mat.mProperties = ctypes.cast(arr, ctypes.POINTER(ctypes.POINTER(structs.MaterialProperty)))
    mat.mNumProperties = n + len(props)
    mat.mNumAllocated = max(mat.mNumAllocated, n + len(props))
    KEEP_ALIVE.append(arr)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cm = pyassimp.load(src, processing=pyassimp.postprocess.aiProcess_Triangulate)
    scene = cm.__enter__()
    for material in scene.materials:
        inject_phong(material)
    pyassimp.export(scene, dst, file_type='fbx')
    with open(dst, 'rb') as stream:
        ok = stream.read(23).startswith(b'Kaydara FBX Binary')
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
