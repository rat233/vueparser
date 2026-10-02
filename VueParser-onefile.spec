# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata
binaries=collect_dynamic_libs('pyassimp')
datas=[]
for package in ('trimesh','numpy','scipy','mapbox-earcut','pyassimp'):
    datas += copy_metadata(package)
a=Analysis(['vue_cli.py'],pathex=[],binaries=binaries,datas=datas,
    hiddenimports=['scipy.spatial.transform._rotation_groups','mapbox_earcut','pyassimp','vue_fbx_phong_material','vue_fbx_from_exported_glb'],
    excludes=['torch','tensorflow','pandas','matplotlib','IPython','pytest','sympy','sklearn','notebook','jupyter','tkinter','PyQt5','PyQt6','PySide6'],
    noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='VueParser',console=True,upx=False)
