# -*- mode: python ; coding: utf-8 -*-
import os
import importlib

block_cipher = None


package_imports = [
    ['dash_loading_spinners', ['package-info.json', "dash_loading_spinners.min.js"]],
    ['dash_daq', ['package-info.json', 'metadata.json', 'dash_daq.min.js']]
    ]

datas = [
    ("RDPMSpecIdentifier/visualize/assets/", "RDPMSpecIdentifier/visualize/assets/"),
    ("RDPMSpecIdentifier/qtInterface/", "RDPMSpecIdentifier/qtInterface/"),
    ]
for package, files in package_imports:
    proot = os.path.dirname(importlib.import_module(package).__file__)
    print(os.path.join(proot, files[0]), package)
    datas.extend((os.path.join(proot, f), package) for f in files)

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas= datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RDPMSpecIdentifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="RDPMSpecIdentifier/visualize/assets/favicon.ico"

)
