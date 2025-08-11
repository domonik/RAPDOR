# -*- mode: python ; coding: utf-8 -*-
import os
import importlib

block_cipher = None

scikit_hiddenimports = [
    'sklearn.utils._cython_blas',
    'sklearn.utils._cython_blas_inner',
    'sklearn._cyutility',
    'sklearn.utils._random',
    'sklearn.utils._logistic_sigmoid',
    'sklearn.utils._heap',
    'sklearn.cluster._k_means_common',
    'sklearn.cluster._k_means_elkan',
    'sklearn.cluster._k_means_lloyd'
]


package_imports = [
    ['dash_daq', ['package-info.json', 'metadata.json', 'dash_daq.min.js', 'async-colorpicker.js']],
    ['dash_extensions', ['package-info.json', 'metadata.json', 'dash_extensions.js']],
    ['RAPDOR', ['visualize/assets/', 'visualize/pages/', "dashConfig.yaml", "tests/testData",  "visualize/callbacks"]],
    ]
rdpsroot = os.path.join(os.getcwd(), "RAPDOR")

datas = [
    ("exestyle.css", ".")
]
for package, files in package_imports:
    if package != "RAPDOR":
        proot = os.path.dirname(importlib.import_module(package).__file__)
    else:
        proot = rdpsroot
    print(os.path.join(proot, files[0]), package)
    for f in files:
        if os.path.isdir(os.path.join(proot, f)):
            datas += [(os.path.join(proot, f), os.path.join(package, f))]
        else:
            datas += [(os.path.join(proot, f), package)]
a = Analysis(
    ['RAPDORexe.py'],
    pathex=[],
    binaries=[],
    datas= datas,
    hiddenimports=[
        "RAPDOR.visualize.pages",
        "RAPDOR.visualize.distributionAndHeatmap",
        "RAPDOR.visualize.clusterAndSettings",
        "RAPDOR.visualize.dataTable",
        "RAPDOR.visualize.modals",
        "RAPDOR.visualize.staticContent",
        "RAPDOR.visualize.callbacks.mainCallbacks",
        "RAPDOR.visualize.callbacks.modalCallbacks",
        "RAPDOR.visualize.callbacks.plotCallbacks",
        "RAPDOR.visualize.callbacks.tableCallbacks",
        "RAPDOR.visualize.callbacks.colorCallbacks",
        "dash_daq"
        ] + scikit_hiddenimports,
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
    [],
    exclude_binaries=True,
    name='RAPDOR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(rdpsroot, 'visualize/assets/favicon.ico')
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RAPDOR',
)
