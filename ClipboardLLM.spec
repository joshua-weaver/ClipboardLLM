# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['src\\main.py'],
    pathex=['src'],  # Add src to the Python path
    binaries=[],
    datas=[
        ("src/clippy.ico", "src"),
        ("src/resource_helper.py", "src")  # Include resource_helper.py
    ],
    hiddenimports=['tkinter', 'win32clipboard', 'win32con', 'requests', 'resource_helper'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ClipboardLLM',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)