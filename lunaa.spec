# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['lunaa.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('lunaa_modules', 'lunaa_modules'),
        ('extensions', 'extensions'),
    ],
    hiddenimports=[
        'ollama',
        'torch',
        'transformers',
        'datasets',
        'numpy',
        'scipy',
        'pandas',
        'PIL',
        'sv_ttk',
        'requests',
        'scrapy',
        'geopy',
        'sounddevice',
        'soundfile',
        'matplotlib',
        'plotly',
    ],
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
    name='Lunaa',
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
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Lunaa',
)
