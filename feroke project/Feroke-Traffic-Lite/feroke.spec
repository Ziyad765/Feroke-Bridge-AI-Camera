# feroke.spec — PyInstaller build specification for Feroke Vanguard
# Run from Feroke-Traffic-Lite directory:
#   pyinstaller feroke.spec
#
# Output: dist/FerokeVanguard/FerokeVanguard.exe (folder mode)
#         dist/FerokeVanguard.exe                (one-file mode — VERY slow to start)
# Folder mode is STRONGLY recommended for AI apps — keeps startup fast.

import sys
import os
from pathlib import Path

block_cipher = None

# ── Paths ────────────────────────────────────────────────────────────────
HERE = Path(spec_root)           # Feroke-Traffic-Lite/
MODELS = HERE / "models"

# ── Data files to bundle ─────────────────────────────────────────────────
# Format: (source, dest_folder_inside_bundle)
datas = [
    # Model weights
    (str(MODELS / "nano_edge.pt"),  "models"),
    # Settings & state JSON (bundled as defaults; the app re-writes them on disk)
    (str(HERE / "feroke_settings.json"), "."),
    (str(HERE / "feroke_state.json"),    "."),
    # Web templates (used by feroke_inference.py web server, kept for completeness)
    (str(HERE / "templates"),       "templates"),
]

# ── Hidden imports needed by ultralytics / PySide6 / pyqtgraph ──────────
hiddenimports = [
    # Ultralytics
    "ultralytics",
    "ultralytics.nn.tasks",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.utils",
    "ultralytics.utils.ops",
    # PyTorch
    "torch",
    "torchvision",
    # PySide6 essentials
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # pyqtgraph
    "pyqtgraph",
    "pyqtgraph.graphicsItems",
    # qdarkstyle
    "qdarkstyle",
    # OpenCV / numpy
    "cv2",
    "numpy",
    # stdlib
    "psutil",
    "csv",
    "json",
    "threading",
]

a = Analysis(
    ["feroke_desktop.py"],
    pathex=[str(HERE)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Strip unused heavy stuff to reduce bundle size
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "notebook",
        "fastapi",
        "uvicorn",
    ],
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
    exclude_binaries=True,        # Folder mode — much faster startup
    name="FerokeVanguard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                     # Compress binaries (optional, requires UPX)
    console=False,                # No console window — pure GUI app
    # icon="feroke_icon.ico",     # Uncomment if you have an .ico file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FerokeVanguard",        # → dist/FerokeVanguard/ folder
)
