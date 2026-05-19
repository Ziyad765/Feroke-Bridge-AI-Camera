#!/bin/bash

echo "🚀 FEROKE AI: STARTING JETSON VANGUARD SYSTEM..."

# 1. INSTALLATION & DEPENDENCY CHECK
echo "-----------------------------------------------"
echo "📦 Checking and installing dependencies..."
echo "-----------------------------------------------"

# Update and install system-level GUI dependencies for PySide6 if they're missing
# Using sudo -n to avoid blocking if sudo is not configured for passwordless, 
# but usually Jetson users have sudo access.
if command -v apt-get &> /dev/null; then
    echo "🔍 Checking system libraries for GUI..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
        python3-pip python3-setuptools
fi

# Install python requirements
echo "🐍 Installing Python requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "-----------------------------------------------"
echo "⚙️  Optimizing GPU Clocks..."
sudo jetson_clocks
echo "-----------------------------------------------"

# 2. AUTO-COMPILE TENSORRT ENGINE
MODEL_PT="../Feroke-Traffic-Lite/models/nano_edge.pt"
MODEL_ENGINE="../Feroke-Traffic-Lite/models/nano_edge.engine"

# Ensure models directory exists
mkdir -p ../Feroke-Traffic-Lite/models

if [ ! -f "$MODEL_ENGINE" ]; then
    echo "🆕 Compiling HPC Model to TensorRT (.engine)..."
    echo "⚠️  This takes ~5-10 minutes on Jetson Nano. PLEASE WAIT."
    # We must be in the Lite directory for imports to work correctly during export if any
    cd ../Feroke-Traffic-Lite
    python3 -c "from ultralytics import YOLO; model = YOLO('models/nano_edge.pt'); model.export(format='engine', device=0, half=True, imgsz=416)"
    cd ../Feroke-Traffic-Jetson
    
    if [ $? -eq 0 ]; then
        echo "✅ COMPILE SUCCESS: System is now optimized for hardware."
    else
        echo "❌ COMPILE FAILED: Continuing with PyTorch mode."
    fi
else
    echo "⚡ Optimized Engine Found: Skipping compilation."
fi

# 3. LAUNCH THE DESKTOP GUI
echo "-----------------------------------------------"
echo "🛰️  Launching Feroke Vanguard Command Center..."
echo "-----------------------------------------------"

cd ../Feroke-Traffic-Lite
export DISPLAY=:0

# Start the Desktop GUI
python3 feroke_desktop.py

if [ $? -ne 0 ]; then
    echo "❌ GUI Launch failed."
    echo "💡 Note: Desktop environment must be active (HDMI/DP display)."
    echo "⚠️  Falling back to Web Dashboard..."
    python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
fi
