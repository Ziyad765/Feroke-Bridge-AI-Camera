#!/bin/bash
echo ""
echo "====================================================="
echo "  FEROKE VANGUARD -- JETSON NATIVE EDITION"
echo "====================================================="
echo ""
echo "Initializing AI Core on Jetson Hardware..."
echo "(Ensure you are running on an active HDMI Display or Desktop Environment,"
echo " as GUI applications require a visual compositor like X11 or Wayland.)"
echo ""

# Jetson usually routes system python to python3
export DISPLAY=:0
python3 feroke_desktop.py
