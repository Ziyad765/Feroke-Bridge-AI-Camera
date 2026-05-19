@echo off
echo Installing NVIDIA GPU Support for Feroke Traffic System...
echo This ensures detection runs FAST on RTX cards.

pip uninstall -y torch torchvision ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics

echo.
echo GPU Setup Complete! You can now use run.bat normally.
pause
