# 🚀 Jetson Orin Nano Deployment Guide - Feroke Traffic System

This guide explains how to set up the **Feroke Traffic System** on an NVIDIA Jetson Orin Nano Developer Kit.

## 📦 Prerequisites

1.  **NVIDIA Jetson Orin Nano** (Running JetPack 5.1.1 or 6.0+)
2.  **Internet Connection** (Ethernet or WiFi)
3.  **Camera** (USB Webcam or IP Webcam URL)

---

## 🛠️ Step 1: System Setup

Open a terminal on your Jetson and run:

```bash
sudo apt update
sudo apt install python3-pip python3-dev libopenblas-dev
```

## 🐍 Step 2: Install PyTorch (The Tricky Part)

**Do NOT use `pip install torch`**. It installs the slow CPU version.
Jetson requires a specific version compiled for ARM64 + CUDA.

**Check if you already have it:**
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
*If it prints a version (e.g., `2.1.0a0+...`) and `True`, you are good! Skip to Step 3.*

**If not installed:**
Follow NVIDIA's official guide to install PyTorch for your JetPack version.
Generally for **JetPack 5.1 (L4T 35.x)**:
```bash
# Install dependencies
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev

# Install PyTorch (Example for v2.1.0)
wget https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# Install Torchvision (Match version!)
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.0
python3 setup.py install --user
```
*(Check [NVIDIA Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) for the exact link matching your JetPack version)*

---

## 📥 Step 3: Install Project Dependencies

Navigate to this folder (`Feroke-Traffic-Jetson`) and run:

```bash
# Don't reinstall torch/torchvision if we just did it manually!
pip3 install ultralytics opencv-python-headless fastapi uvicorn jinja2 python-multipart websockets numpy<2.0.0
```

*Note: `ultralytics` might try to upgrade torch. If it breaks things, reinstall the NVIDIA torch whl.*

---

## 🚀 Step 4: Run the System

1.  **Make the script executable:**
    ```bash
    chmod +x run.sh
    ```

2.  **Start the App:**
    ```bash
    ./run.sh
    ```

3.  **Access the Dashboard:**
    *   Open a browser on any PC in the same network.
    *   Go to: `http://<JETSON_IP_ADDRESS>:8000`
    *   Find IP using `ifconfig` (look for `eth0` or `wlan0`).

---

## 📹 Configuring Cameras

Edit `config.py` to set your camera sources:

**Option A: USB Camera** (Plugged into Jetson)
```python
CAM1_URL = 0  # /dev/video0
CAM2_URL = 1  # /dev/video1
```

**Option B: IP Webcam** (Same as Lite version)
```python
CAM1_URL = "http://192.168.1.100:8080/video"
```

**Option C: CSI Camera** (Rashpberry Pi Cam via ribbon cable)
You might need a GStreamer string.
```python
CAM1_URL = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
```

---

## ⚡ Performance Tips for Orin Nano

1.  **Maximize Power Mode:**
    ```bash
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```
    This forces the CPU/GPU to run at max detection speed.

2.  **Resolution:**
    In the Dashboard "Settings", keep resolution at `640x480` or `480x360` for highest FPS.
