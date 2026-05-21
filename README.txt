# Feroke Bridge Smart Traffic System (Lite)

Welcome to the **Feroke Bridge AI Traffic Monitoring System**. This application uses Artificial Intelligence (YOLOv8) to monitor traffic density on the bridge and intelligently control traffic signals to prevent congestion.

![Dashboard Preview](templates/preview.png)

## 🚀 Quick Start (Windows)

1.  **Install Python**: Ensure you have Python 3.10 or newer installed.
2.  **Run the System**:
    Double-click the **`run.bat`** file.
    
    *   *First run:* It will automatically install all dependencies.
    *   *Subsequent runs:* It will launch the server immediately.

3.  **Open Dashboard**:
    Open your browser and go to: **[http://localhost:8000](http://localhost:8000)**

---

## ⚡ NVIDIA GPU Acceleration (Optional)

If you have an NVIDIA Graphics Card (RTX 2060, A2000, 3060, etc.), you can make detection **5x-10x faster**:

1.  Close the running system.
2.  Double-click **`install_gpu.bat`**.
3.  Wait for it to download and install PyTorch with CUDA support (~2.5GB).
4.  Once done, use **`run.bat`** as normal. The system will auto-detect your GPU.

---

## 🎥 Camera Setup

This system is designed to work with **IP Webcam** apps on Android phones.

1.  Connect your PC and Phones to the **Same Wi-Fi**.
2.  Install "IP Webcam" on phones and "Start Server".
3.  Edit **`config.py`** in this folder:
    ```python
    CAM1_URL = "http://192.168.x.x:8080/video"
    CAM2_URL = "http://192.168.x.y:8080/video"
    ```

---

## 🛠️ Features

*   **Zone Editor**: Click "✏️ Edit Zone" on the dashboard to draw custom detection zones on the live video.
*   **Smart Signal Logic**: Prioritizes the busier side, but ensures fair waiting times.
*   **Buffer Time**: Adds an "All Red" phase to allow vehicles to clear the 150m bridge.
*   **Settings Menu**: Adjust Green Times, Buffer Time, and AI Confidence directly from the UI.

## 📁 File Structure

*   `app.py`: Main web server and API.
*   `detector.py`: AI Logic (YOLOv8).
*   `signal_controller.py`: Traffic Light State Machine.
*   `config.py`: Configuration file.
*   `run.bat`: Launcher for Windows.
*   `install_gpu.bat`: GPU Driver installer.

---
*Created for Feroke Bridge Traffic Management Project*
