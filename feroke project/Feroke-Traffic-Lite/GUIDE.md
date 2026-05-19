# Feroke Bridge AI Traffic Monitoring System - User Guide

## 1. Project Overview
This system provides intelligent traffic management for the Feroke Bridge using AI computer vision. It uses live video feeds from two smartphones to detect vehicle density and dynamically control traffic signals.

### Key Features
*   **Live Vehicle Detection**: Uses YOLOv8 AI to detect cars, buses, trucks, and motorcycles in real-time.
*   **Smart Signal Control**: Automatically prioritizes the side with more traffic or waits for a queue to clear.
*   **Live Dashboard**: A web interface showing the live video feed, current signal state, and traffic statistics.
*   **Emergency Mode**: Ability to manually override signals for emergency vehicles.

---

## 2. Technology Stack
We use modern, efficient technologies to ensure fast performance even on standard laptops:

*   **Language**: Python 3.x
*   **AI Model**: YOLOv8 Nano (`yolov8n`) - A lightweight, high-speed object detection model.
*   **Computer Vision**: OpenCV - For video processing and drawing visual overlays.
*   **Backend Server**: FastAPI - A modern, high-performance web framework for building APIs.
*   **Frontend**: HTML5, JavaScript, and TailwindCSS for the dashboard.
*   **Camera Source**: IP Webcam (Android App) - Turns smartphones into network cameras.

---

## 3. Hardware Requirements
*   **PC/Laptop**: Windows 10/11 with Python installed.
*   **Cameras**: 2 x Android Smartphones connected to the same Wi-Fi network as the PC.
*   **Network**: A stable Wi-Fi connection (Routers are preferred over Hotspots for lower latency).

---

## 4. Step-by-Step Setup Guide

### Step A: Mobile Camera Setup (Do this for TWO phones)
1.  **Install App**: Download **"IP Webcam"** from the Google Play Store (by Pavel Khlebovich) on both phones.
2.  **Connect to Wi-Fi**: Ensure both phones and your laptop are connected to the **SAME Wi-Fi network**.
3.  **Configure App**:
    *   Open IP Webcam.
    *   Scroll down to **Local Broadcasting**.
    *   Set **Login/Password** to empty (or remember them if you set them).
    *   Scroll to the bottom and tap **Start Server**.
4.  **Get the IP Address**:
    *   On the camera screen, you will see an IP address like `http://192.168.1.10:8080`.
    *   **Phone 1 (Side A/Left)**: Note this URL.
    *   **Phone 2 (Side B/Right)**: Note this URL.

### Step B: System Configuration
1.  Open the folder `Feroke-Traffic-Lite`.
2.  Right-click `config.py` and open it with Notepad or a Code Editor.
3.  Update the **Camera URLs** with the IPs you got from Step A:
    ```python
    # Example
    CAM1_URL = "http://192.168.1.5:8080/video"  # Phone 1
    CAM2_URL = "http://192.168.1.6:8080/video"  # Phone 2
    ```
    *Note: Keep `/video` at the end to get the video stream.*
4.  Save the file.

### Step C: Running the System
1.  Go to the `Feroke-Traffic-Lite` folder.
2.  Double-click the **`run.bat`** file.
3.  A black terminal window will appear. It will install necessary libraries (first time only) and then start the server.
4.  Wait until you see: `Uvicorn running on http://0.0.0.0:8000`.

---

## 5. Using the Dashboard
1.  Open your web browser (Chrome/Edge).
2.  Go to: **[http://localhost:8000](http://localhost:8000)**
3.  You will see:
    *   **Live Feeds**: Real-time video from both phones.
    *   **Signal Lights**: Red/Green indicators for Side A and Side B.
    *   **Statistics**: Current queue length, time elapsed in current signal, and a history log of signal changes.

---

## 6. How It Works (The Logic)
1.  **Detection**: The AI counts vehicles in the defined "Zones" (config in `config.py`).
2.  **Decision**:
    *   If **Side A** has more cars than **Side B**, Side A gets Green.
    *   It waits for a minimum time (e.g., 10s) before switching.
    *   If a side waits too long (Max Green Time), it forces a switch to be fair.
    *   If both sides are empty, it stays Red or holds the last state.

## 7. Troubleshooting
*   **Video is laggy**: Lower the resolution in the IP Webcam app settings (Video Preferences -> Video Resolution). 640x480 or 1280x720 is recommended.
*   **"Connection Refused"**: Check if the IP address changed. Phones often get new IPs if reconnected to Wi-Fi. Update `config.py`.
*   **AI not detecting**: Ensure the phones are positioned high up, looking down at the road for a clear view.
