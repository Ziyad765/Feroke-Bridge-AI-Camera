<div align="center">

# 🚦 Feroke Bridge AI Camera System

### *Intelligent AI-Powered Traffic Flow Management*
### Feroke Railway Overbridge · Kerala, India

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=for-the-badge&logo=github&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Jetson_Orin-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-Optimized-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Web_Dashboard-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6-Desktop_GUI-41CD52?style=for-the-badge&logo=qt&logoColor=white)

<br/>

> **An end-to-end, production-grade AI traffic monitoring and adaptive signal control system.**
> Built with YOLOv8 · Deployed on NVIDIA Jetson · Trained on a University HPC Supercomputer.

</div>

---

## 🌉 Why We Built This

The **Feroke railway overbridge** is a critical **single-lane crossing** where vehicles from both sides must take turns. Manual/timer-based signals cause:

- ❌ Fixed green times that ignore real traffic density
- ❌ No awareness of whether the bridge has actually been vacated
- ❌ No automatic emergency response to stalled vehicles
- ❌ Peak-hour congestion and unsafe overtaking

**Our AI solution provides:**

- ✅ **Real-time vehicle counting** on both queue sides and the bridge interior
- ✅ **Adaptive green time** — the busier side always gets priority
- ✅ **Bridge clearance verification** — the opposite side is only released after cameras confirm the bridge is empty
- ✅ **Emergency auto-braking** — all signals lock RED if a vehicle is stuck on the bridge for >45 seconds

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│               TVT NVR  (10-Channel)                      │
│  [Cam1: Queue A]  [Cam2: Queue B]                        │
│  [Cam3: Interior A]  [Cam4: Interior B]                  │
└──────────────────────┬───────────────────────────────────┘
                       │  RTSP over TCP (FFmpeg)
                       ▼
┌──────────────────────────────────────────────────────────┐
│              NVIDIA Jetson Orin Nano                     │
│                                                          │
│  4× CameraGrabber ──► AI Dispatcher ──► SignalController │
│   (Threaded, ~150      (Sequential         (State FSM:   │
│    fps grab rate)       YOLOv8 TRT)         ALL_RED /    │
│                                             SIDE_A_GREEN │
│                                             SIDE_B_GREEN │
│                                             EMERGENCY)   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Desktop GUI (PySide6)  OR  Web Dashboard (MJPEG)│    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 📸 Camera Roles

| Slot | Camera | Role |
|:---:|:---|:---|
| `0` | **Queue Side A** | Counts vehicles waiting to enter from Side A |
| `1` | **Queue Side B** | Counts vehicles waiting to enter from Side B |
| `2` | **Bridge Interior A** | Verifies Side A vehicles have fully cleared the bridge |
| `3` | **Bridge Interior B** | Verifies Side B vehicles have fully cleared the bridge |

---

## 🔄 How It Works

<details>
<summary><b>Step 1 — Frame Capture</b></summary>
<br/>

Each of the 4 streams is handled by a dedicated `CameraGrabber` thread. It connects to the TVT NVR via **RTSP over TCP** using OpenCV's FFmpeg backend. Frames are stored in a thread-safe lock at stream FPS. On stream loss, it auto-reconnects in 2 seconds.

```
rtsp://admin:password@192.168.5.100:554/main?channel=1
```

> **Why FFmpeg over GStreamer?** The TVT NVR appends `?channel=N` to its RTSP URLs. GStreamer's `rtspsrc` misparses the `?` and never opens the stream. FFmpeg handles query-string URLs correctly.

</details>

<details>
<summary><b>Step 2 — AI Inference</b></summary>
<br/>

A single `AIDispatcher` loops **sequentially** through all 4 cameras (0→1→2→3) to avoid GPU contention. Each camera has its own **isolated YOLO model instance** to prevent tracker state contamination (eliminating a 200ms slowdown bug).

- Model runs on the Jetson GPU with `half=True` (FP16)
- Interior cameras **only run during `ALL_RED`** — skipped otherwise to save GPU cycles

</details>

<details>
<summary><b>Step 3 — Zone-Based Counting</b></summary>
<br/>

Each camera has a **configurable polygon detection zone** drawn interactively in the GUI. Only vehicles whose bounding-box centroid falls *inside* the zone are counted toward signal logic. Vehicles outside are rendered but ignored.

</details>

<details>
<summary><b>Step 4 — Signal State Machine</b></summary>
<br/>

```
ALL_RED (Bridge Clearance)
 ├─ Wait minimum clearance time
 ├─ Interior cameras: bridge empty?
 │   ├─ YES → switch to busier side's GREEN
 │   └─ NO  → wait until MAX_CLEARANCE timeout
 └─ If no vehicles anywhere → stay ALL_RED

SIDE_A_GREEN / SIDE_B_GREEN
 ├─ Wait MIN_GREEN_TIME
 ├─ If queue empties AND opposite side has cars → early ALL_RED
 └─ If MAX_GREEN_TIME reached → forced ALL_RED

EMERGENCY 🚨
 └─ Bridge has vehicles for >45s → ALL signals lock RED
    Auto-clears when bridge empties
```

</details>

---

## 🧠 AI Training Journey

The custom model was purpose-built to classify **Light (Class 0)** and **Heavy (Class 1)** vehicles from a bridge-mounted camera perspective across **4 evolutionary phases**.

```
Phase 1           Phase 2             Phase 3              Phase 4
────────          ────────            ────────             ────────
29 hand-curated   Bulk Auto-Label     HPC Supercomputer    Jetson Edge
images            482 frames          Training (A100)      Optimization
    │                  │                   │                    │
    └──► seed.pt ──────► reviewed ─────────► 82.2% mAP ────────► TensorRT
                        dataset             heavy_hpc.pt        nano_edge
                                                                76.7% mAP
                                                                imgsz=416
```

| Phase | Dataset | Platform | Result |
|:---:|:---|:---|:---|
| **1 — Seed** | 29 curated images | Local GPU | Initial bridge-aware model |
| **2 — Scale** | 482 auto-labeled frames | Local + Streamlit review UI | Clean large dataset in minutes |
| **3 — HPC** | Full merged dataset | University DGX A100 (SSH + Docker) | `heavy_hpc.pt` · **mAP50 = 82.2%** |
| **4 — Edge** | Same dataset @ 320px | Jetson TensorRT export | `nano_edge.engine` · **mAP50 = 76.7%** |

**HPC Training Stack:** WinSCP (upload) → SSH → `docker run --gpus device=0 nvcr.io/nvidia/pytorch:22.11-py3` → Jupyter Notebook → WinSCP (download `best.pt`)

---

## 📁 Project Structure

```
cam-export/
│
├── 📄 README.md
├── 📄 AI_TRAINING_GUIDE.md          ← Full AI pipeline masterclass
├── 📄 MASTER_GUIDE.md               ← Operations & Jetson optimization
│
├── 🗂️  Training & Data Scripts
│   ├── bulk_auto_label.py           ← Auto-annotate raw frames with AI
│   ├── auto_label_and_review.py     ← Streamlit label review UI
│   ├── merge_datasets.py            ← Combine datasets
│   ├── train_v2_ultimate.py         ← High-accuracy HPC trainer
│   ├── train_edge_nano.py           ← Edge-optimized trainer (imgsz=320)
│   ├── fast_inference_test.py       ← Live stream test @ 60 FPS
│   └── streamlit_app.py             ← Upload-based visual tester
│
├── 🗂️  HPC_Training_Workspace/
│   ├── 01_HPC_MASTER_GUIDE.md       ← Step-by-step HPC deployment guide
│   ├── 02_Jupyter_Trainer_Code.py   ← Jupyter-ready training code
│   ├── HPC_Dataset_Builder.py       ← Remote dataset preparation
│   └── data.yaml                    ← YOLO dataset config (2 classes)
│
└── 🗂️  feroke project/
    ├── 📄 JETSON_DEPLOYMENT_GUIDE.md
    ├── 📄 JETSON_NVR_INTEGRATION.md
    ├── 📜 install_jetson.sh          ← Full automated Jetson bootstrapper
    │
    ├── 🗂️  Feroke-Traffic-Lite/      ← ⭐ PRIMARY APPLICATION
    │   ├── config.py                 ← All configuration (cameras, AI, timing)
    │   ├── feroke_desktop.py         ← Native Desktop GUI (PySide6)
    │   ├── feroke_inference.py       ← Headless AI engine (FastAPI + MJPEG)
    │   ├── signal_controller.py      ← Traffic signal state machine
    │   ├── app.py                    ← Lightweight FastAPI server
    │   ├── models/
    │   │   ├── nano_edge.pt          ← Primary model (fast, edge-optimized)
    │   │   └── heavy_hpc.pt          ← High-accuracy model
    │   ├── feroke_settings.json      ← Persisted zones & timings
    │   ├── run.bat                   ← Windows launcher
    │   ├── run_jetson.sh             ← Jetson direct launcher
    │   └── build_windows.bat         ← Windows .exe packager
    │
    └── 🗂️  Feroke-Traffic-Jetson/    ← Jetson deployment orchestrator
        ├── run.sh                    ← Master startup script (auto TRT compile)
        └── update.sh                 ← Pull latest code & restart
```

---

## 🚀 Deployment Modes

### 🖥️ Mode 1 — Windows (Development / Demo)

```bash
cd "feroke project/Feroke-Traffic-Lite"
pip install -r requirements.txt

python feroke_inference.py     # Web dashboard → http://localhost:8000
# OR
python feroke_desktop.py       # Native desktop GUI
```

> **GPU acceleration (optional):** Double-click `install_gpu.bat` to install PyTorch with CUDA (~2.5 GB). The system auto-detects your GPU.

---

### 🛰️ Mode 2 — Jetson Orin Nano (Production)

```bash
cd "feroke project/Feroke-Traffic-Jetson"
chmod +x run.sh && ./run.sh
```

**What `run.sh` does automatically:**

| Step | Action |
|:---:|:---|
| 1 | Installs system GUI libraries (`libxcb`, etc.) |
| 2 | Runs `pip install -r requirements.txt` |
| 3 | Locks GPU to max clocks (`sudo jetson_clocks`) |
| 4 | Compiles `nano_edge.pt` → `nano_edge.engine` via TensorRT *(first run, ~10 min)* |
| 5 | Launches **Feroke Vanguard** desktop GUI |
| 6 | Falls back to web dashboard on port `8000` if no display found |

```bash
# Monitor GPU in a second terminal:
sudo jtop
```

---

### 🌐 Mode 3 — Headless Web Dashboard

```bash
python feroke_inference.py
# Open on any LAN device → http://<jetson-ip>:8000
```

---

## 📷 Camera Configuration

Edit `feroke project/Feroke-Traffic-Lite/config.py`:

```python
# ── NVR Mode ──────────────────────────────────────────────
USE_NVR_MODE      = True
NVR_IP            = "192.168.5.100"
NVR_PORT          = "554"
NVR_USER          = "admin"
NVR_PASS          = "your_password"
STREAM_TYPE       = "sub"          # "sub" = fast substream (recommended)

# Map 4 NVR channel numbers to the 4 functional slots:
#   Index 0 → Queue Side A    (vehicle counting)
#   Index 1 → Queue Side B    (vehicle counting)
#   Index 2 → Bridge Interior A  (clearance check)
#   Index 3 → Bridge Interior B  (clearance check)
SELECTED_CHANNELS = [1, 2, 3, 4]  # ← edit with your actual channel numbers

# ── AI ────────────────────────────────────────────────────
MODEL_PATH            = "./models/nano_edge.pt"
USE_TENSORRT          = True        # auto-use .engine if present
CONFIDENCE_THRESHOLD  = 0.25
AI_IMG_SIZE           = 416

# ── Signal Timings (seconds) ──────────────────────────────
MIN_GREEN_TIME    = 10
MAX_GREEN_TIME    = 30
MAX_CLEARANCE_TIME = 15
```

> **Phone/Local Camera (testing):** Set `USE_NVR_MODE = False` and fill `CAM1_URL` with your IP Webcam address.

---

## ⚙️ Key Settings Reference

| Parameter | Default | Description |
|:---|:---:|:---|
| `USE_NVR_MODE` | `True` | Toggle between NVR and phone cameras |
| `STREAM_TYPE` | `main` | `sub` strongly recommended on Jetson |
| `USE_TENSORRT` | `True` | Auto-loads `.engine` for 2–3× speedup |
| `CONFIDENCE_THRESHOLD` | `0.25` | Min detection confidence (0–1) |
| `AI_IMG_SIZE` | `416` | Inference resolution — lower = faster |
| `MIN_GREEN_TIME` | `10s` | Minimum phase duration before logic can switch |
| `MAX_GREEN_TIME` | `30s` | Forced switch after this duration |
| `MAX_CLEARANCE_TIME` | `15s` | ALL_RED timeout before overriding interior check |

---

## 📊 Performance Benchmarks

| Mode | Hardware | FPS (4 Cams) | AI Latency | GPU Load |
|:---|:---|:---:|:---:|:---:|
| PyTorch `.pt` (CPU) | i7 Laptop | 2–4 | 400–800ms | — |
| PyTorch `.pt` (GPU) | RTX 2060 | 15–25 | 40–80ms | 60–80% |
| PyTorch `.pt` | Jetson Orin | 8–15 | 80–150ms | 70–90% |
| **TensorRT `.engine`** | **Jetson Orin** | **30–45** | **20–35ms** | **30–50%** |

| Model | Accuracy (mAP50) | Size | Best For |
|:---|:---:|:---:|:---|
| `heavy_hpc.pt` | **82.2%** | 20 MB | High-accuracy auditing |
| `nano_edge.pt` | **76.7%** | 20 MB | Production 24/7 deployment |
| `nano_edge.engine` | **76.7%** | ~10 MB | TensorRT runtime (same accuracy, faster) |

---

## 🛠️ Technologies Used

| Layer | Technology |
|:---|:---|
| **AI Detection** | YOLOv8 (Ultralytics) — custom-trained 2-class model |
| **Edge Inference** | NVIDIA TensorRT (`.engine` format, FP16 half-precision) |
| **Video Capture** | OpenCV + FFmpeg backend (RTSP/TCP, no-buffer) |
| **Desktop GUI** | PySide6 + QDarkStyle + PyQtGraph |
| **Web Dashboard** | FastAPI + Uvicorn + MJPEG streaming |
| **Night Vision** | CLAHE (Contrast Limited Adaptive Histogram Equalization) |
| **Concurrency** | Python threading (grabbers, AI dispatcher, logic loop) |
| **Training** | Ultralytics YOLO, PyTorch, Streamlit (label review) |
| **Packaging** | PyInstaller (Windows `.exe`), Bash scripts (Jetson service) |
| **System Vitals** | psutil (CPU/RAM monitoring in GUI) |

---

## 🔁 Model Improvement Flywheel

```
New raw footage
      │
      ▼
 bulk_auto_label.py  →  AI draws boxes on every image
      │
      ▼
 auto_label_and_review.py  →  Streamlit UI to fix bad predictions
      │
      ▼
 merge_datasets.py  →  Combine old + new data
      │
      ▼
 train_v2_ultimate.py  →  Fine-tune from current best model
      │
      ▼
 On Jetson:
 yolo export model=best.pt format=engine device=0 half=True imgsz=416
      │
      ▼
 Replace  models/nano_edge.engine  →  Restart system ✅
```

---

## 🔧 Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| No camera feed | Wrong NVR IP or password | Ping NVR, verify `config.py` credentials |
| High lag on Jetson | PyTorch `.pt` mode | Wait for `.engine` compile or verify TensorRT install |
| `Connection lost` loop | Stream drop | Already mitigated — using TCP + auto-reconnect |
| Black screen from NVR | H.265 vs H.264 mismatch | Set `CAMERA_ENCODING = "h265"` in `config.py` |
| NVML assert crash on export | JetPack CUDA allocator bug | Fixed — `run.sh` sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1` |
| Interior cams always idle | Expected behaviour | Interior only runs during `ALL_RED` clearance phase |
| GUI won't open on Jetson | `DISPLAY` not set | `run.sh` exports `DISPLAY=:0` automatically |
| Zone not persisting | Write permission | Ensure `feroke_settings.json` is writable |

---

## 💻 Hardware Requirements

<table>
<tr>
<th>Development / Testing</th>
<th>Production (Deployment)</th>
<th>AI Training</th>
</tr>
<tr>
<td>

- Intel Core i5 (8th gen+)
- 8 GB RAM
- NVIDIA GTX 1060+ *(optional)*
- Python 3.10+
- Windows 10/11 or Ubuntu

</td>
<td>

- **NVIDIA Jetson Orin Nano** (4/8 GB)
- 32 GB+ storage
- Gigabit Ethernet
- TVT 10-ch NVR
- JetPack 5.x / 6.x

</td>
<td>

- University DGX A100
- SSH + Docker access
- WinSCP for file transfer
- `nvcr.io/nvidia/pytorch:22.11-py3`

</td>
</tr>
</table>

---

<div align="center">

**Built for safer, smarter traffic flow on the Feroke Railway Overbridge.**

![Made with YOLOv8](https://img.shields.io/badge/Made%20with-YOLOv8-00BFFF?style=flat-square)
![Runs on Jetson](https://img.shields.io/badge/Runs%20on-NVIDIA%20Jetson-76B900?style=flat-square&logo=nvidia)
![Kerala India](https://img.shields.io/badge/📍-Kerala%2C%20India-FF9933?style=flat-square)

</div>
