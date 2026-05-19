"""
feroke_inference.py — Ultra-Fast 4-Camera Sequential Isolated Engine (Monolithic Enterprise V2)
====================================================================
Architecture (The true fix):
  4× Grabbers  → fill raw_frame at 150fps (non-blocking)
  4× YOLO Models → isolated tracker states per camera (prevents 200ms slowdown)
  1× AI Dispatcher → loops 0→1→2→3 sequentially. No GPU contention.
  1× Web Server → Embedded FastAPI running on Uvicorn accessing RAM directly (0ms File IO lag)

Launch: python feroke_inference.py
"""
import cv2
import time
import json
import os
import threading
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Body, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from signal_controller import SignalController
import config

# Globals for Monolithic UI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

NUM_CAMS = 4
global_grabbers = []
global_stores = []
global_ctrl = None
global_rendered_frames = [np.zeros((360, 480, 3), dtype=np.uint8)] * NUM_CAMS
preview_enabled = True  # Toggle for deep performance
event_log = []

def log_event(msg):
    event_log.insert(0, {"time": time.strftime("%H:%M:%S"), "event": msg})
    if len(event_log) > 50:
        event_log.pop()

# ────────────────────────────────────────────────────────────────────────
# CAMERA GRABBER — fills raw_frame at stream FPS, never blocks
# ────────────────────────────────────────────────────────────────────────
class CameraGrabber:
    def __init__(self, index, source):
        self.index = index
        self.source = source
        self.raw_frame = None
        self.stopped = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._grab, daemon=True, name=f"Grab_{self.index}").start()
        return self

    def _open_cap(self):
        """Open the capture with hardware-proven FFMPEG settings (TCP, no-buffer)."""
        if config.USE_NVR_MODE:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = config.NVR_FFMPEG_OPTIONS
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _grab(self):
        cap = self._open_cap()
        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                print(f"[Cam {self.index}] Stream lost — reconnecting in 2s...")
                cap.release()
                time.sleep(2)
                cap = self._open_cap()
                continue
            frame = cv2.resize(frame, (640, 480))
            with self._lock:
                self.raw_frame = frame
        cap.release()

    def get_frame(self):
        with self._lock:
            return self.raw_frame.copy() if self.raw_frame is not None else None


# ────────────────────────────────────────────────────────────────────────
# DETECTION STORE — per camera, holds latest AI results
# ────────────────────────────────────────────────────────────────────────
class DetectionStore:
    def __init__(self, index, label):
        self.index       = index
        self.label       = label
        self.is_clearance = index in [2, 3]
        self.zone        = None 
        self.boxes       = [] 
        self.count       = 0
        self.inf_ms      = 0.0
        self.ai_fps      = 0.0
        self._lock       = threading.Lock()

    def update(self, boxes, count, inf_ms, ai_fps):
        with self._lock:
            self.boxes    = boxes
            self.count    = count
            self.inf_ms   = inf_ms
            self.ai_fps   = ai_fps

    def read(self):
        with self._lock:
            return (self.boxes[:], self.count, self.inf_ms, self.ai_fps)

    def get_state(self):
        with self._lock:
            return {
                "label":  self.label,
                "count":  self.count,
                "inf_ms": round(self.inf_ms, 1),
                "fps":    round(self.ai_fps, 1),
            }


# ────────────────────────────────────────────────────────────────────────
# AI DISPATCHER
# ────────────────────────────────────────────────────────────────────────
class AIDispatcher:
    def __init__(self, models, grabbers, stores, ctrl):
        self.models   = models    
        self.grabbers = grabbers
        self.stores   = stores
        self.ctrl     = ctrl
        self.dev      = 0 if torch.cuda.is_available() else 'cpu'
        self.stopped  = False
        self._prev_t  = [time.time()] * NUM_CAMS

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="AI_Dispatcher").start()
        return self

    def _run(self):
        while not self.stopped:
            for i in range(NUM_CAMS):
                store   = self.stores[i]
                grabber = self.grabbers[i]
                model   = self.models[i]

                if store.is_clearance and self.ctrl.state.value != "ALL_RED":
                    store.update([], 0, 0, 0)
                    continue

                frame = grabber.get_frame()
                if frame is None:
                    continue

                t0 = time.time()
                results = model.track(
                    frame, persist=True, verbose=False, imgsz=config.AI_IMG_SIZE,
                    conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD,
                    agnostic_nms=config.AGNOSTIC_NMS, device=self.dev, half=(self.dev == 0)
                )
                inf_ms = (time.time() - t0) * 1000

                result = results[0]
                boxes = []
                count = 0
                if result.boxes is not None:
                    zone_np = np.array(store.zone, dtype=np.int32) if store.zone is not None else None
                    for box in result.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        in_zone = (zone_np is not None and cv2.pointPolygonTest(zone_np, (cx, cy), False) >= 0)
                        if in_zone: count += 1
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        name = "Heavy" if cls == 1 else "Light"
                        boxes.append((x1, y1, x2, y2, in_zone, conf, name))

                t_now = time.time()
                ai_fps = 1.0 / (t_now - self._prev_t[i]) if (t_now - self._prev_t[i]) > 0 else 0
                self._prev_t[i] = t_now
                store.update(boxes, count, inf_ms, ai_fps)


def render_frame_for_stream(i, raw_frame, store, ctrl):
    if not preview_enabled:
        return np.zeros((360, 480, 3), dtype=np.uint8) # Blank frame if disabled
        
    frame = raw_frame.copy()
    boxes, count, inf_ms, ai_fps = store.read()
    state = ctrl.state.value
    zone  = store.zone

    if zone is not None:
        zone_np = np.array(zone, dtype=np.int32)
        cv2.polylines(frame, [zone_np], True, (0, 255, 255), 2)

    for (x1, y1, x2, y2, in_zone, conf, name) in boxes:
        color = (0, 0, 255) if in_zone else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    sig_color = (0, 255, 0) if ((store.index == 0 and state == "SIDE_A_GREEN") or (store.index == 1 and state == "SIDE_B_GREEN")) else (0, 0, 255)
    
    cv2.circle(frame, (30, 30), 15, sig_color, -1)
    cv2.putText(frame, f"{store.label} | AI:{inf_ms:.0f}ms", (55, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Count: {count}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return frame

def renderer_loop():
    while True:
        if preview_enabled:
            for i, (g, s) in enumerate(zip(global_grabbers, global_stores)):
                raw = g.get_frame()
                if raw is not None:
                    global_rendered_frames[i] = render_frame_for_stream(i, raw, s, global_ctrl)
        time.sleep(0.06) # ~15 FPS max UI stream to save bandwidth

def logic_loop(stores, ctrl):
    last_state = ctrl.state.value
    while True:
        counts = [s.count for s in stores]
        ctrl.update_counts(counts[0], counts[1], counts[2], counts[3])
        ctrl.run_logic()
        
        # Log state changes
        if ctrl.state.value != last_state:
            log_event(f"Signal Changed: {last_state} -> {ctrl.state.value}")
            last_state = ctrl.state.value
            
        time.sleep(1)

# ────────────────────────────────────────────────────────────────────────
# FASTAPI ROUTES (MONOLITHIC IN-MEMORY SERVER)
# ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status():
    if not global_ctrl: return {"error": "System Booting"}
    return {
        "state": global_ctrl.state.value,
        "elapsed": global_ctrl.get_status()["elapsed"],
        "queue_a": global_stores[0].count,
        "queue_b": global_stores[1].count,
        "clear_a": global_stores[2].count,
        "clear_b": global_stores[3].count,
        "cameras": [s.get_state() for s in global_stores],
        "preview_enabled": preview_enabled,
        "history": event_log
    }

def generate_mjpeg(cam_id):
    while True:
        if not preview_enabled:
            time.sleep(1)
            continue
        frame = global_rendered_frames[cam_id]
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.06)

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    return StreamingResponse(generate_mjpeg(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/toggle_preview")
async def toggle_preview():
    global preview_enabled
    preview_enabled = not preview_enabled
    log_event(f"Video Previews {'Enabled' if preview_enabled else 'Disabled'}")
    return {"preview_enabled": preview_enabled}

@app.post("/api/update_settings")
async def update_settings(settings: dict = Body(...)):
    global_ctrl.update_config(
        min_green=settings.get("min_green", config.MIN_GREEN_TIME),
        max_green=settings.get("max_green", config.MAX_GREEN_TIME),
        max_clearance=settings.get("max_clearance", config.MAX_CLEARANCE_TIME)
    )
    log_event("Timing Configurations Updated.")
    return {"status": "success"}

@app.get("/api/zones")
async def get_zones():
    return [s.zone for s in global_stores]

@app.post("/api/zones")
async def set_zones(zones: list = Body(...)):
    for i, z in enumerate(zones):
        if i < len(global_stores) and z:
            global_stores[i].zone = z
            
    # Save to disk permanently
    settings_file = os.path.join(os.path.dirname(__file__), "feroke_settings.json")
    try:
        with open(settings_file, "r") as f: d = json.load(f)
    except: d = {}
    d["zones"] = zones
    with open(settings_file, "w") as f: json.dump(d, f)
    
    log_event("Security Zones Permentantly Saved.")
    return {"status": "success"}

# ────────────────────────────────────────────────────────────────────────
# BOOTSTRAPPER
# ────────────────────────────────────────────────────────────────────────

def run_ai_system():
    global global_grabbers, global_stores, global_ctrl
    print("[Init] Monolithic AI Intelligence Booting...")
    
    dev = 0 if torch.cuda.is_available() else 'cpu'
    global_ctrl = SignalController(
        min_green=config.MIN_GREEN_TIME, 
        max_green=config.MAX_GREEN_TIME, 
        max_clearance=config.MAX_CLEARANCE_TIME
    )
    
    # Pre-Load saved zones
    settings_file = os.path.join(os.path.dirname(__file__), "feroke_settings.json")
    saved_zones = [config.ZONE_A, config.ZONE_B, config.ZONE_C, config.ZONE_D]
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                d = json.load(f)
                if "zones" in d: saved_zones = d["zones"]
                if "min_green" in d: global_ctrl.min_green = d["min_green"]
                if "max_green" in d: global_ctrl.max_green = d["max_green"]
                if "max_clearance" in d: global_ctrl.max_clearance = d["max_clearance"]
        except: pass

    sources = [config.CAM1_URL, config.CAM2_URL, config.CAM3_URL, config.CAM4_URL]
    labels  = ["Queue A", "Queue B", "Interior A", "Interior B"]

    # Print the channel → slot mapping so you can verify the order at startup
    if config.USE_NVR_MODE:
        print("[Init] NVR Channel Mapping:")
        for slot, (ch, lbl) in enumerate(zip(config.SELECTED_CHANNELS, labels)):
            print(f"  Slot {slot} ({lbl}) → NVR Channel {ch}")

    global_grabbers = [CameraGrabber(i, src).start() for i, src in enumerate(sources)]

    model_path = config.MODEL_PATH
    if getattr(config, 'USE_TENSORRT', False) and os.path.exists(model_path.replace('.pt', '.engine')):
        model_path = model_path.replace('.pt', '.engine')
        
    models = [YOLO(model_path) for _ in range(NUM_CAMS)]

    for i, l in enumerate(labels):
        s = DetectionStore(i, l)
        s.zone = saved_zones[i]
        global_stores.append(s)

    AIDispatcher(models, global_grabbers, global_stores, global_ctrl).start()
    threading.Thread(target=logic_loop, args=(global_stores, global_ctrl), daemon=True).start()
    threading.Thread(target=renderer_loop, daemon=True).start()
    log_event("System Initialized and Tracking.")

if __name__ == "__main__":
    run_ai_system()
    print("-" * 64)
    print(" FEROKE HEADLESS ENTERPRISE SERVER STARTED (PORT 8000)")
    print("-" * 64)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
