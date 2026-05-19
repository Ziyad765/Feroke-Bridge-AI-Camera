import sys
import copy
import cv2
import time
import json
import os
import csv
import threading
import torch
import numpy as np
import psutil
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QGridLayout, QGroupBox,
    QFormLayout, QSlider, QListWidget, QComboBox, QFrame, QScrollArea, QSizePolicy, QCheckBox, QMessageBox
)
import qdarkstyle
import pyqtgraph as pg

from ultralytics import YOLO
from signal_controller import SignalController
import config

STATE_FILE = os.path.join(os.path.dirname(__file__), "feroke_state.json")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "feroke_settings.json")

# ────────────────────────────────────────────────────────────────────────
# CAMERA GRABBER 
# ────────────────────────────────────────────────────────────────────────
class CameraGrabber:
    def __init__(self, index, source):
        self.index = index
        self.source = self._prepare_source(source)
        self.raw_frame = None
        self.stopped = False
        self._lock = threading.Lock()

    def _prepare_source(self, src):
        """Wraps RTSP in a hardware-accelerated GStreamer pipeline if on Jetson."""
        # Check if source is RTSP
        if isinstance(src, str) and src.startswith("rtsp://"):
            # Only use hardware acceleration if configured
            if getattr(config, 'USE_TENSORRT', False):
                encoding = getattr(config, 'CAMERA_ENCODING', 'h264').lower()
                depay = "rtph264depay" if encoding == "h264" else "rtph265depay"
                parser = "h264parse" if encoding == "h264" else "h265parse"
                
                print(f"🎬 [Cam {self.index}] Enabling {encoding.upper()} Hardware Pipeline...")
                return (
                    f"rtspsrc location={src} latency=300 ! "
                    f"{depay} ! {parser} ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=(string)BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
                )
        return src

    def start(self):
        threading.Thread(target=self._grab, daemon=True).start()
        return self

    def _grab(self):
        # Use GStreamer backend if using a pipeline string
        is_gst = "!" in str(self.source)
        cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER if is_gst else cv2.CAP_ANY)
        
        if not is_gst:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ [Cam {self.index}] Connection lost. Reconnecting...")
                cap.release(); time.sleep(2)
                cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER if is_gst else cv2.CAP_ANY)
                if not is_gst: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            # Standardize frame size for GUI and AI
            frame = cv2.resize(frame, (640, 480))
            with self._lock:
                self.raw_frame = frame
        cap.release()

    def get_frame(self):
        with self._lock:
            return self.raw_frame.copy() if self.raw_frame is not None else None


# ────────────────────────────────────────────────────────────────────────
# DETECTION STORE (With Light/Heavy Splits)
# ────────────────────────────────────────────────────────────────────────
class DetectionStore:
    def __init__(self, index, label):
        self.index       = index
        self.label       = label
        self.is_clearance = index in [2, 3]
        self.zone        = None 
        self.boxes       = [] 
        self.t_count     = 0
        self.l_count     = 0  # Light Vehicles
        self.h_count     = 0  # Heavy Vehicles
        self.inf_ms      = 0.0
        self._lock       = threading.Lock()

    def update(self, boxes, t_count, l_count, h_count, inf_ms):
        with self._lock:
            self.boxes    = boxes
            self.t_count  = t_count
            self.l_count  = l_count
            self.h_count  = h_count
            self.inf_ms   = inf_ms

    def read(self):
        with self._lock:
            return (self.boxes[:], self.t_count, self.l_count, self.h_count, self.inf_ms)


# ────────────────────────────────────────────────────────────────────────
# AI ENGINE QTHREAD (Master Predictor + CLAHE Filters)
# ────────────────────────────────────────────────────────────────────────
class AIEngineThread(QThread):
    frame_ready = Signal(int, object) 
    telemetry_ready = Signal(dict) # Passes dynamic JSON payload to UI

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stopped = False
        self.dev = 0 if torch.cuda.is_available() else 'cpu'
        
        # UI Toggles config
        self.night_vision_enabled = False
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

        self.ctrl = SignalController(
            min_green=config.MIN_GREEN_TIME, 
            max_green=config.MAX_GREEN_TIME, 
            max_clearance=config.MAX_CLEARANCE_TIME
        )
        
        # Load local configs
        self.saved_zones = [config.ZONE_A, config.ZONE_B, config.ZONE_C, config.ZONE_D]
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    d = json.load(f)
                    if "zones" in d: self.saved_zones = d["zones"]
                    if "min_green" in d: self.ctrl.min_green = d["min_green"]
                    if "max_green" in d: self.ctrl.max_green = d["max_green"]
                    if "max_clearance" in d: self.ctrl.max_clearance = d["max_clearance"]
                    if "min_clearance" in d: self.ctrl.min_clearance = d["min_clearance"]
                    if "night_vision" in d: self.night_vision_enabled = d["night_vision"]
                    if "auto_alarm" in d: self.ctrl.auto_alarm_enabled = d["auto_alarm"]
            except: pass

        sources = [config.CAM1_URL, config.CAM2_URL, config.CAM3_URL, config.CAM4_URL]
        labels = ["Queue A", "Queue B", "Interior A", "Interior B"]
        
        cv2.setNumThreads(1)
        self.grabbers = [CameraGrabber(i, src).start() for i, src in enumerate(sources)]
        
        model_path = config.MODEL_PATH
        if getattr(config, 'USE_TENSORRT', False) and os.path.exists(model_path.replace('.pt', '.engine')):
            model_path = model_path.replace('.pt', '.engine')
            
        self.master_model = YOLO(model_path)
        
        self.stores = []
        for i, l in enumerate(labels):
            s = DetectionStore(i, l)
            s.zone = self.saved_zones[i]
            self.stores.append(s)

        threading.Thread(target=self._logic_loop, daemon=True).start()

    def update_settings(self, mn, mx, clr, min_clr, nv, alarm):
        self.ctrl.update_config(min_green=mn, max_green=mx, max_clearance=clr, min_clearance=min_clr)
        self.night_vision_enabled = nv
        self.ctrl.auto_alarm_enabled = alarm

    def _logic_loop(self):
        while not self.stopped:
            # Aggregate store data seamlessly
            tc = [s.t_count for s in self.stores]
            self.ctrl.update_counts(tc[0], tc[1], tc[2], tc[3])
            self.ctrl.run_logic()
            
            elapsed = (datetime.now() - self.ctrl.last_switch).total_seconds()
            
            payload = {
                "state": self.ctrl.state.value,
                "elapsed": elapsed,
                "qA_t": tc[0], "qA_l": self.stores[0].l_count, "qA_h": self.stores[0].h_count,
                "qB_t": tc[1], "qB_l": self.stores[1].l_count, "qB_h": self.stores[1].h_count,
                "history": self.ctrl.history[-50:] # Limit log packet to prevent UI lockup
            }
            self.telemetry_ready.emit(payload)
            time.sleep(0.5)

    def run(self):
        while not self.stopped:
            for i in range(4):
                store = self.stores[i]
                grabber = self.grabbers[i]

                if store.is_clearance and self.ctrl.state.value != "ALL_RED":
                    store.update([], 0, 0, 0, 0)
                    self._render_and_emit(i, grabber.get_frame(), store)
                    continue

                raw = grabber.get_frame()
                if raw is None: continue

                # APPLY NIGHT VISION CLAHE ALGORITHM IF ENABLED
                if self.night_vision_enabled:
                    lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l2 = self.clahe.apply(l)
                    lab = cv2.merge((l2,a,b))
                    ai_input = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    ai_input = raw

                t0 = time.time()
                results = self.master_model.predict(
                    ai_input, verbose=False, imgsz=config.AI_IMG_SIZE,
                    conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD,
                    agnostic_nms=config.AGNOSTIC_NMS, device=self.dev, half=(self.dev == 0)
                )
                inf_ms = (time.time() - t0) * 1000

                res = results[0]
                boxes = []
                t_count = 0
                l_count = 0
                h_count = 0

                if res.boxes is not None:
                    z_np = np.array(store.zone, dtype=np.int32) if store.zone is not None else None
                    for box in res.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        inz = (z_np is not None and cv2.pointPolygonTest(z_np, (cx, cy), False) >= 0)
                        
                        cls = int(box.cls[0].cpu().numpy())
                        nm = "Heavy" if cls == 1 else "Light"
                        conf = float(box.conf[0].cpu().numpy())
                        
                        if inz:
                            t_count += 1
                            if cls == 1: h_count += 1
                            else: l_count += 1

                        boxes.append((x1, y1, x2, y2, inz, nm, conf))
                
                store.update(boxes, t_count, l_count, h_count, inf_ms)
                # Pass original physical frame to renderer (not heavily flattened CLAHE frame which looks weird to humans sometimes)
                self._render_and_emit(i, ai_input if self.night_vision_enabled else raw, store)
            
            time.sleep(0.01)

    def _render_and_emit(self, i, raw, store):
        if raw is None: return
        frame = raw.copy()
        bxs, tc, lc, hc, ms = store.read()
        z = store.zone

        if z is not None:
            z_np = np.array(z, dtype=np.int32)
            cv2.polylines(frame, [z_np], True, (0, 255, 255), 2)

        for (x1, y1, x2, y2, inz, nm, cf) in bxs:
            color = (0, 0, 255) if inz else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{nm} {cf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.putText(frame, f"AI Base: {ms:.0f}ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Density: {tc}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        px = QPixmap.fromImage(qt_img)
        self.frame_ready.emit(i, px)


# ────────────────────────────────────────────────────────────────────────
# ZONE CANVAS (Click and Draw)
# ────────────────────────────────────────────────────────────────────────
class InteractiveZoneCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.latest_pixmap = None
        self.recording = False

    def mousePressEvent(self, event):
        if not self.recording: return
        pos = event.position()
        w_scale = 640 / self.width()
        h_scale = 480 / self.height()
        rx, ry = int(pos.x() * w_scale), int(pos.y() * h_scale)
        self.points.append([rx, ry])
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.recording and not self.points: return
        
        painter = QPainter(self)
        pen = QPen(QColor(16, 185, 129), 2)
        painter.setPen(pen)
        
        inv_w = self.width() / 640
        inv_h = self.height() / 480
        adj_pts = [QPoint(int(p[0]*inv_w), int(p[1]*inv_h)) for p in self.points]
        
        for i in range(len(adj_pts)):
            if i > 0: painter.drawLine(adj_pts[i-1], adj_pts[i])
            painter.setBrush(QColor(239, 68, 68))
            painter.drawEllipse(adj_pts[i], 5, 5)
        
        if len(adj_pts) > 2:
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(adj_pts[-1], adj_pts[0])


# ────────────────────────────────────────────────────────────────────────
# MAIN GUI WINDOW
# ────────────────────────────────────────────────────────────────────────
class FerokeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feroke Vanguard - Government Analytics Command")
        self.resize(1300, 850)
        self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))

        self.ai = AIEngineThread()
        self.ai.frame_ready.connect(self.update_video)
        self.ai.telemetry_ready.connect(self.update_telemetry)
        
        # Telemetry History variables for the Live Graphing
        self.hist_time_q = []
        self.hist_qa_q = []
        self.hist_qb_q = []
        self.plot_idx = 0

        self._build_ui()
        
        # Hardware Vitals loop setup independent of AI Loop
        self.hardware_timer = QTimer()
        self.hardware_timer.timeout.connect(self._update_hardware_vitals)
        self.hardware_timer.start(1000)

        self.ai.start()

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Ribbon
        header = QHBoxLayout()
        title = QLabel("FEROKE VANGUARD: ENTERPRISE ANALYTICS")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #3b82f6;")
        self.status_lbl = QLabel("⚫ BOOTING")
        self.status_lbl.setStyleSheet("color: yellow; font-weight: bold;")
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_lbl)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- TAB 1: Dashboard Matrix ---
        dash_widget = QWidget()
        dash_layout = QHBoxLayout(dash_widget)
        
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setSpacing(5)
        # Camera labels (top-left of each feed)
        cam_labels = [
            "SIDE A — Queue",
            "SIDE B — Queue",
            "BRIDGE INTERIOR — Clearance A",
            "BRIDGE INTERIOR — Clearance B",
        ]
        label_colors = ["#3b82f6", "#f59e0b", "#10B981", "#10B981"]

        self.cams = []
        self.cam_overlays = []
        for i in range(4):
            # Wrap each feed in a container so we can overlay the label
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(0)

            header = QLabel(cam_labels[i])
            header.setAlignment(Qt.AlignCenter)
            header.setFixedHeight(22)
            header.setStyleSheet(
                f"background-color: {label_colors[i]}; color: white; "
                "font-weight: bold; font-size: 11px; letter-spacing: 1px;"
            )

            lbl = QLabel(f"Booting {cam_labels[i]}...")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setMinimumSize(320, 218)  # Slightly less height to account for header
            lbl.setStyleSheet("background-color: black; border: 1px solid #1E293B;")

            cell_layout.addWidget(header)
            cell_layout.addWidget(lbl)
            grid.addWidget(cell, i//2, i%2)
            self.cams.append(lbl)
            self.cam_overlays.append(header)
        
        tele_panel = QFrame()
        tele_panel.setFixedWidth(300)
        tele_layout = QVBoxLayout(tele_panel)
        
        self.t_state = QLabel("ALL_RED")
        self.t_state.setFont(QFont("Consolas", 24, QFont.Bold))
        self.t_state.setStyleSheet("color: #EF4444;")
        self.t_time = QLabel("0.0s")
        self.t_time.setFont(QFont("Consolas", 18))
        self.t_time.setStyleSheet("color: #10B981;")

        # Heavy vs Light Expansion
        self.t_qa = QLabel("Queue A Total: 0")
        self.t_qa.setFont(QFont("Arial", 12, QFont.Bold))
        self.t_qa_sub = QLabel("  ↳ 0 Light | 0 Heavy")
        
        self.t_qb = QLabel("Queue B Total: 0")
        self.t_qb.setFont(QFont("Arial", 12, QFont.Bold))
        self.t_qb_sub = QLabel("  ↳ 0 Light | 0 Heavy")

        # Telemetry CPU/RAM Box
        self.t_hardware = QLabel("CPU: 0% | RAM: 0%")
        self.t_hardware.setStyleSheet("color: #64748B; font-weight: bold;")

        tele_layout.addWidget(QLabel("BRIDGE STATE:"))
        tele_layout.addWidget(self.t_state)
        tele_layout.addSpacing(15)
        tele_layout.addWidget(QLabel("TIME ELAPSED:"))
        tele_layout.addWidget(self.t_time)
        tele_layout.addSpacing(15)
        tele_layout.addWidget(self.t_qa)
        tele_layout.addWidget(self.t_qa_sub)
        tele_layout.addSpacing(5)
        tele_layout.addWidget(self.t_qb)
        tele_layout.addWidget(self.t_qb_sub)
        tele_layout.addStretch()
        tele_layout.addWidget(self.t_hardware)
        
        dash_layout.addWidget(grid_container, stretch=4)
        dash_layout.addWidget(tele_panel, stretch=1)
        self.tabs.addTab(dash_widget, "Live Operations Matrix")

        # --- TAB 2: Government Analytics (Graphs & Logs) ---
        anal_widget = QWidget()
        anal_layout = QVBoxLayout(anal_widget)
        
        # PyQtGraph Plot
        self.plot_widget = pg.PlotWidget(title="Live Traffic Accumulation Density")
        self.plot_widget.setLabel('left', 'Total Vehicles In Queue')
        self.plot_widget.setLabel('bottom', 'Time Blocks')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.line_qa = self.plot_widget.plot(pen=pg.mkPen('g', width=2), name="Queue A")
        self.line_qb = self.plot_widget.plot(pen=pg.mkPen('y', width=2), name="Queue B")

        # Logs
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("background-color: #0F172A; color: #10B981; font-family: Consolas;")

        anal_layout.addWidget(self.plot_widget, stretch=2)
        anal_layout.addWidget(QLabel("Signal Shift Historical Record (Last 60 Phasing Events):"))
        anal_layout.addWidget(self.log_list, stretch=1)
        self.tabs.addTab(anal_widget, "Analytics & CSV Subsystems")

        # --- TAB 3: Global System Settings ---
        set_widget = QWidget()
        set_layout = QFormLayout(set_widget)
        set_layout.setContentsMargins(40, 40, 40, 40)
        
        self.sl_min = QSlider(Qt.Horizontal)
        self.sl_min.setRange(5, 30); self.sl_min.setValue(self.ai.ctrl.min_green)
        self.sl_max = QSlider(Qt.Horizontal)
        self.sl_max.setRange(15, 90); self.sl_max.setValue(self.ai.ctrl.max_green)
        self.sl_clr = QSlider(Qt.Horizontal)
        self.sl_clr.setRange(5, 45); self.sl_clr.setValue(self.ai.ctrl.max_clearance)
        self.sl_min_clr = QSlider(Qt.Horizontal)
        self.sl_min_clr.setRange(2, 30); self.sl_min_clr.setValue(self.ai.ctrl.min_clearance)
        
        self.chk_nv = QCheckBox("Enable Night-Vision AI Contrast Engine (CLAHE)")
        self.chk_nv.setChecked(self.ai.night_vision_enabled)
        
        self.chk_alarm = QCheckBox("Enable Autonomous Emergency Auto-Braking (Clearance Block Alarm)")
        self.chk_alarm.setChecked(self.ai.ctrl.auto_alarm_enabled)
        self.chk_alarm.setStyleSheet("color: #EF4444; font-weight: bold;")

        btn_save_time = QPushButton("Flash Settings to Memory")
        btn_save_time.setMinimumHeight(40)
        btn_save_time.setStyleSheet("background-color: #3b82f6; font-weight: bold;")
        btn_save_time.clicked.connect(self._flash_settings)

        btn_csv = QPushButton("EXPORT DAILY LOGS (CSV)")
        btn_csv.setMinimumHeight(40)
        btn_csv.setStyleSheet("background-color: #10B981; color: black; font-weight: bold;")
        btn_csv.clicked.connect(self._export_csv)

        set_layout.addRow(QLabel("Minimum Green Bounds (s):"), self.sl_min)
        set_layout.addRow(QLabel("Maximum Green Bounds (s):"), self.sl_max)
        set_layout.addRow(QLabel("Max Bridge Clearance / All Red (s):"), self.sl_clr)
        set_layout.addRow(QLabel("Min Clearance Wait — Even If Bridge Empty (s):"), self.sl_min_clr)
        set_layout.addRow(QLabel(""), QLabel("")) # Spacer
        set_layout.addRow(QLabel("Vision Modules:"), self.chk_nv)
        set_layout.addRow(QLabel("Security Overrides:"), self.chk_alarm)
        set_layout.addRow(QLabel(""), btn_save_time)
        set_layout.addRow(QLabel(""), QLabel("")) # Spacer
        set_layout.addRow(QLabel("Auditing:"), btn_csv)
        self.tabs.addTab(set_widget, "System Globals")

        # --- TAB 4: Zone Override ---
        zone_widget = QWidget()
        zl = QVBoxLayout(zone_widget)
        z_toolbar = QHBoxLayout()
        self.zone_combo = QComboBox()
        self.zone_combo.addItems(["Side A (View 0)", "Side B (View 1)", "Interior A (View 2)", "Interior B (View 3)"])
        self.zone_combo.currentIndexChanged.connect(self._load_zone_points)
        btn_record = QPushButton("Start Drawing Mode")
        btn_record.clicked.connect(self._toggle_drawing)
        btn_clear = QPushButton("Clear Current Points")
        btn_clear.clicked.connect(self._clear_drawing)
        btn_save_zone = QPushButton("COMMIT ZONE")
        btn_save_zone.clicked.connect(self._flash_zones)

        z_toolbar.addWidget(QLabel("Select Hardware Lens:"))
        z_toolbar.addWidget(self.zone_combo)
        z_toolbar.addStretch()
        z_toolbar.addWidget(btn_record)
        z_toolbar.addWidget(btn_clear)
        z_toolbar.addWidget(btn_save_zone)
        
        self.zone_canvas = InteractiveZoneCanvas()
        self.zone_canvas.setAlignment(Qt.AlignCenter)
        self.zone_canvas.setMinimumSize(640, 480)
        self.zone_canvas.setStyleSheet("background-color: black; border: 2px dashed #374151;")

        zl.addLayout(z_toolbar)
        zl.addWidget(self.zone_canvas, 1)
        self.tabs.addTab(zone_widget, "Zone Plotting")
        
        self.tabs.currentChanged.connect(self._tab_switch)

    def _update_hardware_vitals(self):
        c = psutil.cpu_percent()
        m = psutil.virtual_memory().percent
        self.t_hardware.setText(f"System Load | CPU: {c}% | RAM: {m}%")

    def update_video(self, idx, pixmap):
        if self.tabs.currentIndex() == 0:
            self.cams[idx].setPixmap(pixmap.scaled(self.cams[idx].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif self.tabs.currentIndex() == 3:
            if idx == self.zone_combo.currentIndex():
                self.zone_canvas.setPixmap(pixmap.scaled(self.zone_canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.zone_canvas.latest_pixmap = pixmap

    def update_telemetry(self, p):
        # UI Status updating
        if p["state"] == "EMERGENCY":
            self.status_lbl.setText("🚨 EMERGENCY BLOCK DETECTED 🚨")
            self.status_lbl.setStyleSheet("color: white; background-color: red; font-weight: bold;")
        else:
            self.status_lbl.setText("🟢 ACTIVE & TRACKING")
            self.status_lbl.setStyleSheet("color: #10B981; font-weight: bold; background-color: transparent;")
        
        self.t_state.setText(p["state"])
        if "GREEN" in p["state"]: self.t_state.setStyleSheet("color: #10B981;")
        else: self.t_state.setStyleSheet("color: #EF4444;")
        
        self.t_time.setText(f"{p['elapsed']:.1f}s")
        
        self.t_qa.setText(f"Queue A Total: {p['qA_t']}")
        self.t_qa_sub.setText(f"  ↳ {p['qA_l']} Light | {p['qA_h']} Heavy")
        
        self.t_qb.setText(f"Queue B Total: {p['qB_t']}")
        self.t_qb_sub.setText(f"  ↳ {p['qB_l']} Light | {p['qB_h']} Heavy")

        # PyqtGraph Dynamic Array Pushing
        self.hist_time_q.append(self.plot_idx)
        self.hist_qa_q.append(p['qA_t'])
        self.hist_qb_q.append(p['qB_t'])
        self.plot_idx += 1
        
        if len(self.hist_time_q) > 60: # Limit history visible string length purely for graphing
            self.hist_time_q.pop(0)
            self.hist_qa_q.pop(0)
            self.hist_qb_q.pop(0)

        # Ensure we only physically redraw if physically looking at it, massive optimization secret
        if self.tabs.currentIndex() == 1:
            self.line_qa.setData(self.hist_time_q, self.hist_qa_q)
            self.line_qb.setData(self.hist_time_q, self.hist_qb_q)
            
            # Repopulate Log safely
            if len(p['history']) != self.log_list.count():
                self.log_list.clear() # Dump and restack
                for item in reversed(p['history']):
                    self.log_list.addItem(f"[{item['time']}] {item['event']}")

    def _flash_settings(self):
        self.ai.update_settings(
            self.sl_min.value(), self.sl_max.value(), self.sl_clr.value(),
            self.sl_min_clr.value(),
            self.chk_nv.isChecked(), self.chk_alarm.isChecked()
        )
        self._write_settings()
        self.status_lbl.setText("⚙️ PARAMETERS FLASHED ⚙️")

    def _export_csv(self):
        filename = f"Feroke_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(filename, 'w', newline='') as csvfile:
                w = csv.writer(csvfile)
                w.writerow(["Timestamp", "Internal Event Log"])
                for item in self.ai.ctrl.history:
                    w.writerow([item['time'], item['event']])
            QMessageBox.information(self, "Export Success", f"Secure Log {filename} successfully written to local disk.")
        except Exception as e: pass

    def _tab_switch(self, idx):
        if idx == 3:
            self._load_zone_points()
            self.zone_canvas.recording = False

    def _toggle_drawing(self):
        self.zone_canvas.recording = not self.zone_canvas.recording
        if self.zone_canvas.recording: self.status_lbl.setText("✏️ DRAWING MODE ACTIVE (Click Video)")

    def _clear_drawing(self):
        self.zone_canvas.points = []
        self.zone_canvas.update()

    def _load_zone_points(self):
        cid = self.zone_combo.currentIndex()
        if self.ai.stores[cid].zone is not None:
            self.zone_canvas.points = copy.deepcopy(self.ai.stores[cid].zone)
        else:
            self.zone_canvas.points = []
        self.zone_canvas.update()

    def _flash_zones(self):
        cid = self.zone_combo.currentIndex()
        if len(self.zone_canvas.points) > 2:
            self.ai.stores[cid].zone = copy.deepcopy(self.zone_canvas.points)
            self._write_settings()
            self.status_lbl.setText("🌐 ZONE MAP LOCKED TO FIRMWARE 🌐")
            self.zone_canvas.recording = False

    def _write_settings(self):
        d = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f: d = json.load(f)
            except: pass
        
        d["min_green"] = self.sl_min.value()
        d["max_green"] = self.sl_max.value()
        d["max_clearance"] = self.sl_clr.value()
        d["min_clearance"] = self.sl_min_clr.value()
        d["night_vision"] = self.chk_nv.isChecked()
        d["auto_alarm"] = self.chk_alarm.isChecked()
        d["zones"] = [s.zone for s in self.ai.stores]
        with open(SETTINGS_FILE, "w") as f: json.dump(d, f)

    def closeEvent(self, event):
        self.ai.stopped = True
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = FerokeApp()
    window.show()
    sys.exit(app.exec())
