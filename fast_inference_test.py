import cv2
import time
import threading
from ultralytics import YOLO
from queue import Queue
import numpy as np
import os
import sys

# --- CONFIGURATION ---
# Using your NEW 'Best Ever' model (83% mAP50)
MODEL_PATH = r"runs\detect\runs\detect\vehicle_counter_final\weights\best.pt"
# Fallback to previous best if final is missing for some reason
# if not os.path.exists(MODEL_PATH):
#     MODEL_PATH = r"runs\detect\vehicle_counter_best4\weights\best.pt"

STREAM_URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.144.234.81:8080/video"
IMG_SIZE = 320 # Keep at 320 or drop to 160 for MAX speed
WINDOW_NAME = "REAL-TIME 60FPS DETECTION"

class AsyncInference:
    def __init__(self, model_path, source):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.annotated_frame = None
        self.stopped = False
        self.count = 0
        self.current_count = 0
        self.tracked_ids = set()
        
        # Performance metrics
        self.inf_time = 0
        self.lock = threading.Lock()

    def start(self):
        # Thread 1: Grab frames as fast as possible
        threading.Thread(target=self.grab_frames, daemon=True).start()
        # Thread 2: Run AI in background
        threading.Thread(target=self.run_ai, daemon=True).start()
        return self

    def grab_frames(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.frame = frame

    def run_ai(self):
        while not self.stopped:
            if self.frame is not None:
                tmp_frame = None
                with self.lock:
                    tmp_frame = self.frame.copy()
                
                start = time.time()
                # Run tracking in background with strict IoU and Agnostic NMS (merges overlapping boxes)
                results = self.model.track(tmp_frame, persist=True, verbose=False, imgsz=IMG_SIZE, iou=0.45, conf=0.25, agnostic_nms=True)[0]
                self.inf_time = (time.time() - start) * 1000 # ms
                
                # Count logic
                self.current_count = 0
                if results.boxes.id is not None:
                    ids = results.boxes.id.cpu().numpy().astype(int)
                    self.current_count = len(ids)
                    for obj_id in ids:
                        if obj_id not in self.tracked_ids:
                            self.tracked_ids.add(obj_id)
                            self.count += 1
                
                # Update the annotated overlay
                with self.lock:
                    self.annotated_frame = results.plot()
            else:
                time.sleep(0.01)

    def show(self):
        # Setup Resizable Window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720) # Set default readable size
        
        prev_time = time.time()
        
        while not self.stopped:
            display_frame = None
            
            with self.lock:
                # Use annotated frame if available, otherwise raw frame
                if self.annotated_frame is not None:
                    display_frame = self.annotated_frame.copy()
                else:
                    display_frame = self.frame
            
            if display_frame is not None:
                # Calculate Display FPS (should be 60 if stream is 60)
                curr_time = time.time()
                time_diff = curr_time - prev_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                prev_time = curr_time
                
                # Draw high-speed metrics (Current Occupancy)
                cv2.putText(display_frame, f"Stream FPS: {self.fps:.1f}", (20, 60), 1, 2, (0, 255, 0), 2)
                cv2.putText(display_frame, f"AI Speed: {self.inf_time:.1f}ms", (20, 110), 1, 2, (0, 255, 255), 2)
                
                # LARGE display for Real-time vehicle count
                cv2.putText(display_frame, f"VEHICLES: {self.current_count}", (20, 200), 1, 5, (0, 0, 255), 5)
                
                cv2.imshow(WINDOW_NAME, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Starting Async Inference on: {STREAM_URL}")
    engine = AsyncInference(MODEL_PATH, STREAM_URL).start()
    engine.show()
