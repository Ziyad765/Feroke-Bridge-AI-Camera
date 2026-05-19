import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

# Configuration
# 1. Path to your exported TensorRT model (.engine) for best performance
# or just the PyTorch model (.pt) while testing.
MODEL_PATH = r"runs\detect\runs\detect\vehicle_counter_final\weights\best.pt" 

# 2. List of 4 CCTV / RTSP URLs or local video files
CCTV_SOURCES = [
    "rtsp://user:pass@camera1_ip:554/stream",
    "rtsp://user:pass@camera2_ip:554/stream",
    "rtsp://user:pass@camera3_ip:554/stream",
    "rtsp://user:pass@camera4_ip:554/stream"
]

# 3. Line coordinates for counting (adjust for each camera if needed)
# Format: [(x1, y1), (x2, y2)]
COUNT_LINES = [
    [(1000, 1500), (3000, 1500)],  # Camera 1
    [(1000, 1500), (3000, 1500)],  # Camera 2
    [(1000, 1500), (3000, 1500)],  # Camera 3
    [(1000, 1500), (3000, 1500)],  # Camera 4
]

def run_multi_stream_counting():
    # Load model (use .engine on Jetson for maximum speed)
    model = YOLO(MODEL_PATH)
    
    # Initialize 4 counters
    counters = []
    for i in range(4):
        counter = object_counter.ObjectCounter()
        counter.set_args(
            view_img=True,          # Set to False in production
            reg_pts=COUNT_LINES[i], # Line for counting
            classes_names=model.names,
            draw_tracks=True,
            line_thickness=3
        )
        counters.append(counter)

    # Initialize 4 VideoCaptures
    caps = [cv2.VideoCapture(src) for src in CCTV_SOURCES]
    
    print("Multi-stream counting started. Press 'q' to stop.")

    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # Optional: Handle stream disconnection/reconnect logic
                continue

            # Run YOLO inference with strict NMS to prevent double detections
            # imgsz=640 is standard; decrease to 320 for MAX speed on Nano
            results = model.track(frame, persist=True, show=False, imgsz=640, iou=0.45, conf=0.25, agnostic_nms=True)

            # Update counting logic
            frame = counters[i].start_counting(frame, results)

            # Display individual camera feed (Optional)
            cv2.imshow(f"Camera {i+1}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_multi_stream_counting()
