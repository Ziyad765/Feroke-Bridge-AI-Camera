import cv2
from ultralytics import YOLO
import os
import time

# 1. Configuration
MODEL_PATH = r"c:\Users\Ziyad\Downloads\cam-export\HPC trained model\runs\detect\nano_edge_train\weights\best.pt"
VIDEO_SOURCE = r"c:\Users\Ziyad\Downloads\cam-export\test_video.mp4"
OUTPUT_DIR = "hpc_test_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def test_nano():
    print(f"🚀 Loading HPC Nano (Edge) Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        return

    # Load the model
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # 2. Open Video Source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {VIDEO_SOURCE}")
        return

    print(f"📽️ Processing video: {VIDEO_SOURCE}")
    
    frame_count = 0
    max_frames = 50  # We only process 50 frames for a quick benchmark
    
    start_bench = time.time()

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        # Run inference at 416p (Native training resolution for Nano)
        results = model.predict(frame, imgsz=416, conf=0.25, verbose=False)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Save every 10th frame to visually check quality
        if frame_count % 10 == 0:
            save_path = os.path.join(OUTPUT_DIR, f"nano_frame_{frame_count}.jpg")
            cv2.imwrite(save_path, annotated_frame)
            print(f"💾 Saved sample Result: {save_path}")

        frame_count += 1
        
    end_bench = time.time()
    
    total_time = end_bench - start_bench
    avg_fps = frame_count / total_time
    
    print("\n" + "="*40)
    print("📊 BENCHMARK RESULTS: NANO MODEL (416p)")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Speed: {avg_fps:.2f} FPS")
    print("="*40)
    
    cap.release()
    print(f"\n✅ Testing Complete. Samples are in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    test_nano()
