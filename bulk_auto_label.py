import cv2
import os
import shutil
import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"runs\detect\runs\detect\vehicle_counter_final\weights\best.pt"
INPUT_DIR = "clean_frames1"
OUTPUT_DIR = "yolo_dataset_v2"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

def init_folders():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

def auto_label_bulk():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    init_folders()
    
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
    print(f"Found {len(all_files)} images. Starting auto-labeling...")
    
    for filename in tqdm.tqdm(all_files):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        # Stricter IoU=0.45 and Agnostic NMS to merge overlapping double-detections
        results = model(img, verbose=False, iou=0.45, conf=0.25, agnostic_nms=True)[0]
        
        label_path = os.path.join(LABELS_DIR, filename.replace(".jpg", ".txt"))
        
        with open(label_path, "w") as f:
            for box in results.boxes:
                # YOLO Format: class x_center y_center width height (normalized)
                cls = int(box.cls[0])
                xywhn = box.xywhn[0].cpu().numpy()
                f.write(f"{cls} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")
                
        # Copy image to new dataset folder
        shutil.copy(img_path, os.path.join(IMAGES_DIR, filename))

    print(f"\nSUCCESS: {len(all_files)} images auto-labeled in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    auto_label_bulk()
