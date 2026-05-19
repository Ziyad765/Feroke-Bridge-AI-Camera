import cv2
import os
import random
import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"c:\Users\Ziyad\Downloads\cam-export\feroke project\Feroke-Traffic-Lite\best.pt"
OUTPUT_DIR = r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset_v2"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# We want to save 10 previews into the conversation's artifact directory so the user can see them
ARTIFACT_DIR = r"C:\Users\Ziyad\.gemini\antigravity\brain\61609460-4566-47b2-826b-a3ccacf84e9d"

def auto_label_v2():
    print(f"Loading new model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    os.makedirs(LABELS_DIR, exist_ok=True)
    
    all_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    print(f"Found {len(all_files)} images in '{IMAGES_DIR}'. Starting auto-labeling...")
    
    # Pick 10 random files to save previews for
    random.seed(42)
    preview_files = set(random.sample(all_files, min(10, len(all_files))))
    preview_count = 0
    
    for filename in tqdm.tqdm(all_files):
        img_path = os.path.join(IMAGES_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Stricter IoU=0.45 and Agnostic NMS to merge overlapping double-detections
        results = model(img, verbose=False, iou=0.45, conf=0.25, agnostic_nms=True)[0]
        
        label_path = os.path.join(LABELS_DIR, filename.replace(".jpg", ".txt"))
        
        with open(label_path, "w") as f:
            for box in results.boxes:
                # YOLO Format: class x_center y_center width height (normalized)
                cls = int(box.cls[0])
                xywhn = box.xywhn[0].cpu().numpy()
                f.write(f"{cls} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")
                
        # Save preview if it's in our random list
        if filename in preview_files:
            preview_count += 1
            annotated_img = results.plot()
            preview_path = os.path.join(ARTIFACT_DIR, f"preview_{preview_count}.jpg")
            cv2.imwrite(preview_path, annotated_img)

    print(f"\nSUCCESS: {len(all_files)} images auto-labeled. Saved {preview_count} previews to artifacts.")

if __name__ == "__main__":
    auto_label_v2()
