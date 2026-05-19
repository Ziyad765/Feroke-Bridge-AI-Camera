import os
import random
import cv2

img_dir = r"C:\Users\Ziyad\Downloads\cam-export\HPC_Training_Workspace\dataset\images\train"
lbl_dir = r"C:\Users\Ziyad\Downloads\cam-export\HPC_Training_Workspace\dataset\labels\train"
out_dir = r"C:\Users\Ziyad\.gemini\antigravity\brain\61609460-4566-47b2-826b-a3ccacf84e9d"

# Validate paths
if not os.path.exists(img_dir):
    print("Dataset directory not found yet.")
    exit(1)

imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not imgs:
    print("No images found in train dir.")
    exit(1)

# Pick 6 random images
random.seed()
random.shuffle(imgs)

count = 0
for img_name in imgs[:6]:
    img_path = os.path.join(img_dir, img_name)
    base = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(lbl_dir, base + '.txt')
    
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    # Math conversion from normalized to raw pixels
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    label = "Light" if cls == 0 else "Heavy"
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        # Add a note if no detections exist
        cv2.putText(img, "NO VEHICLES DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Save directly to artifact pipeline
    out_file = os.path.join(out_dir, f"hpc_visual_{count}.jpg")
    cv2.imwrite(out_file, img)
    count += 1

print("Visualizations generated.")
