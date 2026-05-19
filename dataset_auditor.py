import os
import cv2

# Hardcoded paths matching our dataset build
dataset_base = r"C:\Users\Ziyad\Downloads\cam-export\HPC_Training_Workspace\dataset"
img_dirs = [os.path.join(dataset_base, "images", "train"), os.path.join(dataset_base, "images", "val")]
lbl_dirs = [os.path.join(dataset_base, "labels", "train"), os.path.join(dataset_base, "labels", "val")]

# Aggregate all files
image_paths = []
for d in img_dirs:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(d, f))

if not image_paths:
    print("NO IMAGES FOUND!")
    exit(1)

print(f"Total Dataset Loaded: {len(image_paths)} photos.")
print("Controls: -> [D] | <- [A] | +50 [E] | +100 [R] | QUIT [Q]")

current_idx = 0
cv2.namedWindow("HPC Dataset Auditor", cv2.WINDOW_NORMAL)

while True:
    img_path = image_paths[current_idx]
    
    # Intelligently find the matching label path regardless of train/val state
    lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')

    img = cv2.imread(img_path)
    if img is not None:
        h, w, _ = img.shape
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                        label = "Light" if cls == 0 else "Heavy"
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                        cv2.putText(img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        cv2.putText(img, f"Image {current_idx + 1} of {len(image_paths)} | E: +50 | R: +100", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imshow("HPC Dataset Auditor", img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d') or key == 83:  # 'd' or Right Arrow key
        current_idx = (current_idx + 1) % len(image_paths)
    elif key == ord('a') or key == 81:  # 'a' or Left Arrow key
        current_idx = (current_idx - 1) % len(image_paths)
    elif key == ord('e'):               # 'e' to fast forward 50
        current_idx = (current_idx + 50) % len(image_paths)
    elif key == ord('r'):               # 'r' to super skip 100
        current_idx = (current_idx + 100) % len(image_paths)

cv2.destroyAllWindows()
