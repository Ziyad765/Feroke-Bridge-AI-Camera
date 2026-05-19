import json
import os
import shutil
from pathlib import Path
from PIL import Image

# Configuration
DATASET_PATH = Path(r"c:\Users\Ziyad\Downloads\cam-export")
OUTPUT_PATH = Path(r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset")
IMAGE_SIZE = (3840, 2160)  # Width, Height

# Class mapping (Merge all to one class: vehicle)
CLASS_MAP = {
    "light": 0,
    "heavy": 0
}

def convert_to_yolo():
    # Load labels
    labels_file = DATASET_PATH / "info.labels"
    with open(labels_file, "r") as f:
        data = json.load(f)

    # Create output directories
    for split in ["train", "val"]:
        (OUTPUT_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process files
    for entry in data["files"]:
        # Edge Impulse category to YOLO split
        EI_category = entry["category"]
        yolo_split = "train" if EI_category == "training" else "val"
        
        # Source image path
        src_img_path = DATASET_PATH / entry["path"]
        if not src_img_path.exists():
            print(f"Warning: {src_img_path} not found.")
            continue

        # Target image path
        img_name = Path(entry["name"]).with_suffix(".jpg")
        dst_img_path = OUTPUT_PATH / "images" / yolo_split / img_name
        
        # Copy image
        shutil.copy(src_img_path, dst_img_path)

        # Label path
        label_path = OUTPUT_PATH / "labels" / yolo_split / img_name.with_suffix(".txt")
        
        # Write YOLO labels
        with open(label_path, "w") as f:
            for bbox in entry.get("boundingBoxes", []):
                label_str = bbox.get("label")
                if label_str not in CLASS_MAP:
                    continue
                
                class_id = CLASS_MAP[label_str]
                
                # Normalize coordinates
                x = bbox["x"]
                y = bbox["y"]
                w = bbox["width"]
                h = bbox["height"]
                
                # YOLO format: x_center, y_center, width, height (normalized)
                x_center = (x + w / 2) / IMAGE_SIZE[0]
                y_center = (y + h / 2) / IMAGE_SIZE[1]
                w_norm = w / IMAGE_SIZE[0]
                h_norm = h / IMAGE_SIZE[1]
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    # Create data.yaml
    yaml_content = f"""
path: {OUTPUT_PATH.absolute()}
train: images/train
val: images/val

names:
  0: vehicle
"""
    with open(OUTPUT_PATH / "data.yaml", "w") as f:
        f.write(yaml_content.strip())

    print(f"Data conversion complete. Dataset located at: {OUTPUT_PATH}")

if __name__ == "__main__":
    convert_to_yolo()
