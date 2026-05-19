import os
import shutil
import random
from pathlib import Path

# Paths
INPUT_DIR = Path(r"c:\Users\Ziyad\Downloads\cam-export\gooddata")
OUTPUT_DIR = Path(r"c:\Users\Ziyad\Downloads\cam-export\yolo_gooddata")

def main():
    print(f"Preparing YOLO Dataset from: {INPUT_DIR}")
    
    # Read train.txt to find all relative paths
    train_txt_path = INPUT_DIR / "train.txt"
    if not train_txt_path.exists():
        print(f"Error: {train_txt_path} not found.")
        return

    with open(train_txt_path, "r") as f:
        # e.g., "data/obj_train_data/frame_1124.jpg"
        lines = [x.strip() for x in f.readlines() if x.strip()]

    # Extract base names (e.g., frame_1124)
    base_names = []
    for line in lines:
        basename = os.path.basename(line).replace('.jpg', '')
        base_names.append(basename)

    # Shuffle for train/val split
    random.seed(42)
    random.shuffle(base_names)
    
    split_index = int(len(base_names) * 0.8) # 80% Train, 20% Val
    train_names = base_names[:split_index]
    val_names = base_names[split_index:]

    # Directories setup
    for split in ['train', 'val']:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print(f"Total Images: {len(base_names)} (Train: {len(train_names)}, Val: {len(val_names)})")

    # Move files
    success_count = 0
    for file_basename in base_names:
        split = 'train' if file_basename in train_names else 'val'
        
        # Source paths
        src_img = INPUT_DIR / "photos" / f"{file_basename}.jpg"
        src_lbl = INPUT_DIR / "obj_train_data" / f"{file_basename}.txt"
        
        # Check they exist
        if not src_img.exists() or not src_lbl.exists():
            print(f"Missing file for {file_basename}, skipping.")
            continue
            
        # Target paths
        dst_img = OUTPUT_DIR / "images" / split / f"{file_basename}.jpg"
        dst_lbl = OUTPUT_DIR / "labels" / split / f"{file_basename}.txt"
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)
        success_count += 1
        
    print(f"Successfully processed {success_count} valid image-label pairs.")

    # Create data.yaml mapping both classes
    yaml_content = f"""
path: {(OUTPUT_DIR).absolute()}
train: images/train
val: images/val

names:
  0: light
  1: heavy
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content.strip())
        
    print(f"Created data.yaml. Dataset fully prepared at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
