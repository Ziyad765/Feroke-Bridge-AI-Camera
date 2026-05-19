import os
import shutil
import random
from pathlib import Path

# Paths
INPUT_DIR = Path(r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset_v2")
OUTPUT_DIR = Path(r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset_v2_split")

def main():
    print(f"Preparing YOLO V2 Splitted Dataset from: {INPUT_DIR}")
    
    in_images = INPUT_DIR / "images"
    in_labels = INPUT_DIR / "labels"
    
    # We will only use files that have BOTH image and label
    all_label_files = [f for f in os.listdir(in_labels) if f.endswith(".txt")]
    
    base_names = []
    for label_file in all_label_files:
        basename = label_file.replace('.txt', '')
        img_path = in_images / f"{basename}.jpg"
        if img_path.exists():
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
        
    print(f"Total Valid Images: {len(base_names)} (Train: {len(train_names)}, Val: {len(val_names)})")

    # Move files
    success_count = 0
    for file_basename in base_names:
        split = 'train' if file_basename in train_names else 'val'
        
        # Source paths
        src_img = in_images / f"{file_basename}.jpg"
        src_lbl = in_labels / f"{file_basename}.txt"
            
        # Target paths
        dst_img = OUTPUT_DIR / "images" / split / f"{file_basename}.jpg"
        dst_lbl = OUTPUT_DIR / "labels" / split / f"{file_basename}.txt"
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)
        success_count += 1
        
    print(f"Successfully processed {success_count} image-label pairs into split directory.")

    # Create data.yaml mapping both classes
    yaml_content = f"""
path: {(OUTPUT_DIR).absolute()}
train: images/train
val: images/val

names:
  0: light
  1: heavy
"""
    with open(OUTPUT_DIR / "data_v2.yaml", "w") as f:
        f.write(yaml_content.strip())
        
    print(f"Created data_v2.yaml. Dataset V2 fully prepared at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
