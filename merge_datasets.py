import os
import shutil
import random

# --- CONFIGURATION ---
ORIGINAL_DATA = "yolo_dataset"
NEW_DATA = "yolo_dataset_v2"
FINAL_DATA = "yolo_final_dataset"

# Ratios
VAL_RATIO = 0.2

def prepare_folder(path):
    os.makedirs(os.path.join(path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "val"), exist_ok=True)

def merge_and_split():
    prepare_folder(FINAL_DATA)
    
    # Collect all image-label pairs from both sources
    all_pairs = []
    
    # 1. From original (might be nested or flat, let's look for images)
    for root, dirs, files in os.walk(ORIGINAL_DATA):
        if "labels" in root: continue
        for f in files:
            if f.endswith(".jpg"):
                img_path = os.path.join(root, f)
                # find matching label
                lbl_name = f.replace(".jpg", ".txt")
                lbl_path = os.path.join(root.replace("images", "labels"), lbl_name)
                if os.path.exists(lbl_path):
                    all_pairs.append((img_path, lbl_path))

    # 2. From new auto-labeled data
    new_imgs = os.path.join(NEW_DATA, "images")
    new_lbls = os.path.join(NEW_DATA, "labels")
    for f in os.listdir(new_imgs):
        if f.endswith(".jpg"):
            img_path = os.path.join(new_imgs, f)
            lbl_path = os.path.join(new_lbls, f.replace(".jpg", ".txt"))
            if os.path.exists(lbl_path):
                all_pairs.append((img_path, lbl_path))

    print(f"Total pairs found: {len(all_pairs)}")
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * (1 - VAL_RATIO))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    def copy_set(pairs, set_name):
        for img_orig, lbl_orig in pairs:
            name = os.path.basename(img_orig)
            shutil.copy(img_orig, os.path.join(FINAL_DATA, "images", set_name, name))
            shutil.copy(lbl_orig, os.path.join(FINAL_DATA, "labels", set_name, name.replace(".jpg", ".txt")))

    print("Copying training set...")
    copy_set(train_pairs, "train")
    print("Copying validation set...")
    copy_set(val_pairs, "val")
    
    # Use absolute, forward-slashed paths for maximum reliability on Windows YOLO
    final_abs = os.path.abspath(FINAL_DATA).replace("\\", "/")
    yaml_content = f"""
train: {final_abs}/images/train
val: {final_abs}/images/val

nc: 1
names: ['vehicle']
"""
    with open(os.path.join(FINAL_DATA, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"\nDONE! Final dataset ready in '{FINAL_DATA}' with {len(all_pairs)} images.")

if __name__ == "__main__":
    merge_and_split()
