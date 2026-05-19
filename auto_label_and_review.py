import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
import shutil

# --- CONFIGURATION ---
MODEL_PATH = r"c:\Users\Ziyad\Downloads\cam-export\feroke project\Feroke-Traffic-Lite\best.pt"
INPUT_DIR = r"yolo_dataset_v2\images"
OUTPUT_DIR = "yolo_dataset_v2"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Page Config
st.set_page_config(page_title="AI Auto-Labeler & Reviewer", layout="wide")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def init_folders():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

def auto_label_image(model, image_path, img_name):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    # Add iou=0.45 and agnostic_nms to merge overlapping double-detections
    results = model(img, verbose=False, iou=0.45, conf=0.25, agnostic_nms=True)[0]
    
    label_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))
    
    with open(label_path, "w") as f:
        for box in results.boxes:
            # YOLO Format: class x_center y_center width height (normalized)
            cls = int(box.cls[0])
            xywhn = box.xywhn[0].cpu().numpy()
            f.write(f"{cls} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")
            
    # Copy image to new dataset folder
    shutil.copy(image_path, os.path.join(IMAGES_DIR, img_name))
    return results.plot()

# --- UI START ---
st.title("🏷️ YOLOv26 Auto-Labeling & Review")
st.markdown("Use your current model to label new data and build an upgraded dataset.")

model = load_model()
init_folders()

# 1. SCAN FILES
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
st.sidebar.info(f"Found {len(all_files)} images in `{INPUT_DIR}`")

# 2. BULK AUTO-LABEL MODE
if st.sidebar.button("🚀 Start Bulk Auto-Labeling"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(all_files):
        img_path = os.path.join(INPUT_DIR, filename)
        auto_label_image(model, img_path, filename)
        progress_bar.progress((i + 1) / len(all_files))
        status_text.text(f"Auto-labeling: {i+1}/{len(all_files)} ({filename})")
    
    st.sidebar.success("Bulk Auto-Labeling Complete!")

# 3. REVIEW MODE
st.divider()
st.header("🔍 Human-in-the-loop Review")

if len(all_files) > 0:
    idx = st.number_input("Browse Image #", min_value=0, max_value=len(all_files)-1, step=1, value=0)
    current_file = all_files[idx]
    
    img_path = os.path.join(INPUT_DIR, current_file)
    label_file = current_file.replace(".jpg", ".txt")
    label_path = os.path.join(LABELS_DIR, label_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        img = cv2.imread(img_path)
        # Show detections if labels exist
        if os.path.exists(label_path):
            res = model(img, verbose=False, iou=0.45, conf=0.25, agnostic_nms=True)[0]
            annotated = res.plot()
            st.image(annotated, channels="BGR", caption=f"Preview: {current_file}", use_container_width=True)
        else:
            st.image(img, channels="BGR", caption=f"Raw: {current_file}", use_container_width=True)

    with col2:
        st.subheader("Actions")
        if os.path.exists(label_path):
            st.success("✅ Auto-labeled")
            with open(label_path, "r") as f:
                st.text_area("YOLO Labels:", f.read(), height=200)
            
            if st.button("🗑️ Delete Label (Bad AI work)"):
                os.remove(label_path)
                st.rerun()
        else:
            st.warning("⚠️ Not labeled yet")
            if st.button("🪄 Auto-label this image"):
                auto_label_image(model, img_path, current_file)
                st.rerun()

    st.info("PRO TIP: After bulk labeling, use this view to quickly skip through and delete any frames where the AI made big mistakes.")
