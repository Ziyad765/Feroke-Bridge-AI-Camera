# 🚗 Master Operations & Jetson Optimization Guide

This guide covers everything you need to run, maintain, and upgrade your 83% accuracy vehicle detection system.

---

## 📂 1. Script Directory & How to Run
Every script is pre-configured with the **"Best Ever" (83% mAP)** model settings.

| File | Purpose | How to Run |
| :--- | :--- | :--- |
| `fast_inference_test.py` | **Live Stream Test** (60 FPS, Large UI) | `python fast_inference_test.py "http://IP:8080/video"` |
| `streamlit_app.py` | **Visual Dashboard** (Upload Image/Video) | `streamlit run streamlit_app.py` |
| `jetson_inference_multistream.py` | **Production Code** (4 Streams + Counting) | `python jetson_inference_multistream.py` |
| `bulk_auto_label.py` | **Mass Labeling** new unlabelled images | `python bulk_auto_label.py` |
| `auto_label_and_review.py` | **Review UI** for auto-labels | `streamlit run auto_label_and_review.py` |
| `merge_datasets.py` | **Dataset Joiner** for new labels | `python merge_datasets.py` |
| `final_train_model.py` | **Model Trainer** for new datasets | `python final_train_model.py` |

---

## 🔄 2. The "Flywheel" Workflow (Upgrade Your Model)
Follow these steps whenever you have a new folder of images (e.g., `new_photos`) and want to improve the model:

1.  **Auto-Label**: Update `INPUT_DIR` in `bulk_auto_label.py` to your `new_photos` folder and run it. This creates a standard YOLO dataset automatically.
2.  **Review (Optional)**: Run `auto_label_and_review.py` to visually confirm or delete any bad labels made by the AI.
3.  **Merge**: Run `merge_datasets.py`. It combines your original data, the previous 482 images, and your *new* set into one master folder (`yolo_final_dataset`).
4.  **Retrain**: Run `final_train_model.py`. It will start from your current 83% model and fine-tune it with the new data.
5.  **Deploy**: Take the new `best.pt` from `runs\detect\...` and copy it to your Jetson.

---

## ⚡ 3. Jetson Nano Optimization (TensorRT)
Your current `.pt` files will work on Jetson, but they aren't "optimized" for pure speed. To get maximum performance on the Orin Nano, you MUST convert to **TensorRT (`.engine`)**.

### How to Optimize on Jetson:
1.  Copy `best.pt` to the Jetson.
2.  Install Ultralytics on the Jetson if not there (`pip install ultralytics`).
3.  Run this command in the terminal:
    ```bash
    yolo export model=best.pt format=engine device=0 imgsz=640
    ```
4.  This generates `best.engine`. 
5.  Update `MODEL_PATH` in `jetson_inference_multistream.py` to point to `best.engine`. 

**This optimization makes the AI run directly on the Jetson's GPU hardware for roughly 2x-3x faster speeds.**

---

## 🛡️ 4. Tips for Success
- **Paths**: Always use absolute paths (like `C:\Users\...`) or the `r"path\to\file"` format on Windows to avoid errors.
- **Double Boxes**: All scripts now include `iou=0.45` and `agnostic_nms=True`. Never lower these unless you want duplicate boxes to reappear.
- **Backups**: Your `vehicle_counter_best4` is your original safe backup. The `vehicle_counter_final` is your current champion.

🏆 **You are now a YOLO Master!** 🚀
