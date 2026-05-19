# The Feroke AI Pipeline: From Zero to Edge 🚀

This document is your permanent masterclass on exactly how we sculpted the incredibly lightweight, accurate AI that powers the Feroke Bridge routing system from absolute scratch. 

Keep this documentation on-hand. If the bridge environment changes (new lighting setups, new camera angles, or weirdly shaped vehicles), you can use these exact steps to evolve the model!

---

## 🏗️ Phase 1: The Initial Seed ("Gooddata") 
**The Goal:** Train a model perfectly customized to identifying specifically "Light" (Class 0) and "Heavy" (Class 1) vehicles from the bridge's perspective.
**The Data:** We started with just **29 highly curated images** (the `gooddata` folder) manually generated and labeled by hand.

1. **Structuring:** Ultralytics YOLO strictly demands data be nested into `images/train` and `labels/train`. We wrote a parser to split those 29 images into an 80/20 train/validation split.
2. **Initial Training:** We pulled the generic `yolo26n.pt` base structure and taught it your two distinct classes over 300 epochs. Because the dataset was small, we minimized augmentations. 
3. **Result:** An initial `best.pt` model that was smart enough to recognize standard bridge traffic reasonably well.

---

## 🤖 Phase 2: Autonomous Scaling ("Bulk Auto-Labeling")
**The Goal:** A dataset of 29 images is mathematically prone to "overfitting" (memorizing the images rather than learning to track dynamically). We needed thousands of representations. 
**The Data:** You provided hundreds of completely raw, unannotated bridge frames inside `clean_frames1`.

1. **Auto-Labeling:** We built a script (`bulk_auto_label.py`) that forced our Phase 1 `best.pt` model to look at every single raw image and systematically draw bounding boxes where it *thought* the vehicles were.
2. **Human-in-the-loop Validation:** Those raw AI outputs were dumped into `yolo_dataset_v2`. We generated a Streamlit Interface allowing you to physically audit, correct, and delete any bad assumptions the Phase 1 model made.
3. **Result:** A massive, 482-image pristine dataset (`yolo_dataset_v2`) created in minutes instead of manually drawing thousands of boxes by hand for hours.

---

## 🧠 Phase 3: The Ultimate Checkpoint ("V2 Ultimate")
**The Goal:** Leverage the new 482-frame dataset to build an incredibly potent, unshakeable tracker.

1. **Transfer Learning:** We did *not* begin from scratch! We took our Phase 1 model and loaded it as the base. It retained its prior localized knowledge and exploded in accuracy as the new 482 images reinforced it.
2. **Hyper-parameter Tuning:** Because the dataset was larger, we significantly scaled batch sizes up and maximized augmentation variances (Mosaic, Mixup, Scaling).
3. **Result:** After hours on the GPU, we produced the `vehicle_counter_v2_ultimate\best.pt` achieving **mAP50 = 0.822** precision!

---

## ⚡ Phase 4: Extreme Jetson/IoT Optimization ("Edge Nano")
**The Goal:** Models built for 8GB Server GPUs natively crush tiny Single Board Computers (SBCs) like Jetsons and Raspberry Pis due to thermal/RAM constraints over 24/7 lifetimes. We must physically shrink the mathematics.

1. **Squinting (`imgsz=320`):** By executing `train_edge_nano.py`, we reran the whole training sequence forcing the model to calculate convolutions at exactly **320x320 resolution** rather than 640x640. This permanently deletes about 75% of the VRAM requirement and matrix allocation footprint.
2. **Retained Accuracy:** Despite the massive reduction in size, it retained ~`0.767 mAP50`, achieving nearly identical bounding box capability while utilizing exponentially fewer system resources.
3. **C++ Compilation:** We broke the PyTorch dependency entirely. We successfully translated the model into `NCNN` format (Tencent's C++ Edge framework), stripping out the heavy Python runtime.

---

## 🛠️ How to Evolve the Model Further (Next Steps)

If you gather a new folder of raw images (e.g., night-time footage or rainy weather) and want to make the AI smarter:

1. **Bulk Label:** Run `python bulk_auto_label.py` to let the current AI guess the boxes.
2. **Review:** Run `streamlit run auto_label_and_review.py` and fix any mistakes.
3. **Re-Train:** Run `python train_v2_ultimate.py` (making sure it points to the new dataset output)!
4. **Deploy:** When deploying on a production NVIDIA Jetson for 24/7 survival:
   - Run this specific command **organically on the Jetson** to compile TensorRT: 
   - `yolo export model=best.pt format=engine imgsz=320 dynamic=True workspace=2`
   - This creates a `.engine` blob optimized specifically for that individual Jetson's hardware, unlocking hyper-efficient native Nvidia operations!
