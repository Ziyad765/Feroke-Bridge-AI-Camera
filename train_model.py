from ultralytics import YOLO
import os

# Configuration
DATASET_YAML = r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset\data.yaml"
# Upgrading to YOLOv26 (Latest and Fastest Architecture - March 2026)
MODEL_VARIANT = 'yolo26n.pt' 

def train():
    # Load a pre-trained YOLOv26 model
    model = YOLO(MODEL_VARIANT)

    # Final High-Accuracy Training Settings
    results = model.train(
        data=DATASET_YAML,
        epochs=300,
        imgsz=640,
        batch=4,            # Safe batch size
        device=0,
        name='vehicle_counter_best',
        patience=50,
        save=True,
        augment=True,
        workers=0,          # Windows stability
        amp=False,          # Stability
        plots=False,        # Disable to avoid silent Matplotlib crashes on Windows
        mosaic=1.0,         # Accurate for small objects
        mixup=0.1,          # Good for small datasets
    )

    # Export the final model to ONNX
    print("Optimization: Exporting model to ONNX...")
    model.export(format='onnx', imgsz=640)

if __name__ == '__main__':
    train()
