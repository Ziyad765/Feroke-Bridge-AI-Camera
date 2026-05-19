from ultralytics import YOLO
import os
import torch

def train_best_ever():
    # Use the absolute path to our processed data
    data_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\yolo_gooddata\data.yaml")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    print(f"Starting FINAL 'Best Ever' training using data at: {data_path}")
    
    # Windows-specific stability fixes
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # Starting from the clean YOLOv26 base for maximum stability and speed
    # We use yolo26n.pt since this was previously downloaded and used as a base variant
    model_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\yolo26n.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: Base model {model_path} not found. Attempting to fallback to 'yolov8n.pt' or downlaod it.")
        model_path = "yolov8n.pt" # Safe fallback

    model = YOLO(model_path)
    
    # Train with ULTIMATE stability and accuracy settings for Windows
    try:
        results = model.train(
            data=data_path,
            epochs=300,
            imgsz=640,
            batch=4,
            device=0 if torch.cuda.is_available() else 'cpu',
            project="runs/detect",
            name="vehicle_counter_fresh",
            save=True,
            exist_ok=True,
            plots=False,        # Disable to avoid Matplotlib worker crashes on Windows
            workers=0,          # Disable multiprocessing for stability
            amp=False,          # Disable AMP for stability
            patience=50,
            augment=True,
            mosaic=1.0,         # Accurate for small objects
            mixup=0.1,          # Good for small datasets
            verbose=True
        )
        print("--- 🏁 FRESH TRAINING COMPLETE 🏁 ---")
        print(f"Your new 'Best Ever' model has been saved at: {results.save_dir}/weights/best.pt")
        
        # Exporting to ONNX based on previous training optimization patterns
        print("Optimization: Exporting model to ONNX format...")
        model.export(format='onnx', imgsz=640)
        
    except Exception as e:
        print(f"FATAL ERROR during training: {e}")

if __name__ == "__main__":
    train_best_ever()
