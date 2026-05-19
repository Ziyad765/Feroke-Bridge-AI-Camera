from ultralytics import YOLO
import os
import torch

def train_v2_ultimate():
    # Path to our newly generated V2 Dataset
    data_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset_v2_split\data_v2.yaml")
    
    if not os.path.exists(data_path):
        print(f"ERROR: V2 Dataset not found at {data_path}")
        return

    print(f"Starting ULTIMATE V2 training using data at: {data_path}")
    
    # Windows-specific stability fixes
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # Starting from the previously trained 'best.pt' for transfer learning (This preserves existing knowledge)
    model_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\feroke project\Feroke-Traffic-Lite\best.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: Previous Best model {model_path} not found. Attempting fallback.")
        model_path = "yolov8n.pt"

    print(f"Transfer Learning base model loaded from: {model_path}")
    model = YOLO(model_path)
    
    # Train with ULTIMATE stability and heavily optimized accuracy settings for larger datasets
    try:
        results = model.train(
            data=data_path,
            epochs=300,
            imgsz=640,
            batch=8,            # Increased batch size safely since data is larger and fits in 8GB A2000
            device=0 if torch.cuda.is_available() else 'cpu',
            project="runs/detect",
            name="vehicle_counter_v2_ultimate",
            save=True,
            exist_ok=True,
            plots=False,        # Disable to avoid Matplotlib worker crashes on Windows
            workers=0,          # Disable multiprocessing for 100% stability
            amp=False,          # Disable AMP for stability
            patience=100,       # Increased patience for large dataset to assure full convergence
            # Heavy Augmentations to improve generalizability:
            augment=True,
            mosaic=1.0,         
            mixup=0.2,          
            degrees=10.0,
            translate=0.2,
            scale=0.5,
            verbose=True
        )
        print("--- 🏁 ULTIMATE V2 TRAINING COMPLETE 🏁 ---")
        print(f"Your V2 'Best Ever' model has been saved at: {results.save_dir}/weights/best.pt")
        
        # Expert Optimization: Exporting model to ONNX format natively for the main Feroke loop
        print("Optimization: Exporting V2 model to ONNX format...")
        model.export(format='onnx', imgsz=640)
        
    except Exception as e:
        print(f"FATAL ERROR during V2 training: {e}")

if __name__ == "__main__":
    train_v2_ultimate()
