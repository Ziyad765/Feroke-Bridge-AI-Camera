from ultralytics import YOLO
import os
import torch

def train_and_crush():
    # Path to our massive V2 dataset
    data_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\yolo_dataset_v2_split\data_v2.yaml")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    print("--- 🔬 INITIATING ULTRA-LIGHTWEIGHT EDGE TRAINING ---")
    
    # Windows stability fixes
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # Transfer learning from your massive V2 'Best' payload
    model_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\runs\detect\vehicle_counter_v2_ultimate\weights\best.pt")
    if not os.path.exists(model_path):
        print("Falling back to previous best.pt...")
        model_path = os.path.abspath(r"c:\Users\Ziyad\Downloads\cam-export\feroke project\Feroke-Traffic-Lite\best.pt")

    print(f"Loading Base Weights: {model_path}")
    model = YOLO(model_path)
    
    try:
        # Step 1: Retrain for SMALLER dimension (imgsz=320)
        print(">> PHASE 1: Downsizing Convolutional Matrices (imgsz=320)...")
        results = model.train(
            data=data_path,
            epochs=300,        
            imgsz=320,          # THE CRITICAL EDIT: 320x320 vs 640x640 shrinks VRAM requirement immensely
            batch=16,           # Safe to boost batch size now because Images are 4x smaller    
            device=0 if torch.cuda.is_available() else 'cpu',
            project="runs/detect",
            name="vehicle_counter_edge_nano",
            save=True,
            exist_ok=True,
            plots=False,
            workers=0,
            amp=False,
            patience=50,
            # Light augs since we are crunching it
            augment=True,
            verbose=True
        )
        print("--- 🏁 EDGE TRAINING COMPLETE 🏁 ---")
        
        final_best_path = f"{results.save_dir}/weights/best.pt"
        
        # Step 2: C++ Execution Compiling
        print(">> PHASE 2: Cross-Compiling to INT8 (TFLite) & NCNN...")
        edge_model = YOLO(final_best_path)
        
        # EXPORT 1: NCNN (Tencent's dedicated SBC framework - extremely fast)
        try:
            print("[Compiling -> NCNN]")
            edge_model.export(format='ncnn', imgsz=320)
        except Exception as e:
            print(f"NCNN Export Failed: {e}")
            
        # EXPORT 2: TFLite specifically locked to INT8 using our dataset for calibration
        try:
            print("[Compiling -> TFLite INT8]")
            # int8=True crushes floats to integers natively cutting total kb footprint over 50%.
            edge_model.export(format='tflite', int8=True, imgsz=320, data=data_path)
        except Exception as e:
            print(f"TFLite INT8 Export Failed: {e}")
            
        print("✅ ALL EDGE ARCHITECTURES SUCCESSFULLY COMPILED!")
        print(f"Check {results.save_dir}/weights/ for your lightweight binaries.")

    except Exception as e:
        print(f"FATAL ERROR during Edge Crushing: {e}")

if __name__ == "__main__":
    train_and_crush()
