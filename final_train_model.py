from ultralytics import YOLO
import os

def train_final_best_ever():
    # Use the absolute path to our combined data
    data_path = os.path.abspath("yolo_final_dataset/data.yaml")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    print(f"Starting FINAL 'Best Ever' training using data at: {data_path}")
    
    # Windows-specific stability fixes
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    # Starting from the clean YOLOv26 base for maximum stability
    model = YOLO("yolo26n.pt")
    
    print(f"Starting FINAL 'Best Ever' training using data at: {data_path}")
    
    # Train with ULTIMATE stability settings for Windows
    try:
        results = model.train(
            data=data_path,
            epochs=300,
            imgsz=640,
            batch=4,
            device=0,
            project="runs/detect",
            name="vehicle_counter_final",
            save=True,
            exist_ok=True,
            plots=False,        # Disable to avoid Matplotlib worker crashes
            workers=0,          # Disable multiprocessing for stability
            amp=False,          # Disable AMP for stability
            patience=50,
            augment=True,
            verbose=True
        )
        print("--- 🏁 FINAL TRAINING COMPLETE 🏁 ---")
    except Exception as e:
        print(f"FATAL ERROR during training: {e}")
    print(f"Your 'Best Ever' model is saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_final_best_ever()
