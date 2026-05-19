import os
from ultralytics import YOLO
import cv2
import sys

# Path to your model
MODEL_PATH = "runs/detect/vehicle_counter_best4/weights/best.pt"

def test():
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return

    # Check for images to test
    test_dir = r"yolo_dataset/images/val"
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return
        
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print("No images found in validation set to test.")
        return
        
    test_image_path = os.path.join(test_dir, images[0])
    print(f"Testing inference on: {test_image_path}")
    
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print("Failed to read image with OpenCV.")
            return
            
        results = model(img)
        print("Inference successful!")
        print(f"Found {len(results[0].boxes)} vehicles.")
        
        # Try to save a result
        results[0].save(filename="test_result.jpg")
        print("Result saved to 'test_result.jpg'. The model works!")
        
    except Exception as e:
        print(f"CRITICAL ERROR during inference: {e}")

if __name__ == "__main__":
    test()
