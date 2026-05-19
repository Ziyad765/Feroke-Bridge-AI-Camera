import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your custom-trained 'Best Ever' YOLOv26 model
# MODEL_PATH = "runs/detect/vehicle_counter_best4/weights/best.pt"
MODEL_PATH = r"runs\detect\runs\detect\vehicle_counter_final\weights\best.pt"
model = YOLO(MODEL_PATH)

def detect_vehicles(input_img):
    try:
        if input_img is None:
            return None, "Please upload an image."
        
        # Perform inference
        results = model(input_img, iou=0.45)[0]
        
        # Process results
        annotated_frame = results.plot()
        count = len(results.boxes)
        
        # Convert BGR to RGB for Gradio
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        return annotated_frame_rgb, f"Total Vehicles Detected: {count}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Simplified Interface API for Gradio 6.x stability
demo = gr.Interface(
    fn=detect_vehicles,
    inputs=gr.Image(type="numpy", label="Upload Photo"),
    outputs=[gr.Image(label="Detections"), gr.Textbox(label="Result Summary")],
    title="🚗 YOLOv26 Vehicle Detection",
    description="Upload an image to see detection results."
)

if __name__ == "__main__":
    # Launch without complex features
    demo.launch(share=False)
