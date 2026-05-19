import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="YOLOv26 Vehicle Detection", page_icon="🚗", layout="wide")

st.title("🚗 YOLOv26 Vehicle Detection & Counting")
st.markdown("""
    This app uses your custom-trained YOLOv26 model to detect and count vehicles in your CCTV images.
    Upload an image in the sidebar to get started. 🚀
""")

# Load Model (Cache to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    # MODEL_PATH = "runs/detect/vehicle_counter_best4/weights/best.pt"
    MODEL_PATH = r"runs\detect\runs\detect\vehicle_counter_final\weights\best.pt"
    return YOLO(MODEL_PATH)

model = load_model()

# Sidebar for Upload
st.sidebar.header("Configuration")
app_mode = st.sidebar.selectbox("Choose Mode", ["Image Detection", "Video Tracking"])

if app_mode == "Image Detection":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        with st.spinner('Detecting...'):
            results = model(opencv_image, iou=0.45, conf=0.25, agnostic_nms=True)[0]
        annotated_frame = results.plot()
        st.image(annotated_frame, channels="BGR", use_container_width=True)
        st.success(f"Detected {len(results.boxes)} vehicles.")

elif app_mode == "Video Tracking":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    # Performance Settings in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Performance Settings")
    skip_frames = st.sidebar.slider("Skip Frames (Increase for speed)", 1, 10, 3)
    img_size = st.sidebar.select_slider("Inference Resolution", options=[320, 480, 640], value=480)
    
    if uploaded_video is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        st_metrics = st.empty()
        
        count = 0
        tracked_ids = set()
        
        import time
        frame_idx = 0
        
        stop_btn = st.button("Stop Processing")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            # Skip frames to keep up with real-time video speed
            if frame_idx % skip_frames != 0:
                continue
                
            start_time = time.time()
            
            # Run Tracking with lower resolution and strict IoU for speed/accuracy
            results = model.track(frame, persist=True, verbose=False, imgsz=img_size, iou=0.45, conf=0.25, agnostic_nms=True)[0]
            
            if results.boxes.id is not None:
                ids = results.boxes.id.cpu().numpy().astype(int)
                for obj_id in ids:
                    if obj_id not in tracked_ids:
                        tracked_ids.add(obj_id)
                        count += 1
            
            # FPS Calculation
            inference_time = time.time() - start_time
            fps = 1 / inference_time if inference_time > 0 else 0
            
            # Plot results
            annotated_frame = results.plot()
            
            # Draw Metrics
            cv2.putText(annotated_frame, f"Count: {count} | FPS: {fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            st_metrics.write(f"**Unique Vehicles:** {count} | **Est. Inference FPS:** {fps:.1f} | **Skipping:** {skip_frames} frames")
            
        cap.release()
        st.success(f"Processing Complete! Total unique vehicles: {count}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using YOLOv26 and Streamlit.")
