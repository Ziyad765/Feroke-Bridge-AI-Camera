# config.py — Global System Configuration

# 1. OPERATION MODE
# Set to True to use the real TVT NVR. Set to False for Phone/Local Test Cam.
USE_NVR_MODE = True

# 2. NVR SETTINGS (TVT Company Standard)
NVR_IP           = "192.168.5.100"  # NVR IP address
NVR_PORT         = "554"            # Default RTSP port
NVR_USER         = "admin"
NVR_PASS         = "user%402019"    # URL-encoded: admin / user@2019

# Stream type: "main" (High-Res) | "sub" (Fast/Stable)
STREAM_TYPE      = "main"

# ────────────────────────────────────────────────────────────────────────
# CAMERA SELECTION  (choose any 4 out of the 10 channels on the NVR)
#
# List exactly 4 NVR channel numbers in this order:
#   Index 0 → Queue Side A      (traffic counting, Side A)
#   Index 1 → Queue Side B      (traffic counting, Side B)
#   Index 2 → Bridge Interior A (clearance check, Side A)
#   Index 3 → Bridge Interior B (clearance check, Side B)
#
# The NVR has channels 1-10. Pick whichever 4 cameras are physically
# installed at the correct positions.
#
# Examples:
#   All four in sequence :  SELECTED_CHANNELS = [1, 2, 3, 4]
#   Skip unused channels :  SELECTED_CHANNELS = [2, 5, 8, 10]
#   Physically swapped   :  SELECTED_CHANNELS = [3, 1, 9, 6]
# ────────────────────────────────────────────────────────────────────────
SELECTED_CHANNELS = [1, 2, 3, 4]  # <<< EDIT: put your 4 channel numbers here

# FFMPEG capture options — proven stable on TVT hardware (TCP, no buffer)
NVR_FFMPEG_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|stimeout=2000000"

if USE_NVR_MODE:
    # Working TVT RTSP URL format (verified against your hardware scan)
    # Format: rtsp://user:pass@ip:port/main?channel=N
    def _nvr_url(channel: int) -> str:
        return f"rtsp://{NVR_USER}:{NVR_PASS}@{NVR_IP}:{NVR_PORT}/{STREAM_TYPE}?channel={channel}"

    CAM1_URL = _nvr_url(SELECTED_CHANNELS[0])   # Queue A       ← channel SELECTED_CHANNELS[0]
    CAM2_URL = _nvr_url(SELECTED_CHANNELS[1])   # Queue B       ← channel SELECTED_CHANNELS[1]
    CAM3_URL = _nvr_url(SELECTED_CHANNELS[2])   # Interior A    ← channel SELECTED_CHANNELS[2]
    CAM4_URL = _nvr_url(SELECTED_CHANNELS[3])   # Interior B    ← channel SELECTED_CHANNELS[3]
else:
    # LOCAL TEST CONFIGURATION (Phone IP cam via DroidCam / IP Webcam)
    CAM1_URL = "http://10.69.123.206:8080/video"  # Queue Side A
    CAM2_URL = "http://10.69.123.206:8080/video"  # Queue Side B
    CAM3_URL = "http://10.69.123.206:8080/video"  # Bridge Interior A
    CAM4_URL = "http://10.69.123.206:8080/video"  # Bridge Interior B

# 3. AI CONFIGURATION
MODEL_PATH = "./models/nano_edge.pt"
USE_TENSORRT = True        # Set to True for Jetson optimization (Engine file used if present)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
AGNOSTIC_NMS = True
AI_IMG_SIZE = 416          # Native resolution for HPC Nano model

# 4. SIGNAL TIMINGS (Seconds)
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 30
MAX_CLEARANCE_TIME = 15

# 5. DETECTION ZONES (Standard Python Lists for JSON compatibility)
# Format: [ [x1, y1], [x2, y2], ... ]
ZONE_A = [ [50, 480], [50, 100], [300, 100], [300, 480] ]
ZONE_B = [ [340, 480], [340, 100], [600, 100], [600, 480] ]
ZONE_C = [ [100, 480], [100, 100], [540, 100], [540, 480] ]
ZONE_D = [ [100, 480], [100, 100], [540, 100], [540, 480] ]
