import cv2
import time
import psutil
from supabase import create_client
from ultralytics import YOLO
import os

# -------------- CONFIG --------------

SUPABASE_URL = "https://zpezidrqlotoyequnywe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwZXppZHJxbG90b3llcXVueXdlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQyMTE5MzQsImV4cCI6MjA3OTc4NzkzNH0.mWBIhSNpmSaoPN1rytp9JlXdcv_kG9i6aNqBwxGo4q0"

ROOM_NAME = "CL305"

# logic timers
UNOCCUPIED_TIMEOUT = 60
REGULAR_UPDATE_INTERVAL = 10

# defaults (will be overwritten by speed mode)
INFERENCE_SIZE = 320
FRAME_SKIP = 1
CAM_WIDTH = 640
CAM_HEIGHT = 480
WEBCAM_INDEX = 0
SHOW_WINDOW = True

# -------------- SPEED CONTROL --------------

def set_speed_mode(mode, size=None, skip=None, cam_w=None, cam_h=None):
    global INFERENCE_SIZE, FRAME_SKIP, CAM_WIDTH, CAM_HEIGHT

    if mode == "slow_demo":
        INFERENCE_SIZE = 416
        FRAME_SKIP = 0
        CAM_WIDTH = 1280
        CAM_HEIGHT = 720
        print("[Speed Mode] slow_demo activated")

    elif mode == "balanced":
        INFERENCE_SIZE = 320
        FRAME_SKIP = 1
        CAM_WIDTH = 640
        CAM_HEIGHT = 480
        print("[Speed Mode] balanced activated")

    elif mode == "fast":
        INFERENCE_SIZE = 256
        FRAME_SKIP = 1
        CAM_WIDTH = 640
        CAM_HEIGHT = 480
        print("[Speed Mode] fast activated")

    elif mode == "max":
        INFERENCE_SIZE = 224
        FRAME_SKIP = 2
        CAM_WIDTH = 320
        CAM_HEIGHT = 240
        print("[Speed Mode] max activated")

    elif mode == "custom":
        INFERENCE_SIZE = size
        FRAME_SKIP = skip
        CAM_WIDTH = cam_w
        CAM_HEIGHT = cam_h
        print("[Speed Mode] custom settings applied")

    else:
        print("Invalid speed mode. Defaulting to balanced.")
        INFERENCE_SIZE = 320
        FRAME_SKIP = 1
        CAM_WIDTH = 640
        CAM_HEIGHT = 480

# choose speed mode
set_speed_mode("slow_demo")


# -------------- SUPABASE SETUP --------------

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_jetson_temp():
    try:
        temps = psutil.sensors_temperatures()
        for key in ["cpu-thermal", "CPU", "gpu", "GPU"]:
            if key in temps and len(temps[key]) > 0:
                return float(temps[key][0].current)
    except Exception:
        pass
    return None

def insert_log(count_value, status, event_type, fps_value):
    jetson_temp = get_jetson_temp()

    # ADD TIMESTAMP FIX HERE
    timestamp_value = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    data = {
        "timestamp": timestamp_value,          # <--- FIXED
        "room": ROOM_NAME,
        "count": int(count_value),
        "status": status,
        "event_type": event_type,
        "jetson_temp": jetson_temp,
        "jetson_fps": float(fps_value) if fps_value is not None else None,
    }

    try:
        supabase.table("occupancy_logs").insert(data).execute()
        print("Supabase log:", data)
    except Exception as e:
        print("Supabase insert failed:", e)


# -------------- YOLO SETUP --------------

print("Loading YOLO model (trying TensorRT engine first)...")

model_path = "yolov8n.engine" if os.path.exists("yolov8n.engine") else "yolov8n.pt"
print(f"Using model file: {model_path}")

model = YOLO(model_path)

try:
    model.to("cuda")
    try:
        import torch
        model.to(dtype=torch.float16)
        print("Using CUDA with half precision")
    except Exception:
        print("Using CUDA with default precision")
except Exception:
    print("CUDA not available, running on CPU")


# -------------- CAMERA SETUP --------------

print("Opening webcam:", WEBCAM_INDEX)
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print("Error: cannot open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


# -------------- MAIN LOOP --------------

last_nonzero_time = time.time()
last_regular_update_time = 0.0
last_status = "unoccupied"

fps = 0.0
last_frame_time = time.time()
frame_index = 0

print("Starting optimized detection loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: frame grab failed")
            time.sleep(0.05)
            continue

        frame_index += 1

        if FRAME_SKIP > 0 and (frame_index % (FRAME_SKIP + 1) != 0):
            now_time = time.time()
            dt = now_time - last_frame_time
            if dt > 0:
                fps = 1.0 / dt
            last_frame_time = now_time

            if SHOW_WINDOW:
                text = f"Skipping frame | FPS: {fps:.1f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                cv2.imshow("CAMPUS YOLOv8", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        results = model.predict(frame, imgsz=INFERENCE_SIZE, verbose=False)

        people_count = 0
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.cls is not None:
                for cls in r.boxes.cls:
                    if int(cls.item()) == 0:
                        people_count += 1

        now = time.time()
        dt = now - last_frame_time
        if dt > 0:
            fps = 1.0 / dt
        last_frame_time = now

        status = "occupied" if people_count > 0 else "unoccupied"
        if people_count > 0:
            last_nonzero_time = now

        if (
            people_count > 0
            and now - last_regular_update_time >= REGULAR_UPDATE_INTERVAL
        ):
            insert_log(people_count, "occupied", "regular_update", fps)
            last_regular_update_time = now
            last_status = "occupied"

        if (
            people_count == 0
            and now - last_nonzero_time >= UNOCCUPIED_TIMEOUT
            and last_status != "unoccupied"
        ):
            insert_log(0, "unoccupied", "unoccupied_event", fps)
            last_status = "unoccupied"

        annotated_frame = results[0].plot() if len(results) > 0 else frame

        text = f"People: {people_count}  Status: {status}  FPS: {fps:.1f}"
        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        if SHOW_WINDOW:
            cv2.imshow("CAMPUS YOLOv8", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Stopping by keyboard interrupt")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Bye.")

