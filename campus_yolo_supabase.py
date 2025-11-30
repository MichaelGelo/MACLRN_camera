import cv2
import time
import psutil
from supabase import create_client
from ultralytics import YOLO
import os
import io
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
import torch

# -------------- CONFIG --------------

SUPABASE_URL = "https://zpezidrqlotoyequnywe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwZXppZHJxbG90b3llcXVueXdlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQyMTE5MzQsImV4cCI6MjA3OTc4NzkzNH0.mWBIhSNpmSaoPN1rytp9JlXdcv_kG9i6aNqBwxGo4q0"

ROOM_NAME = "GK407"

UNOCCUPIED_TIMEOUT = 60
REGULAR_UPDATE_INTERVAL = 10

INFERENCE_SIZE = 320
FRAME_SKIP = 1
CAM_WIDTH = 640
CAM_HEIGHT = 480
WEBCAM_INDEX = 0
SHOW_WINDOW = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE)

# -------------- SPEED CONTROL --------------

def set_speed_mode(mode):
    global INFERENCE_SIZE, FRAME_SKIP, CAM_WIDTH, CAM_HEIGHT

    if mode == "slow_demo":
        INFERENCE_SIZE = 416
        FRAME_SKIP = 0
        CAM_WIDTH = 1280
        CAM_HEIGHT = 720
    else:
        INFERENCE_SIZE = 320
        FRAME_SKIP = 1
        CAM_WIDTH = 640
        CAM_HEIGHT = 480

set_speed_mode("slow_demo")

# -------------- START LOCAL WEB SERVER --------------

def start_dashboard_server():
    """Serve index.html and all files in this folder on port 8000."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("0.0.0.0", 8000), SimpleHTTPRequestHandler)
    print("\n📡 Dashboard available at: http://<JETSON_IP>:8000\n")
    server.serve_forever()

def launch_webserver():
    t = threading.Thread(target=start_dashboard_server)
    t.daemon = True
    t.start()

# Launch server automatically
launch_webserver()

# -------------- SUPABASE SETUP --------------

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_image_supabase(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None

    img_bytes = buffer.tobytes()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"{ROOM_NAME}/{ts}.jpg"

    try:
        supabase.storage.from_("room_images").upload(
            path=file_name,
            file=img_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        return supabase.storage.from_("room_images").get_public_url(file_name)
    except Exception as e:
        print("Supabase upload failed:", e)
        return None


def insert_log(count_value, status, fps_value, image_url=None):
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "room": ROOM_NAME,
        "count": int(count_value),
        "status": status,
        "jetson_fps": round(float(fps_value), 2),
        "image_url": image_url,
        "model_name": model_name,
    }

    try:
        supabase.table("occupancy_logs").insert(data).execute()
        print("Supabase log:", data)
    except Exception as e:
        print("Supabase insert failed:", e)


# -------------- YOLO SETUP --------------

model_path = "yolov8n.engine" if os.path.exists("yolov8n.engine") else "yolov8n.pt"
model_name = os.path.basename(model_path)

print("Using model:", model_name)

model = YOLO(model_path)

try:
    model.to("cuda")
    model.to(dtype=torch.float16)
    print("Using CUDA FP16")
except:
    print("Using CPU")


# -------------- CAMERA SETUP --------------

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("Error: cannot open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# -------------- MAIN LOOP --------------

last_nonzero_time = time.time()
last_regular_update_time = 0
last_status = "unoccupied"
last_frame_time = time.time()
frame_index = 0

print("Starting detection loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_index += 1

        if FRAME_SKIP > 0 and frame_index % (FRAME_SKIP + 1) != 0:
            now = time.time()
            fps = 1.0 / (now - last_frame_time)
            last_frame_time = now
            continue

        results = model.predict(
            frame,
            classes=[0],
            imgsz=INFERENCE_SIZE,
            verbose=False
        )

        people_count = sum(1 for c in results[0].boxes.cls if int(c.item()) == 0)
        now = time.time()
        fps = 1.0 / (now - last_frame_time)
        last_frame_time = now

        status = "occupied" if people_count > 0 else "unoccupied"

        annotated_frame = results[0].plot()

        if people_count > 0:
            last_nonzero_time = now

            if now - last_regular_update_time >= REGULAR_UPDATE_INTERVAL:
                img_url = upload_image_supabase(annotated_frame)
                insert_log(people_count, status, fps, img_url)
                last_regular_update_time = now
                last_status = "occupied"

        if people_count == 0 and now - last_nonzero_time >= UNOCCUPIED_TIMEOUT and last_status != "unoccupied":
            insert_log(0, "unoccupied", fps, None)
            last_status = "unoccupied"

        if SHOW_WINDOW:
            text = f"People: {people_count}  Status: {status}  FPS: {fps:.1f}"
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("CAMPUS YOLOv8", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
