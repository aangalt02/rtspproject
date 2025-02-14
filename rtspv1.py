import json
import cv2
import torch
import numpy as np
import time
import os
import threading
from ultralytics import YOLO
import sys
from pydub.generators import Sine
from pydub.playback import play
import atexit

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

print("\U0001F504 Memuat model YOLOv5 di CPU...")
model = YOLO("yolov5s.pt")  
print("\u2705 Model berhasil dimuat!")

SAVE_DIR = "/home/acer/yolov5/data"
os.makedirs(SAVE_DIR, exist_ok=True)

RTSP_URL = "rtsp://admin:bojonegoro123@192.168.1.64:554/Streaming/Channels/102"
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("\U0001F6A8 Gagal membuka stream RTSP! Periksa koneksi kamera.")
    sys.exit()

last_positions = {}
start_time = {}
last_shot_time = {}
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
photo_interval = 20    
is_beeping = threading.Event()  
BEEP_COOLDOWN = 0.5  # Cooldown time between beeps

# Coba load green_zone_pts dari file JSON jika ada
json_file = "green_zone_pts.json"
if os.path.exists(json_file):
    try:
        with open(json_file, "r") as file:
            green_zone_pts = np.array(json.load(file), np.int32)
        print("✅ Koordinat green_zone_pts berhasil dimuat dari JSON!")
    except Exception as e:
        print(f"⚠️ Gagal memuat green_zone_pts dari JSON: {e}")
        green_zone_pts = np.array([[0, 100], [640, 100], [640, 480], [0, 480]], np.int32)  # Default
else:
    green_zone_pts = np.array([[0, 100], [640, 100], [640, 480], [0, 480]], np.int32)  # Default

dragging = -1  

# Mode detection
mode = "motormobil"  # Default mode
if len(sys.argv) > 1:
    if sys.argv[1] == "--mode" and len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        print("\U0001F6A8 Mode tidak valid! Gunakan --mode manusia atau --mode motormobil")
        sys.exit()

def mouse_callback(event, x, y, flags, param):
    global dragging, green_zone_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(green_zone_pts):
            if np.linalg.norm([x - px, y - py]) < 10:
                dragging = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging != -1:
        green_zone_pts[dragging] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1

cv2.namedWindow("YOLOv5 Detection")
cv2.setMouseCallback("YOLOv5 Detection", mouse_callback)

def draw_green_zone(frame):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [green_zone_pts], (0, 255, 0))  
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return cv2.boundingRect(green_zone_pts)[1]

def play_beep():
    beep_sound = Sine(1000).to_audio_segment(duration=500)
    while is_beeping.is_set():
        print("▶️  Playing beep...")
        play(beep_sound)
        time.sleep(BEEP_COOLDOWN)  # Add cooldown to prevent overlapping beeps

beep_thread = None

target_classes = [2, 3] if mode == "motormobil" else [0]  # 2,3 for vehicles, 0 for person

# Fungsi untuk menyimpan green_zone_pts ke file JSON saat program berakhir
def save_green_zone_on_exit():
    with open("green_zone_pts.json", "w") as file:
        json.dump(green_zone_pts.tolist(), file)
    print("Koordinat green_zone_pts disimpan ke green_zone_pts.json")

# Daftarkan fungsi save_green_zone_on_exit untuk dijalankan saat program berakhir
atexit.register(save_green_zone_on_exit)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Gagal membaca frame, mencoba kembali...")
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL)  
            continue

        start_process_time = time.time()
        results = model(frame)  
        green_zone_top = draw_green_zone(frame)  

        annotated_frame = frame.copy()
        current_positions = []
        should_beep = False

        for r in results:
            boxes = r.boxes  
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = box.conf[0]
                
                if cls in target_classes:  
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    current_positions.append((center_x, center_y))
                    inside_green_zone = cv2.pointPolygonTest(green_zone_pts, (center_x, center_y), False) >= 0
                    color = (0, 0, 255) if inside_green_zone else (255, 0, 0)
                    label_name = "Car" if cls == 2 else "Motorcycle" if cls == 3 else "Person"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if inside_green_zone:
                        found = False
                        for key in list(last_positions.keys()):
                            last_x, last_y = last_positions[key]
                            if np.linalg.norm([center_x - last_x, center_y - last_y]) < 30:
                                found = True
                                if key not in start_time:
                                    start_time[key] = time.time()  # Start timer for new object
                                    last_shot_time[key] = time.time()
                                if time.time() - start_time[key] >= 20:  # Check if 20 seconds have passed
                                    should_beep = True
                                    if time.time() - last_shot_time[key] >= photo_interval:
                                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                                        filename = f"{SAVE_DIR}/{label_name.lower()}_{timestamp}.jpg"
                                        cv2.imwrite(filename, annotated_frame)
                                        last_shot_time[key] = time.time()
                                        print(f"📸 Gambar disimpan: {filename}")
                        if not found:
                            new_id = len(last_positions)
                            last_positions[new_id] = (center_x, center_y)
                            start_time[new_id] = time.time()  # Reset timer for new object
                            last_shot_time[new_id] = time.time()
                    else:
                        # If object leaves the green zone, reset its timer
                        for key in list(last_positions.keys()):
                            last_x, last_y = last_positions[key]
                            if np.linalg.norm([center_x - last_x, center_y - last_y]) < 30:
                                start_time[key] = time.time()  # Reset timer

        # Manage beep logic
        if should_beep and not is_beeping.is_set():
            is_beeping.set()
            if beep_thread is None or not beep_thread.is_alive():
                beep_thread = threading.Thread(target=play_beep, daemon=True)
                beep_thread.start()
        elif not should_beep and is_beeping.is_set():
            is_beeping.clear()

        last_positions = {i: pos for i, pos in enumerate(current_positions)}
        annotated_display = r.plot()
        cv2.polylines(annotated_display, [green_zone_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("YOLOv5 Detection", annotated_display)

        process_time = time.time() - start_process_time
        wait_time = max(1, int((1 / fps - process_time) * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Menutup proses...")
    sys.exit()

cap.release()
cv2.destroyAllWindows()