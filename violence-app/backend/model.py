import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import subprocess
from collections import deque
from torchvision.models.video import r3d_18
from ultralytics import YOLO

# ========================== CONFIG ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 112
CLIP_LEN = 16
YOLO_STRIDE = 2
SMOOTHING_WINDOW = 5

WEAPON_CLASS_ID = 1
WEAPON_CONF_THRESHOLD = 0.5

# 🔥 MANUAL PATHS (UNCHANGED)
MODEL_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/live_violence.pth"
YOLO_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/backend/yolo11m.pt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed_videos")
TEMP_FOLDER = os.path.join(BASE_DIR, "temp_clean")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ========================== LOAD MODELS ==========================
print("Loading 3D CNN...")
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]

model.to(DEVICE)
model.eval()

print("Loading YOLO...")
weapon_model = YOLO(YOLO_PATH)

# ========================== PREPROCESS ==========================
def preprocess_clip(frames):
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ========================== CLEAN INPUT VIDEO ==========================
def convert_video_ffmpeg(input_path):
    clean_path = os.path.join(TEMP_FOLDER, f"clean_{int(time.time())}.mp4")

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        clean_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return clean_path


# ========================== MAIN FUNCTION ==========================
def predict_video(video_path):

    # Step 1: Clean input video
    clean_video_path = convert_video_ffmpeg(video_path)

    vid = cv2.VideoCapture(clean_video_path)
    if not vid.isOpened():
        raise RuntimeError("Failed to open video after FFmpeg conversion.")

    fps = vid.get(cv2.CAP_PROP_FPS) or 30

    frame_buffer = deque(maxlen=CLIP_LEN)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    frame_count = 0
    writer = None
    W, H = None, None

    timestamp = int(time.time())
    avi_path = os.path.join(OUTPUT_FOLDER, f"processed_{timestamp}.avi")
    mp4_path = os.path.join(OUTPUT_FOLDER, f"processed_{timestamp}.mp4")

    current_label = "NonFight"
    current_confidence = 0.0

    while True:
        grabbed, frame = vid.read()
        if not grabbed:
            break

        frame_count += 1

        if W is None:
            H, W = frame.shape[:2]
            writer = cv2.VideoWriter(
                avi_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                fps,
                (W, H)
            )

        output = frame.copy()

        # ================= YOLO DETECTION =================
        weapon_detected = False
        person_boxes = []
        weapon_boxes = []

        if frame_count % YOLO_STRIDE == 0:
            results = weapon_model(frame, imgsz=640, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == 0:
                        person_boxes.append((x1, y1, x2, y2, conf))

                    elif cls_id == WEAPON_CLASS_ID and conf > WEAPON_CONF_THRESHOLD:
                        weapon_detected = True
                        weapon_boxes.append((x1, y1, x2, y2, conf))

        # ================= HIERARCHICAL LOGIC =================
        if weapon_detected:
            final_label = "Weaponized"
            final_confidence = 1.0
            color = (0, 0, 255)

        else:
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_buffer.append(resized)

            if len(frame_buffer) == CLIP_LEN:
                clip = preprocess_clip(list(frame_buffer))

                with torch.inference_mode():
                    outputs = model(clip)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                predicted_label = class_names[pred.item()]
                current_label = predicted_label
                current_confidence = conf.item()

                final_label = current_label
                final_confidence = current_confidence
                color = (0, 165, 255) if current_label != "NonFight" else (0, 255, 0)
            else:
                final_label = "Processing..."
                final_confidence = 0
                color = (255, 255, 0)

        # Draw boxes
        for (x1, y1, x2, y2, conf) in person_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for (x1, y1, x2, y2, conf) in weapon_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

        display_text = f"{final_label} ({final_confidence*100:.2f}%)"
        cv2.putText(output,
                    display_text,
                    (int(0.03 * W), int(0.08 * H)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3)

        writer.write(output)

    vid.release()
    writer.release()

    # Convert AVI → MP4
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", avi_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        mp4_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(avi_path):
        os.remove(avi_path)

    return final_label, final_confidence, os.path.basename(mp4_path)