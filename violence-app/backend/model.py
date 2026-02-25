import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from torchvision.models.video import r3d_18
from ultralytics import YOLO
import subprocess

# ==========================
# CONFIG
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 112
CLIP_LEN = 16
YOLO_STRIDE = 3
SMOOTHING_WINDOW = 5

ALERT_CLASSES = ["Fight", "HockeyFight"]

MODEL_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/live_violence.pth"
YOLO_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/yolov8n.pt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed_videos")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ==========================
# LOAD MODELS (LOAD ONCE)
# ==========================

print("Loading 3D CNN model...")
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]

model.to(DEVICE)
model.eval()

print("Classes:", class_names)

weapon_model = YOLO(YOLO_PATH)

# ==========================
# PREPROCESS
# ==========================

def preprocess_clip(frames):
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==========================
# MAIN FUNCTION
# ==========================

def predict_video(video_path):

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)

    if fps is None or fps == 0:
        fps = 30

    frame_buffer = deque(maxlen=CLIP_LEN)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    confidence_history = deque(maxlen=SMOOTHING_WINDOW)

    frame_count = 0
    last_yolo_boxes = []

    current_label = "NonFight"
    current_confidence = 0.0

    writer = None
    (W, H) = (None, None)

    timestamp = int(time.time())
    avi_filename = f"processed_{timestamp}.avi"
    mp4_filename = f"processed_{timestamp}.mp4"

    avi_path = os.path.join(OUTPUT_FOLDER, avi_filename)
    mp4_path = os.path.join(OUTPUT_FOLDER, mp4_filename)

    while True:
        grabbed, frame = vid.read()
        if not grabbed:
            break

        frame_count += 1

        if W is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()

        # ================= CNN =================
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_buffer.append(resized)

        if len(frame_buffer) == CLIP_LEN:
            clip = preprocess_clip(list(frame_buffer))

            with torch.inference_mode():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            predicted_label = class_names[pred.item()]
            predicted_conf = conf.item()

            prediction_history.append(predicted_label)
            confidence_history.append(predicted_conf)

            current_label = max(set(prediction_history), key=prediction_history.count)

            relevant_conf = [
                confidence_history[i]
                for i in range(len(prediction_history))
                if prediction_history[i] == current_label
            ]

            if relevant_conf:
                current_confidence = sum(relevant_conf) / len(relevant_conf)

        # ================= YOLO =================
        if frame_count % YOLO_STRIDE == 0:
            yolo_results = weapon_model(frame, imgsz=640, verbose=False)
            last_yolo_boxes = []

            for r in yolo_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    last_yolo_boxes.append((x1, y1, x2, y2))

        is_violence = current_label in ALERT_CLASSES
        color = (0, 0, 255) if is_violence else (0, 255, 0)

        for (x1, y1, x2, y2) in last_yolo_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        display_text = f"{current_label} ({current_confidence*100:.2f}%)"

        cv2.putText(
            output,
            display_text,
            (int(0.03 * W), int(0.08 * H)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3,
        )

        # ================= VIDEO WRITER =================
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(avi_path, fourcc, fps, (W, H))

            if not writer.isOpened():
                raise Exception("VideoWriter failed to open.")

        writer.write(output)

    # ================= CLEANUP =================
    if writer:
        writer.release()

    vid.release()

    # ================= CONVERT AVI → MP4 =================
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", avi_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        mp4_path
    ]

    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Remove AVI
    if os.path.exists(avi_path):
        os.remove(avi_path)

    print("Saved video:", mp4_path)
    print("File exists:", os.path.exists(mp4_path))

    return current_label, current_confidence, mp4_filename