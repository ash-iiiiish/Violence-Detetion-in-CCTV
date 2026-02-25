import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import winsound
from collections import deque
from torchvision.models.video import r3d_18
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 112
CLIP_LEN = 16

VIDEO_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/Videos/demo2.mp4"
OUTPUT_PATH = "output.mp4"

MODEL_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/live_violence.pth"
YOLO_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/yolov8n.pt"

SMOOTHING_WINDOW = 5
ALERT_COOLDOWN = 3  # seconds

# ONLY THESE CLASSES TRIGGER ALERT
ALERT_CLASSES = ["Fight", "HockeyFight"]

# ==========================
# LOAD CNN MODEL
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

# ==========================
# LOAD YOLO MODEL (Only for box drawing)
# ==========================

weapon_model = YOLO(YOLO_PATH)

# ==========================
# PREPROCESS FUNCTION
# ==========================

def preprocess_clip(frames):
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))  # C,T,H,W
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==========================
# VIDEO PROCESSING
# ==========================

vid = cv2.VideoCapture(VIDEO_PATH)
fps = vid.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

print(f"FPS: {fps}")

writer = None
(W, H) = (None, None)

frame_buffer = []
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

alert_cooldown_until = 0
current_label = "NonFight"

while True:

    grabbed, frame = vid.read()
    if not grabbed:
        print("Video ended.")
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    output = frame.copy()

    # Resize for CNN
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_buffer.append(resized)

    # ===============================
    # CNN Prediction Every CLIP_LEN
    # ===============================

    if len(frame_buffer) >= CLIP_LEN:

        clip = preprocess_clip(frame_buffer[-CLIP_LEN:])

        with torch.no_grad():
            outputs = model(clip)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        current_label = class_names[pred.item()]
        prediction_history.append(current_label)

        # Majority vote smoothing
        current_label = max(set(prediction_history), key=prediction_history.count)

        frame_buffer.pop(0)

    # ===============================
    # STRICT ALERT LOGIC
    # ===============================

    is_violence = current_label in ALERT_CLASSES

    # ===============================
    # COLOR SELECTION
    # ===============================

    color = (0, 0, 255) if is_violence else (0, 255, 0)

    # ===============================
    # YOLO BOXES (Just for visualization)
    # ===============================

    yolo_results = weapon_model(frame, verbose=False)

    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    # ===============================
    # DISPLAY LABEL
    # ===============================

    cv2.putText(
        output,
        f"{current_label}",
        (int(0.03 * W), int(0.08 * H)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3,
    )

    # ===============================
    # BEEP ALERT (Only Fight/HockeyFight)
    # ===============================

    current_time = time.time()

    if is_violence and current_time > alert_cooldown_until:
        alert_cooldown_until = current_time + ALERT_COOLDOWN
        winsound.Beep(1000, 1000)  # 1 sec beep

    # ===============================
    # SAVE VIDEO
    # ===============================

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H), True)

    writer.write(output)
    cv2.imshow("Output", output)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
        print("ESC pressed.")
        break

print("Releasing memory...")

if writer is not None:
    writer.release()

vid.release()
cv2.destroyAllWindows()