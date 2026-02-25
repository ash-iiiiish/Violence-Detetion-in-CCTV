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
# PERFORMANCE BOOST
# ==========================

torch.backends.cudnn.benchmark = True

# ==========================
# CONFIG
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 112
CLIP_LEN = 16
YOLO_STRIDE = 3  # Run YOLO every 3 frames

VIDEO_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/Videos/demo5.mp4"
OUTPUT_PATH = "output.mp4"

MODEL_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/live_violence.pth"
YOLO_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/yolov8n.pt"

SMOOTHING_WINDOW = 5
ALERT_COOLDOWN = 3

ALERT_CLASSES = ["Fight", "HockeyFight"]

# ==========================
# LOAD 3D CNN MODEL
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
# LOAD YOLO MODEL
# ==========================

weapon_model = YOLO(YOLO_PATH)

# ==========================
# PREPROCESS
# ==========================

def preprocess_clip(frames):
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==========================
# VIDEO
# ==========================

vid = cv2.VideoCapture(VIDEO_PATH)
fps = vid.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

print("Video FPS:", fps)

writer = None
(W, H) = (None, None)

frame_buffer = deque(maxlen=CLIP_LEN)
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
confidence_history = deque(maxlen=SMOOTHING_WINDOW)

alert_cooldown_until = 0
current_label = "NonFight"
current_confidence = 0.0

frame_count = 0
last_yolo_boxes = []

start_time = time.time()

while True:
    grabbed, frame = vid.read()
    if not grabbed:
        break

    frame_count += 1

    if W is None:
        (H, W) = frame.shape[:2]

    output = frame.copy()

    # ===============================
    # CNN PROCESSING
    # ===============================

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

        # Majority vote smoothing
        current_label = max(set(prediction_history), key=prediction_history.count)

        # Average confidence of smoothed label
        relevant_conf = [
            confidence_history[i]
            for i in range(len(prediction_history))
            if prediction_history[i] == current_label
        ]

        if len(relevant_conf) > 0:
            current_confidence = sum(relevant_conf) / len(relevant_conf)

    # ===============================
    # ALERT LOGIC
    # ===============================

    is_violence = current_label in ALERT_CLASSES
    color = (0, 0, 255) if is_violence else (0, 255, 0)

    # ===============================
    # YOLO (RUN EVERY N FRAMES)
    # ===============================

    if frame_count % YOLO_STRIDE == 0:
        yolo_results = weapon_model(frame, imgsz=640, verbose=False)

        last_yolo_boxes = []
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_yolo_boxes.append((x1, y1, x2, y2))

    # Draw stored boxes
    for (x1, y1, x2, y2) in last_yolo_boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    # ===============================
    # DISPLAY LABEL + CONFIDENCE
    # ===============================

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

    # ===============================
    # BEEP ALERT
    # ===============================

    current_time = time.time()
    if is_violence and current_time > alert_cooldown_until:
        alert_cooldown_until = current_time + ALERT_COOLDOWN
        winsound.Beep(1000, 500)

    # ===============================
    # SAVE VIDEO
    # ===============================

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H), True)

    writer.write(output)
    cv2.imshow("Output", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================
# CLEANUP
# ==========================

end_time = time.time()
total_time = end_time - start_time
print("Total Time:", total_time)
print("Average FPS:", frame_count / total_time)

if writer:
    writer.release()

vid.release()
cv2.destroyAllWindows()