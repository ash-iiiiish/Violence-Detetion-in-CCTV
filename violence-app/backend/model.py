import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from torchvision.models.video import r3d_18
from ultralytics import YOLO

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

OUTPUT_FOLDER = "processed_videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

torch.backends.cudnn.benchmark = True

# ==========================
# LOAD 3D CNN MODEL (LOAD ONCE)
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
# LOAD YOLO MODEL (LOAD ONCE)
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
# MAIN PREDICTION FUNCTION
# ==========================

def predict_video(video_path):

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    if fps == 0:
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

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"processed_{int(time.time())}.mp4"
    )

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
            current_label = max(
                set(prediction_history),
                key=prediction_history.count
            )

            relevant_conf = [
                confidence_history[i]
                for i in range(len(prediction_history))
                if prediction_history[i] == current_label
            ]

            if len(relevant_conf) > 0:
                current_confidence = sum(relevant_conf) / len(relevant_conf)

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

        # Draw boxes
        is_violence = current_label in ALERT_CLASSES
        color = (0, 0, 255) if is_violence else (0, 255, 0)

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
        # SAVE VIDEO
        # ===============================

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (W, H),
                True
            )

        writer.write(output)

    # Cleanup
    if writer:
        writer.release()

    vid.release()

    return current_label, current_confidence, output_path