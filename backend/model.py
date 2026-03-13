import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import subprocess
from collections import deque, Counter
from torchvision.models.video import r3d_18
from ultralytics import YOLO

# ========================== CONFIG ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 112
CLIP_LEN = 16
YOLO_STRIDE = 2

WEAPON_CONF_THRESHOLD = 0.5
WEAPON_RELAX_FRAMES   = 30
VIOLENCE_SMOOTH_COUNT = 5

# ── Permanent violence activation config ─────────────────────
# Once any fight class is detected with confidence >= this threshold,
# it locks in permanently for the rest of the video.
VIOLENCE_LOCK_THRESHOLD = 0.55   # minimum confidence to trigger lock
VIOLENCE_LOCK_CLASSES   = ["Fight", "HockeyFight", "MovieFight"]  # classes that trigger lock

# 🔥 MANUAL PATHS
MODEL_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/backend/best-violence.pth"
YOLO_PATH  = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/backend/best-yolo.pt"

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER  = os.path.join(BASE_DIR, "processed_videos")
TEMP_FOLDER    = os.path.join(BASE_DIR, "temp_clean")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER,   exist_ok=True)

# ========================== LOAD MODELS ==========================
print("Loading 3D CNN...")
model    = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]

model.to(DEVICE)
model.eval()

print("Loading YOLO Weapon Model...")
weapon_model = YOLO(YOLO_PATH)

# ========================== PREPROCESS ==========================
def preprocess_clip(frames):
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ========================== VIDEO CLEAN ==========================
def convert_video_ffmpeg(input_path):
    clean_path = os.path.join(TEMP_FOLDER, f"clean_{int(time.time())}.mp4")
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        clean_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return clean_path

# ========================== MAIN FUNCTION ==========================
def predict_video(video_path):

    clean_video_path = convert_video_ffmpeg(video_path)

    vid = cv2.VideoCapture(clean_video_path)
    if not vid.isOpened():
        raise RuntimeError("Failed to open video.")

    fps = vid.get(cv2.CAP_PROP_FPS) or 30

    frame_buffer      = deque(maxlen=CLIP_LEN)
    violence_history  = deque(maxlen=VIOLENCE_SMOOTH_COUNT)

    frame_count = 0
    writer      = None
    W, H        = None, None

    # ── Weapon state ─────────────────────────────────────────
    weapon_relax_counter  = 0
    weapon_mode_activated = False   # permanent once weapon seen

    # ── Violence lock state ───────────────────────────────────
    # Once a fight class is detected with enough confidence,
    # these lock in permanently for the rest of the video.
    violence_mode_activated  = False   # permanent flag
    locked_violence_label    = None    # e.g. "Fight"
    locked_violence_conf     = 0.0     # confidence at the time of locking

    timestamp  = int(time.time())
    avi_path   = os.path.join(OUTPUT_FOLDER, f"processed_{timestamp}.avi")
    mp4_path   = os.path.join(OUTPUT_FOLDER, f"processed_{timestamp}.mp4")

    # These track the final per-frame values returned at the end
    final_label      = "NonFight"
    final_confidence = 0.0

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

        # ================= YOLO WEAPON DETECTION =================
        weapon_boxes = []

        if frame_count % YOLO_STRIDE == 0:
            results = weapon_model(frame, imgsz=640, verbose=False)

            for r in results:
                for box in r.boxes:
                    conf_w = float(box.conf[0])
                    if conf_w > WEAPON_CONF_THRESHOLD:
                        weapon_mode_activated = True
                        weapon_relax_counter  = WEAPON_RELAX_FRAMES
                        cls_id     = int(box.cls[0])
                        class_name = weapon_model.names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        weapon_boxes.append((x1, y1, x2, y2, conf_w, class_name))

            # Relax weapon mode after N frames of no detection
            if weapon_mode_activated and weapon_relax_counter > 0:
                weapon_relax_counter -= YOLO_STRIDE
            if weapon_relax_counter <= 0:
                weapon_mode_activated = False

        # ================= VIOLENCE DETECTION =================
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_buffer.append(resized)

        if len(frame_buffer) == CLIP_LEN:
            clip = preprocess_clip(list(frame_buffer))

            with torch.inference_mode():
                outputs = model(clip)
                probs   = torch.softmax(outputs, dim=1)
                conf_v, pred = torch.max(probs, 1)

            predicted_label = class_names[pred.item()]
            predicted_conf  = conf_v.item()

            violence_history.append(predicted_label)
            smoothed_label = Counter(violence_history).most_common(1)[0][0]

            # ── PERMANENT VIOLENCE LOCK ───────────────────────
            # If a fight class is detected with sufficient confidence
            # AND we haven't already locked a higher-priority class,
            # lock it in for the rest of the video.
            if (
                not violence_mode_activated
                and predicted_label in VIOLENCE_LOCK_CLASSES
                and predicted_conf >= VIOLENCE_LOCK_THRESHOLD
            ):
                violence_mode_activated = True
                locked_violence_label   = predicted_label
                locked_violence_conf    = predicted_conf

            # Once locked, keep updating confidence with the smoothed
            # window so the % shown is meaningful (not just one frame).
            if violence_mode_activated:
                # Recalculate average confidence for the locked label
                # across recent history so it reflects recent frames.
                relevant = [
                    list(violence_history)[i]
                    for i in range(len(violence_history))
                    if list(violence_history)[i] == locked_violence_label
                ]
                # Use the raw predicted_conf when the current frame
                # agrees with locked label, else keep last known value.
                if predicted_label == locked_violence_label:
                    locked_violence_conf = (locked_violence_conf * 0.7
                                            + predicted_conf * 0.3)

                violence_label = locked_violence_label
                violence_conf  = locked_violence_conf
            else:
                violence_label = smoothed_label
                violence_conf  = predicted_conf

        else:
            violence_label = "Processing..."
            violence_conf  = 0.0

        # ================= FINAL LABEL LOGIC =================
        if weapon_mode_activated:
            # Weapon takes highest priority
            final_label      = f"Weaponized - {violence_label}"
            final_confidence = violence_conf
            label_color      = (0, 0, 255)         # red
        elif violence_mode_activated:
            # Permanent fight lock
            final_label      = violence_label
            final_confidence = violence_conf
            label_color      = (0, 120, 255)        # orange
        else:
            # Normal smoothed prediction
            final_label      = violence_label
            final_confidence = violence_conf
            label_color      = (0, 255, 0) if violence_label == "NonFight" else (0, 165, 255)

        # ================= DRAW WEAPON BOXES =================
        for (x1, y1, x2, y2, conf_w, class_name) in weapon_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(output,
                        f"{class_name} {conf_w:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        # ================= DRAW LABEL =================
        display_text = f"{final_label} ({final_confidence * 100:.1f}%)"
        cv2.putText(output,
                    display_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    label_color,
                    3)

        writer.write(output)

    vid.release()
    writer.release()

    # Re-encode to browser-compatible mp4
    subprocess.run([
        "ffmpeg", "-y",
        "-i", avi_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        mp4_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(avi_path):
        os.remove(avi_path)

    return final_label, final_confidence, os.path.basename(mp4_path)