import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque, Counter
from torchvision.models.video import r3d_18
from ultralytics import YOLO




# ============================================================
# NOTE: winsound is Windows-only. We handle it gracefully.
# ============================================================
try:
    import winsound
    BEEP_AVAILABLE = True
except ImportError:
    BEEP_AVAILABLE = False

# ==========================
# PERFORMANCE BOOST
# ==========================
torch.backends.cudnn.benchmark = True

# ==========================
# CONFIG
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE    = 112
CLIP_LEN    = 16
YOLO_STRIDE = 2         # Run YOLO every N frames

# ── Model paths ───────────────────────────────────────────────
MODEL_PATH = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/backend/best-violence.pth"
YOLO_PATH  = "C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/backend/best-yolo.pt"

OUTPUT_PATH = "output_live.mp4"   # only used when SOURCE is a video file

# ── Detection config ──────────────────────────────────────────
SMOOTHING_WINDOW      = 5    # majority-vote window
WEAPON_CONF_THRESHOLD = 0.8  # min YOLO confidence to count as weapon
WEAPON_RELAX_FRAMES   = 30   # frames before weapon mode can deactivate
ALERT_COOLDOWN        = 3    # seconds between beep alerts

ALERT_CLASSES        = ["Fight", "HockeyFight"]
WEAPON_CLASSES_NAMES = None  # filled from YOLO model at runtime

# ── Camo detection config ──────────────────────────────────────
# Camo Studio virtual camera is identified by higher resolution
# than a typical built-in webcam. We scan indices 0–5 and pick
# the one with the highest resolution; if it differs from index 0
# we treat it as the Camo camera. You can also hardcode a fallback.
MAX_CAMERA_SCAN = 6          # how many indices to probe


# ==========================
# AUTO CAMERA SELECTOR
# ==========================
def find_best_camera():
    """
    Scans camera indices 0 to MAX_CAMERA_SCAN-1.
    Returns (index, label) where label is 'Camo/Android' or 'Built-in Webcam'.

    Strategy:
      1. Collect all available cameras and their resolutions.
      2. If more than one camera is found, pick the one with the
         highest resolution that is NOT index 0 — that is almost
         certainly the Camo virtual camera.
      3. If only one camera exists (index 0), use that as fallback.
    """
    available = []  # list of (index, width, height)

    print("[CAMERA SCAN] Probing available cameras...")
    for i in range(MAX_CAMERA_SCAN):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)   # CAP_DSHOW = faster init on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, w, h))
                print(f"  [✓] Index {i}: {w}x{h}")
            cap.release()
        else:
            print(f"  [✗] Index {i}: not available")

    if not available:
        raise RuntimeError("[ERROR] No camera found at all!")

    if len(available) == 1:
        idx, w, h = available[0]
        print(f"\n[CAMERA] Only one camera found. Using index {idx} ({w}x{h}) — Built-in Webcam\n")
        return idx, "Built-in Webcam"

    # Prefer non-zero index with highest resolution (Camo virtual cam)
    non_default = [(i, w, h) for i, w, h in available if i != 0]
    if non_default:
        # pick highest resolution among non-default cameras
        best = max(non_default, key=lambda x: x[1] * x[2])
        idx, w, h = best
        print(f"\n[CAMERA] Camo/Android camera detected at index {idx} ({w}x{h}). Using it!\n")
        return idx, "Camo/Android Camera"

    # All found cameras are index 0 (shouldn't happen but safe fallback)
    idx, w, h = available[0]
    print(f"\n[CAMERA] Falling back to index {idx} ({w}x{h}) — Built-in Webcam\n")
    return idx, "Built-in Webcam"


# ==========================
# LOAD 3D CNN MODEL
# ==========================
print(f"[INFO] Device: {DEVICE}")
print("[INFO] Loading R3D-18 violence model...")

model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
class_names = checkpoint["class_names"]

model.to(DEVICE)
model.eval()
print(f"[INFO] Violence classes: {class_names}")

# ==========================
# LOAD YOLO MODEL
# ==========================
print("[INFO] Loading YOLOv8 weapon model...")
weapon_model = YOLO(YOLO_PATH)
print(f"[INFO] Weapon classes: {weapon_model.names}")

# ==========================
# PREPROCESS CLIP
# ==========================
def preprocess_clip(frames):
    frames = np.array(frames) / 255.0
    frames = np.transpose(frames, (3, 0, 1, 2))   # (C, T, H, W)
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==========================
# DRAW HUD OVERLAY
# ==========================
def draw_hud(frame, label, confidence, weapon_active, fps_val, W, H, cam_label):
    overlay = frame.copy()

    # ── colour logic ─────────────────────────────────────────
    if weapon_active:
        bar_color  = (0, 0, 220)
        text_color = (0, 0, 255)
        threat_txt = "THREAT: CRITICAL"
    elif label in ALERT_CLASSES:
        bar_color  = (0, 120, 255)
        text_color = (0, 165, 255)
        threat_txt = "THREAT: HIGH"
    else:
        bar_color  = (30, 180, 30)
        text_color = (0, 220, 60)
        threat_txt = "THREAT: NONE"

    # ── top label box ────────────────────────────────────────
    box_h = int(H * 0.085)
    cv2.rectangle(overlay, (0, 0), (W, box_h), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    display = f"{label}  {confidence*100:.1f}%"
    font_scale = max(0.55, W / 1200)
    cv2.putText(frame, display,
                (int(W * 0.02), int(box_h * 0.72)),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale, text_color, 2, cv2.LINE_AA)

    # ── Camera source label (top-centre) ─────────────────────
    cam_txt = f"[ {cam_label} ]"
    (cw, _), _ = cv2.getTextSize(cam_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, cam_txt,
                ((W - cw) // 2, int(box_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # ── FPS top-right ─────────────────────────────────────────
    fps_txt = f"FPS: {fps_val:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, fps_txt,
                (W - tw - int(W * 0.02), int(box_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (120, 120, 120), 1, cv2.LINE_AA)

    # ── bottom threat strip ───────────────────────────────────
    strip_h = int(H * 0.055)
    cv2.rectangle(frame, (0, H - strip_h), (W, H), bar_color, -1)
    (tw2, th2), _ = cv2.getTextSize(threat_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, threat_txt,
                ((W - tw2) // 2, H - strip_h + (strip_h + th2) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # ── WEAPON badge ─────────────────────────────────────────
    if weapon_active:
        badge = "WEAPON DETECTED"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx = W - bw - int(W * 0.02)
        by = H - strip_h - int(H * 0.04)
        cv2.rectangle(frame,
                      (bx - 6, by - bh - 4),
                      (bx + bw + 6, by + 6),
                      (0, 0, 180), -1)
        cv2.putText(frame, badge, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ==========================
# AUTO-SELECT CAMERA
# ==========================
SOURCE, CAM_LABEL = find_best_camera()
is_webcam = True   # always True since we're always using a camera index (int)

print(f"[INFO] Opening camera index {SOURCE} ({CAM_LABEL}) ...")
cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Cannot open camera at index: {SOURCE}")

fps_src = cap.get(cv2.CAP_PROP_FPS)
if not fps_src or fps_src == 0:
    fps_src = 30

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Resolution: {W}x{H}  |  FPS: {fps_src:.1f}")
print(f"[INFO] Active camera: {CAM_LABEL}\n")

# ==========================
# STATE
# ==========================
frame_buffer       = deque(maxlen=CLIP_LEN)
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
confidence_history = deque(maxlen=SMOOTHING_WINDOW)

current_label      = "Processing..."
current_confidence = 0.0

weapon_active        = False
weapon_relax_counter = 0
last_yolo_boxes      = []   # [(x1,y1,x2,y2, conf, cls_name), ...]

alert_cooldown_until = 0
frame_count          = 0

# FPS tracking
fps_timer   = time.time()
fps_display = 0.0
fps_counter = 0

print("[INFO] Running — press ESC or Q to quit.\n")

# ==========================
# MAIN LOOP
# ==========================
while True:
    grabbed, frame = cap.read()
    if not grabbed:
        print("[WARN] Frame grab failed — retrying...")
        time.sleep(0.05)
        continue

    frame_count += 1
    fps_counter  += 1

    # ── Rolling FPS ─────────────────────────────────────────
    now = time.time()
    if now - fps_timer >= 1.0:
        fps_display = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer   = now

    output = frame.copy()

    # ── YOLOv8 Weapon Detection ──────────────────────────────
    if frame_count % YOLO_STRIDE == 0:
        results = weapon_model(frame, imgsz=640, verbose=False)
        last_yolo_boxes = []

        for r in results:
            for box in r.boxes:
                conf_w = float(box.conf[0])
                if conf_w >= WEAPON_CONF_THRESHOLD:
                    weapon_active        = True
                    weapon_relax_counter = WEAPON_RELAX_FRAMES
                    cls_id   = int(box.cls[0])
                    cls_name = weapon_model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    last_yolo_boxes.append((x1, y1, x2, y2, conf_w, cls_name))

        # relax weapon mode if nothing detected for N frames
        if weapon_active and weapon_relax_counter > 0:
            weapon_relax_counter -= YOLO_STRIDE
        if weapon_relax_counter <= 0:
            weapon_active = False

    # ── Draw YOLO boxes ──────────────────────────────────────
    for (x1, y1, x2, y2, conf_w, cls_name) in last_yolo_boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output,
                    f"{cls_name} {conf_w:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 2, cv2.LINE_AA)

    # ── R3D-18 Violence Classification ──────────────────────
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_buffer.append(resized)

    if len(frame_buffer) == CLIP_LEN:
        clip = preprocess_clip(list(frame_buffer))

        with torch.inference_mode():
            out   = model(clip)
            probs = torch.softmax(out, dim=1)
            conf_v, pred = torch.max(probs, 1)

        predicted_label = class_names[pred.item()]
        prediction_history.append(predicted_label)
        confidence_history.append(conf_v.item())

        # Majority-vote smoothing
        current_label = Counter(prediction_history).most_common(1)[0][0]

        # Average confidence for the winning label
        relevant = [
            confidence_history[i]
            for i in range(len(prediction_history))
            if prediction_history[i] == current_label
        ]
        current_confidence = sum(relevant) / len(relevant) if relevant else 0.0

    # ── Final label string ───────────────────────────────────
    display_label = f"Weaponized - {current_label}" if weapon_active else current_label

    # ── HUD Overlay ──────────────────────────────────────────
    output = draw_hud(output, display_label, current_confidence,
                      weapon_active, fps_display, W, H, CAM_LABEL)

    # ── Beep Alert (Windows only) ─────────────────────────────
    if BEEP_AVAILABLE:
        is_violent = current_label in ALERT_CLASSES or weapon_active
        if is_violent and now > alert_cooldown_until:
            alert_cooldown_until = now + ALERT_COOLDOWN
            winsound.Beep(1000, 400)

    # ── Display ───────────────────────────────────────────────
    window_title = f"VIGIL.AI — {CAM_LABEL}  |  ESC / Q to quit"
    cv2.imshow(window_title, output)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        print("[INFO] Quit signal received.")
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Processed {frame_count} frames. Done.")