import cv2
import torch
from ultralytics import YOLO

# ==========================================
# 🔧 MODEL & VIDEO PATHS
# ==========================================

MODEL_PATH = r"C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/backend/best-yolo.pt"

VIDEO_PATH = r"C:/Users/kumar/OneDrive/Desktop/TRY-3/Violence-Detetion-in-CCTV/violence-app/demo-videos/demo6.mp4"

# ==========================================
# 🚀 LOAD MODEL
# ==========================================

print("Loading custom weapon model...")
model = YOLO(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using device:", device)

print("\nModel Classes:")
print(model.names)

# ==========================================
# 🎥 OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video.")
    exit()

print("\nStarting detection... Press Q to exit.\n")

# ==========================================
# 🔎 DETECTION LOOP
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    results = model.predict(
        source=frame,
        conf=0.30,
        iou=0.6,
        verbose=False
    )

    for result in results:
        image = result.orig_img

        if result.boxes is not None:

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # Label text
                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # Alert text
                cv2.putText(
                    image,
                    "WEAPON DETECTED",
                    (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

        cv2.imshow("Weapon Detection Output", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")