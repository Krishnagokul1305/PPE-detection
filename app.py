from pathlib import Path
import cv2
from ultralytics import YOLO
import pathlib
import platform

# Patch for Windows-trained models on Linux
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Helper functions
def center_in_box(item_box, person_box):
    vx1, vy1, vx2, vy2 = item_box
    px1, py1, px2, py2 = person_box

    cx = (vx1 + vx2) / 2
    cy = (vy1 + vy2) / 2

    return px1 <= cx <= px2 and py1 <= cy <= py2

# ✅ Load YOLOv8 model
model = YOLO("models/best.pt")

# Paths
video_path = Path("/home/tr_25-043_gokul/Downloads/PPE/ppe_personal_protection_equipment_dataset_for_object_detection_720P.mp4")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_video_path = output_dir / "ppe_detection_output_22.mp4"

# Video setup
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ✅ YOLOv8 inference
    result = model(frame, verbose=False)[0]

    persons = []
    vests = []
    hardhats = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()
            confidence = float(box.conf[0])

            if "person" in class_name:
                # Skip person detections with confidence <= 0.35
                if confidence > 0.35:
                    persons.append((x1, y1, x2, y2))
            elif "vest" in class_name and "no" not in class_name:
                vests.append((x1, y1, x2, y2))
            elif "hardhat" in class_name and "no" not in class_name:
                hardhats.append((x1, y1, x2, y2))

    # Draw results
    for idx, person_box in enumerate(persons):
        has_vest = any(center_in_box(vb, person_box) for vb in vests)
        has_hardhat = any(center_in_box(hb, person_box) for hb in hardhats)

        # Green if has BOTH vest AND hardhat, Red otherwise
        color = (0, 255, 0) if (has_vest and has_hardhat) else (0, 0, 255)

        x1, y1, x2, y2 = person_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add label showing PPE status
        ppe_label = "PPE present" if (has_vest and has_hardhat) else "No PPE"
        cv2.putText(frame, ppe_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        vest_status = "HAS VEST ✓" if has_vest else "NO VEST ✗"
        hardhat_status = "HAS HARDHAT ✓" if has_hardhat else "NO HARDHAT ✗"
        print(f"Person {idx + 1}: {vest_status} | {hardhat_status}")

    print(f"Frame {frame_count}: {len(persons)} persons, {len(vests)} vests, {len(hardhats)} hardhats")

    writer.write(frame)

cap.release()
writer.release()

print("\n✅ Detection completed. Output saved to:", output_video_path)