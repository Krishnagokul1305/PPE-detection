from pathlib import Path
import cv2
from ultralytics import YOLO
import pathlib
import platform

# Patch for Windows-trained models on Linux
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# ✅ Load YOLOv8 model
model = YOLO("models/best.pt")

# Paths
video_path = Path("data/ppe3.mp4")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_video_path = output_dir / "ppe_detection_output_15.mp4"

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

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()
            confidence = float(box.conf[0])

            # Skip person detections with confidence <= 0.35
            if class_name == "person" and confidence <= 0.35:
                continue

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'box': (x1, y1, x2, y2)
            })

    # Print all detections and draw on frame
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['box']
        class_name = detection['class']
        confidence = detection['confidence']

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label with class name and confidence
        label = f"{class_name} {confidence:.2f}"

        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # Draw background rectangle for text
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 5, y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

        print(f"Detection {idx + 1}: {class_name} (Confidence: {confidence:.2f})")

    print(f"Frame {frame_count}: {len(detections)} detections")

    writer.write(frame)

cap.release()
writer.release()

print("\n✅ Detection completed. Output saved to:", output_video_path)
