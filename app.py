from pathlib import Path
import cv2
import torch
import pathlib

# Patch for loading Windows-trained models on Linux
import platform
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Helper function to check if vest center is inside person box
def center_in_box(vest_box, person_box):
    vest_x1, vest_y1, vest_x2, vest_y2 = vest_box
    person_x1, person_y1, person_x2, person_y2 = person_box
    vest_center_x = (vest_x1 + vest_x2) / 2
    vest_center_y = (vest_y1 + vest_y2) / 2
    return person_x1 <= vest_center_x <= person_x2 and person_y1 <= vest_center_y <= person_y2

# Load YOLO model (YOLOv5 format for ppe1.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/ppe1.pt', force_reload=False)

# Input/output paths
video_path = Path("data/ppe.mp4")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_video_path = output_dir / "ppe_detection_output4.mp4"

# Open video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer
writer = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# Process frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Run detection
    results = model(frame)
    
    # Data structures for this frame
    persons = []  # List of (x1, y1, x2, y2) tuples
    vests = []    # List of (x1, y1, x2, y2) tuples
    
    # Extract person and vest detections (YOLOv5 format)
    detections = results.xyxy[0]  # Get predictions for first (only) image
    if len(detections) > 0:
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            class_name = results.names[class_id].lower()
            
            if "person" in class_name:
                persons.append((x1, y1, x2, y2))
            elif "vest" in class_name and "no" not in class_name:
                vests.append((x1, y1, x2, y2))
    
    # Draw person boxes with color logic (vest assignment)
    for idx, person_box in enumerate(persons):
        # Check if this person has a vest
        has_vest = any(center_in_box(vest_box, person_box) for vest_box in vests)
        
        # Color: green if has vest, red if no vest
        color = (0, 255, 0) if has_vest else (0, 0, 255)
        
        x1, y1, x2, y2 = person_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Print status for each person
        status = "HAS VEST ✓" if has_vest else "NO VEST ✗"
        print(f"  Person {idx + 1}: {status}")
    
    # Print frame summary
    print(f"Frame {frame_count}: {len(persons)} persons, {len(vests)} vests")
    
    # Write frame to output video
    writer.write(frame)
    
cap.release()
writer.release()

print("\nDetection completed. Output saved to:", output_video_path)
