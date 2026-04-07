# PPE Detection System

A real-time Personal Protective Equipment (PPE) detection system using YOLOv8 that identifies workers wearing proper safety gear (vests and hardhats) in video streams.

## Overview

This project uses YOLOv8 object detection to:

- Detect persons in video frames
- Identify safety vests and hardhats
- Determine if each person has complete PPE (both vest AND hardhat)
- Display real-time compliance status with visual labels

## Features

- ✅ Real-time PPE detection using YOLOv8
- ✅ Confidence score filtering for accurate detections
- ✅ Visual feedback: Green boxes for compliant workers, Red for non-compliant
- ✅ Console logging with detailed detection information
- ✅ MP4 video output with annotations
- ✅ Linux/Windows compatibility

## Project Structure

```
.
├── app.py                              # Main PPE detection script with confidence filtering
├── app2.py                             # Alternative detection script (all detections)
├── models/                             # Pre-trained YOLOv8 models
│   ├── best.pt                         # Primary detection model
│   ├── best1.pt, best2.pt, best4.pt   # Alternative models
│   ├── ppe.pt, ppe1.pt, ppe2.pt       # PPE-specific models
├── data/                               # Input data
│   ├── ppe3.mp4                        # Sample video file
├── output/                             # Output directory
│   ├── ppe_detection/                  # Detection results
│   └── ppe_video_detection/            # Video outputs
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- Ultralytics YOLOv8
- NumPy
- PyTorch

## Installation

1. **Clone or navigate to the project directory:**

   ```bash
   cd /home/tr_25-043_gokul/Desktop/Practise
   ```

2. **Create and activate a virtual environment (optional):**

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Main Detection Script (app.py)

Runs PPE detection with confidence filtering for person detections (> 0.35):

```bash
python3 app.py
```

**Features:**

- Filters person detections with confidence > 0.35
- No confidence threshold for vest/hardhat detection
- Displays "WITH PPE" (green) or "WITHOUT PPE" (red) labels
- Console output shows individual person compliance status

### Alternative Script (app2.py)

Runs PPE detection with all detections and confidence display:

```bash
python3 app2.py
```

**Features:**

- Confidence score filtering for all detection classes
- Shows confidence percentage for each detection
- Different visual labeling approach

## Configuration

### Video Input

Update the `video_path` variable in the script:

```python
video_path = Path("data/ppe3.mp4")
```

### Video Output

Modify the `output_video_path`:

```python
output_video_path = output_dir / "ppe_detection_output_21.mp4"
```

### Confidence Threshold

Adjust the person detection confidence threshold in app.py:

```python
if confidence > 0.35:  # Change this value
    persons.append((x1, y1, x2, y2))
```

### Model Selection

Change the model in the script:

```python
model = YOLO("models/best.pt")  # Try other models like ppe.pt
```

## Output

The script generates:

- **Video file** with bounding boxes and PPE status labels
- **Console logs** with per-frame detection statistics
- **Frame-by-frame person compliance status**

Example console output:

```
Person 1: HAS VEST ✓ | HAS HARDHAT ✓
Person 2: NO VEST ✗ | HAS HARDHAT ✓
Frame 1: 2 persons, 2 vests, 2 hardhats
```

## Key Functions

- `center_in_box(item_box, person_box)`: Checks if an equipment item (vest/hardhat) overlaps with a person's bounding box

## Detection Classes

The model detects the following classes:

- `person`: Workers/individuals
- `vest`: Safety vests
- `hardhat`: Safety helmets
- `no_vest`: Absence of safety vest
- `no_hardhat`: Absence of safety helmet

## Color Coding

- **Green box + "WITH PPE ✓"**: Person has both vest and hardhat
- **Red box + "WITHOUT PPE ✗"**: Person missing vest or hardhat

## Notes

- The script uses a platform compatibility patch for Windows-trained models on Linux
- Video processing may take time depending on video length and frame resolution
- Ensure sufficient disk space for video output

## Troubleshooting

**1. Model not found error:**

```bash
FileNotFoundError: models/best.pt
```

Solution: Ensure the model file exists in the `models/` directory.

**2. Video file not found:**

```bash
FileNotFoundError: data/ppe3.mp4
```

Solution: Verify the video path is correct.

**3. YOLO inference errors on Linux:**
The script includes a compatibility patch for Windows-trained models. No additional action needed.

## Future Enhancements

- [ ] Real-time camera feed detection
- [ ] Multi-person PPE analytics dashboard
- [ ] Frame-by-frame compliance report generation
- [ ] Database logging for compliance tracking
- [ ] SMS/Email alerts for non-compliance
- [ ] Support for different PPE types (gloves, safety glasses, etc.)

## License

This project uses YOLOv8 from Ultralytics. Please refer to their licensing terms.

## Support

For issues or questions, ensure all dependencies are installed and models are in the correct location.
