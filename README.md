# Vision-Based Fall Detection System for Elderly Care

A professional computer-vision application for detecting human falls from a live camera stream using YOLO pose estimation, posture/trajectory analysis, a Flask dashboard, and SQLite event logging.

## Project Modules

1. **Image Enhancement** – improves incoming frames before analysis.
2. **Pose Feature Extraction** – extracts human pose/keypoint information with YOLO pose estimation.
3. **Activity / Fall Classification** – analyzes body geometry and temporal motion to identify likely falls.
4. **User Interface Design** – provides a live monitoring dashboard through Flask.
5. **Emergency Alert Module** – raises a confirmed fall alert and stores a snapshot/event record.

## Requirements

- Windows 10/11
- Python 3.11 (recommended)
- Webcam
- Internet connection for first-time package installation

## Installation

Open PowerShell in this folder:

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open the dashboard at:

`http://127.0.0.1:5000`

## Notes

- `yolo11n-pose.pt` is included in the project root.
- Fall snapshots are stored under `data/fall_events/`.
- The SQLite database is created automatically at `data/fall_detection.db`.
- Allow camera access when Windows asks for permission.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

Make sure the virtual environment is active, then run:

```powershell
python -m pip install opencv-python
```

### `ModuleNotFoundError: No module named 'ultralytics'`

Run:

```powershell
python -m pip install ultralytics
```

### Camera does not open

Close other applications using the webcam and try again. On Windows, check **Settings → Privacy & security → Camera** and allow desktop apps to access the camera.


## Emergency Alert Message

When a confirmed fall is detected, the dashboard displays a prominent **🚨 EMERGENCY ALERT** containing:
- Person number
- Detection confidence
- Detection time
- A clear instruction to check the person immediately

The alert also triggers the existing server-side sound alarm and a browser alert tone when supported.  
The **Acknowledge** button clears the dashboard alert and returns Module 5 to `READY`.
