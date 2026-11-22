from ultralytics import YOLO

print("====================================")
print("VISION-BASED FALL DETECTION SYSTEM")
print("====================================")

print("Loading YOLO Pose model...")

model = YOLO("yolo11n-pose.pt")

print("YOLO Pose model loaded successfully!")
print("Model is ready for inference.")