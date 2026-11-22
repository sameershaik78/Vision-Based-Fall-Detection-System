import cv2
from ultralytics import YOLO

print("Loading YOLO Pose model...")

model = YOLO("yolo11n-pose.pt")

print("Opening camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    print("Check whether another application is using your camera.")
    exit()

print("Camera started successfully.")
print("Press Q to exit.")

while True:

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Vision-Based Fall Detection - YOLO Pose",
        annotated_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Camera stopped.")