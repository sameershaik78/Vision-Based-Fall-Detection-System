import cv2
import os
import time
from fall_detector import FallDetector
from database import (
    initialize_database,
    add_fall_event
)
from datetime import datetime

from ultralytics import YOLO

from fall_detector import FallDetector


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n-pose.pt"

FALL_THRESHOLD = 0.70

FALL_CONFIRMATION_TIME = 2.0

ALARM_COOLDOWN = 10


# ============================================================
# CREATE REQUIRED FOLDER
# ============================================================

EVENT_FOLDER = os.path.join(
    "data",
    "fall_events"
)

os.makedirs(
    EVENT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO Pose model...")

model = YOLO(MODEL_PATH)

print("YOLO Pose model loaded successfully.")
# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()

# ============================================================
# CREATE FALL DETECTOR
# ============================================================

detector = FallDetector(
    fall_threshold=FALL_THRESHOLD,
    confirmation_time=FALL_CONFIRMATION_TIME
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


print()
print("==============================================")
print(" VISION-BASED FALL DETECTION SYSTEM")
print("==============================================")
print("Camera started.")
print("Press Q to exit.")
print()


# ============================================================
# ALARM FUNCTION
# ============================================================

def play_alarm():

    try:

        import winsound

        winsound.Beep(
            1000,
            1000
        )

        winsound.Beep(
            1200,
            1000
        )

        winsound.Beep(
            1000,
            1000
        )

    except Exception as error:

        print(
            "Alarm could not be played:",
            error
        )


# ============================================================
# EVENT VARIABLES
# ============================================================

last_alarm_time = 0

fall_event_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    frame_height, frame_width = frame.shape[:2]


    # ========================================================
    # YOLO POSE
    # ========================================================

    results = model(
        frame,
        verbose=False
    )


    annotated_frame = results[0].plot()


    # ========================================================
    # PROCESS PEOPLE
    # ========================================================

    if results[0].keypoints is not None:

        keypoints_data = (
            results[0]
            .keypoints
            .xy
            .cpu()
            .numpy()
        )


        for person_id, keypoints in enumerate(
            keypoints_data
        ):

            person_keypoints = [
                tuple(point)
                for point in keypoints
            ]


            # =================================================
            # FALL DETECTION
            # =================================================

            result = detector.detect(
                person_keypoints,
                frame_width,
                frame_height
            )


            status = result["status"]

            posture = result["posture"]

            confidence = result["confidence"]


            # =================================================
            # DISPLAY PERSON INFORMATION
            # =================================================

            y = 40 + person_id * 120


            cv2.putText(
                annotated_frame,
                f"Person: {person_id + 1}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )


            cv2.putText(
                annotated_frame,
                f"Posture: {posture}",
                (20, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            cv2.putText(
                annotated_frame,
                f"Fall Confidence: {confidence * 100:.1f}%",
                (20, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            cv2.putText(
                annotated_frame,
                f"Status: {status}",
                (20, y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # =================================================
            # FALL CONFIRMED
            # =================================================

            if result["fall_detected"]:

                current_time = time.time()


                # ---------------------------------------------
                # FALL EVENT
                # ---------------------------------------------

                if (
                    current_time - last_alarm_time
                    > ALARM_COOLDOWN
                ):

                    fall_event_count += 1


                    # -----------------------------------------
                    # TIMESTAMP
                    # -----------------------------------------

                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )


                    # -----------------------------------------
                    # SAVE FALL SNAPSHOT
                    # -----------------------------------------

                    filename = (
                        f"fall_{timestamp}_"
                        f"person_{person_id + 1}.jpg"
                    )


                    filepath = os.path.join(
                        EVENT_FOLDER,
                        filename
                    )


                    cv2.imwrite(
                        filepath,
                        annotated_frame
                    )


                    # -----------------------------------------
                    # DISPLAY EVENT
                    # -----------------------------------------

                    print()
                    print(
                        "=========================================="
                    )
                    print(
                        "🚨 FALL DETECTED!"
                    )
                    print(
                        "=========================================="
                    )

                    print(
                        f"Person: {person_id + 1}"
                    )

                    print(
                        f"Confidence: "
                        f"{confidence * 100:.2f}%"
                    )

                    print(
                        f"Time: "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    print(
                        f"Snapshot: {filepath}"
                    )

                    print(
                        "=========================================="
                    )


                    # -----------------------------------------
                    # PLAY ALARM
                    # -----------------------------------------

                    play_alarm()


                    # -----------------------------------------
                    # UPDATE COOLDOWN
                    # -----------------------------------------

                    last_alarm_time = current_time


                # ---------------------------------------------
                # VISUAL EMERGENCY ALERT
                # ---------------------------------------------

                cv2.putText(
                    annotated_frame,
                    "!!! FALL DETECTED !!!",
                    (120, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    4
                )


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    cv2.putText(
        annotated_frame,
        f"Fall Events: {fall_event_count}",
        (20, frame_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        "AI MONITORING: ACTIVE",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "Vision-Based Fall Detection System",
        annotated_frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Camera stopped.")
print(
    f"Total fall events detected: {fall_event_count}"
)