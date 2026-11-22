import cv2
import os
import time
import threading
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, send_from_directory
from ultralytics import YOLO

from fall_detector import FallDetector
from image_enhancement import ImageEnhancer
from database import initialize_database, add_fall_event, get_all_fall_events, get_statistics, acknowledge_event

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")

FALL_THRESHOLD = 0.70
FALL_CONFIRMATION_TIME = 2.0
ALARM_COOLDOWN = 10

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

EVENT_FOLDER = os.path.join(BASE_DIR, "data", "fall_events")

os.makedirs(
    EVENT_FOLDER,
    exist_ok=True
)

initialize_database()
# ============================================================
# MODULE 1 - IMAGE ENHANCEMENT
# ============================================================

enhancer = ImageEnhancer(
    width=1280,
    height=720
)

# ============================================================
# MODULE 1 - IMAGE ENHANCEMENT
# ============================================================

def enhance_frame(frame):
    """
    Module 1:
    Enhances the raw camera frame before sending it
    to the YOLO pose model.

    Processing:
    1. Resize
    2. Gaussian noise reduction
    3. LAB color conversion
    4. CLAHE contrast enhancement
    5. Convert back to BGR
    """

    try:

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        enhanced = cv2.resize(
            frame,
            (CAMERA_WIDTH, CAMERA_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Gaussian Blur
        # ----------------------------------------------------

        enhanced = cv2.GaussianBlur(
            enhanced,
            (5, 5),
            0
        )

        # ----------------------------------------------------
        # Convert BGR -> LAB
        # ----------------------------------------------------

        lab = cv2.cvtColor(
            enhanced,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        # ----------------------------------------------------
        # CLAHE Contrast Enhancement
        # ----------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(
            l_channel
        )

        # ----------------------------------------------------
        # Merge channels
        # ----------------------------------------------------

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel
            )
        )

        # ----------------------------------------------------
        # LAB -> BGR
        # ----------------------------------------------------

        enhanced_frame = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        return enhanced_frame

    except Exception as error:

        print(
            "Image enhancement error:",
            error
        )

        # If enhancement fails, return original frame
        return frame


# ============================================================
# LOAD YOLO POSE MODEL
# ============================================================

print()
print("=" * 60)
print("Loading YOLO Pose model...")
print("=" * 60)

try:

    model = YOLO(MODEL_PATH)

    print(
        "YOLO Pose model loaded successfully."
    )

except Exception as error:

    print(
        "ERROR: Could not load YOLO Pose model."
    )

    print(
        "Reason:",
        error
    )

    model = None


# ============================================================
# CAMERA
# ============================================================

print()
print("Opening webcam...")

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
)


if camera.isOpened():

    print(
        "Camera connected successfully."
    )

    camera_status = "CONNECTED"

else:

    print(
        "WARNING: Could not open webcam."
    )

    camera_status = "DISCONNECTED"


# ============================================================
# FALL DETECTOR
# ============================================================

detector = FallDetector(
    fall_threshold=FALL_THRESHOLD,
    confirmation_time=FALL_CONFIRMATION_TIME
)


# ============================================================
# SYSTEM STATE
# ============================================================

system_state = {

    # --------------------------------------------------------
    # System
    # --------------------------------------------------------

    "system": "ACTIVE",

    # --------------------------------------------------------
    # Five project modules
    # --------------------------------------------------------

    "module1": "ACTIVE",
    "module2": "WAITING",
    "module3": "WAITING",
    "module4": "ACTIVE",
    "module5": "READY",

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    "camera": camera_status,

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    "fall_status": "NORMAL",
    "posture": "UNKNOWN",
    "confidence": 0.0,

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    "fps": 0.0,
    "uptime": 0,

    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    "person_detected": False,
    "person_count": 0,

    # --------------------------------------------------------
    # Fall events
    # --------------------------------------------------------

    "fall_events": 0,
    "last_event": None,
    "last_snapshot": None,

    # --------------------------------------------------------
    # Emergency alert message
    # --------------------------------------------------------

    "alert_active": False,
    "alert_message": "",
    "alert_time": None,
    "alert_person": None,
    "alert_confidence": 0.0,

    # --------------------------------------------------------
    # Module 2 information
    # --------------------------------------------------------

    "pose_keypoints": 0,
    "body_angle": 0.0,
    "aspect_ratio": 0.0,
    "vertical_speed": 0.0,

    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    "activity": "UNKNOWN"

}


# ============================================================
# GLOBAL VARIABLES
# ============================================================

running = True

last_alarm_time = 0

fall_event_count = 0

latest_frame = None

frame_lock = threading.Lock()

state_lock = threading.Lock()

start_time = time.time()


# ============================================================
# ALARM
# ============================================================

def play_alarm():

    try:

        import winsound

        winsound.Beep(
            1000,
            800
        )

        winsound.Beep(
            1200,
            800
        )

        winsound.Beep(
            1000,
            800
        )

    except Exception as error:

        print(
            "Alarm could not be played:",
            error
        )


# ============================================================
# FALL EVENT HANDLER
# ============================================================

def handle_fall_event(
    annotated_frame,
    person_id,
    confidence
):

    global last_alarm_time
    global fall_event_count

    current_time = time.time()

    # --------------------------------------------------------
    # Alarm cooldown
    # --------------------------------------------------------

    if (
        current_time - last_alarm_time
        <= ALARM_COOLDOWN
    ):

        return

    # --------------------------------------------------------
    # Increment event counter
    # --------------------------------------------------------

    fall_event_count += 1

    timestamp = datetime.now()

    timestamp_string = timestamp.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    filename = (
        f"fall_{timestamp_string}_"
        f"person_{person_id + 1}.jpg"
    )

    filepath = os.path.join(
        EVENT_FOLDER,
        filename
    )

    # --------------------------------------------------------
    # Save evidence
    # --------------------------------------------------------

    try:

        saved = cv2.imwrite(
            filepath,
            annotated_frame
        )

        if not saved:

            print(
                "WARNING: Could not save fall evidence."
            )

    except Exception as error:

        print(
            "Evidence save error:",
            error
        )

    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    with state_lock:

        system_state["fall_status"] = (
            "FALL DETECTED"
        )

        system_state["module5"] = (
            "ALERT TRIGGERED"
        )

        system_state["last_event"] = (
            timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        system_state["last_snapshot"] = filename

        system_state["fall_events"] = (
            fall_event_count
        )

        # Emergency alert shown on the dashboard
        system_state["alert_active"] = True
        system_state["alert_message"] = (
            f"🚨 EMERGENCY ALERT: Fall detected for Person {person_id + 1}! "
            "Please check the person immediately."
        )
        system_state["alert_time"] = timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        system_state["alert_person"] = person_id + 1
        system_state["alert_confidence"] = round(
            confidence * 100, 2
        )

    # --------------------------------------------------------
    # Console alert
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("🚨 FALL DETECTED!")
    print("=" * 60)

    print(
        f"Person      : {person_id + 1}"
    )

    print(
        f"Confidence  : {confidence * 100:.2f}%"
    )

    print(
        "Time        :",
        timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        f"Snapshot    : {filepath}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Sound alarm
    # --------------------------------------------------------

    alarm_thread = threading.Thread(
        target=play_alarm,
        daemon=True
    )

    alarm_thread.start()

    # --------------------------------------------------------
    # Update cooldown
    # --------------------------------------------------------

    last_alarm_time = current_time


# ============================================================
# ACTIVITY ESTIMATION
# ============================================================

def determine_activity(
    posture,
    vertical_speed
):

    """
    Converts posture information into a simple
    human-readable activity label.
    """

    if posture == "STANDING":

        return "STANDING"

    if posture == "UPRIGHT":

        if abs(vertical_speed) > 40:

            return "WALKING"

        return "STANDING"

    if posture == "BENDING / SITTING":

        return "SITTING"

    if posture == "HORIZONTAL":

        return "LYING"

    if posture == "LYING / LOW":

        return "LYING"

    return "ANALYZING"


# ============================================================
# CAMERA PROCESSING THREAD
# ============================================================

def process_camera():

    global latest_frame
    global running

    previous_time = time.time()

    print()
    print(
        "Camera processing thread started."
    )

    while running:

        # ====================================================
        # READ CAMERA FRAME
        # ====================================================

        success, frame = camera.read()
        

        if not success:

            with state_lock:

                system_state["camera"] = (
                    "DISCONNECTED"
                )

            time.sleep(0.1)

            continue

        with state_lock:

            system_state["camera"] = (
                "CONNECTED"
            )

        # ====================================================
        # ORIGINAL FRAME SIZE
        # ====================================================

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ====================================================
        # MODULE 1
        # IMAGE ENHANCEMENT
        # ====================================================

        enhanced_frame = enhance_frame(
            frame
        )

        with state_lock:

            system_state["module1"] = (
                "ACTIVE"
            )

        # ====================================================
        # MODULE 2
        # POSE FEATURE EXTRACTION
        # ====================================================

        if model is None:

            with state_lock:

                system_state["module2"] = (
                    "ERROR"
                )

                system_state["module3"] = (
                    "WAITING"
                )

            annotated_frame = enhanced_frame.copy()

            cv2.putText(
                annotated_frame,
                "YOLO MODEL NOT LOADED",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        else:

            try:

                # ------------------------------------------------
                # YOLO POSE INFERENCE
                # ------------------------------------------------

                results = model(
                    enhanced_frame,
                    verbose=False
                )

                # ------------------------------------------------
                # Draw YOLO annotations
                # ------------------------------------------------

                annotated_frame = (
                    results[0].plot()
                )

                with state_lock:

                    system_state["module2"] = (
                        "ACTIVE"
                    )

            except Exception as error:

                print(
                    "YOLO processing error:",
                    error
                )

                annotated_frame = (
                    enhanced_frame.copy()
                )

                with state_lock:

                    system_state["module2"] = (
                        "ERROR"
                    )

                    system_state["module3"] = (
                        "WAITING"
                    )

                # Continue loop

                continue

            # ====================================================
            # RESET CURRENT FRAME STATE
            # ====================================================

            current_fall_status = "NORMAL"

            current_posture = "UNKNOWN"

            current_confidence = 0.0

            current_activity = "UNKNOWN"

            current_body_angle = 0.0

            current_aspect_ratio = 0.0

            current_vertical_speed = 0.0

            person_detected = False

            person_count = 0

            keypoint_count = 0

            # ====================================================
            # EXTRACT POSE KEYPOINTS
            # ====================================================

            if results[0].keypoints is not None:

                try:

                    keypoints_data = (
                        results[0]
                        .keypoints
                        .xy
                        .cpu()
                        .numpy()
                    )

                except Exception as error:

                    print(
                        "Keypoint extraction error:",
                        error
                    )

                    keypoints_data = []

                person_count = len(
                    keypoints_data
                )

                if person_count > 0:

                    person_detected = True

                    with state_lock:

                        system_state["module2"] = (
                            "ACTIVE"
                        )

                # =================================================
                # PROCESS EACH PERSON
                # =================================================

                for person_id, keypoints in enumerate(
                    keypoints_data
                ):

                    # ------------------------------------------------
                    # Convert keypoints
                    # ------------------------------------------------

                    person_keypoints = []

                    for point in keypoints:

                        try:

                            x = float(point[0])
                            y = float(point[1])

                            # Check invalid coordinates

                            if (
                                x == 0
                                and y == 0
                            ):

                                person_keypoints.append(
                                    None
                                )

                            else:

                                person_keypoints.append(
                                    (x, y)
                                )

                        except Exception:

                            person_keypoints.append(
                                None
                            )

                    keypoint_count = max(
                        keypoint_count,
                        len(person_keypoints)
                    )

                    # ------------------------------------------------
                    # MODULE 3
                    # FALL CLASSIFICATION
                    # ------------------------------------------------

                    with state_lock:

                        system_state["module3"] = (
                            "ACTIVE"
                        )

                    try:

                        result = detector.detect(
                            person_keypoints,
                            frame_width,
                            frame_height
                        )

                    except Exception as error:

                        print(
                            "Fall detector error:",
                            error
                        )

                        continue

                    status = result.get(
                        "status",
                        "UNKNOWN"
                    )

                    posture = result.get(
                        "posture",
                        "UNKNOWN"
                    )

                    confidence = float(
                        result.get(
                            "confidence",
                            0.0
                        )
                    )

                    body_angle = float(
                        result.get(
                            "body_angle",
                            0.0
                        )
                    )

                    aspect_ratio = float(
                        result.get(
                            "aspect_ratio",
                            0.0
                        )
                    )

                    vertical_speed = float(
                        result.get(
                            "vertical_speed",
                            0.0
                        )
                    )

                    fall_detected = bool(
                        result.get(
                            "fall_detected",
                            False
                        )
                    )

                    # ------------------------------------------------
                    # Activity
                    # ------------------------------------------------

                    activity = determine_activity(
                        posture,
                        vertical_speed
                    )

                    # ------------------------------------------------
                    # Select strongest result
                    # ------------------------------------------------

                    if confidence >= current_confidence:

                        current_fall_status = (
                            status
                        )

                        current_posture = (
                            posture
                        )

                        current_confidence = (
                            confidence
                        )

                        current_activity = (
                            activity
                        )

                        current_body_angle = (
                            body_angle
                        )

                        current_aspect_ratio = (
                            aspect_ratio
                        )

                        current_vertical_speed = (
                            vertical_speed
                        )

                    # ------------------------------------------------
                    # Person information
                    # ------------------------------------------------

                    y_position = (
                        40 +
                        person_id * 130
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Person: {person_id + 1}",
                        (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Activity: {activity}",
                        (20, y_position + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Posture: {posture}",
                        (20, y_position + 53),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        (
                            f"Fall Confidence: "
                            f"{confidence * 100:.1f}%"
                        ),
                        (20, y_position + 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Status: {status}",
                        (20, y_position + 103),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                    # ------------------------------------------------
                    # FALL CONFIRMED
                    # ------------------------------------------------

                    if fall_detected:

                        current_fall_status = (
                            "FALL DETECTED"
                        )

                        with state_lock:

                            system_state["module5"] = (
                                "ALERT TRIGGERED"
                            )

                        handle_fall_event(
                            annotated_frame,
                            person_id,
                            confidence
                        )

            # ====================================================
            # UPDATE SYSTEM STATE
            # ====================================================

            with state_lock:

                system_state["person_detected"] = (
                    person_detected
                )

                system_state["person_count"] = (
                    person_count
                )

                system_state["pose_keypoints"] = (
                    keypoint_count
                )

                system_state["fall_status"] = (
                    current_fall_status
                )

                system_state["posture"] = (
                    current_posture
                )

                system_state["activity"] = (
                    current_activity
                )

                system_state["confidence"] = round(
                    current_confidence * 100,
                    2
                )

                system_state["body_angle"] = round(
                    current_body_angle,
                    2
                )

                system_state["aspect_ratio"] = round(
                    current_aspect_ratio,
                    3
                )

                system_state["vertical_speed"] = round(
                    current_vertical_speed,
                    2
                )

                # ------------------------------------------------
                # Emergency module state
                # ------------------------------------------------

                if (
                    current_fall_status
                    != "FALL DETECTED"
                ):

                    if (
                        system_state["module5"]
                        == "ALERT TRIGGERED"
                    ):

                        system_state["module5"] = (
                            "READY"
                        )

            # ====================================================
            # FPS
            # ====================================================

            current_time = time.time()

            fps = 1 / max(
                current_time - previous_time,
                0.001
            )

            previous_time = current_time

            with state_lock:

                system_state["fps"] = round(
                    fps,
                    1
                )

                system_state["uptime"] = int(
                    time.time() - start_time
                )

            # ====================================================
            # MODULE INFORMATION ON VIDEO
            # ====================================================

            cv2.putText(
                annotated_frame,
                "VISION-BASED FALL DETECTION",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            # ----------------------------------------------------
            # Module labels
            # ----------------------------------------------------

            cv2.putText(
                annotated_frame,
                "M1: IMAGE ENHANCEMENT",
                (20, frame_height - 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                annotated_frame,
                "M2: POSE EXTRACTION",
                (20, frame_height - 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                annotated_frame,
                "M3: FALL CLASSIFICATION",
                (20, frame_height - 59),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            cv2.putText(
                annotated_frame,
                f"Fall Events: {fall_event_count}",
                (20, frame_height - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                "AI MONITORING: ACTIVE",
                (20, frame_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1
            )

            # ====================================================
            # EMERGENCY DISPLAY
            # ====================================================

            if (
                current_fall_status
                == "FALL DETECTED"
            ):

                cv2.putText(
                    annotated_frame,
                    "!!! FALL DETECTED !!!",
                    (120, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    4
                )

            elif (
                current_fall_status
                == "POSSIBLE FALL"
            ):

                cv2.putText(
                    annotated_frame,
                    "POSSIBLE FALL",
                    (150, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 165, 255),
                    3
                )

        # ====================================================
        # ENCODE FRAME
        # ====================================================

        try:

            success, buffer = cv2.imencode(
                ".jpg",
                annotated_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85
                ]
            )

            if success:

                with frame_lock:

                    latest_frame = (
                        buffer.tobytes()
                    )

        except Exception as error:

            print(
                "Frame encoding error:",
                error
            )

            time.sleep(0.01)

    print(
        "Camera processing thread stopped."
    )


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    while running:

        with frame_lock:

            frame = latest_frame

        if frame is None:

            time.sleep(0.01)

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ============================================================
# ROUTE - DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# ROUTE - LIVE VIDEO
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# ROUTE - SYSTEM STATUS
# ============================================================

@app.route("/api/status")
def status():

    with state_lock:

        current_state = dict(
            system_state
        )

    return jsonify(
        current_state
    )


# ============================================================
# ROUTE - ACKNOWLEDGE EMERGENCY ALERT
# ============================================================

@app.route("/api/alerts/acknowledge", methods=["POST"])
def acknowledge_alert():

    with state_lock:

        system_state["alert_active"] = False
        system_state["alert_message"] = ""
        system_state["alert_time"] = None
        system_state["alert_person"] = None
        system_state["alert_confidence"] = 0.0

        # Return the system to ready monitoring state.
        system_state["module5"] = "READY"

    return jsonify({
        "success": True,
        "message": "Emergency alert acknowledged."
    })


# ============================================================
# ROUTE - FALL EVENTS
# ============================================================

@app.route("/api/events")
def events():

    files = []

    if os.path.exists(
        EVENT_FOLDER
    ):

        for filename in os.listdir(
            EVENT_FOLDER
        ):

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                filepath = os.path.join(
                    EVENT_FOLDER,
                    filename
                )

                try:

                    event_time = (
                        datetime.fromtimestamp(
                            os.path.getmtime(
                                filepath
                            )
                        ).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                    files.append({

                        "filename": filename,

                        "time": event_time

                    })

                except Exception:

                    continue

    files.sort(
        key=lambda x: x["time"],
        reverse=True
    )

    return jsonify(
        files
    )


# ============================================================
# ROUTE - FALL EVENT IMAGE
# ============================================================

@app.route(
    "/fall_events/<path:filename>"
)
def fall_event_image(filename):

    return send_from_directory(
        EVENT_FOLDER,
        filename
    )


# ============================================================
# ROUTE - CURRENT TIME
# ============================================================

@app.route("/api/time")
def current_time():

    return jsonify({

        "time":
        datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

    })


# ============================================================
# ROUTE - MODULE INFORMATION
# ============================================================

@app.route("/api/modules")
def modules():

    with state_lock:

        data = {

            "module1": {
                "name": "Image Enhancement",
                "status":
                    system_state["module1"],
                "techniques": [
                    "Frame Resizing",
                    "Gaussian Blur",
                    "CLAHE Contrast Enhancement"
                ]
            },

            "module2": {
                "name":
                    "Pose Feature Extraction",
                "status":
                    system_state["module2"],
                "keypoints":
                    system_state["pose_keypoints"]
            },

            "module3": {
                "name":
                    "Activity / Fall Classification",
                "status":
                    system_state["module3"],
                "activity":
                    system_state["activity"],
                "posture":
                    system_state["posture"],
                "confidence":
                    system_state["confidence"],
                "body_angle":
                    system_state["body_angle"],
                "aspect_ratio":
                    system_state["aspect_ratio"],
                "vertical_speed":
                    system_state["vertical_speed"]
            },

            "module4": {
                "name":
                    "User Interface Design",
                "status":
                    system_state["module4"]
            },

            "module5": {
                "name":
                    "Emergency Alert Module",
                "status":
                    system_state["module5"],
                "fall_events":
                    system_state["fall_events"],
                "last_event":
                    system_state["last_event"],
                "last_snapshot":
                    system_state["last_snapshot"]
            }

        }

    return jsonify(data)


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print(
        "        VISION-BASED FALL DETECTION SYSTEM"
    )
    print("=" * 65)

    print(
        " AI Model       : YOLO11n Pose"
    )

    print(
        " Module 1       : Image Enhancement"
    )

    print(
        " Module 2       : Pose Feature Extraction"
    )

    print(
        " Module 3       : Activity / Fall Classification"
    )

    print(
        " Module 4       : User Interface"
    )

    print(
        " Module 5       : Emergency Alert"
    )

    print(
        f" Fall Threshold : "
        f"{FALL_THRESHOLD * 100:.0f}%"
    )

    print(
        f" Confirmation   : "
        f"{FALL_CONFIRMATION_TIME} seconds"
    )

    print(
        f" Alarm Cooldown : "
        f"{ALARM_COOLDOWN} seconds"
    )

    print(
        " Camera         :",
        system_state["camera"]
    )

    print()
    print(
        " Dashboard      : "
        "http://127.0.0.1:5000"
    )

    print("=" * 65)
    print()

    # ========================================================
    # START CAMERA THREAD
    # ========================================================

    camera_thread = threading.Thread(
        target=process_camera,
        daemon=True
    )

    camera_thread.start()

    # ========================================================
    # START FLASK
    # ========================================================

    try:

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True
        )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping system..."
        )

    finally:

        running = False

        try:

            camera.release()

        except Exception:
            pass

        print(
            "Camera stopped."
        )

        print(
            f"Total fall events detected: "
            f"{fall_event_count}"
        )

        print(
            "System shutdown complete."
        )