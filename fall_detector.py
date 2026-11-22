import math
import time
from collections import deque


class FallDetector:
    """
    Vision-based fall detection using YOLO Pose keypoints.

    The detector combines:
    1. Body orientation
    2. Body aspect ratio
    3. Vertical movement
    4. Persistence of fallen posture
    """

    def __init__(
        self,
        fall_threshold=0.70,
        confirmation_time=2.0
    ):
        self.fall_threshold = fall_threshold
        self.confirmation_time = confirmation_time

        self.previous_center_y = None
        self.previous_time = None

        self.fall_start_time = None
        self.fall_confirmed = False

        self.movement_history = deque(maxlen=10)

    # ---------------------------------------------------------
    # Calculate distance between two points
    # ---------------------------------------------------------

    @staticmethod
    def distance(p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # ---------------------------------------------------------
    # Calculate body angle
    # ---------------------------------------------------------

    @staticmethod
    def calculate_body_angle(shoulder_center, hip_center):

        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]

        angle = math.degrees(math.atan2(abs(dy), abs(dx)))

        return angle

    # ---------------------------------------------------------
    # Calculate body bounding box
    # ---------------------------------------------------------

    @staticmethod
    def calculate_bbox(keypoints):

        xs = [point[0] for point in keypoints]
        ys = [point[1] for point in keypoints]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        return min_x, min_y, max_x, max_y, width, height

    # ---------------------------------------------------------
    # Main fall detection
    # ---------------------------------------------------------

    def detect(self, keypoints, frame_width, frame_height):

        if keypoints is None or len(keypoints) < 5:
            return {
                "status": "NO PERSON",
                "posture": "UNKNOWN",
                "confidence": 0.0,
                "fall_detected": False
            }

        try:

            # YOLO Pose COCO keypoints
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            # Check whether important points exist
            required_points = [
                nose,
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            ]

            valid_points = [
                p for p in required_points
                if p is not None
            ]

            if len(valid_points) < 4:
                return {
                    "status": "ANALYZING",
                    "posture": "UNKNOWN",
                    "confidence": 0.0,
                    "fall_detected": False
                }

            # -------------------------------------------------
            # Shoulder center
            # -------------------------------------------------

            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )

            # -------------------------------------------------
            # Hip center
            # -------------------------------------------------

            hip_center = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )

            # -------------------------------------------------
            # Body angle
            # -------------------------------------------------

            body_angle = self.calculate_body_angle(
                shoulder_center,
                hip_center
            )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            visible_points = [
                p for p in keypoints
                if p is not None
            ]

            (
                min_x,
                min_y,
                max_x,
                max_y,
                width,
                height
            ) = self.calculate_bbox(visible_points)

            if height <= 0:
                height = 1

            aspect_ratio = width / height

            # -------------------------------------------------
            # Body center
            # -------------------------------------------------

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # -------------------------------------------------
            # Calculate downward movement
            # -------------------------------------------------

            current_time = time.time()

            vertical_speed = 0.0

            if (
                self.previous_center_y is not None
                and self.previous_time is not None
            ):

                time_difference = current_time - self.previous_time

                if time_difference > 0:

                    vertical_speed = (
                        center_y - self.previous_center_y
                    ) / time_difference

            self.previous_center_y = center_y
            self.previous_time = current_time

            self.movement_history.append(vertical_speed)

            # -------------------------------------------------
            # Determine posture
            # -------------------------------------------------

            if body_angle > 60 and aspect_ratio < 0.75:
                posture = "STANDING"

            elif body_angle > 40 and aspect_ratio < 1.0:
                posture = "UPRIGHT"

            elif body_angle < 35 and aspect_ratio > 1.2:
                posture = "HORIZONTAL"

            elif aspect_ratio > 1.0:
                posture = "LYING / LOW"

            else:
                posture = "BENDING / SITTING"

            # -------------------------------------------------
            # Fall indicators
            # -------------------------------------------------

            horizontal_score = 0.0
            movement_score = 0.0

            # Horizontal body
            if body_angle < 35 and aspect_ratio > 1.2:
                horizontal_score = 0.65

            elif body_angle < 45 and aspect_ratio > 1.0:
                horizontal_score = 0.40

            # Sudden downward movement
            if vertical_speed > 120:

                movement_score = 0.35

            elif vertical_speed > 70:

                movement_score = 0.20

            # -------------------------------------------------
            # Combined confidence
            # -------------------------------------------------

            fall_confidence = min(
                horizontal_score + movement_score,
                1.0
            )

            # -------------------------------------------------
            # Fall candidate
            # -------------------------------------------------

            fall_candidate = (
                fall_confidence >= self.fall_threshold
            )

            # -------------------------------------------------
            # Confirm fall only if posture persists
            # -------------------------------------------------

            if fall_candidate:

                if self.fall_start_time is None:
                    self.fall_start_time = current_time

                elapsed = (
                    current_time - self.fall_start_time
                )

                if elapsed >= self.confirmation_time:

                    self.fall_confirmed = True

            else:

                # Reset only if person returns to normal
                if posture in [
                    "STANDING",
                    "UPRIGHT"
                ]:

                    self.fall_start_time = None
                    self.fall_confirmed = False

            # -------------------------------------------------
            # Final status
            # -------------------------------------------------

            if self.fall_confirmed:

                status = "FALL DETECTED"

            elif fall_candidate:

                status = "POSSIBLE FALL"

            else:

                status = "NORMAL"

            return {
                "status": status,
                "posture": posture,
                "confidence": fall_confidence,
                "fall_detected": self.fall_confirmed,
                "body_angle": body_angle,
                "aspect_ratio": aspect_ratio,
                "vertical_speed": vertical_speed,
                "center": (center_x, center_y)
            }

        except Exception as error:

            print("Fall detection error:", error)

            return {
                "status": "ERROR",
                "posture": "UNKNOWN",
                "confidence": 0.0,
                "fall_detected": False
            }