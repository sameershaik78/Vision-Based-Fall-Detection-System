from fall_detector import FallDetector


detector = FallDetector()

print("===================================")
print(" FALL DETECTION ALGORITHM TEST")
print("===================================")

# Simulated upright person
upright_keypoints = [
    (320, 100),   # nose
    (300, 120),
    (340, 120),
    (280, 160),
    (360, 160),
    (285, 180),   # left shoulder
    (355, 180),   # right shoulder
    (290, 250),
    (350, 250),
    (295, 320),
    (345, 320),
    (300, 330),   # left hip
    (340, 330)    # right hip
]

result = detector.detect(
    upright_keypoints,
    640,
    480
)

print("\nUpright Test:")
print(result)

print("\nFall detector initialized successfully!")