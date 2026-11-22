import cv2


class ImageEnhancer:
    """
    Module 1: Image Enhancement

    Performs:
    - Noise reduction
    - Brightness and contrast enhancement
    - Image resizing
    """

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

    def enhance(self, frame):

        if frame is None:
            return None

        # -------------------------------------------------
        # 1. Resize frame
        # -------------------------------------------------

        resized = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR
        )

        # -------------------------------------------------
        # 2. Noise reduction
        # -------------------------------------------------

        denoised = cv2.GaussianBlur(
            resized,
            (5, 5),
            0
        )

        # -------------------------------------------------
        # 3. Brightness and contrast enhancement
        # -------------------------------------------------

        enhanced = cv2.convertScaleAbs(
            denoised,
            alpha=1.15,
            beta=8
        )

        # -------------------------------------------------
        # 4. CLAHE contrast enhancement
        # -------------------------------------------------

        lab = cv2.cvtColor(
            enhanced,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l_channel = clahe.apply(l_channel)

        enhanced_lab = cv2.merge(
            (
                l_channel,
                a_channel,
                b_channel
            )
        )

        final_frame = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        return final_frame