import numpy as np
import cv2


def enhance_document(img):
    # 1. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. denoise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. sharpen
    blur = cv2.GaussianBlur(denoised, (0,0), 3)
    sharp = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)

    # 4. contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(sharp)

    # 5. restore 3-channel RGB for CRAFT
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced_rgb




