import numpy as np
import cv2

from handwriting_ocr_pipeline.config.paths import OUTPUTS_DIR
import handwriting_ocr_pipeline.config.settings as settings

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


def print_progress_bar(progress: float, end: int) -> None:
    filled = int(progress)
    print(
        "PROGRESS: [{0}{1}] {2:.1f}%".format(
            "█" * int((filled / end) * 25),
            "-" * int(((end - filled) / end) * 25),
            progress,
        ),
        end="\r",
    )
    if progress == end:
        print()

def draw_boxes(image: np.typing.NDArray, boxes: np.typing.NDArray, save_path=OUTPUTS_DIR / settings.wordsOutputImageFile):
    img = image.copy()
    for b in boxes:
        # b = np.concatenate(b, axis=0)
        x_min = int(b[:, 0].min())
        y_min = int(b[:, 1].min())
        x_max = int(b[:, 0].max())
        y_max = int(b[:, 1].max())

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"[+] Saved raw box image: {save_path}")

def draw_boxes_of_lines(image: np.typing.NDArray, boxes: list, save_path=OUTPUTS_DIR / settings.linesOutputImage):
    img = image.copy()
    for b in boxes:
        b = np.concatenate(b, axis=0)
        x_min = int(b[:, 0].min())
        y_min = int(b[:, 1].min())
        x_max = int(b[:, 0].max())
        y_max = int(b[:, 1].max())

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"[+] Saved raw box image: {save_path}")

def save_text(output_path, text):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[+] Saved as text file: {output_path}")