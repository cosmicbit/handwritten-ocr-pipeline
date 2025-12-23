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

def compute_curved_line_envelopes(lines):
    """
    Returns a list of closed contours (Nx2 int arrays),
    one per text line.
    """
    contours = []

    for line in lines:
        if len(line) < 2:
            continue

        # sort left → right
        line = sorted(line, key=lambda b: b[:, 0].mean())

        upper, lower = [], []

        for box in line:
            pts = box[np.argsort(box[:, 1])]
            upper.append(pts[:2].mean(axis=0))
            lower.append(pts[-2:].mean(axis=0))

        # endpoint anchors
        left_top, left_bot = endpoint_anchors(line[0], "left")
        right_top, right_bot = endpoint_anchors(line[-1], "right")

        upper = [left_top] + upper + [right_top]
        lower = [left_bot] + lower + [right_bot]

        # densify & smooth
        upper = chaikin(densify(upper, 6), iterations=2)
        lower = chaikin(densify(lower, 6), iterations=2)

        # controlled looseness
        line_height = estimate_line_height(line)
        padding = 0.2 * line_height

        upper[:, 1] -= padding
        lower[:, 1] += padding

        contour = np.vstack([upper, lower[::-1]]).astype(np.int32)
        contours.append(contour)

    return contours


def draw_line_envelopes(image, contours, save_path,
                        color=(0, 255, 0), thickness=2):
    img = image.copy()

    for contour in contours:
        cv2.polylines(img, [contour], True, color, thickness)

    cv2.imwrite(save_path, img)


def estimate_line_height(line):
    heights = [(b[:, 1].max() - b[:, 1].min()) for b in line]
    return np.median(heights)

def endpoint_anchors(box, side="left"):
    if side == "left":
        x = box[:, 0].min()
    else:
        x = box[:, 0].max()

    top_y = box[:, 1].min()
    bot_y = box[:, 1].max()

    return np.array([x, top_y]), np.array([x, bot_y])


def top_bottom_midpoints(box):
    # sort points by y
    pts = box[np.argsort(box[:, 1])]

    top_edge = pts[:2]
    bottom_edge = pts[-2:]

    top_mid = top_edge.mean(axis=0)
    bottom_mid = bottom_edge.mean(axis=0)

    return top_mid, bottom_mid

def densify(points, samples_per_segment=5):
    dense = []
    for p1, p2 in zip(points[:-1], points[1:]):
        for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
            dense.append(p1 * (1 - t) + p2 * t)
    dense.append(points[-1])
    return dense

def chaikin(points, iterations=2):
    pts = np.array(points)
    for _ in range(iterations):
        new_pts = []
        for p, q in zip(pts[:-1], pts[1:]):
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        pts = np.array(new_pts)
    return pts

def white_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0

    out = np.ones_like(img) * 255
    out[mask] = img[mask]

    return out

def extract_contour_region(image, contour):
    h, w = image.shape[:2]

    contour = np.ascontiguousarray(contour, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)

    masked = cv2.bitwise_and(image, image, mask=mask)

    x, y, bw, bh = cv2.boundingRect(contour)
    cropped = masked[y:y+bh, x:x+bw]

    return cropped

