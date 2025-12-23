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

def draw_curved_line_envelopes(image, lines, save_path):
    img = image.copy()

    for line in lines:
        if len(line) < 2:
            continue

        line = sorted(line, key=lambda b: b[:, 0].mean())

        upper, lower = [], []

        for box in line:
            pts = box[np.argsort(box[:, 1])]
            top_mid = pts[:2].mean(axis=0)
            bot_mid = pts[-2:].mean(axis=0)

            upper.append(top_mid)
            lower.append(bot_mid)

        # first and last boxes
        first_box = line[0]
        last_box = line[-1]

        # anchor endpoints
        left_top, left_bot = endpoint_anchors(first_box, "left")
        right_top, right_bot = endpoint_anchors(last_box, "right")

        # prepend & append
        upper = [left_top] + upper + [right_top]
        lower = [left_bot] + lower + [right_bot]

        # densify
        upper = densify(upper, samples_per_segment=6)
        lower = densify(lower, samples_per_segment=6)

        # smooth
        upper = chaikin(upper, iterations=2)
        lower = chaikin(lower, iterations=2)

        line_height = estimate_line_height(line)
        padding = 0.2 * line_height   # tune: 0.25–0.45 works well

        upper[:, 1] -= padding
        lower[:, 1] += padding

        contour = np.vstack([upper, lower[::-1]]).astype(np.int32)

        cv2.polylines(img, [contour], True, (0, 255, 0), 2)

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