import numpy as np
import cv2
import alphashape
from shapely.geometry import Polygon, MultiPolygon
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

def draw_curved_line_envelopes(image, lines, save_path, thickness=2):
    img = image.copy()

    for line in lines:
        if len(line) < 2:
            continue

        # left → right
        line = sorted(line, key=lambda b: b[:, 0].mean())

        upper = []
        lower = []

        for box in line:
            top_mid, bottom_mid = top_bottom_midpoints(box)
            upper.append(top_mid)
            lower.append(bottom_mid)

        contour = np.array(
            upper + lower[::-1],
            dtype=np.int32
        )

        cv2.polylines(
            img,
            [contour],
            isClosed=True,
            color=(0, 255, 0),
            thickness=thickness
        )

    cv2.imwrite(save_path, img)


def top_bottom_midpoints(box):
    # sort points by y
    pts = box[np.argsort(box[:, 1])]

    top_edge = pts[:2]
    bottom_edge = pts[-2:]

    top_mid = top_edge.mean(axis=0)
    bottom_mid = bottom_edge.mean(axis=0)

    return top_mid, bottom_mid


def draw_concave_hull_lines(
        image, 
        lines, 
        save_path=OUTPUTS_DIR / settings.hullLinesOutputImage, 
        alpha=0.02):
    img = image.copy()

    for line in lines:
        # collect all points from all boxes
        points = np.concatenate(line, axis=0)

        # convert to list of tuples for alphashape
        pts = [(float(x), float(y)) for x, y in points]

        if len(pts) < 4:
            continue  # hull not possible

        hull = alphashape.alphashape(pts, alpha)

        if hull.is_empty:
            continue

        # alphashape may return Polygon or MultiPolygon
        polygons = (
            [hull] if isinstance(hull, Polygon)
            else list(hull.geoms) if isinstance(hull, MultiPolygon)
            else []
        )

        for poly in polygons:
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(save_path, img)
