import numpy as np

def merge_boxes_into_lines(boxes, overlap_ratio=0.40):
    # Sort by center Y
    boxes = sorted(boxes, key=lambda b: b[:, 1].mean())

    lines = []
    current_line = [boxes[0]]

    def vertical_overlap(b1, b2):
        y1_min, y1_max = b1[:, 1].min(), b1[:, 1].max()
        y2_min, y2_max = b2[:, 1].min(), b2[:, 1].max()

        overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        min_height = min(y1_max - y1_min, y2_max - y2_min)

        return (overlap / min_height) >= overlap_ratio

    for b in boxes[1:]:
        if vertical_overlap(current_line[-1], b):
            current_line.append(b)
        else:
            lines.append(current_line)
            current_line = [b]

    lines.append(current_line)
    return lines
