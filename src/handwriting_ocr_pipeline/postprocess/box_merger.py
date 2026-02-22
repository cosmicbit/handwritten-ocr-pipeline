import numpy as np

def merge_boxes_into_lines(boxes, y_threshold=25):

    boxes = sorted(boxes, key=lambda b: b[:,1].mean())
    lines = []

    for box in boxes:
        cy = box[:,1].mean()

        placed = False

        for line in lines:
            ly = np.mean([b[:,1].mean() for b in line])

            if abs(cy - ly) < y_threshold:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    return lines