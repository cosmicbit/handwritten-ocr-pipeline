import numpy as np

def merge_boxes_into_lines(boxes, y_threshold=10):
    """
    Merge word-level boxes into lines based on vertical proximity.
    """
    # Ensure all boxes are numpy arrays
    boxes = [np.array(b) for b in boxes]

    # sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[:, 1].min())
    lines = []
    current_line = []

    for box in boxes:
        box_y = box[:,1].mean()
        if not current_line:
            current_line.append(box)
            continue
        prev_y = np.array(current_line)[:,:,1].mean()
        if abs(box_y - prev_y) < y_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    # merge each line's boxes into a single bounding rectangle
    merged_lines = []
    for line in lines:
        line = np.vstack(line)
        x_min, y_min = line.min(axis=0)
        x_max, y_max = line.max(axis=0)
        merged_lines.append(np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]))

    return merged_lines
