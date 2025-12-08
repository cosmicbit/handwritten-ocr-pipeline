import cv2
from detectors.craft_detector import CraftDetector
from postprocess.box_merger import merge_boxes_into_lines
from recognizers.trocr_recognizer import TrocrRecognizer
from PIL import Image
import numpy as np

input = "inputs/image.jpg"
output = "outputs/output.jpg"

def draw_raw_boxes(image, boxes, save_path="outputs/raw_boxes.png"):
    img = image.copy()

    for b in boxes:
        x_min = int(b[:, 0].min())
        y_min = int(b[:, 1].min())
        x_max = int(b[:, 0].max())
        y_max = int(b[:, 1].max())

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"[+] Saved raw box image: {save_path}")


if __name__ == "__main__":
    # Initialize modules
    craft = CraftDetector(
        "weights/craft_mlt_25k.pth",
        cuda=False
    )
    recognizer = TrocrRecognizer()

    # Detect words
    img, boxes = craft.detect(input)

    # Draw raw boxes
    draw_raw_boxes(img, boxes)

    # Merge words into lines
    lines = merge_boxes_into_lines(boxes)

    # Draw boxes and recognize text
    for line_box in lines:
        line_array = np.concatenate(line_box, axis=0)
        x_min, y_min = line_array[:,0].min(), line_array[:,1].min()
        x_max, y_max = line_array[:,0].max(), line_array[:,1].max()

        #x_min, y_min = line_box[:,0].min(), line_box[:,1].min()
        #x_max, y_max = line_box[:,0].max(), line_box[:,1].max()
        cv2.rectangle(img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0,255,0), 2)

        # Crop line for recognition
        line_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        pil_img = Image.fromarray(line_img)
        text = recognizer.recognize(pil_img)
        print("Recognized:", text)

    cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Output saved as output.jpg")
