import cv2
from detectors.craft_detector import CraftDetector
from postprocess.box_merger import merge_boxes_into_lines
from recognizers.trocr_recognizer import TrocrRecognizer
from detectors.imgproc import loadImage
from PIL import Image
import numpy as np
from utils.file_utils import enhance_document

input = "inputs/image2.jpg"
output = "outputs/output.jpg"
outputText = "outputs/outputTextfile.txt"

def draw_boxes(image: np.typing.NDArray, boxes: np.typing.NDArray, save_path="outputs/words_output.png"):
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

def draw_boxes_of_lines(image: np.typing.NDArray, boxes: list, save_path="outputs/lines_output.png"):
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

if __name__ == "__main__":
    # Initialize modules
    craft = CraftDetector(
        "weights/craft_mlt_25k.pth",
        cuda=False
    )
    recognizer = TrocrRecognizer()

    #load image
    image = loadImage(input)
    #image = enhance_document(image)

    # Detect words
    boxes = craft.detect(image)

    # Merge words into lines
    lines = merge_boxes_into_lines(boxes)

    # draw boxes
    draw_boxes(image, boxes)
    draw_boxes_of_lines(image, lines)

    recognized_lines = []

    # recognize text
    for line_box in lines:
        line_array = np.concatenate(line_box, axis=0)
        x_min, y_min = line_array[:,0].min(), line_array[:,1].min()
        x_max, y_max = line_array[:,0].max(), line_array[:,1].max()

        # Crop line for recognition
        line_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        pil_img = Image.fromarray(line_img)
        text = recognizer.recognize(pil_img)
        print("Recognized:", text)
        recognized_lines.append(text)

    #save results
    save_text(outputText, "\n".join(recognized_lines))
    print("Output saved as outputTextfile.txt")