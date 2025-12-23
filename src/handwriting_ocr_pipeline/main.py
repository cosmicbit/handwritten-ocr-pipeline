from handwriting_ocr_pipeline.detectors.craft_detector import CraftDetector
from handwriting_ocr_pipeline.postprocess.box_merger import merge_boxes_into_lines
from handwriting_ocr_pipeline.recognizers.trocr_recognizer import TrocrRecognizer
from handwriting_ocr_pipeline.detectors.imgproc import loadImage
from handwriting_ocr_pipeline.utils.file_utils import print_progress_bar
import handwriting_ocr_pipeline.utils.file_utils as utils

from handwriting_ocr_pipeline.config.paths import INPUTS_DIR
from handwriting_ocr_pipeline.config.paths import OUTPUTS_DIR
from handwriting_ocr_pipeline.config.paths import MODELS_DIR
import handwriting_ocr_pipeline.config.settings as settings

from PIL import Image
import numpy as np

input = INPUTS_DIR / settings.inputFileTwo
outputText = OUTPUTS_DIR / settings.recognisedTextFile

def main():
    # Initialize modules
    craft = CraftDetector(
        MODELS_DIR / settings.craftModelFile,
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
    utils.draw_boxes(image, boxes, save_path=OUTPUTS_DIR / settings.wordsOutputImageFile)
    utils.draw_boxes_of_lines(image, lines, save_path=OUTPUTS_DIR / settings.linesOutputImage)
    utils.draw_curved_line_envelopes(image, lines, save_path=OUTPUTS_DIR / settings.curvedLinesOutputImage)

    recognized_lines = []

    # recognize text
    for i in range(len(lines)):
        line_array = np.concatenate(lines[i], axis=0)
        x_min, y_min = line_array[:,0].min(), line_array[:,1].min()
        x_max, y_max = line_array[:,0].max(), line_array[:,1].max()

        # Crop line for recognition
        line_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        pil_img = Image.fromarray(line_img)
        text = recognizer.recognize(pil_img)
        progress = (i / len(lines)) * 100
        print_progress_bar(progress, 100)
        recognized_lines.append(text)
    print_progress_bar(100, 100)
    #save results
    utils.save_text(outputText, "\n".join(recognized_lines))

if __name__ == "__main__":
    main()