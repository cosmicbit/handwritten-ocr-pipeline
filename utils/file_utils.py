


import numpy as np
import cv2
from detectors.craft_utils import getDetBoxes_core

def getDetBoxes(textmap, linkmap, text_thresh, link_thresh, low_text):
    boxes, polys = getDetBoxes_core(textmap, linkmap, text_thresh, link_thresh, low_text)
    return boxes, polys
