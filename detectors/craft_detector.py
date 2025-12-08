import torch
import cv2
from detectors.craft import CRAFT
from detectors.imgproc import resize_aspect_ratio, normalizeMeanVariance
from utils.weights import remove_module_prefix
from detectors.craft_utils import getDetBoxes ,adjustResultCoordinates

class CraftDetector:

    def __init__(self, weight_path, cuda=False):
        self.cuda = cuda
        
        # Load model
        self.model = CRAFT()

        state_dict = torch.load(weight_path, map_location="cpu")
        state_dict = remove_module_prefix(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        
        if cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()

    def detect(self, image):
        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            image, 1280, cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        img_resized = normalizeMeanVariance(img_resized)

        # convert to float32
        x = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()

        if self.cuda and torch.cuda.is_available():
            x = x.cuda()

        # forward
        with torch.no_grad():
            y, _ = self.model(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # get boxes
        boxes, polys = getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4)

        # adjust coordinates back to image size
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

        return boxes
