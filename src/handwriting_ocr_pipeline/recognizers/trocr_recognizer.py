from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class TrocrRecognizer:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("models/trocr-local")
        self.model = VisionEncoderDecoderModel.from_pretrained("models/trocr-local")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def recognize(self, image):
        """
        image: PIL.Image of a single word or line
        """
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
