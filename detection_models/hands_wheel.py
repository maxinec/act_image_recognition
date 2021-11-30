import logging
import torch
import cv2
import sys
from detection_models.base_model import BaseModel
from detection_models.constants import *

logger = logging.getLogger(__name__)

class HandsWheel(BaseModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def run_recognition(self, image):
        results = self.model(image)
        logger.debug(results)
        return results.pandas().xyxy[0].to_dict(orient="records")

    def decorate_image(self, image, result):
        results = self.model(image)
        results.render()
        return results.imgs[0]