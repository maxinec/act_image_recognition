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
        detection = self.model(image)
        detection_df = detection.pandas()
        if len(detection_df.xyxy) == 0:
            return []
        logger.debug(detection_df.xyxy[0].to_json(orient="records"))
        return detection_df.xyxy[0].to_dict(orient="records")

    def decorate_image(self, image, result):
        for detection in result:
            if detection['confidence'] > 0.7:
                xmin = int(detection['xmin'])
                ymin = int(detection['ymin'])
                xmax = int(detection['xmax'])
                ymax = int(detection['ymax'])
                box_corner_1 = (xmin, ymin)
                box_corner_2 = (xmax, ymax)
                cv2.rectangle(image, box_corner_1, box_corner_2, HIGHLIGHT_GREEN_COLOR)
                cv2.putText(image, "{} {:.2f}".format(detection['name'], detection['confidence']), (xmin, ymin - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, HIGHLIGHT_GREEN_COLOR)
        return image