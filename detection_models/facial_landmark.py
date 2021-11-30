import logging
import face_recognition
import cv2

from detection_models.base_model import BaseModel
from detection_models.constants import *

logger = logging.getLogger(__name__)

class FacialLandmark(BaseModel):
    def __init__(self):
        super().__init__()

    def run_recognition(self, image):
        faces = face_recognition.face_landmarks(image)
        logger.debug("Facial detection: {} face(s)".format(len(faces)))
        for i in range(len(faces)):
            logger.debug("Face {}:".format(i + 1))
            landmarks = faces[i]
            for face_feature in landmarks:
                logger.debug("{}: {}".format(face_feature, landmarks[face_feature]))
        return faces

    def decorate_image(self, image, result):
        for face_landmarks in result:
            for facial_feature in face_landmarks.keys():
                for point in face_landmarks[facial_feature]:
                    cv2.circle(image, point, radius=0, color=HIGHLIGHT_TEAL_COLOR, thickness=-1)
        return image