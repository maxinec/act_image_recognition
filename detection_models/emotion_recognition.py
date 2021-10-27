from fer import FER
import cv2
import logging

from detection_models.base_model import BaseModel
from detection_models.constants import *

logger = logging.getLogger(__name__)

class EmotionRecognition(BaseModel):
    def __init__(self):
        super().__init__()
        self.emotion_detector = FER(mtcnn=True)

    def run_recognition(self, image):
        results = self.emotion_detector.detect_emotions(image)
        dominant_emotion, emotion_score = self.emotion_detector.top_emotion(image)
        logger.debug("Emotion detection: {} face(s)".format(len(results)))
        logger.debug("Primary result: {} {}".format(dominant_emotion, emotion_score))
        for i in range(len(results)):
            face_bounds = results[i]['box']
            emotions = results[i]['emotions']
            logger.debug("Location: {}".format(face_bounds))
            logger.debug(emotions)
        return results

    def get_top_emotion(self, emotions):
        top_emotion = 'neutral'
        top_score = 0
        for emotion, score in emotions.items():
            if score > top_score:
                top_emotion = emotion
                top_score = score
        return top_emotion, top_score


    def decorate_image(self, image, result):
        for emotion_result in result:
            face_bounds = emotion_result['box']
            emotions = emotion_result['emotions']
            box_corner_1 = (face_bounds[0], face_bounds[1]) # x, y coordinates
            box_corner_2 = (face_bounds[0] + face_bounds[2], face_bounds[1] + face_bounds[3]) # x + width, y + height
            cv2.rectangle(image, box_corner_1, box_corner_2, HIGHLIGHT_RED_COLOR)
            top_emotion, score = self.get_top_emotion(emotions)
            cv2.putText(image, "{} {}".format(top_emotion, score), (face_bounds[0], face_bounds[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_RED_COLOR)

        return image
