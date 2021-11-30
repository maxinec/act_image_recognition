import argparse
import logging

import cv2
import math
import numpy as np
import json
import matplotlib.pyplot as plt
from detection_models.facial_landmark import FacialLandmark
from detection_models.emotion_recognition import EmotionRecognition
from detection_models.hands_wheel import HandsWheel

logger = logging.getLogger(__name__)

def run_video_against_models(video_path, write_test_images=False):
    models = [FacialLandmark(), EmotionRecognition(), HandsWheel('./yolo/results/train/stock_image_train/weights/best.pt')]

    vidcap = cv2.VideoCapture(video_path)
    try:
        success, image = vidcap.read()
        jsonres = {}
        seconds = 0
        while success:
            logger.debug("video position: {} seconds".format(seconds))
            decorated_image = image
            jsonr = {}

            for model in models:
                result = model.run_recognition(image)
                print(result, type(result), type(model))
                if type(model) == FacialLandmark:
                    jsonr["facial_landmark"] = result
                elif type(model) == EmotionRecognition:
                    jsonr["emotion_recognition"] = result
                elif type(model) == HandsWheel:
                    jsonr["hands_wheel"] = result
                if write_test_images:
                    decorated_image = model.decorate_image(decorated_image, result)

            if write_test_images:
                cv2.imwrite("test/frame%d.jpg" % seconds, decorated_image)

            jsonres[seconds] = jsonr

            logger.debug("---------------------------------------------------")
            seconds += args.process_seconds
            vidcap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, image = vidcap.read()

        json_res = json.dumps(jsonres)

        with open("test.json", "w") as outfile:
            outfile.write(json_res)
        
    except:
        logging.error("Failed to process video", exc_info=True)
    finally:
        vidcap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a driving simulator video through various models.')
    parser.add_argument('video', help='input video to process')
    parser.add_argument('-s', '--process_seconds', type=int, default=10, help='process a frame every X seconds')
    parser.add_argument('-v', '--verbose', action="store_true", help='prints extra information and writes test images')
    args = parser.parse_args()

    if args.video is None:
        parser.print_help()
        exit(0)

    write_test_images = False
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        write_test_images = True

    run_video_against_models(args.video, write_test_images)
