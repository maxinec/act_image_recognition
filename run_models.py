import argparse
import face_recognition
import numpy as np

import cv2
import math
import numpy as np
from fer import FER
import matplotlib.pyplot as plt

EMOTION_DETECTOR = FER(mtcnn=True)
HIGHLIGHT_BLUE_COLOR = (255, 255, 0)
HIGHLIGHT_RED_COLOR = (255, 49, 49)

def get_facial_features(image, debug=False):
    faces = face_recognition.face_landmarks(image)
    if debug:
        print("Facial detection: {} face(s)".format(len(faces)))
        for i in range(len(faces)):
            print("Face {}:".format(i + 1))
            landmarks = faces[i]
            for face_feature in landmarks:
                print("{}: {}".format(face_feature, landmarks[face_feature]))
        print("")
    return faces

def get_emotions(image, debug=False):
    results = EMOTION_DETECTOR.detect_emotions(image)
    dominant_emotion, emotion_score = EMOTION_DETECTOR.top_emotion(image)
    if debug:
        print("Emotion detection: {} face(s)".format(len(results)))
        print("Primary result: {} {}".format(dominant_emotion, emotion_score))
        for i in range(len(results)):
            face_bounds = results[i]['box']
            emotions = results[i]['emotions']
            print("Location: {}".format(face_bounds))
            print(emotions)
        print("")
    return results

def get_top_emotion(emotions):
    top_emotion = 'neutral'
    top_score = 0
    for emotion, score in emotions.items():
        if score > top_score:
            top_emotion = emotion
            top_score = score
    return top_emotion, top_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a driving simulator video through various models.')
    parser.add_argument('video', help='input video to process')
    parser.add_argument('-s', '--process_seconds', type=int, default=10, help='process a frame every X seconds')
    args = parser.parse_args()

    if args.video is None:
        parser.print_help()
        exit(0)

    debug = True
    vidcap = cv2.VideoCapture(args.video)
    success, image = vidcap.read()
    seconds = 0
    while success:
        print("video position: {} seconds".format(seconds))
        face_feature_results = get_facial_features(image, debug)
        emotion_results = get_emotions(image, debug)

        for face_landmarks in face_feature_results:
            for facial_feature in face_landmarks.keys():
                for point in face_landmarks[facial_feature]:
                    cv2.circle(image, point, radius=0, color=HIGHLIGHT_BLUE_COLOR, thickness=-1)

        for emotion_result in emotion_results:
            face_bounds = emotion_result['box']
            emotions = emotion_result['emotions']
            box_corner_1 = (face_bounds[0], face_bounds[1]) # x, y coordinates
            box_corner_2 = (face_bounds[0] + face_bounds[2], face_bounds[1] + face_bounds[3]) # x + width, y + height
            cv2.rectangle(image, box_corner_1, box_corner_2, HIGHLIGHT_RED_COLOR)
            top_emotion, score = get_top_emotion(emotions)
            cv2.putText(image, "{} {}".format(top_emotion, score), (face_bounds[0], face_bounds[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT_RED_COLOR)

        cv2.imwrite("test/frame%d.jpg" % seconds, image)
        print("---------------------------------------------------")
        seconds += args.process_seconds
        vidcap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, image = vidcap.read()

    vidcap.release()
    print("Complete")

#YOLO
'''
1. convert to "saved model" format: python export.py --weights /Users/maxichan/dev/act_computer_vision/yolo_fall_model.pt --include saved_model

'''