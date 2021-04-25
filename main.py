import pandas as pd
from preprocess import Preprocess
from predict import Predict
from pose_estimation import PoseEstimation
from load_json import load_json
import cv2 as cv
import mediapipe as mp
import time
config = load_json('config.json')
pose = PoseEstimation()

# Initiate Classes
pose = PoseEstimation()
predict = Predict(model_weights=config['model'])

# Input Video file
cap = cv.VideoCapture('toothbrush.mp4')
pTime = 0
posePoints = []
while True:
    succes, img = cap.read()
    if succes is False:
        break
    pose.pose_estimation(img)
    posePoints.append(pose.points)

    # Preprocess pose points for deep learning prediction
    prepro = Preprocess(input=posePoints)

    # Predict from PosePoints
    predict.execute(input=prepro.prepro_input)


    # todo: restart on idlenes when switched to live


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50),cv.FONT_HERSHEY_PLAIN,1,(255,255,255),3)
    if predict.prob > 0.95 and len(posePoints) > 150:
        cv.putText(img, str(predict.label[0]), (120,50),cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))
    cv.imshow('Image', img)

    cv.waitKey(1)









