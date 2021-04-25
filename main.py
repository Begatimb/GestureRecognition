from preprocess import Preprocess
from predict import Predict
from pose_estimation import PoseEstimation
from load_json import load_json
import cv2 as cv
import time

config = load_json('config.json')

# Initiate Classes
pose = PoseEstimation()
predict = Predict(model_weights=config['model'])

# Input Video file
cap = cv.VideoCapture('toothbrush.mp4')
pTime = 0
posePoints = []
while True:
    success, img = cap.read()
    if success is False:
        break
    # Extract Upper Body Poses
    points = pose.pose_estimation(img)
    posePoints.append(points)

    # Preprocess pose points for deep learning classification
    preprocess = Preprocess(input_p=posePoints)

    # Predict from PosePoints
    pred = predict.execute(input_p=preprocess.preprocess_input)

    # todo: restart on idleness when switched to live

    # Show frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)
    if pred[1] > 0.95 and len(posePoints) > 150:
        cv.putText(img, str(pred[0]), (120, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
    cv.imshow('Image', img)

    cv.waitKey(1)
