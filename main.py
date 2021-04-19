import pandas as pd
from preprocess import Preprocess
from predict import Predict
from pose_estimation import PoseEstimation
import os
from load_json import load_json
import matplotlib.pyplot as plt
import cv2 as cv
config = load_json('config.json')


# Input
label = 'toothbrush'
file = 'N16001_2201807281037223902.csv'
drivePath = 'Thesis/'

im = cv.imread('image.jpeg')

"""
# Extract Pose Estimations using OpenPose and OpenCV
pose = PoseEstimation()
estimated_img = pose.pose_estimation(im)
plt.imshow(estimated_img)
plt.show()
"""

# Preprocess Pose Estimations for Gesture Classification
for f in os.listdir('Thesis/lowlands_openpose/{}/'.format(label)):
    o_ex = pd.read_csv('Thesis/lowlands_openpose/{}/{}'.format(label, f), sep=';')[:494]
    prepro = Preprocess(input_df=o_ex,
                        drivePath = drivePath,
                        keepcols = list(config['BODY_PARTS'].keys()))
    prepro.execute()
    prepro_input = prepro.prepro_input


    # Predict Label from preprocessed input
    predict = Predict(prepro_input= prepro_input,
                drivePath=drivePath,
                model_weights=config['model'])
    predict.execute()


#todo: check frame size of videos on gdrive

#todo: restart on idlenes when switched to live






