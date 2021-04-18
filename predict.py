def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


class Predict:
    """
    Data cleaning and preprocessing
    """
    def __init__(self,prepro_input,drivePath, model_weights):
        self.prepro_input = prepro_input
        self.drivePath = drivePath
        self.model_weights = model_weights

    def execute(self):

        self.model = load_model(
            '{}weights_openpose/{}'.format(
                self.drivePath, self.model_weights))

        self.le = LabelEncoder()
        self.le.classes_ = np.load('classes.npy')
        self.pred = self.model.predict(self.prepro_input)
        self.label = self.le.inverse_transform([np.argmax(self.pred)])
        self.prob = np.max(self.pred)


        print('Label', self.label[0], self.prob)
