def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


class Predict:
    """
    Data cleaning and preprocessing
    """
    def __init__(self, model_weights):
        self.model = load_model(
            'Thesis/weights_openpose/{}'.format(model_weights))

    def execute(self, input):
        self.prepro_input = input
        self.le = LabelEncoder()
        self.le.classes_ = np.load('classes.npy')
        self.pred = self.model.predict(self.prepro_input)
        self.label = self.le.inverse_transform([np.argmax(self.pred)])
        self.prob = np.max(self.pred)

