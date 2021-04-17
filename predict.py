import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical

class Predict:
    """
    Data cleaning and preprocessing
    """
    def __init__(self):
        print("Initialized Processinf")

    def execute(self,prepro_input,drivePath):
        self.prepro_input = prepro_input
        self.drivePath = drivePath
        self.model_openpose = load_model(
            '{}weights_openpose/weights.4CNN no LSTM_optimizer-<keras.optimizers.Adamax object at 0x7fa9481bc5d0>_filtersize-64_activ2d-tanh_activ3dtanh.hdf5'.format(
                self.drivePath))

        self.p_labels = np.load('{}p_labels.npy'.format(self.drivePath))
        le = LabelEncoder()
        self.yy = to_categorical(le.fit_transform(self.p_labels))

        self.label = le.inverse_transform([np.argmax(self.model_openpose.predict(self.prepro_input))])
        print('Label', self.label[0])
