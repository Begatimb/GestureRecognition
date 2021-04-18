def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import load
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import json


class Preprocess:
    """
    Data cleaning and preprocessing
    """
    def __init__(self, input_df, drivePath, remove_cols):
        self.input_df = input_df
        self.drivePath = drivePath
        self.remove_o = remove_cols

    def execute(self):
        # a list of body joints to remove from openpose


        # Get indexes of columns to keep, and split X and Y
        self.openpose_x_columns = []
        self.openpose_y_columns = []
        for i in range(len(self.input_df.columns)):
            if self.input_df.columns[i][-1] == 'X' and self.input_df.columns[i][:-1] not in self.remove_o:
                self.openpose_x_columns.append(i)
            if self.input_df.columns[i][-1] == 'Y' and self.input_df.columns[i][:-1] not in self.remove_o:
                self.openpose_y_columns.append(i)


        self.input_x = self.input_df[self.input_df.columns[self.openpose_x_columns]]
        self.input_y = self.input_df[self.input_df.columns[self.openpose_y_columns]]

        self.concat_input = np.zeros((1,2,494,42))
        for s in range(self.input_df.shape[0]):
            self.concat_input[0, 0, s] = self.input_df.values[s, self.openpose_x_columns]
            self.concat_input[0, 1, s] = self.input_df.values[s, self.openpose_x_columns]

        self.scaler = load('std_scaler.bin')

        self.flattened = self.concat_input.reshape(self.concat_input.shape[0],
                                                   self.concat_input.shape[1] *
                                                   self.concat_input.shape[2] *
                                                   self.concat_input.shape[3])
        self.scaled = self.scaler.transform(self.flattened)


        self.prepro_input = self.scaled.reshape(1, self.concat_input.shape[1],
                                                   self.concat_input.shape[2],
                                                   self.concat_input.shape[3]).transpose(0, 2, 3, 1)
        
        #self.prepro_input = scaler.fit_transform(self.concat_input.reshape(494,84))
        #self.prepro_input = self.prepro_input.reshape(1,494,42,2)


