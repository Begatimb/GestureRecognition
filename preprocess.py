from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
import numpy as np
import pandas as pd

class Preprocess:
    """
    Data cleaning and preprocessing
    """
    def __init__(self):
        print("Initialized Processing")

    def execute(self,input_df,drivePath):
        self.input_df = input_df
        self.drivePath = drivePath

        # a list of body joints to remove from openpose
        self.remove_o = ['REye', 'LEye', 'LEar', 'REar', 'RKnee', 'RAnkle', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe',
                    'LHeel', 'RBigToe', 'RSmallToe', 'RHeel', 'Face0', 'Face1', 'Face2', 'Face3', 'Face4', 'Face5',
                    'Face6', 'Face7', 'Face8', 'Face9', 'Face10', 'Face11', 'Face12', 'Face13', 'Face14', 'Face15',
                    'Face16', 'Face17', 'Face18', 'Face19', 'Face20', 'Face21', 'Face22', 'Face23', 'Face24', 'Face25',
                    'Face26', 'Face27', 'Face28', 'Face29', 'Face30', 'Face31', 'Face32', 'Face33', 'Face34', 'Face35',
                    'Face36', 'Face37', 'Face38', 'Face39', 'Face40', 'Face41', 'Face42', 'Face43', 'Face44', 'Face45',
                    'Face46', 'Face47', 'Face48', 'Face49', 'Face50', 'Face51', 'Face52', 'Face53', 'Face54', 'Face55',
                    'Face56', 'Face57', 'Face58', 'Face59', 'Face60', 'Face61', 'Face62', 'Face63', 'Face64', 'Face65',
                    'Face66', 'Face67', 'Face68', 'Face69', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
                    'LWrist', 'MidHip', 'RHip', 'LHip', 'Nose']

        # Get indexes of columns to keep, and split X and Y
        self.openpose_x_columns = []
        self.openpose_y_columns = []
        for i in range(len(self.input_df.columns)):
            if self.input_df.columns[i][-1] == 'X' and self.input_df.columns[i][:-1] not in self.remove_o:
                self.openpose_x_columns.append(i)
            if self.input_df.columns[i][-1] == 'Y' and self.input_df.columns[i][:-1] not in self.remove_o:
                self.openpose_y_columns.append(i)

        # Print columns to keep
        # print(self.input_df.columns[self.openpose_x_columns])
        # print(self.input_df.columns[self.openpose_y_columns])
        self.input_x = self.input_df[self.input_df.columns[self.openpose_x_columns]]
        self.input_y = self.input_df[self.input_df.columns[self.openpose_x_columns]]

        self.concat_input = np.zeros((1,494,42,2))
        self.concat_input[0, :len(self.input_x), :, 0] = self.input_x.values
        self.concat_input[0, :len(self.input_x), :, 1] = self.input_y.values

        scaler = StandardScaler()
        self.prepro_input = scaler.fit_transform(self.concat_input.reshape(494,84))
        self.prepro_input = self.prepro_input.reshape(1,494,42,2)

        self.p_labels = np.load('{}p_labels.npy'.format(drivePath))

