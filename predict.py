import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


class Predict:
    """
    Returns predicted class from trained deep learning model
    """
    def __init__(self, model_weights):
        self.model = load_model(
            'Thesis/weights_openpose/{}'.format(model_weights))

    def execute(self, input_p):
        le = LabelEncoder()
        le.classes_ = np.load('classes.npy', allow_pickle=True)
        pred = self.model.predict(input_p)
        label = le.inverse_transform([np.argmax(pred)])
        prob = np.max(pred)

        return label[0], prob
