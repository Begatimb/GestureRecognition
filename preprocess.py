import numpy as np


class Preprocess:
    """
    Data cleaning and preprocessing
    """

    def __init__(self, input_p):
        self.input = input_p[-494:]
        self.concat_input = np.zeros((1, 2, 494, 12))
        for i in range(len(self.input)):
            for p in range(len(self.input[i])):
                if self.input[i][p] is not None:
                    self.concat_input[0, 0, i, p] = self.input[i][p][0]
                    self.concat_input[0, 1, i, p] = self.input[i][p][1]
        self.preprocess_input = self.concat_input.transpose((0, 2, 3, 1))
