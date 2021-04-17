import pandas as pd
from preprocess import Preprocess
from predict import Predict
# Input example
o_ex = pd.read_csv('Thesis/nemo_openpose/guitar/N16013_2201807281452262553.csv',sep=';')
drivePath = 'Thesis/'

preprocess = Preprocess()
preprocess.execute(input_df=o_ex,
                   drivePath = drivePath)
prepro_input = preprocess.prepro_input

# Predict Label from preprocessed input
predict = Predict()
predict.execute(prepro_input= prepro_input,
                drivePath=drivePath)






