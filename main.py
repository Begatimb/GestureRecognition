import pandas as pd
from preprocess import Preprocess
from predict import Predict
from load_json import load_json


config = load_json('config.json')
# Input example
o_ex = pd.read_csv('Thesis/nemo_openpose/guitar/N15016_2201807271342280681.csv',sep=';')
drivePath = 'Thesis/'


preprocess = Preprocess(input_df=o_ex,
                        drivePath = drivePath,
                        remove_cols = config['remove_columns'])
preprocess.execute()


prepro_input = preprocess.prepro_input

# Predict Label from preprocessed input
predict = Predict()
predict.execute(prepro_input= prepro_input,
                drivePath=drivePath)






