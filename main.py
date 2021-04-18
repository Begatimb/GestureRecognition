import pandas as pd
from preprocess import Preprocess
from predict import Predict
from load_json import load_json

import os
label = 'toothbrush'
file = 'N16001_2201807281037223902.csv'

config = load_json('config.json')
drivePath = 'Thesis/'
o_ex = pd.read_csv('Thesis/nemo_openpose/{}/{}'.format(label,file), sep=';')

# Input example
start = 0
end = 200
for i in range(494-200):
    o_ex = pd.read_csv('Thesis/nemo_openpose/{}/{}'.format(label,file), sep=';')[start:end]

    preprocess = Preprocess(input_df=o_ex,
                        drivePath = drivePath,
                        remove_cols = config['remove_columns'])
    preprocess.execute()


    prepro_input = preprocess.prepro_input

    # Predict Label from preprocessed input
    predict = Predict(prepro_input= prepro_input,
                drivePath=drivePath,
                model_weights=config['model'])
    predict.execute()
    end+=1






