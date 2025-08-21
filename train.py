#!/usr/bin/env python3

import sys
import pandas as pd

from utils.ZScoreScaler import ZScoreScaler
from utils.LinearRegressionModel import LinearRegressionModel

if len(sys.argv) != 3:
    print('Invalid arguements')
    print('Usage: ./train.py <csv_data_path> <x_label> <y_label>')

csv_path = sys.argv[1]
x_label = sys.argv[2]
y_label = sys.argv[3]

df = pd.read_csv(csv_path)

scaler_x = ZScoreScaler()
scaler_y = ZScoreScaler()
scaler_x.fit(df[x_label])
scaler_y.fit(df[y_label])

df['km'] = scaler_x.transform(df[x_label])
df['price'] = scaler_y.transform(df[y_label])

model = LinearRegressionModel(epoch=1000)
model.fit(df[x_label], df[y_label])

model.to_json()
scaler_x.to_json('scaler_x.json')
scaler_y.to_json('scaler_y.json')
