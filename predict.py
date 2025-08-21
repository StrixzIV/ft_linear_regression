#!/usr/bin/env python3

import sys

from utils.ZScoreScaler import ZScoreScaler
from utils.LinearRegressionModel import LinearRegressionModel

if len(sys.argv) in {3, 4}:
    print('Invalid arguements')
    print('Usage: ./predict.py <value> <model>')

model = LinearRegressionModel()

scaler_x = ZScoreScaler()
scaler_x.from_json('scaler_x.json')

scaler_y = ZScoreScaler()
scaler_y.from_json('scaler_y.json')

if len(sys.argv) == 3:
    model.from_json(sys.argv[2])

else:
    model.from_json()
    
predicted_value = scaler_y.inverse_transform(model.predict(scaler_x.transform(float(sys.argv[1]))))
print(f'f({float(sys.argv[1]):.2f}) = {predicted_value:.2f}')
