#!/usr/bin/env python3

import os
import sys
import argparse

from utils.ZScoreScaler import ZScoreScaler
from utils.LinearRegressionModel import LinearRegressionModel

parser = argparse.ArgumentParser(description="A prediction script for ft_linear_regression")

parser.add_argument("value", type=float, help="Input value to predict")
parser.add_argument("--model-path", default=".", help="Directory containing the model and scaler JSON files (default: current directory)")

args = parser.parse_args()

input_value = args.value
model_dir = args.model_path

model_path = os.path.join(model_dir, "model.json")
scaler_x_path = os.path.join(model_dir, "scaler_x.json")
scaler_y_path = os.path.join(model_dir, "scaler_y.json")

if not os.path.exists(model_path):
    print(f'Error: model.json does not exists at {model_path}')
    sys.exit(1)

if not os.path.exists(scaler_x_path):
    print(f'Error: scaler_x.json does not exists at {scaler_x_path}')
    sys.exit(1)

if not os.path.exists(scaler_y_path):
    print(f'Error: scaler_y.json does not exists at {scaler_y_path}')
    sys.exit(1)

model = LinearRegressionModel()

scaler_x = ZScoreScaler()
scaler_x.from_json(scaler_x_path)

scaler_y = ZScoreScaler()
scaler_y.from_json(scaler_y_path)

model.from_json(model_path)
    
predicted_value = scaler_y.inverse_transform(model.predict(scaler_x.transform(input_value)))
print(f'f({input_value:.2f}) = {predicted_value:.2f}')
