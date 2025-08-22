#!/usr/bin/env python3

import os
import sys
import argparse

from utils.ZScoreScaler import ZScoreScaler, load_pkl_scaler
from utils.LinearRegressionModel import LinearRegressionModel, load_pkl_model

parser = argparse.ArgumentParser(description="A prediction script for ft_linear_regression")

parser.add_argument("value", type=float, help="Input value to predict")
parser.add_argument("--model-path", default=".", help="Directory containing the model and scaler JSON files (default: current directory)")
parser.add_argument("--pkl", action="store_true", help="Load model and scalers from pickle (.pkl) instead of JSON")

args = parser.parse_args()

input_value = args.value
model_dir = args.model_path

model_path = os.path.join(model_dir, f"model.{'pkl' if args.pkl else 'json'}")
scaler_x_path = os.path.join(model_dir, f"scaler_x.{'pkl' if args.pkl else 'json'}")
scaler_y_path = os.path.join(model_dir, f"scaler_y.{'pkl' if args.pkl else 'json'}")

if not os.path.exists(model_path):
    print(f'Error: model.{"pkl" if args.pkl else "json"} does not exists at {model_path}')
    sys.exit(1)

if not os.path.exists(scaler_x_path):
    print(f'Error: scaler_x.{"pkl" if args.pkl else "json"} does not exists at {scaler_x_path}')
    sys.exit(1)

if not os.path.exists(scaler_y_path):
    print(f'Error: scaler_y.{"pkl" if args.pkl else "json"} does not exists at {scaler_y_path}')
    sys.exit(1)

model = LinearRegressionModel()
scaler_x = ZScoreScaler()
scaler_y = ZScoreScaler()

if args.pkl:
    model = load_pkl_model(model_path)
    scaler_x = load_pkl_model(scaler_x_path)
    scaler_y = load_pkl_model(scaler_y_path)

else:
    model.from_json(model_path)
    scaler_x.from_json(scaler_x_path)
    scaler_y.from_json(scaler_y_path)
    
predicted_value = scaler_y.inverse_transform(model.predict(scaler_x.transform(input_value)))
print(f'f({input_value:.2f}) = {predicted_value:.2f}')
