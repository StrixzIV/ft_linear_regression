#!/usr/bin/env python3

import os
import sys
import argparse

import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.ZScoreScaler import ZScoreScaler
from utils.LinearRegressionModel import LinearRegressionModel

parser = argparse.ArgumentParser(description="A training script for ft_linear_regression")

parser.add_argument("csv_path", help="Path to CSV dataset")
parser.add_argument("x_label", help="Column name for the independent variable (X)")
parser.add_argument("y_label", help="Column name for the dependent variable (Y)")
parser.add_argument("--output", default="model", help="Output directory to save model and scalers (default: ./model)")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--as-pkl", action="store_true", help="Save model and scalers as pickle (.pkl) instead of JSON")

args = parser.parse_args()

csv_path = args.csv_path

try:
    df = pd.read_csv(csv_path)
    
except Exception:
    print(f'Failed to open {csv_path}')
    sys.exit(1)
    
x_label = args.x_label
y_label = args.y_label

if x_label not in df.columns:
    print(f'Invalid x-label: Column "{x_label}" does not exists in the CSV file')
    sys.exit(1)
    
if y_label not in df.columns:
    print(f'Invalid y-label: Column "{y_label}" does not exists in the CSV file')
    sys.exit(1)

if not is_numeric_dtype(df[x_label]):
    print(f'Error: x values column "{x_label}" should only contains numerical values.')
    sys.exit(1)

if not is_numeric_dtype(df[y_label]):
    print(f'Error: y values column "{y_label}" should only contains numerical values.')
    sys.exit(1)

scaler_x = ZScoreScaler()
scaler_y = ZScoreScaler()
scaler_x.fit(df[x_label])
scaler_y.fit(df[y_label])

df[x_label] = scaler_x.transform(df[x_label])
df[y_label] = scaler_y.transform(df[y_label])

model = LinearRegressionModel(epoch=args.epochs)
model.fit(df[x_label], df[y_label])

os.makedirs(args.output, exist_ok=True)

model_path = os.path.join(args.output, f"model.{'pkl' if args.as_pkl else 'json'}")
scaler_x_path = os.path.join(args.output, f"scaler_x.{'pkl' if args.as_pkl else 'json'}")
scaler_y_path = os.path.join(args.output, f"scaler_y.{'pkl' if args.as_pkl else 'json'}")

if args.as_pkl:
    model.to_pkl(model_path)
    scaler_x.to_pkl(scaler_x_path)
    scaler_y.to_pkl(scaler_y_path)

else:
    model.to_json(model_path)
    scaler_x.to_json(scaler_x_path)
    scaler_y.to_json(scaler_y_path)

print(f"Training completed. Artifacts saved in {os.path.abspath(args.output)}")
