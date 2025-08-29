#!/usr/bin/env python3

import os
import sys
import argparse

import pandas as pd
from pandas.api.types import is_numeric_dtype

from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from utils.ZScoreScaler import ZScoreScaler, load_pkl_scaler
from utils.LinearRegressionModel import LinearRegressionModel, load_pkl_model

console = Console()
parser = argparse.ArgumentParser(description="A metrics script for ft_linear_regression")

parser.add_argument("csv_path", help="Path to CSV dataset to evaluate")
parser.add_argument("x_label", help="Column name for the independent variable (X)")
parser.add_argument("y_label", help="Column name for the dependent variable (Y)")
parser.add_argument("--model-path", default=".", help="Directory containing the model and scaler files (default: current directory)")
parser.add_argument("--pkl", action="store_true", help="Load model and scalers from pickle (.pkl) instead of JSON")

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

if not is_numeric_dtype(df[x_label]):
    print(f'Error: x values column "{x_label}" should only contains numerical values.')
    sys.exit(1)

if not is_numeric_dtype(df[y_label]):
    print(f'Error: y values column "{y_label}" should only contains numerical values.')
    sys.exit(1)

model = LinearRegressionModel()
scaler_x = ZScoreScaler()
scaler_y = ZScoreScaler()

if args.pkl:
    model = load_pkl_model(model_path)
    scaler_x = load_pkl_scaler(scaler_x_path)
    scaler_y = load_pkl_scaler(scaler_y_path)

else:
    model.from_json(model_path)
    scaler_x.from_json(scaler_x_path)
    scaler_y.from_json(scaler_y_path)

x_scaled = scaler_x.transform(df[args.x_label].to_numpy())
y_scaled = scaler_y.transform(df[args.y_label].to_numpy())
predictions_scaled = model.predict(x_scaled)
model.calculate_metrics(y_scaled, predictions_scaled, len(df))

equation_text = Text()
equation_text.append(f"Normal form -> f(x) = {model.get_slope()}x + {model.get_y_intercept()}\n")

inverse_slope = scaler_x.inverse_transform(model.get_slope())
inverse_y_intercept = scaler_y.inverse_transform(model.get_y_intercept())
equation_text.append(f"Inverse transform -> f(x) = {inverse_slope:.2f}x + {inverse_y_intercept:.2f}", style="bold yellow")

console.print(Panel(equation_text, title="Model Equations", border_style="blue"))

metrics_table = Table(title_style="bold underline", show_header=False, show_lines=True)
metrics_table.add_column("Metric", justify="right", style="cyan")
metrics_table.add_column("Value", justify="left", style="white")

metrics = [
    ("Mean Squared Error (MSE)", f"{model.mse_history[-1]:.4f}"),
    ("Root Mean Squared Error (RMSE)", f"{model.rmse_history[-1]:.4f}"),
    ("Mean Absolute Error (MAE)", f"{model.mae_history[-1]:.4f}"),
    ("R-squared (R^2)", f"{model.r2_history[-1]:.4f}"),
    ("Huber Loss", f"{model.huber_loss_history[-1]:.4f}"),
    ("Mean Absolute Percentage Error (MAPE)", f"{model.mape_history[-1]:.2f}%")
]

for metric, value in metrics:
    metrics_table.add_row(metric, value)

console.print(Panel(metrics_table, title="Performance Metrics", border_style="purple"))
