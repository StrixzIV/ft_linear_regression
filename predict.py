#!/usr/bin/env python3

import os
import sys
import argparse

from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from utils.ZScoreScaler import ZScoreScaler, load_pkl_scaler
from utils.LinearRegressionModel import LinearRegressionModel, load_pkl_model

console = Console()
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

if not os.path.exists(scaler_x_path):
    console.print(f'[bold red]Error:[/] scaler_x.{"pkl" if args.pkl else "json"} file not found at [bold]\'{scaler_x_path}\'[/bold].')
    sys.exit(1)

if not os.path.exists(scaler_y_path):
    console.print(f'[bold red]Error:[/] scaler_y.{"pkl" if args.pkl else "json"} file not found at [bold]\'{scaler_y_path}\'[/bold].')
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
    
predicted_value = model.predict(scaler_x.transform(input_value))
predicted_value_scaled = scaler_y.inverse_transform(model.predict(scaler_x.transform(input_value)))

slope = scaler_x.inverse_transform(model.get_slope())
y_intercept = scaler_y.inverse_transform(model.get_y_intercept())

equation_text = Text()
equation_text.append(f"Normal form -> f(x) = {model.get_slope()}x + {model.get_y_intercept()}\n")

inverse_slope = scaler_x.inverse_transform(model.get_slope())
inverse_y_intercept = scaler_y.inverse_transform(model.get_y_intercept())
equation_text.append(f"Inverse transform -> f(x) = {inverse_slope:.2f}x + {inverse_y_intercept:.2f}", style="bold yellow")

console.print(Panel(equation_text, title="Model Equations", border_style="blue"))

table = Table()
table.add_column("Input Value", justify="right", style="cyan")
table.add_column("Predicted Value", justify="right", style="magenta")
table.add_column("Predicted Value (Scaled)", justify="right", style="magenta")
table.add_row(f"{input_value:.2f}", f"{predicted_value:.2f}", f"{predicted_value_scaled:.2f}")

console.print(Panel(table, title=f"Predictions from f(x) = {slope:.2f}x {'-' if y_intercept < 0 else '+'} {abs(y_intercept):.2f}", border_style="green"))
