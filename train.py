#!/usr/bin/env python3

import os
import sys
import argparse

import pandas as pd
from pandas.api.types import is_numeric_dtype

from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from utils.ZScoreScaler import ZScoreScaler
from utils.LinearRegressionModel import LinearRegressionModel

console = Console()
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
    
except Exception as e:
    console.print(f"[bold red]Error:[/] Failed to open '{args.csv_path}'. Reason: {e}")
    sys.exit(1)
    
x_label = args.x_label
y_label = args.y_label

if x_label not in df.columns:
    console.print(f"[bold red]Error:[/] Invalid x-label: Column '{args.x_label}' does not exist.")
    sys.exit(1)
    
if y_label not in df.columns:
    console.print(f"[bold red]Error:[/] Invalid y-label: Column '{args.y_label}' does not exist.")
    sys.exit(1)

if not is_numeric_dtype(df[x_label]):
    console.print(f"[bold red]Error:[/] Column '{args.x_label}' must contain numerical values.")
    sys.exit(1)

if not is_numeric_dtype(df[y_label]):
    console.print(f"[bold red]Error:[/] Column '{args.y_label}' must contain numerical values.")
    sys.exit(1)

scaler_x = ZScoreScaler()
scaler_y = ZScoreScaler()
scaler_x.fit(df[x_label])
scaler_y.fit(df[y_label])

df[x_label] = scaler_x.transform(df[x_label])
df[y_label] = scaler_y.transform(df[y_label])

model = LinearRegressionModel(epoch=args.epochs)
model.fit(df[x_label], df[y_label])

table = Table()
table.add_column("Parameter", style="cyan")
table.add_column("Value", style="magenta")
table.add_row("Slope (θ₁)", f"{model.get_slope()}")
table.add_row("Y-Intercept (θ₀)", f"{model.get_y_intercept()}")
console.print(Panel(table, title="Final Model Parameters", border_style="green"))

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

console.print(f"[bold green]Training completed[/bold green]\nArtifacts saved in {os.path.abspath(args.output)}")
