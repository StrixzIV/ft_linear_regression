import os
import json
import tqdm
import pickle
import numpy as np
import pandas as pd

class LinearRegressionModel:

    def __init__(self, learning_rate: float = 1e-3, epoch: int = 100):

        '''
            * theta_0 is y-intercept
            * theta_1 is slope
        '''

        self.slope = 0
        self.y_intercept = 0

        self.lr = learning_rate
        self.epoch = epoch

        self.slope_history: list[float] = []
        self.y_intercept_history: list[float] = []

        self.mse_history: list[float] = []
        self.rmse_history: list[float] = []
        self.mae_history: list[float] = []
        self.mape_history: list[float] = []
        self.r2_history: list[float] = []
        self.huber_loss_history: list[float] = []


    def get_slope(self) -> float:
        return self.slope
    

    def get_y_intercept(self) -> float:
        return self.y_intercept


    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame({
            'slope': self.slope_history,
            'y_intercept': self.y_intercept_history,
            'MSE': self.mse_history,
            'RMSE': self.rmse_history,
            'MAE': self.mae_history,
            'MAPE': self.mape_history,
            'R^2': self.r2_history,
            'huber_loss': self.huber_loss_history
        })


    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, m: int) -> None:

        errors = y_pred - y_true
        
        # Mean Squared Error
        mse = np.sum(errors ** 2) / (2 * m)
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.sum(errors ** 2) / m)
        
        # Mean Absolute Error
        mae = np.sum(np.abs(errors)) / m
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs(errors / np.where(y_true != 0, y_true, 1e-8))) * 100 # Add small epsilon to avoid division by zero
        
        # R-squared (coefficient of determination)
        y_mean = np.mean(y_true)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
        
        # Huber Loss (robust to outliers)

        # Huber loss parameter
        delta = 1.0
        huber_loss = np.where(
            np.abs(errors) <= delta,
            0.5 * errors ** 2,
            delta * (np.abs(errors) - 0.5 * delta)
        )

        huber_loss = np.sum(huber_loss) / m
        
        self.mse_history.append(mse)
        self.rmse_history.append(rmse)
        self.mae_history.append(mae)
        self.mape_history.append(mape)
        self.r2_history.append(r2)
        self.huber_loss_history.append(huber_loss)


    def predict(self, x: float | np.ndarray[float]) -> float:
        return (self.slope * x) + self.y_intercept
    

    def fit(self, x: pd.Series, y: pd.Series) -> None:
        
        m = x.size

        if m == 0:
            raise ValueError('Training data is empty')
        
        x_ndarray = x.to_numpy()
        y_ndarray = y.to_numpy()

        for _ in tqdm.tqdm(range(self.epoch)):

            predictions = self.predict(x_ndarray)
            self.calculate_metrics(y_ndarray, predictions, m)

            errors = predictions - y_ndarray

            tmp_slope = self.lr * (1 / m) * np.sum(errors * x_ndarray)
            tmp_y_intercept = self.lr * (1 / m) * np.sum(errors)

            self.slope -= tmp_slope
            self.y_intercept -= tmp_y_intercept

            self.slope_history.append(self.slope)
            self.y_intercept_history.append(self.y_intercept)


    def to_pkl(self, filename: str = 'model.pkl') -> None:

        try:

            with open(filename, 'wb') as f:
                pickle.dump(self, f)

            print(f"Model successfully saved to {os.path.abspath(filename)}")

        except IOError as e:
            print(f"Error saving model to {os.path.abspath(filename)}: {e}")


    def to_json(self, filename: str = 'model.json') -> None:

        params = {
            'theta_0': self.y_intercept,
            'theta_1': self.slope
        }

        try:

            with open(filename, 'w') as f:
                json.dump(params, f, indent=4)

            print(f"Parameters successfully saved to: {os.path.abspath(filename)}")

        except IOError as e:
            print(f"Error saving parameters to {os.path.abspath(filename)}: {e}")


    def from_json(self, filename: str = 'model.json') -> bool:

        try:

            with open(filename, 'r') as f:
                params = json.load(f)
                self.y_intercept  = params.get('theta_0', 0.0)
                self.slope = params.get('theta_1', 0.0)

            print(f"Parameters successfully loaded from {filename}")
            return True

        except (IOError, json.JSONDecodeError) as e:

            print(f"Warning: Could not load parameters from {filename}. Using default values (0, 0). Reason: {e}")

            self.y_intercept = 0.0
            self.slope = 0.0

            return False


def load_pkl_model(filename: str = 'model.pkl') -> LinearRegressionModel:
    
    try:
    
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        print(f"Model successfully loaded from {os.path.abspath(filename)}")
        return model
    
    except (IOError, pickle.PickleError) as e:
        print(f"Error loading model from {os.path.abspath(filename)}: {e}")
        return None
