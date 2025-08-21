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
        self.error_history: list[float] = []

    def get_slope(self) -> float:
        return self.slope
    

    def get_y_intercept(self) -> float:
        return self.y_intercept


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
            errors = predictions - y_ndarray

            mean_squared_error = np.sum(errors ** 2) / (2 * m)
            self.error_history.append(mean_squared_error)

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
