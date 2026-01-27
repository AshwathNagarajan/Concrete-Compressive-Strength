from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


class Evaluator:
    """
    Computes regression performance metrics.
    """

    def evaluate(self, y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
