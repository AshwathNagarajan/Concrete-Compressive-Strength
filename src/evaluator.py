from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


class Evaluator:
    """
    Computes regression performance metrics and selects best model.
    """

    def evaluate(self, y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def select_best_model(self, results):
        """Select the best model based on R2 score."""
        best_model_name = None
        best_model = None
        best_metrics = None
        best_score = float("-inf")
        
        for model_name, result in results.items():
            metrics = result["metrics"]
            if metrics["R2"] > best_score:
                best_score = metrics["R2"]
                best_model_name = model_name
                best_model = result["model"]
                best_metrics = metrics
        
        return best_model_name, best_model, best_metrics
