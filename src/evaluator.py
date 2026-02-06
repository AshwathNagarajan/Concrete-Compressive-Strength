import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)
from config import Config

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Computes regression performance metrics and selects best model.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary with R2, RMSE, MAE, MAPE metrics
        """
        try:
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            
            return {
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape
            }
        except Exception as e:
            logger.error(f"Error evaluating metrics: {str(e)}")
            raise
    
    @staticmethod
    def evaluate_cv(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for cross-validation results.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        return Evaluator.evaluate(y_true, y_pred)
    
    def select_best_model(
        self, 
        results: Dict[str, dict]
    ) -> Tuple[str, object, Dict[str, float]]:
        """
        Select the best model based on configured metric.
        
        Args:
            results: Dictionary with model results
                    {model_name: {"model": model_obj, "metrics": {...}}}
            
        Returns:
            Tuple of (best_model_name, best_model_object, best_metrics)
            
        Raises:
            ValueError: If invalid selection metric or empty results
        """
        try:
            if not results:
                raise ValueError("No models provided for selection")
            
            selection_metric = Config.SELECTION_METRIC
            if selection_metric not in ["R2", "RMSE", "MAE", "MAPE"]:
                raise ValueError(f"Invalid selection metric: {selection_metric}")
            
            best_model_name = None
            best_model = None
            best_metrics = None
            
            if selection_metric == "R2":
                # Higher R2 is better
                best_score = float("-inf")
                for model_name, result in results.items():
                    metrics = result["metrics"]
                    if metrics["R2"] > best_score:
                        best_score = metrics["R2"]
                        best_model_name = model_name
                        best_model = result["model"]
                        best_metrics = metrics
            else:
                # Lower RMSE, MAE, MAPE is better
                best_score = float("inf")
                for model_name, result in results.items():
                    metrics = result["metrics"]
                    if metrics[selection_metric] < best_score:
                        best_score = metrics[selection_metric]
                        best_model_name = model_name
                        best_model = result["model"]
                        best_metrics = metrics
            
            logger.info(f"Best model selected: {best_model_name} with {selection_metric}={best_score:.4f}")
            return best_model_name, best_model, best_metrics
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            raise
    
    @staticmethod
    def get_model_comparison_df(results: Dict[str, dict]) -> pd.DataFrame:
        """
        Create a comparison DataFrame of all models.
        
        Args:
            results: Dictionary with model results
            
        Returns:
            DataFrame with all models and metrics
        """
        comparison_data = []
        for model_name, result in results.items():
            metrics = result["metrics"]
            metrics["Model"] = model_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        # Reorder columns
        cols = ["Model"] + [col for col in df.columns if col != "Model"]
        df = df[cols]
        
        # Sort by R2 descending
        df = df.sort_values("R2", ascending=False).reset_index(drop=True)
        
        return df
