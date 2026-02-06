import logging
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from evaluator import Evaluator
from config import Config

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Advanced model selection strategies beyond simple metric-based selection.
    """

    def __init__(self):
        self.evaluator = Evaluator()

    def select_by_metric(
        self,
        results: Dict[str, dict],
        metric: str = Config.SELECTION_METRIC
    ) -> Tuple[str, object, Dict[str, float]]:
        """
        Select model by single metric (wrapper around evaluator).
        
        Args:
            results: Model results dictionary
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_name, model_object, metrics)
        """
        return self.evaluator.select_best_model(results)

    def select_by_pareto_front(
        self,
        results: Dict[str, dict],
        metrics: List[str] = ["R2", "RMSE"]
    ) -> List[Tuple[str, object, Dict[str, float]]]:
        """
        Select models on Pareto front (maximize R2, minimize RMSE/MAE).
        
        Args:
            results: Model results dictionary
            metrics: Metrics to consider for Pareto front
            
        Returns:
            List of (model_name, model_object, metrics) on Pareto front
        """
        try:
            comparison_df = pd.DataFrame([
                {"model": name, **result["metrics"]}
                for name, result in results.items()
            ])
            
            # Normalize metrics (higher is better)
            normalized = comparison_df.copy()
            normalized["R2"] = comparison_df["R2"]  # higher is better
            normalized["RMSE"] = 1 / (1 + comparison_df["RMSE"])  # lower is better (inverted)
            normalized["MAE"] = 1 / (1 + comparison_df["MAE"])  # lower is better (inverted)
            
            # Find Pareto front
            pareto_models = []
            for idx, row in comparison_df.iterrows():
                dominated = False
                for other_idx, other_row in comparison_df.iterrows():
                    if idx != other_idx:
                        # Check if other_row dominates this row
                        dominates = True
                        for metric in metrics:
                            if metric == "R2":
                                if other_row[metric] <= row[metric]:
                                    dominates = False
                                    break
                            else:  # RMSE, MAE - lower is better
                                if other_row[metric] >= row[metric]:
                                    dominates = False
                                    break
                        if dominates:
                            dominated = True
                            break
                
                if not dominated:
                    pareto_models.append(row["model"])
            
            logger.info(f"Pareto front models: {pareto_models}")
            
            # Return models on Pareto front
            pareto_results = [
                (name, results[name]["model"], results[name]["metrics"])
                for name in pareto_models
            ]
            
            return pareto_results
        except Exception as e:
            logger.error(f"Error selecting Pareto front: {str(e)}")
            return []

    def select_by_ensemble_score(
        self,
        results: Dict[str, dict],
        weights: Dict[str, float] = None
    ) -> Tuple[str, object, Dict[str, float]]:
        """
        Select model by weighted ensemble score of multiple metrics.
        
        Args:
            results: Model results dictionary
            weights: Weights for each metric {R2: 0.4, RMSE: 0.3, MAE: 0.3}
            
        Returns:
            Tuple of (model_name, model_object, metrics)
        """
        try:
            if weights is None:
                weights = {"R2": 0.5, "RMSE": 0.25, "MAE": 0.25}
            
            comparison_df = pd.DataFrame([
                {"model": name, **result["metrics"]}
                for name, result in results.items()
            ])
            
            # Normalize metrics to [0, 1] scale
            normalized = comparison_df.copy()
            
            # R2: higher is better, scale to [0, 1]
            r2_min, r2_max = comparison_df["R2"].min(), comparison_df["R2"].max()
            normalized["R2"] = (comparison_df["R2"] - r2_min) / (r2_max - r2_min + 1e-6)
            
            # RMSE: lower is better, invert the scale
            rmse_min, rmse_max = comparison_df["RMSE"].min(), comparison_df["RMSE"].max()
            normalized["RMSE"] = 1 - (comparison_df["RMSE"] - rmse_min) / (rmse_max - rmse_min + 1e-6)
            
            # MAE: lower is better, invert the scale
            mae_min, mae_max = comparison_df["MAE"].min(), comparison_df["MAE"].max()
            normalized["MAE"] = 1 - (comparison_df["MAE"] - mae_min) / (mae_max - mae_min + 1e-6)
            
            # Calculate weighted score
            normalized["ensemble_score"] = (
                weights.get("R2", 0) * normalized["R2"] +
                weights.get("RMSE", 0) * normalized["RMSE"] +
                weights.get("MAE", 0) * normalized["MAE"]
            )
            
            best_idx = normalized["ensemble_score"].idxmax()
            best_model_name = comparison_df.iloc[best_idx]["model"]
            
            logger.info(f"Best model by ensemble score: {best_model_name}")
            
            return (
                best_model_name,
                results[best_model_name]["model"],
                results[best_model_name]["metrics"]
            )
        except Exception as e:
            logger.error(f"Error selecting by ensemble score: {str(e)}")
            # Fallback to simple R2 selection
            return self.evaluator.select_best_model(results)

    def get_top_k_models(
        self,
        results: Dict[str, dict],
        k: int = 3,
        metric: str = "R2"
    ) -> List[Tuple[str, object, Dict[str, float]]]:
        """
        Get top K models by specified metric.
        
        Args:
            results: Model results dictionary
            k: Number of top models to return
            metric: Metric to sort by
            
        Returns:
            List of top K models with their metrics
        """
        try:
            comparison_df = pd.DataFrame([
                {"model": name, **result["metrics"]}
                for name, result in results.items()
            ])
            
            # Sort by metric
            if metric in ["R2"]:
                comparison_df = comparison_df.sort_values(metric, ascending=False)
            else:  # RMSE, MAE, MAPE - lower is better
                comparison_df = comparison_df.sort_values(metric, ascending=True)
            
            top_k = comparison_df.head(k)
            
            top_k_results = [
                (row["model"], results[row["model"]]["model"], results[row["model"]]["metrics"])
                for _, row in top_k.iterrows()
            ]
            
            logger.info(f"Top {k} models by {metric}: {[m[0] for m in top_k_results]}")
            
            return top_k_results
        except Exception as e:
            logger.error(f"Error getting top K models: {str(e)}")
            return []

    def analyze_model_stability(
        self,
        results: Dict[str, dict]
    ) -> pd.DataFrame:
        """
        Analyze model stability using CV scores variance.
        
        Args:
            results: Model results dictionary with CV scores
            
        Returns:
            DataFrame with stability analysis
        """
        try:
            stability_data = []
            
            for model_name, result in results.items():
                metrics = result["metrics"]
                
                if "CV_R2_mean" in metrics and "CV_R2_std" in metrics:
                    cv_r2_mean = metrics["CV_R2_mean"]
                    cv_r2_std = metrics["CV_R2_std"]
                    test_r2 = metrics["R2"]
                    
                    stability_data.append({
                        "model": model_name,
                        "CV_R2_mean": cv_r2_mean,
                        "CV_R2_std": cv_r2_std,
                        "Test_R2": test_r2,
                        "Generalization_Gap": abs(cv_r2_mean - test_r2),
                        "Stability": 1 - (cv_r2_std / max(cv_r2_mean, 0.1))  # Higher is more stable
                    })
            
            df = pd.DataFrame(stability_data)
            
            if not df.empty:
                logger.info(f"Model Stability Analysis:\n{df.to_string()}")
            
            return df
        except Exception as e:
            logger.error(f"Error analyzing model stability: {str(e)}")
            return pd.DataFrame()
