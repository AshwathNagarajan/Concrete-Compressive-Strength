import unittest
import numpy as np
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import ModelFactory
from evaluator import Evaluator
from config import Config


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
    
    def test_get_linear_model(self):
        """Test linear regression model creation."""
        model = self.factory.get_model("linear")
        self.assertIsNotNone(model)
    
    def test_get_ridge_model(self):
        """Test Ridge model creation."""
        model = self.factory.get_model("ridge")
        self.assertIsNotNone(model)
    
    def test_get_random_forest_model(self):
        """Test Random Forest model creation."""
        model = self.factory.get_model("random_forest")
        self.assertIsNotNone(model)
    
    def test_get_xgboost_model(self):
        """Test XGBoost model creation."""
        model = self.factory.get_model("xgboost")
        self.assertIsNotNone(model)
    
    def test_invalid_model_name(self):
        """Test error with invalid model name."""
        with self.assertRaises(ValueError):
            self.factory.get_model("invalid_model_name")
    
    def test_all_configured_models(self):
        """Test that all configured models can be created."""
        for model_name in Config.MODELS:
            model = self.factory.get_model(model_name)
            self.assertIsNotNone(model, f"Failed to create {model_name} model")


class TestEvaluator(unittest.TestCase):
    """Test cases for Evaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = Evaluator()
        
        # Create sample predictions
        np.random.seed(42)
        self.y_true = np.array([30.0, 35.0, 40.0, 45.0, 50.0])
        self.y_pred = self.y_true + np.random.normal(0, 2, 5)
    
    def test_evaluate_metrics(self):
        """Test evaluation metrics computation."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)
        
        required_metrics = ["R2", "RMSE", "MAE", "MAPE"]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_r2_range(self):
        """Test that R2 is in valid range."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)
        
        # R2 should be between -1 and 1
        self.assertGreaterEqual(metrics["R2"], -1)
        self.assertLessEqual(metrics["R2"], 1)
    
    def test_perfect_prediction(self):
        """Test metrics with perfect prediction."""
        y_true = np.array([30.0, 35.0, 40.0])
        y_pred = np.array([30.0, 35.0, 40.0])
        
        metrics = self.evaluator.evaluate(y_true, y_pred)
        
        self.assertAlmostEqual(metrics["R2"], 1.0, places=5)
        self.assertAlmostEqual(metrics["RMSE"], 0.0, places=5)
        self.assertAlmostEqual(metrics["MAE"], 0.0, places=5)
    
    def test_select_best_model_by_r2(self):
        """Test best model selection by R2."""
        Config.SELECTION_METRIC = "R2"
        
        results = {
            "model1": {
                "model": object(),
                "metrics": {"R2": 0.8, "RMSE": 5.0, "MAE": 4.0, "MAPE": 10.0}
            },
            "model2": {
                "model": object(),
                "metrics": {"R2": 0.9, "RMSE": 3.0, "MAE": 2.5, "MAPE": 7.0}
            }
        }
        
        best_name, best_model, best_metrics = self.evaluator.select_best_model(results)
        
        self.assertEqual(best_name, "model2")
        self.assertEqual(best_metrics["R2"], 0.9)
    
    def test_empty_results(self):
        """Test error with empty results."""
        with self.assertRaises(ValueError):
            self.evaluator.select_best_model({})
    
    def test_get_model_comparison_df(self):
        """Test model comparison DataFrame creation."""
        results = {
            "model1": {
                "model": object(),
                "metrics": {"R2": 0.8, "RMSE": 5.0, "MAE": 4.0, "MAPE": 10.0}
            },
            "model2": {
                "model": object(),
                "metrics": {"R2": 0.9, "RMSE": 3.0, "MAE": 2.5, "MAPE": 7.0}
            }
        }
        
        df = self.evaluator.get_model_comparison_df(results)
        
        self.assertEqual(len(df), 2)
        self.assertIn("Model", df.columns)
        self.assertIn("R2", df.columns)
        # Should be sorted by R2 descending
        self.assertEqual(df.iloc[0]["Model"], "model2")


if __name__ == '__main__':
    unittest.main()
