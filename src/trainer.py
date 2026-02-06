import logging
import joblib
import warnings
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    cross_val_score, cross_validate, GridSearchCV, learning_curve
)
from model import ModelFactory
from evaluator import Evaluator
from config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles model training, evaluation, cross-validation, hyperparameter tuning,
    and persistence.
    """

    def __init__(self) -> None:
        self.factory = ModelFactory()
        self.evaluator = Evaluator()
        self.cv_folds: int = Config.CV_FOLDS

    # -------------------------------------------------
    # Train All Models with Optional CV
    # -------------------------------------------------
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, dict]:
        """
        Train all models with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with all model results
        """
        results = {}

        for model_name in Config.MODELS:
            try:
                logger.info(f"Training {model_name}...")
                model = self.factory.get_model(model_name)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    
                    # Apply hyperparameter tuning if enabled
                    if Config.ENABLE_HYPERPARAMETER_TUNING:
                        model = self._tune_hyperparameters(model, model_name, X_train, y_train)
                    
                    # Train final model
                    model.fit(X_train, y_train)

                # Evaluate on test set
                y_pred = model.predict(X_test)
                metrics = self.evaluator.evaluate(y_test, y_pred)
                
                # Cross-validation scores
                cv_scores = None
                if Config.USE_CV:
                    cv_scores = self._compute_cv_scores(model, X_train, y_train)
                    metrics["CV_R2_mean"] = cv_scores["R2_mean"]
                    metrics["CV_R2_std"] = cv_scores["R2_std"]
                    logger.info(f"{model_name} - CV R2: {cv_scores['R2_mean']:.4f} (+/- {cv_scores['R2_std']:.4f})")

                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "cv_scores": cv_scores
                }
                
                logger.info(f"Completed {model_name} - Test R2: {metrics['R2']:.4f}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue

        return results

    # -------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------
    def _tune_hyperparameters(
        self, 
        model: object, 
        model_name: str, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> object:
        """
        Tune hyperparameters using GridSearchCV for specified models.
        
        Args:
            model: Base model instance
            model_name: Name of the model
            X: Training features
            y: Training target
            
        Returns:
            Model with best hyperparameters
        """
        try:
            param_grid = self._get_param_grid(model_name)
            if not param_grid:
                return model
            
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=self.cv_folds,
                scoring="r2",
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best CV R2 for {model_name}: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed for {model_name}: {str(e)}. Using default parameters.")
            return model

    def _get_param_grid(self, model_name: str) -> Dict:
        """
        Get parameter grid for GridSearchCV based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with parameter grid
        """
        param_grids = {
            "ridge": {
                "alpha": Config.RIDGE_ALPHA_RANGE
            },
            "lasso": {
                "alpha": Config.LASSO_ALPHA_RANGE
            },
            "elasticnet": {
                "alpha": Config.ELASTICNET_ALPHA_RANGE,
                "l1_ratio": Config.ELASTICNET_L1_RATIO_RANGE
            },
            "random_forest": {
                "n_estimators": Config.RF_N_ESTIMATORS_RANGE,
                "max_depth": Config.RF_MAX_DEPTH_RANGE
            },
            "gradient_boosting": {
                "n_estimators": Config.GB_N_ESTIMATORS_RANGE,
                "learning_rate": Config.GB_LEARNING_RATE_RANGE,
                "max_depth": Config.GB_MAX_DEPTH_RANGE
            },
            "xgboost": {
                "n_estimators": Config.XGB_N_ESTIMATORS_RANGE,
                "learning_rate": Config.XGB_LEARNING_RATE_RANGE,
                "max_depth": Config.XGB_MAX_DEPTH_RANGE
            }
        }
        
        return param_grids.get(model_name, {})

    # -------------------------------------------------
    # Cross-Validation
    # -------------------------------------------------
    def _compute_cv_scores(
        self, 
        model: object, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute cross-validation scores.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            
        Returns:
            Dictionary with CV mean and std
        """
        try:
            scores = cross_val_score(
                model, X, y, 
                cv=self.cv_folds, 
                scoring="r2",
                n_jobs=-1
            )
            
            return {
                "R2_mean": scores.mean(),
                "R2_std": scores.std(),
                "R2_scores": scores
            }
        except Exception as e:
            logger.error(f"Error computing CV scores: {str(e)}")
            return None

    # -------------------------------------------------
    # Learning Curves
    # -------------------------------------------------
    def compute_learning_curves(
        self, 
        model: object, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute learning curves to detect overfitting/underfitting.
        
        Args:
            model: Trained model
            X: Training features
            y: Training target
            
        Returns:
            Tuple of (train_sizes, train_scores, val_scores)
        """
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model,
                X, y,
                cv=self.cv_folds,
                scoring="r2",
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5)
            )
            
            return train_sizes, train_scores, val_scores
        except Exception as e:
            logger.error(f"Error computing learning curves: {str(e)}")
            return None, None, None

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Predictions
        """
        try:
            return model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------
    def save_model(self, model: object, path: str = Config.MODEL_SAVE_PATH) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Model to save
            path: Save path
        """
        try:
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            joblib.dump(model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str = Config.MODEL_SAVE_PATH) -> object:
        """
        Load trained model from disk.
        
        Args:
            path: Model file path
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
