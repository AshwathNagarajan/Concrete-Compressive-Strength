import logging
from typing import Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from config import Config

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating regression models dynamically.
    """

    def __init__(self) -> None:
        self.random_state: int = Config.RANDOM_STATE

    def get_model(self, model_name: str) -> BaseEstimator:
        """
        Create a regression model by name.
        
        Args:
            model_name: Name of the model (linear, ridge, lasso, elasticnet,
                       decision_tree, random_forest, gradient_boosting, xgboost)
            
        Returns:
            sklearn BaseEstimator model instance
            
        Raises:
            ValueError: If model_name is not supported
        """
        try:
            if model_name == "linear":
                return LinearRegression()

            elif model_name == "ridge":
                return Ridge(
                    alpha=Config.RIDGE_ALPHA,
                    random_state=self.random_state
                )

            elif model_name == "lasso":
                return Lasso(
                    alpha=Config.LASSO_ALPHA,
                    max_iter=10000,
                    random_state=self.random_state
                )

            elif model_name == "elasticnet":
                return ElasticNet(
                    alpha=Config.ELASTICNET_ALPHA,
                    l1_ratio=Config.ELASTICNET_L1_RATIO,
                    max_iter=10000,
                    random_state=self.random_state
                )

            elif model_name == "decision_tree":
                return DecisionTreeRegressor(
                    max_depth=Config.TREE_MAX_DEPTH,
                    random_state=self.random_state,
                    min_samples_split=2
                )

            elif model_name == "random_forest":
                return RandomForestRegressor(
                    n_estimators=Config.RF_N_ESTIMATORS,
                    max_depth=Config.RF_MAX_DEPTH,
                    random_state=self.random_state,
                    n_jobs=Config.RF_N_JOBS,
                    min_samples_split=Config.RF_MIN_SAMPLES_SPLIT
                )

            elif model_name == "gradient_boosting":
                return GradientBoostingRegressor(
                    n_estimators=Config.GB_N_ESTIMATORS,
                    learning_rate=Config.GB_LEARNING_RATE,
                    max_depth=Config.GB_MAX_DEPTH,
                    random_state=self.random_state,
                    validation_fraction=0.1,
                    n_iter_no_change=10
                )

            elif model_name == "xgboost":
                return XGBRegressor(
                    n_estimators=Config.XGB_N_ESTIMATORS,
                    learning_rate=Config.XGB_LEARNING_RATE,
                    max_depth=Config.XGB_MAX_DEPTH,
                    subsample=Config.XGB_SUBSAMPLE,
                    colsample_bytree=Config.XGB_COLSAMPLE_BYTREE,
                    random_state=self.random_state,
                    n_jobs=Config.XGB_N_JOBS,
                    verbosity=0,
                    tree_method='auto'
                )

            else:
                raise ValueError(f"Unsupported model: {model_name}. "
                               f"Supported models: {', '.join(Config.MODELS)}")
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {str(e)}")
            raise

