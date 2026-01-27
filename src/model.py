from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from config import Config


class ModelFactory:
    """
    Factory class for creating regression models dynamically.
    """

    def __init__(self):
        self.random_state = Config.RANDOM_STATE

    def get_model(self, model_name: str):
        if model_name == "linear":
            return LinearRegression()

        elif model_name == "ridge":
            return Ridge(alpha=Config.RIDGE_ALPHA)

        elif model_name == "lasso":
            return Lasso(alpha=Config.LASSO_ALPHA)

        elif model_name == "elasticnet":
            return ElasticNet(
                alpha=Config.ELASTICNET_ALPHA,
                l1_ratio=Config.ELASTICNET_L1_RATIO,
                random_state=self.random_state
            )

        elif model_name == "decision_tree":
            return DecisionTreeRegressor(
                max_depth=Config.TREE_MAX_DEPTH,
                random_state=self.random_state
            )

        elif model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=Config.RF_N_ESTIMATORS,
                max_depth=Config.RF_MAX_DEPTH,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif model_name == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=Config.GB_N_ESTIMATORS,
                learning_rate=Config.GB_LEARNING_RATE,
                max_depth=Config.GB_MAX_DEPTH,
                random_state=self.random_state
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")
