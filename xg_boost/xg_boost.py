
from xgboost import XGBRegressor

    


class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def train_xgboost(self, X_train, y_train):
        print("\nTraining XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        print("XGBoost training completed!")
        self.model = xgb_model