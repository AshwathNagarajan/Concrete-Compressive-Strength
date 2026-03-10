from sklearn.ensemble import RandomForestRegressor

class RandomForest:
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        
    def train_random_forest(self, X_train, y_train):
        print("\nTraining Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        print("Random Forest training completed!")
        
        self.model = rf_model