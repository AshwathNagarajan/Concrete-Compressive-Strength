from xgboost import XGBRegressor


def get_model():
    return XGBRegressor(
        n_estimators=100,
        random_state=42,
        objective="reg:squarederror",
        verbosity=0,
    )