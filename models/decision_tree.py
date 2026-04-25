from sklearn.tree import DecisionTreeRegressor


def get_model():
    return DecisionTreeRegressor(max_depth=10, random_state=42)
