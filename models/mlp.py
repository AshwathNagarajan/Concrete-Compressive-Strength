from sklearn.neural_network import MLPRegressor


def get_model():
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)