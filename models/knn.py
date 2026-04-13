from sklearn.neighbors import KNeighborsRegressor


def get_model():
    return KNeighborsRegressor(n_neighbors=5, weights="distance")