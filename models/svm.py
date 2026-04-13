from sklearn.svm import SVR


def get_model():
    return SVR(kernel="rbf", C=100, gamma=0.1)