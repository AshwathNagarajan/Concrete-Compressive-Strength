import joblib
from config import Config


class Trainer:
    """
    Handles model training, prediction, and persistence.
    """

    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def save_model(self, model, path=Config.MODEL_SAVE_PATH):
        joblib.dump(model, path)

    def load_model(self, path=Config.MODEL_SAVE_PATH):
        return joblib.load(path)
