import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning
from model import ModelFactory
from evaluator import Evaluator
from config import Config


class Trainer:
    """
    Handles model training, evaluation, and persistence.
    """

    def __init__(self):
        self.factory = ModelFactory()
        self.evaluator = Evaluator()

    # -------------------------------------------------
    # Train All Models
    # -------------------------------------------------
    def train(self, X_train, y_train, X_test, y_test):
        results = {}

        for model_name in Config.MODELS:
            print(f"  🔹 Training {model_name}...")
            model = self.factory.get_model(model_name)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = self.evaluator.evaluate(y_test, y_pred)

            results[model_name] = {
                "model": model,
                "metrics": metrics
            }

        return results

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, model, X):
        return model.predict(X)

    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------
    def save_model(self, model, path=Config.MODEL_SAVE_PATH):
        joblib.dump(model, path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path=Config.MODEL_SAVE_PATH):
        return joblib.load(path)
