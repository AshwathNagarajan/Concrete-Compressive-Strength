from model import ModelFactory
from trainer import Trainer
from evaluator import Evaluator
from config import Config


class ModelSelector:
    """
    Trains and evaluates multiple models and selects the best-performing one.
    """

    def __init__(self):
        self.factory = ModelFactory()
        self.trainer = Trainer()
        self.evaluator = Evaluator()

    def select_best_model(self, X_train, X_test, y_train, y_test):
        results = {}
        best_model = None
        best_score = float("-inf")

        for model_name in Config.MODELS:
            print(f"Training {model_name}...")

            model = self.factory.get_model(model_name)
            model = self.trainer.train(model, X_train, y_train)

            y_pred = self.trainer.predict(model, X_test)
            metrics = self.evaluator.evaluate(y_test, y_pred)

            results[model_name] = metrics

            if metrics["R2"] > best_score:
                best_score = metrics["R2"]
                best_model = model

        return best_model, results
