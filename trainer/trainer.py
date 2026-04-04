import os
from preprocessing.preprocess import Preprocessing
from random_forest.random_forest import RandomForest
from xg_boost.xg_boost import XGBoostModel
from evaluation.evaluator import Evaluator
from visualizations.visual import Plot
from models.model import ModelSaver
from report.report import Report


RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = "data/concrete.csv"
TARGET_COLUMN = "strength"
MODEL_DIR = "models"
VIZ_DIR = "visualizations"


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

class Trainer:
    def __init__(self):
        self.preprocessor = Preprocessing(DATA_PATH, TARGET_COLUMN)
        self.rf_trainer = RandomForest(n_estimators=100, max_depth=20, random_state=RANDOM_STATE)
        self.xgb_trainer = XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE)
        self.evaluator = Evaluator()
        self.plotter = Plot(VIZ_DIR)
        self.saver = ModelSaver(MODEL_DIR)
        self.reporter = Report()

    def trainer(self):

        print("=" * 70)
        print("CONCRETE COMPRESSIVE STRENGTH PREDICTION PIPELINE")
        print("=" * 70)
        
        preprocessor = self.preprocessor
        
        df = preprocessor.load_data()
        
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        feature_names = X.columns.tolist()
        print(f"\nFeatures: {feature_names}")
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        X_train_scaled = preprocessor.preprocess_data(X_train, fit=True)
        X_test_scaled = preprocessor.preprocess_data(X_test, fit=False)
        
        
        rf_trainer = self.rf_trainer
        rf_trainer.train_random_forest(X_train_scaled, y_train)
        rf_model = rf_trainer.model
        
        xgb_trainer = self.xgb_trainer
        xgb_trainer.train_xgboost(X_train_scaled, y_train)
        xgb_model = xgb_trainer.model
        
        
        rf_results = self.evaluator.evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
        xgb_results = self.evaluator.evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
        
        
        plotter = self.plotter
        
        plotter.plot_correlation_matrix(X, y)
        plotter.plot_predictions_vs_actual(y_test, [rf_results, xgb_results])
        
        models_dict = {"Random Forest": rf_model, "XGBoost": xgb_model}
        plotter.plot_feature_importance(models_dict, feature_names)
        
        saver = self.saver
        saver.save_models(models_dict)
        
        reporter = self.reporter
        reporter.save_results([rf_results, xgb_results])

        