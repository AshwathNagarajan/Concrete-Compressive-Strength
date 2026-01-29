from preprocessing import DataPreprocessor
from features import FeatureEngineer
from trainer import Trainer
from evaluator import Evaluator
from visualize import Visualizer
from config import Config


def main():
    print("\nConcrete Compressive Strength Prediction System\n")

    # -------------------------------------------------
    # 1. Load + Feature Engineering
    # -------------------------------------------------
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    visualizer = Visualizer()

    print("Loading dataset...")
    data = preprocessor.load_data(Config.DATA_PATH)

    print("Performing feature engineering...")
    data = feature_engineer.transform(data)

    # -------------------------------------------------
    # 2. Exploratory Visualizations
    # -------------------------------------------------
    '''print("Visualizing dataset...")
    visualizer.plot_distributions(data)
    visualizer.plot_boxplots(data)
    visualizer.plot_correlation_heatmap(data)
    visualizer.plot_feature_vs_target(data)'''

    # -------------------------------------------------
    # 3. Preprocessing
    # -------------------------------------------------
    print("Preprocessing data...")
    X, y = preprocessor.split_feature_target(data)
    preprocessor.build_pipeline()
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X_processed, y)

    # -------------------------------------------------
    # 4. Model Training (Auto Model Selection)
    # -------------------------------------------------
    print("Training all models...")
    trainer = Trainer()
    results = trainer.train(X_train, y_train, X_test, y_test)

    # -------------------------------------------------
    # 5. Model Evaluation & Selection
    # -------------------------------------------------
    evaluator = Evaluator()
    best_model_name, best_model, best_metrics = evaluator.select_best_model(results)

    print("\nBest Model Selected")
    print(f"Model : {best_model_name}")
    for k, v in best_metrics.items():
        print(f"{k} : {v:.4f}")

    # -------------------------------------------------
    # 6. Visualization of Best Model
    # -------------------------------------------------
    print("\nVisualizing best model predictions...")
    y_pred = best_model.predict(X_test)
    visualizer.plot_predictions(y_test, y_pred)
    visualizer.plot_residuals(y_test, y_pred)

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
