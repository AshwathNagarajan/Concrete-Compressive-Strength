import os
import sys
import logging
import argparse
import joblib
from datetime import datetime
from preprocessing import DataPreprocessor
from features import FeatureEngineer
from trainer import Trainer
from evaluator import Evaluator
from visualize import Visualizer
from config import Config

# -------------------------------------------------
# Configure Logging
# -------------------------------------------------
def setup_logging(log_file: str = Config.LOG_FILE) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Logger instance
    """
    log_dir = os.path.dirname(log_file) or "."
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# -------------------------------------------------
# Parse Command-Line Arguments
# -------------------------------------------------
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Concrete Compressive Strength ML Pipeline"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=Config.DATA_PATH,
        help="Path to input data CSV file"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=Config.MODELS,
        help="List of models to train"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=Config.CV_FOLDS,
        help="Number of cross-validation folds"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=Config.TEST_SIZE,
        help="Test set fraction"
    )
    
    parser.add_argument(
        "--disable-visualizations",
        action="store_true",
        help="Disable visualizations"
    )
    
    parser.add_argument(
        "--disable-tuning",
        action="store_true",
        help="Disable hyperparameter tuning"
    )
    
    parser.add_argument(
        "--disable-cv",
        action="store_true",
        help="Disable cross-validation"
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Concrete Compressive Strength Prediction System")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Update config from arguments
        Config.CV_FOLDS = args.cv_folds
        Config.TEST_SIZE = args.test_size
        Config.MODELS = args.models
        Config.ENABLE_VISUALIZATIONS = not args.disable_visualizations
        Config.ENABLE_HYPERPARAMETER_TUNING = not args.disable_tuning
        Config.USE_CV = not args.disable_cv
        
        logger.info(f"Configuration: CV_FOLDS={Config.CV_FOLDS}, TEST_SIZE={Config.TEST_SIZE}")
        logger.info(f"Models to train: {', '.join(Config.MODELS)}")
        
        # -------------------------------------------------
        # 1. Load Data + Feature Engineering
        # -------------------------------------------------
        logger.info("Starting data loading and feature engineering...")
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        visualizer = Visualizer()

        logger.info(f"Loading dataset from {args.data_path}...")
        data = preprocessor.load_data(args.data_path)
        logger.info(f"Dataset shape: {data.shape}")

        logger.info("Performing feature engineering...")
        data = feature_engineer.transform(data)
        logger.info(f"After feature engineering shape: {data.shape}")

        # -------------------------------------------------
        # 2. Exploratory Visualizations
        # -------------------------------------------------
        if Config.ENABLE_VISUALIZATIONS:
            logger.info("Generating exploratory visualizations...")
            try:
                os.makedirs("visualizations", exist_ok=True)
                visualizer.plot_distributions(data, "visualizations/distributions.png")
                visualizer.plot_boxplots(data, "visualizations/boxplots.png")
                visualizer.plot_correlation_heatmap(data, "visualizations/correlation.png")
                visualizer.plot_feature_vs_target(data, "visualizations/feature_vs_target.png")
            except Exception as e:
                logger.warning(f"Error generating exploratory visualizations: {str(e)}")

        # -------------------------------------------------
        # 3. Preprocessing
        # -------------------------------------------------
        logger.info("Preprocessing data...")
        X, y = preprocessor.split_feature_target(data)
        preprocessor.build_pipeline()
        X_processed = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = preprocessor.split_train_test(X_processed, y)
        logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # -------------------------------------------------
        # 4. Model Training (with CV and Hyperparameter Tuning)
        # -------------------------------------------------
        logger.info("Starting model training...")
        trainer = Trainer()
        results = trainer.train(X_train, y_train, X_test, y_test)

        if not results:
            logger.error("No models trained successfully!")
            return

        # -------------------------------------------------
        # 5. Model Evaluation & Selection
        # -------------------------------------------------
        logger.info("Evaluating and selecting best model...")
        evaluator = Evaluator()
        best_model_name, best_model, best_metrics = evaluator.select_best_model(results)

        logger.info("\n" + "=" * 60)
        logger.info("BEST MODEL SELECTED")
        logger.info("=" * 60)
        logger.info(f"Model: {best_model_name}")
        for k, v in best_metrics.items():
            if "CV" not in k:  # Skip CV metrics in main output
                logger.info(f"{k:15s}: {v:.4f}")
        if "CV_R2_mean" in best_metrics:
            logger.info(f"CV R2 (mean ± std): {best_metrics['CV_R2_mean']:.4f} ± {best_metrics['CV_R2_std']:.4f}")
        logger.info("=" * 60 + "\n")

        # -------------------------------------------------
        # 6. Generate Model Comparison Report
        # -------------------------------------------------
        logger.info("Generating model comparison report...")
        comparison_df = evaluator.get_model_comparison_df(results)
        comparison_df.to_csv(Config.RESULTS_SAVE_PATH, index=False)
        logger.info(f"Model comparison saved to {Config.RESULTS_SAVE_PATH}")
        logger.info("\nModel Comparison Table:")
        logger.info(comparison_df.to_string())

        # -------------------------------------------------
        # 7. Visualizations of Best Model
        # -------------------------------------------------
        if Config.ENABLE_VISUALIZATIONS:
            logger.info("Generating best model visualizations...")
            try:
                os.makedirs("visualizations", exist_ok=True)
                y_pred = trainer.predict(best_model, X_test)
                
                visualizer.plot_predictions(y_test, y_pred, "visualizations/predictions.png")
                visualizer.plot_residuals(y_test, y_pred, "visualizations/residuals.png")
                visualizer.plot_model_comparison(comparison_df, "R2", "visualizations/model_comparison.png")
                
                # Feature importance (if applicable)
                if hasattr(best_model, "feature_importances_"):
                    feature_names = preprocessor.feature_names
                    visualizer.plot_feature_importance(best_model, feature_names, "visualizations/feature_importance.png")
                
                # Learning curves (if enabled)
                if Config.USE_CV and Config.PLOT_LEARNING_CURVES:
                    logger.info("Computing learning curves for best model...")
                    train_sizes, train_scores, val_scores = trainer.compute_learning_curves(
                        best_model, X_train, y_train
                    )
                    if train_sizes is not None:
                        visualizer.plot_learning_curves(
                            train_sizes, train_scores, val_scores, 
                            "visualizations/learning_curves.png"
                        )
            except Exception as e:
                logger.warning(f"Error generating visualizations: {str(e)}")

        # -------------------------------------------------
        # 8. Save Models and Pipeline
        # -------------------------------------------------
        logger.info("Saving models and pipeline...")
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH) or ".", exist_ok=True)
        
        # Save best model
        trainer.save_model(best_model, Config.MODEL_SAVE_PATH)
        
        # Save preprocessing pipeline for inference
        joblib.dump(preprocessor.pipeline, Config.PIPELINE_SAVE_PATH)
        logger.info(f"Preprocessing pipeline saved to {Config.PIPELINE_SAVE_PATH}")

        # -------------------------------------------------
        # 9. Summary Report
        # -------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Model saved to: {Config.MODEL_SAVE_PATH}")
        logger.info(f"Pipeline saved to: {Config.PIPELINE_SAVE_PATH}")
        logger.info(f"Results saved to: {Config.RESULTS_SAVE_PATH}")
        if Config.ENABLE_VISUALIZATIONS:
            logger.info("Visualizations saved to: visualizations/")
        logger.info("=" * 60 + "\n")
        
        logger.info(f"Completed at {datetime.now()}")
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
