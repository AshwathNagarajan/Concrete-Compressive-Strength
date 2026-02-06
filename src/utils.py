import os
import logging
import pandas as pd
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class Utils:
    """Utility functions for the ML pipeline."""
    
    @staticmethod
    def save_results(results: Dict, path: str = "model/model_comparison.csv") -> None:
        """
        Save model results to CSV.
        
        Args:
            results: Dictionary with model results
            path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            df = pd.DataFrame(results).T
            df.to_csv(path)
            logger.info(f"Results saved to {path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    @staticmethod
    def create_results_summary(
        best_model_name: str,
        best_metrics: Dict,
        comparison_df: pd.DataFrame,
        output_file: str = "model/training_summary.txt"
    ) -> None:
        """
        Create a text summary of training results.
        
        Args:
            best_model_name: Name of best model
            best_metrics: Best model metrics
            comparison_df: DataFrame with all model metrics
            output_file: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            
            with open(output_file, "w") as f:
                f.write("=" * 70 + "\n")
                f.write("CONCRETE COMPRESSIVE STRENGTH ML TRAINING SUMMARY\n")
                f.write("=" * 70 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("BEST MODEL\n")
                f.write("-" * 70 + "\n")
                f.write(f"Model Name: {best_model_name}\n")
                for metric, value in best_metrics.items():
                    if "CV" not in metric:
                        f.write(f"{metric:20s}: {value:.6f}\n")
                if "CV_R2_mean" in best_metrics:
                    f.write(f"CV R2 (mean):        {best_metrics['CV_R2_mean']:.6f}\n")
                    f.write(f"CV R2 (std):         {best_metrics['CV_R2_std']:.6f}\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("ALL MODELS COMPARISON\n")
                f.write("=" * 70 + "\n")
                f.write(comparison_df.to_string())
                f.write("\n")
            
            logger.info(f"Summary saved to {output_file}")
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            raise
    
    @staticmethod
    def verify_data(df: pd.DataFrame) -> bool:
        """
        Verify data integrity.
        
        Args:
            df: DataFrame to verify
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check for null values
            if df.isnull().sum().sum() > 0:
                logger.warning(f"Found {df.isnull().sum().sum()} null values in data")
                return False
            
            # Check for non-numeric columns (except target)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) != len(df.columns):
                logger.warning("Found non-numeric columns in data")
                return False
            
            logger.info("Data verification passed")
            return True
        except Exception as e:
            logger.error(f"Error verifying data: {str(e)}")
            return False
    
    @staticmethod
    def print_configuration() -> None:
        """Print current configuration."""
        from config import Config
        
        logger.info("\n" + "=" * 70)
        logger.info("CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Data path: {Config.DATA_PATH}")
        logger.info(f"Target column: {Config.TARGET_COL}")
        logger.info(f"Test size: {Config.TEST_SIZE}")
        logger.info(f"CV folds: {Config.CV_FOLDS}")
        logger.info(f"Random state: {Config.RANDOM_STATE}")
        logger.info(f"Scaler type: {Config.SCALER_TYPE}")
        logger.info(f"Distribution transform: {Config.DISTRIBUTION_TRANSFORM}")
        logger.info(f"Selection metric: {Config.SELECTION_METRIC}")
        logger.info(f"Models: {', '.join(Config.MODELS)}")
        logger.info("=" * 70 + "\n")
