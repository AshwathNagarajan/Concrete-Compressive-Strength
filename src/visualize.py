import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    mean_absolute_percentage_error
)
from config import Config

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Handles all visual diagnostics for data, features, and models.
    """

    def __init__(self) -> None:
        self.target_col: str = Config.TARGET_COL
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 12)
        logger.info("Visualizer initialized")

    # -------------------------------------------------
    # Dataset-Level Visualizations (Subplots)
    # -------------------------------------------------
    def plot_distributions(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot distributions of all numeric columns."""
        try:
            cols = [col for col in df.columns if col != self.target_col]
            n_cols = len(cols)
            n_rows = (n_cols + 2) // 3
            
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
            axes = axes.flatten()
            
            for idx, col in enumerate(cols):
                sns.histplot(df[col], kde=True, ax=axes[idx])
                axes[idx].set_title(f"Distribution of {col}")
            
            for idx in range(len(cols), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Distribution plot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting distributions: {str(e)}")

    def plot_boxplots(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot boxplots for outlier detection."""
        try:
            cols = [col for col in df.columns if col != self.target_col]
            n_cols = len(cols)
            n_rows = (n_cols + 2) // 3
            
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
            axes = axes.flatten()
            
            for idx, col in enumerate(cols):
                sns.boxplot(y=df[col], ax=axes[idx])
                axes[idx].set_title(f"Boxplot of {col}")
            
            for idx in range(len(cols), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Boxplot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting boxplots: {str(e)}")

    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot correlation heatmap."""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Correlation heatmap saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting correlation heatmap: {str(e)}")

    # -------------------------------------------------
    # Target Relationship Visualizations
    # -------------------------------------------------
    def plot_feature_vs_target(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot scatter plots of features vs target."""
        try:
            cols = [col for col in df.columns if col != self.target_col]
            n_cols = len(cols)
            n_rows = (n_cols + 2) // 3
            
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
            axes = axes.flatten()
            
            for idx, col in enumerate(cols):
                sns.scatterplot(x=df[col], y=df[self.target_col], ax=axes[idx])
                axes[idx].set_title(f"{col} vs {self.target_col}")
            
            for idx in range(len(cols), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature vs Target plot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting feature vs target: {str(e)}")

    # -------------------------------------------------
    # Model Performance Visualizations
    # -------------------------------------------------
    def plot_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot actual vs predicted and residuals."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Actual vs Predicted scatter
            axes[0].scatter(y_true, y_pred, alpha=0.6)
            axes[0].plot([y_true.min(), y_true.max()],
                         [y_true.min(), y_true.max()],
                         color="red", linestyle="--", linewidth=2)
            axes[0].set_xlabel("Actual")
            axes[0].set_ylabel("Predicted")
            axes[0].set_title("Actual vs Predicted")
            axes[0].grid(True, alpha=0.3)
            
            # Residuals scatter
            residuals = y_true - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6)
            axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Residuals")
            axes[1].set_title("Residuals vs Predicted")
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Predictions plot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")

    def plot_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot residual distribution and Q-Q plot."""
        try:
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram of residuals
            sns.histplot(residuals, kde=True, ax=axes[0])
            axes[0].set_title("Residual Distribution")
            axes[0].set_xlabel("Residuals")
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q Plot")
            
            plt.tight_layout()
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Residuals plot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")

    # -------------------------------------------------
    # Feature Importance Visualization
    # -------------------------------------------------
    def plot_feature_importance(
        self, 
        model: object, 
        feature_names: list,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            save_path: Optional path to save figure
        """
        try:
            if not hasattr(model, "feature_importances_"):
                logger.warning(f"Model {type(model).__name__} doesn't have feature_importances_")
                return
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")

    # -------------------------------------------------
    # Learning Curves
    # -------------------------------------------------
    def plot_learning_curves(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot learning curves to detect overfitting.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            save_path: Optional path to save figure
        """
        try:
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.title("Learning Curves")
            plt.xlabel("Training Set Size")
            plt.ylabel("Score (R²)")
            plt.grid(alpha=0.3)
            
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
            plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
            
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
            plt.plot(train_sizes, val_mean, "o-", color="g", label="Validation score")
            
            plt.legend(loc="best")
            plt.tight_layout()
            
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Learning curves saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting learning curves: {str(e)}")

    # -------------------------------------------------
    # Model Comparison
    # -------------------------------------------------
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "R2",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of all models.
        
        Args:
            comparison_df: DataFrame with model metrics
            metric: Metric to plot
            save_path: Optional path to save figure
        """
        try:
            comparison_df_sorted = comparison_df.sort_values(metric, ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.bar(comparison_df_sorted["Model"], comparison_df_sorted[metric])
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.title(f"Model Comparison - {metric}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if Config.ENABLE_VISUALIZATIONS:
                plt.show()
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Model comparison plot saved to {save_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting model comparison: {str(e)}")

    # -------------------------------------------------
    # Metrics Summary
    # -------------------------------------------------
    def print_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute and print evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        try:
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

            print("\n" + "="*50)
            print("Model Performance Metrics")
            print("="*50)
            print(f"R² Score : {r2:.4f}")
            print(f"RMSE     : {rmse:.4f}")
            print(f"MAE      : {mae:.4f}")
            print(f"MAPE     : {mape:.4f}")
            print("="*50 + "\n")
            
            return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}
        except Exception as e:
            logger.error(f"Error printing metrics: {str(e)}")
            return {}

    # -------------------------------------------------
    # Dataset-Level Visualizations (Subplots)
    # -------------------------------------------------
    def plot_distributions(self, df: pd.DataFrame):
        cols = [col for col in df.columns if col != self.target_col]
        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for idx, col in enumerate(cols):
            sns.histplot(df[col], kde=True, ax=axes[idx])
            axes[idx].set_title(f"Distribution of {col}")
        
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, df: pd.DataFrame):
        cols = [col for col in df.columns if col != self.target_col]
        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for idx, col in enumerate(cols):
            sns.boxplot(y=df[col], ax=axes[idx])
            axes[idx].set_title(f"Boxplot of {col}")
        
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # Target Relationship Visualizations (Subplots)
    # -------------------------------------------------
    def plot_feature_vs_target(self, df: pd.DataFrame):
        cols = [col for col in df.columns if col != self.target_col]
        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for idx, col in enumerate(cols):
            sns.scatterplot(x=df[col], y=df[self.target_col], ax=axes[idx])
            axes[idx].set_title(f"{col} vs {self.target_col}")
        
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # Model Performance Visualizations (Subplots)
    # -------------------------------------------------
    def plot_predictions(self, y_true, y_pred):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted scatter
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     color="red", linestyle="--", linewidth=2)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Actual vs Predicted")
        axes[0].grid(True, alpha=0.3)
        
        # Residuals scatter
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residuals vs Predicted")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=axes[0])
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residuals")
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot")
        
        plt.tight_layout()
        plt.show()

    def print_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        print("\nModel Performance:")
        print(f"R² Score : {r2:.4f}")
        print(f"RMSE     : {rmse:.4f}")
        print(f"MAE      : {mae:.4f}")
        print(f"MAPE     : {mape:.4f}")