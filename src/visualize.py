import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from config import Config


class Visualizer:
    """
    Handles all visual diagnostics for data, features, and models.
    """

    def __init__(self):
        self.target_col = Config.TARGET_COL
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 12)

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