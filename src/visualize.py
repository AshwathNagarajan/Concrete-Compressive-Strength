import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import Config


class Visualizer:
    """
    Handles all visual diagnostics for data, features, and models.
    """

    def __init__(self):
        self.target_col = Config.TARGET_COL

    # -------------------------------------------------
    # Dataset-Level Visualizations
    # -------------------------------------------------
    def plot_distributions(self, df: pd.DataFrame):
        for col in df.columns:
            plt.figure(figsize=(5, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

    def plot_boxplots(self, df: pd.DataFrame):
        for col in df.columns:
            plt.figure(figsize=(5, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    # -------------------------------------------------
    # Target Relationship Visualizations
    # -------------------------------------------------
    def plot_feature_vs_target(self, df: pd.DataFrame):
        for col in df.columns:
            if col != self.target_col:
                plt.figure(figsize=(5, 4))
                sns.scatterplot(x=df[col], y=df[self.target_col])
                plt.title(f"{col} vs {self.target_col}")
                plt.show()

    # -------------------------------------------------
    # Model Performance Visualizations
    # -------------------------------------------------
    def plot_predictions(self, y_true, y_pred):
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 color="red", linestyle="--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True)
        plt.title("Residual Distribution")
        plt.show()

    def print_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)

        print("\n📊 Model Performance:")
        print(f"R² Score : {r2:.4f}")
        print(f"RMSE     : {rmse:.4f}")
