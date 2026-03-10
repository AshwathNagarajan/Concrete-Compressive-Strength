import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class Plot:

    def __init__(self, viz_dir):
        self.viz_dir = viz_dir

    def plot_correlation_matrix(self, X, y):
        print("\nGenerating correlation matrix plot...")
        
        data = pd.concat([X, y], axis=1)
        corr_matrix = data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/01_correlation_matrix.png", dpi=100)
        plt.close()
        print(f"Saved: {self.viz_dir}/01_correlation_matrix.png")


    def plot_predictions_vs_actual(self, y_test, results):
        print("\nGenerating predictions vs actual plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, result in enumerate(results):
            ax = axes[idx]
            y_pred = result["y_pred"]
            model_name = result["model_name"]
            r2 = result["r2"]
            
            ax.scatter(y_test, y_pred, alpha=0.6, s=30)
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
            
            ax.set_xlabel("Actual Strength (MPa)")
            ax.set_ylabel("Predicted Strength (MPa)")
            ax.set_title(f"{model_name}\n(R² = {r2:.4f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/02_predictions_vs_actual.png", dpi=100)
        plt.close()
        print(f"Saved: {self.viz_dir}/02_predictions_vs_actual.png")


    def plot_feature_importance(self, models_dict, feature_names):
        print("\nGenerating feature importance plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            ax = axes[idx]
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            ax.barh(range(len(indices)), importances[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel("Importance Score")
            ax.set_title(f"{model_name}\nFeature Importance (Top 10)")
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/03_feature_importance.png", dpi=100)
        plt.close()
        print(f"Saved: {self.viz_dir}/03_feature_importance.png")
