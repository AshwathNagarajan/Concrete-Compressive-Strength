import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()


def plot_model_comparison(results, title="Model Comparison", output_path="model_comparison.png"):
    names = list(results.keys())
    scores = list(results.values())
    y_min = min(scores)
    y_max = max(scores)
    y_range = max(y_max - y_min, 1e-6)
    label_offset = 0.02 * y_range

    plt.figure(figsize=(8, 5))
    colors = plt.cm.Set2(range(len(names)))
    bars = plt.bar(names, scores, color=colors, edgecolor="black", linewidth=0.8)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylabel("R2 Score")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Strength")
    plt.ylabel("Predicted Strength")
    plt.title(f"Actual vs Predicted - {model_name}")

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{model_name}.png")
    plt.show()


def plot_roc_auc_comparison(y_true_binary, model_scores):
    plt.figure(figsize=(8, 6))
    auc_scores = {}

    for model_name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true_binary, scores)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_auc_comparison.png")
    plt.show()

    return auc_scores



def plot_all_feature_importance(trained_models, feature_names):
    models = [
        (name, model)
        for name, model in trained_models.items()
        if hasattr(model, "feature_importances_")
    ]

    n = len(models)
    if n == 0:
        print("No models support feature importance.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models):
        importance = model.feature_importances_

        ax.barh(feature_names, importance)
        ax.set_title(name)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png")
    plt.show()