import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results):
    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, scores)
    plt.ylabel("R2 Score")
    plt.title("Model Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
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
    plt.show()


import matplotlib.pyplot as plt


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
    plt.show()