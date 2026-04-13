import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from preprocess.preprocess import load_data, feature_engineering, split_data, scale_data

from models.linear import get_model as linear
from models.svm import get_model as svm
from models.knn import get_model as knn
from models.mlp import get_model as mlp
from models.random_forest import get_model as rf
from models.xgboost_model import get_model as xgb

from visualization.plots import (
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_all_feature_importance,
)


df = load_data("data/concrete.csv")
df = feature_engineering(df)
plot_correlation_heatmap(df)

X, y = split_data(df)
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

models = {
    "Linear": linear(),
    "SVM": svm(),
    "KNN": knn(),
    "MLP": mlp(),
    "RandomForest": rf(),
    "XGBoost": xgb(),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
trained_models = {}
predictions = {}

print("\nResults:\n")

scale_sensitive_models = {"SVM", "KNN", "MLP", "Linear"}

for name, model in models.items():
    print(f"Running {name}...")

    if name in scale_sensitive_models:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"CV Mean: {cv_scores.mean():.4f}")
    print("-" * 30)

    results[name] = r2
    trained_models[name] = model
    predictions[name] = preds

print("\nFinal Ranking:\n")
for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")
plot_model_comparison(results)

best_model_name = max(results, key=results.get)
best_preds = predictions[best_model_name]
plot_actual_vs_predicted(y_test, best_preds, best_model_name)

plot_all_feature_importance(trained_models, feature_names)