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

from visualization.plots import plot_results


# -----------------------
# Load + preprocess
# -----------------------
df = load_data("data/concrete.csv")
df = feature_engineering(df)

X, y = split_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_test = scale_data(X_train, X_test)


# -----------------------
# Models
# -----------------------
models = {
    "Linear": linear(),
    "SVM": svm(),
    "KNN": knn(),
    "MLP": mlp(),
    "RandomForest": rf(),
    "XGBoost": xgb(),
}


# -----------------------
# Evaluation
# -----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

print("\nResults:\n")

for name, model in models.items():
    print(f"Running {name}...")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"CV Mean: {cv_scores.mean():.4f}")
    print("-" * 30)

    results[name] = r2


# -----------------------
# Ranking
# -----------------------
print("\nFinal Ranking:\n")
for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")


# -----------------------
# Plot
# -----------------------
plot_results(results)
