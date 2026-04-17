import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from preprocess.preprocess import (
    load_data,
    feature_engineering,
    split_data,
    scale_data,
    save_model_pkl,
    load_model_pkl,
)

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

X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

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
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
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

best_model = trained_models[best_model_name]
model_path = "models/model/model.pkl"
save_model_pkl(best_model, model_path)
loaded_model = load_model_pkl(model_path)

# Use the loaded best model as the final prediction model.
prediction_model = loaded_model

if best_model_name in scale_sensitive_models:
    best_preds = prediction_model.predict(X_test_scaled)
else:
    best_preds = prediction_model.predict(X_test)

print(f"Best model: {best_model_name}")
print(f"Best model saved at: {model_path}")
plot_actual_vs_predicted(y_test, best_preds, best_model_name)

plot_all_feature_importance(trained_models, feature_names)

print("\nEnter values to predict concrete strength:")
cement = float(input("cement: "))
blastfurnaceslag = float(input("blastfurnaceslag: "))
flyash = float(input("flyash: "))
water = float(input("water: "))
superplasticizer = float(input("superplasticizer: "))
coarseaggregate = float(input("coarseaggregate: "))
fineaggregate = float(input("fineaggregate: "))
age = float(input("age: "))

user_features = {
    "cement": cement,
    "blastfurnaceslag": blastfurnaceslag,
    "flyash": flyash,
    "water": water,
    "superplasticizer": superplasticizer,
    "coarseaggregate": coarseaggregate,
    "fineaggregate": fineaggregate,
    "age": age,
}

user_features["water_cement_ratio"] = user_features["water"] / (user_features["cement"] + 1e-6)
user_features["total_binder"] = (
    user_features["cement"]
    + user_features["blastfurnaceslag"]
    + user_features["flyash"]
)
user_features["agg_ratio"] = (
    user_features["coarseaggregate"] / (user_features["fineaggregate"] + 1e-6)
)
user_features["log_age"] = np.log1p(user_features["age"])
user_features["binder_water_ratio"] = (
    user_features["total_binder"] / (user_features["water"] + 1e-6)
)

user_input_row = np.array(
    [[float(user_features[col]) for col in feature_names]],
    dtype=float,
)

if best_model_name in scale_sensitive_models:
    user_input_row = scaler.transform(user_input_row)

user_prediction = prediction_model.predict(user_input_row)[0]
print(f"Predicted concrete strength: {user_prediction:.4f}")