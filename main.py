import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    log_loss,
)

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
from models.decision_tree import get_model as dt
from models.xgboost_model import get_model as xgb

from visualization.plots import (
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_roc_auc_comparison,
    plot_all_feature_importance,
)


def get_models():
    return {
        "Linear": linear(),
        "SVM": svm(),
        "KNN": knn(),
        "MLP": mlp(),
        "RandomForest": rf(),
        "DecisionTree": dt(),
        "XGBoost": xgb(),
    }


def to_probability_scores(preds, train_preds):
    min_pred = float(np.min(train_preds))
    max_pred = float(np.max(train_preds))

    if np.isclose(max_pred - min_pred, 0.0):
        return np.full_like(preds, 0.5, dtype=float)

    probs = (preds - min_pred) / (max_pred - min_pred)
    return np.clip(probs.astype(float), 1e-6, 1 - 1e-6)


def compute_kfold_crossentropy(name, model, X, y, kf, scale_sensitive_models):
    ce_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        if name in scale_sensitive_models:
            X_train_fold_scaled, X_val_fold_scaled, _ = scale_data(X_train_fold, X_val_fold)
            X_fit = X_train_fold_scaled
            X_eval = X_val_fold_scaled
        else:
            X_fit = X_train_fold
            X_eval = X_val_fold

        fold_model = clone(model)
        fold_model.fit(X_fit, y_train_fold)

        train_preds = fold_model.predict(X_fit)
        val_preds = fold_model.predict(X_eval)

        threshold = y_train_fold.median()
        y_val_binary = (np.asarray(y_val_fold) >= threshold).astype(int)
        val_probs = to_probability_scores(np.asarray(val_preds), np.asarray(train_preds))

        ce_scores.append(log_loss(y_val_binary, val_probs, labels=[0, 1]))

    return ce_scores


df_raw = load_data("data/concrete.csv")
df = feature_engineering(df_raw)
plot_correlation_heatmap(df)

X, y = split_data(df)
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

models = get_models()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
trained_models = {}
predictions = {}
actual_vs_pred_df = pd.DataFrame({"Actual": np.asarray(y_test)})

print("\nResults:\n")

scale_sensitive_models = {"SVM", "KNN", "MLP", "Linear"}
strength_threshold = y_train.median()
y_test_binary = (np.asarray(y_test) >= strength_threshold).astype(int)
kfold_crossentropy_values = {}

for name, model in models.items():
    print(f"Running {name}...")

    if name in scale_sensitive_models:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    ce_scores = compute_kfold_crossentropy(name, model, X, y, kf, scale_sensitive_models)

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
    kfold_crossentropy_values[name] = ce_scores
    trained_models[name] = model
    predictions[name] = preds
    actual_vs_pred_df[f"{name}_Predicted"] = np.asarray(preds)
    plot_actual_vs_predicted(y_test, preds, name)

print("\nFinal Ranking:\n")
for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: {v:.4f}")
plot_model_comparison(
    results,
    title="Model Comparison (With Feature Engineering)",
    output_path="model_comparison_with_feature_engineering.png",
)
actual_vs_pred_df.to_csv("actual_vs_predicted_all_models.csv", index=False)
print("Saved: actual_vs_predicted_all_models.csv")
auc_scores = plot_roc_auc_comparison(y_test_binary, predictions)
print(f"ROC threshold (median train strength): {strength_threshold:.4f}")
print("AUC Scores:\n")
for model_name, auc_score in sorted(auc_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {auc_score:.4f}")
print("Saved: roc_auc_comparison.png")

print("\nK-Fold Cross-Entropy Summary (With Feature Engineering):")
for model_name, ce_scores in kfold_crossentropy_values.items():
    print(
        f"{model_name}: values={[round(v, 4) for v in ce_scores]}, mean={np.mean(ce_scores):.4f}"
    )

print("\nRunning pipeline without feature engineering...\n")
X_no_fe, y_no_fe = split_data(df_raw)

X_train_no_fe, X_test_no_fe, y_train_no_fe, y_test_no_fe = train_test_split(
    X_no_fe, y_no_fe, test_size=0.2, random_state=42
)

X_train_no_fe_scaled, X_test_no_fe_scaled, _ = scale_data(X_train_no_fe, X_test_no_fe)
models_no_fe = get_models()
results_no_fe = {}
kfold_crossentropy_no_fe = {}

for name, model in models_no_fe.items():
    if name in scale_sensitive_models:
        model.fit(X_train_no_fe_scaled, y_train_no_fe)
        preds_no_fe = model.predict(X_test_no_fe_scaled)
    else:
        model.fit(X_train_no_fe, y_train_no_fe)
        preds_no_fe = model.predict(X_test_no_fe)

    r2_no_fe = r2_score(y_test_no_fe, preds_no_fe)
    mse_no_fe = mean_squared_error(y_test_no_fe, preds_no_fe)
    rmse_no_fe = np.sqrt(mse_no_fe)
    mae_no_fe = mean_absolute_error(y_test_no_fe, preds_no_fe)
    mape_no_fe = mean_absolute_percentage_error(y_test_no_fe, preds_no_fe)
    cv_scores_no_fe = cross_val_score(model, X_no_fe, y_no_fe, cv=kf, scoring="r2")
    ce_scores_no_fe = compute_kfold_crossentropy(
        name, model, X_no_fe, y_no_fe, kf, scale_sensitive_models
    )

    results_no_fe[name] = r2_no_fe
    kfold_crossentropy_no_fe[name] = ce_scores_no_fe

    print(f"{name} (No Feature Engineering):")
    print(f"R2: {r2_no_fe:.4f}")
    print(f"RMSE: {rmse_no_fe:.4f}")
    print(f"MAE: {mae_no_fe:.4f}")
    print(f"MAPE: {mape_no_fe:.4f}")
    print(f"CV Mean: {cv_scores_no_fe.mean():.4f}")
    print("-" * 30)

plot_model_comparison(
    results_no_fe,
    title="Model Comparison (No Feature Engineering)",
    output_path="model_comparison_no_feature_engineering.png",
)
print("Saved: model_comparison_no_feature_engineering.png")

print("\nK-Fold Cross-Entropy Summary (No Feature Engineering):")
for model_name, ce_scores in kfold_crossentropy_no_fe.items():
    print(
        f"{model_name}: values={[round(v, 4) for v in ce_scores]}, mean={np.mean(ce_scores):.4f}"
    )

print("\n")
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

user_input_row = pd.DataFrame(
    [[float(user_features[col]) for col in feature_names]],
    columns=feature_names
)

if best_model_name in scale_sensitive_models:
    user_input_row = scaler.transform(user_input_row)

user_prediction = prediction_model.predict(user_input_row)[0]
print(f"Predicted concrete strength: {user_prediction:.4f}")