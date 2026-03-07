import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = "data/processed_concrete.csv"
TARGET_COLUMN = "Strength"
MODEL_DIR = "models"
VIZ_DIR = "visualizations"


def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.describe())
    return df


def preprocess_data(X, scaler=None, fit=True):
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Scaler fitted on training data")
    else:
        X_scaled = scaler.transform(X)
        print(f"Scaler applied to test data")
    
    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed!")
    
    return rf_model


def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training completed!")
    
    return xgb_model


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  R² Score: {r2:.4f}   (Explains {r2*100:.1f}% of variance)")
    print(f"  RMSE: {rmse:.4f} MPa (Average prediction error)")
    print(f"  MAE: {mae:.4f} MPa   (Mean absolute error)")
    
    return {
        "model_name": model_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "y_pred": y_pred
    }


def plot_correlation_matrix(X, y):
    print("\nGenerating correlation matrix plot...")
    
    data = pd.concat([X, y], axis=1)
    corr_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/01_correlation_matrix.png", dpi=100)
    plt.close()
    print(f"Saved: {VIZ_DIR}/01_correlation_matrix.png")


def plot_predictions_vs_actual(y_test, results):
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
    plt.savefig(f"{VIZ_DIR}/02_predictions_vs_actual.png", dpi=100)
    plt.close()
    print(f"Saved: {VIZ_DIR}/02_predictions_vs_actual.png")


def plot_feature_importance(models_dict, feature_names):
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
    plt.savefig(f"{VIZ_DIR}/03_feature_importance.png", dpi=100)
    plt.close()
    print(f"Saved: {VIZ_DIR}/03_feature_importance.png")


def save_models(models_dict):
    print("\nSaving models...")
    for model_name, model in models_dict.items():
        path = f"{MODEL_DIR}/{model_name.lower().replace(' ', '_')}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {path}")


def save_results(results):
    print("\nSaving results summary...")
    
    with open("results.txt", "w") as f:
        f.write("CONCRETE COMPRESSIVE STRENGTH PREDICTION - RESULTS SUMMARY\n")
        
        best_result = max(results, key=lambda x: x["r2"])
        f.write(f"BEST MODEL: {best_result['model_name']}\n")
        f.write(f"  R² Score: {best_result['r2']:.4f}\n")
        f.write(f"  RMSE: {best_result['rmse']:.4f} MPa\n")
        f.write(f"  MAE: {best_result['mae']:.4f} MPa\n\n")
        
        f.write("ALL MODELS:\n")
        f.write("-" * 60 + "\n")
        for result in results:
            f.write(f"{result['model_name']}:\n")
            f.write(f"  R² Score: {result['r2']:.4f}\n")
            f.write(f"  RMSE: {result['rmse']:.4f} MPa\n")
            f.write(f"  MAE: {result['mae']:.4f} MPa\n\n")
    
    print("Saved: results.txt")


def main():
    
    df = load_data(DATA_PATH)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = X.columns.tolist()
    print(f"\nFeatures: {feature_names}")
    
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    
    X_train_scaled, scaler = preprocess_data(X_train, fit=True)
    X_test_scaled, _ = preprocess_data(X_test, scaler=scaler, fit=False)
    
    rf_model = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    
    results = [rf_results, xgb_results]
    
    plot_correlation_matrix(X, y)
    plot_predictions_vs_actual(y_test, results)
    plot_feature_importance(
        {"Random Forest": rf_model, "XGBoost": xgb_model},
        feature_names
    )
    
    save_models({"Random Forest": rf_model, "XGBoost": xgb_model})
    save_results(results)
    
    print(f"Visualizations saved to: {VIZ_DIR}/")
    print(f"Models saved to: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
