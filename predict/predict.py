import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


class Predictor:

    def __init__(self, model_dir="models"):
        with open(os.path.join(model_dir, "random_forest.pkl"), "rb") as f:
            self.rf_model = pickle.load(f)

        with open(os.path.join(model_dir, "xgboost.pkl"), "rb") as f:
            self.xgb_model = pickle.load(f)

    def get_user_input(self):
        print("\nEnter concrete mix features:\n")

        cement = float(input("Cement (kg/m³): "))
        slag = float(input("Blast Furnace Slag (kg/m³): "))
        flyash = float(input("Fly Ash (kg/m³): "))
        water = float(input("Water (kg/m³): "))
        superplasticizer = float(input("Superplasticizer (kg/m³): "))
        coarse = float(input("Coarse Aggregate (kg/m³): "))
        fine = float(input("Fine Aggregate (kg/m³): "))
        age = int(input("Age (days): "))

        return [cement, slag, flyash, water, superplasticizer, coarse, fine, age]

    def predict(self, model_choice="both"):

        features = self.get_user_input()

        X = np.array(features).reshape(1, -1)
        X_scaled = StandardScaler().fit_transform(X)

        if model_choice == "rf":
            pred = self.rf_model.predict(X_scaled)[0]
            print(f"Random Forest Prediction: {pred:3.2f}")

        elif model_choice == "xgb":
            pred = self.xgb_model.predict(X_scaled)[0]
            print(f"XGBoost Prediction: {pred:3.2f}")

        elif model_choice == "both":
            rf_pred = self.rf_model.predict(X_scaled)[0]
            xgb_pred = self.xgb_model.predict(X_scaled)[0]

            avg = (rf_pred + xgb_pred) / 2

            print(f"Random Forest Prediction: {rf_pred:3.2f}")
            print(f"XGBoost Prediction: {xgb_pred:3.2f}")
            print(f"Average Prediction: {avg:3.2f}")