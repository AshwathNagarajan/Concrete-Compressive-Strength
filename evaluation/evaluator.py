from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class Evaluator:

    def evaluate_model(self, model, X_test, y_test, model_name):
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