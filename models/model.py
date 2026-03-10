import pickle
import os


class ModelSaver:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_models(self, models_dict):
        print("\nSaving models...")
        for model_name, model in models_dict.items():
            path = f"{self.model_dir}/{model_name.lower().replace(' ', '_')}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {path}")