import pandas as pd


class Utils:
    @staticmethod
    def save_results(results: dict, path: str = "model/model_comparison.csv"):
        df = pd.DataFrame(results).T
        df.to_csv(path)
        print(f"Results saved to {path}")
