class Report:

    def save_results(self, results):
        print("\nSaving results summary...")
        
        with open(f"results.txt", "w") as f:
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
        
        print(f"Saved: results.txt")
