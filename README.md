# Concrete Compressive Strength Prediction

A simple machine learning project for predicting concrete compressive strength based on ingredient ratios and curing time.

## Project Overview

This project implements a straightforward machine learning pipeline that:
1. Loads concrete mixture data (2,114 samples with 8 ingredient features)
2. Preprocesses data by normalizing features to the same scale
3. Trains two regression models: Random Forest and XGBoost
4. Evaluates both models on a held-out test set
5. Generates visualizations to understand model performance and feature importance

**Best Result**: XGBoost or Random Forest typically achieve R² ≈ 0.90 (explains 90% of variance)

## Dataset

- **Total Samples**: 1,030 concrete mixtures
- **Features**: 8 ingredient measurements (Cement, Slag, FlyAsh, Water, Superplasticizer, CoarseAggregate, FineAggregate, Age)
- **Target**: Compressive Strength in MPa (megapascals)
- **Train/Test Split**: 80% training, 20% testing

## Installation

### 1. Clone or download the project
```bash
cd Concrete_CCS
```

### 2. Create a virtual environment (recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the complete pipeline
```bash
python main.py
```

This will:
1. Load and explore the data
2. Split into training (80%) and test (20%) sets
3. Train Random Forest and XGBoost models
4. Evaluate both models on test data
5. Generate 3 visualization plots
6. Save trained models and results

### Output Files

After running, you'll find:
- **results.txt** - Summary of model performance metrics
- **models/** - Saved trained models (.pkl files)
- **visualizations/** - Three PNG plots:
  - `01_correlation_matrix.png` - Feature relationships
  - `02_predictions_vs_actual.png` - Model accuracy visualization
  - `03_feature_importance.png` - Most influential features

## Results Interpretation

### Evaluation Metrics

- **R² Score** (R-squared): Proportion of variance in target explained by model. Range: 0-1, higher is better.
  - R² = 0.90 means model explains 90% of strength variation
  
- **RMSE** (Root Mean Squared Error): Average prediction error in MPa (megapascals). Lower is better.
  - RMSE = 4.5 MPa means predictions are off by ~4.5 MPa on average
  
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual. Lower is better.
  - MAE = 3.2 MPa means typical error is 3.2 MPa

### Expected Performance

| Model | R² Score | RMSE (MPa) |
|-------|----------|-----------|
| Random Forest | 0.88-0.92 | 4.5-6.0 |
| XGBoost | 0.90-0.93 | 4.0-5.5 |

## Code Structure

```
main.py                    Main script with entire pipeline
├── load_data()           Load CSV file
├── split_data()          Train/test split (80/20)
├── preprocess_data()     Normalize features using StandardScaler
├── train_random_forest() Train Random Forest with 100 trees
├── train_xgboost()       Train XGBoost with 100 boosting rounds
├── evaluate_model()      Calculate R², RMSE, MAE metrics
├── plot_*()              Create 3 visualization plots
├── save_models()         Save trained models as .pkl files
├── save_results()        Save metrics summary to text file
└── main()                Orchestrate entire pipeline
```

## Key Concepts Explained

### Data Preprocessing
We normalize features using StandardScaler, which transforms each feature to have mean=0 and standard deviation=1. This helps models converge faster and perform better.

### Train/Test Split
We use 80% of data for training and 20% for testing. The test set simulates real-world, unseen data to evaluate model generalization.

### Random Forest
An ensemble method that trains multiple decision trees and averages their predictions. Good for capturing non-linear relationships.

### XGBoost
A boosting algorithm that trains trees sequentially, where each tree corrects previous tree's errors. Often more accurate than Random Forest.

### Feature Importance
Shows which input features have the most influence on predictions. Helps understand what drives concrete strength.

## Potential Improvements (for future work)

1. Feature Engineering: Create ratios like water-cement ratio
2. Hyperparameter Tuning: Optimize model parameters using GridSearchCV
3. Cross-Validation: Use k-fold CV for more robust evaluation
4. Feature Selection: Remove less important features
5. Model Comparison: Compare with other algorithms like Gradient Boosting
6. Ensemble Methods: Combine predictions from multiple models

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine Learning models and preprocessing
- **XGBoost** - Gradient boosting library
- **Matplotlib & Seaborn** - Data visualization
- **joblib** - Model persistence

## File Sizes

- `main.py`: ~400 lines (single file, easy to understand)
- `requirements.txt`: 7 packages (minimal dependencies)
- Total code: ~400 lines (not counting comments)

## Author

[Your Name]  
B.E./B.Tech Student - 2nd Year  
AI & Data Science Stream

## References

- Dataset: UCI Machine Learning Repository (Concrete Compressive Strength)
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/

---

**Last Updated**: 2024

python src/main.py --data-path data/raw/concrete.csv

# Run only specific models
python src/main.py --models linear ridge random_forest xgboost

# Disable visualization
python src/main.py --no-visualize

# Custom number of CV folds
python src/main.py --cv-folds 10

# Custom test size
python src/main.py --test-size 0.15
```

### Expected Output

```
Concrete Compressive Strength Prediction System

Loading dataset...
Performing feature engineering...
Preprocessing data...
Training all models...
  🔹 Training linear...
  🔹 Training ridge...
  🔹 Training random_forest...
  ...

Best Model Selected
Model : random_forest
R2 : 0.9234
RMSE : 4.5612
MAE : 3.2456
MAPE : 8.1234

Visualizing best model predictions...
✅ Model saved to model/best_model.pkl
✅ Pipeline saved to model/preprocessing_pipeline.pkl
✅ Results saved to model/model_comparison.csv
Training pipeline completed successfully.
```

## Dataset

The project uses the **UCI Concrete Compressive Strength Dataset** with the following features:

| Feature | Unit | Description |
|---------|------|-------------|
| Cement | kg/m³ | Portland cement content |
| BlastFurnaceSlag | kg/m³ | Slag content from blast furnaces |
| FlyAsh | kg/m³ | Fly ash content |
| Water | kg/m³ | Water content |
| Superplasticizer | kg/m³ | Superplasticizer additive |
| CoarseAggregate | kg/m³ | Coarse aggregate weight |
| FineAggregate | kg/m³ | Fine aggregate weight |
| Age | days | Age of concrete sample |
| **Strength** | MPa | **Target: Compressive strength** |

### Feature Engineering

The pipeline creates domain-driven features:
- **Water-Cement Ratio**: Water / Cement (fundamental concrete property)
- **Binder Content**: Cement + Slag + FlyAsh (total binding material)
- **Water-Binder Ratio**: Water / Binder Content (affects strength and durability)
- **Interaction Terms**: Cement×Water, Cement×Age, Water×Age, Coarse/Fine Aggregate Ratio, Superplasticizer×Water

## Models Benchmarked

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Linear | Baseline model |
| Ridge | Linear | L2 regularization |
| Lasso | Linear | L1 regularization (feature selection) |
| ElasticNet | Linear | L1 + L2 regularization |
| Decision Tree | Tree | Single tree regression |
| Random Forest | Ensemble | 300 trees, parallel training |
| Gradient Boosting | Ensemble | Sequential tree boosting |
| XGBoost | Ensemble | Optimized gradient boosting |

## Evaluation Metrics

- **R² Score**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## Key Findings

After training and evaluation, the project produces:
1. **Model Comparison Table** (`model/model_comparison.csv`) - All models ranked by R²
2. **Actual vs Predicted Plot** - Scatter plot with diagonal reference line
3. **Residuals Plot** - Error distribution analysis
4. **Learning Curves** - Training vs validation performance across dataset sizes
5. **Feature Importance** - For tree-based models
6. **Cross-Validation Scores** - Distribution of performance across folds

## Configuration

Edit [src/config.py](src/config.py) to customize:

```python
# Data paths
DATA_PATH = "data/raw/concrete.csv"
TARGET_COL = "Strength"

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing
SCALER_TYPE = "standard"              # standard | robust | minmax | none
DISTRIBUTION_TRANSFORM = "power"      # power | quantile | none

# Hyperparameters
RIDGE_ALPHA = 1.0
RF_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
# ... and many more

# Cross-validation
CV_FOLDS = 5

# Model selection metric
SELECTION_METRIC = "R2"               # R2 | RMSE | MAE | MAPE
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Individual test modules:
```bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_features.py -v
python -m pytest tests/test_models.py -v
```

## Logging

Logs are saved to `logs/concrete_ml.log` with detailed execution information:
```
2026-02-06 10:23:45 - INFO - Loading dataset from data/raw/concrete.csv
2026-02-06 10:23:46 - INFO - Dataset shape: (1030, 8)
2026-02-06 10:23:47 - INFO - Performing feature engineering...
2026-02-06 10:24:12 - INFO - Training random_forest with CV...
```

## Performance

Typical results on the concrete dataset:

| Model | R² | RMSE | MAE |
|-------|-----|--------|--------|
| Linear | 0.56 | 10.22 | 8.15 |
| Ridge | 0.61 | 9.45 | 7.42 |
| Random Forest | **0.92** | **4.56** | **3.24** |
| XGBoost | **0.91** | **4.89** | **3.45** |

*Note: Actual results depend on preprocessing settings and hyperparameters.*

## Development Guidelines

### Adding a New Model

1. **Add hyperparameters to config.py**:
   ```python
   NEWMODEL_PARAM1 = value
   NEWMODEL_PARAM2 = value
   ```

2. **Add to model factory in model.py**:
   ```python
   elif model_name == "newmodel":
       return NewModel(param1=Config.NEWMODEL_PARAM1, ...)
   ```

3. **Add to MODELS list in config.py**:
   ```python
   MODELS = [..., "newmodel"]
   ```

### Adding a New Feature

1. **Create method in features.py**:
   ```python
   def add_new_feature(self, df: pd.DataFrame) -> pd.DataFrame:
       df = df.copy()
       df["NewFeature"] = ...
       return df
   ```

2. **Add to transform() method**:
   ```python
   def transform(self, df: pd.DataFrame) -> pd.DataFrame:
       ...
       df = self.add_new_feature(df)
       return df
   ```

## Troubleshooting

**ImportError: No module named 'xgboost'**
```bash
pip install xgboost
```

**Convergence warnings with Lasso**
- These are safe to ignore (handled in trainer.py)
- Increase `max_iter` in model.py if needed

**Memory issues with large datasets**
- Reduce `RF_N_ESTIMATORS` and `GB_N_ESTIMATORS` in config.py
- Set `n_jobs=1` for models in model.py instead of `-1`

## License

This project is provided as-is for educational and research purposes.

## Contact & Support

For issues, feature requests, or improvements, please open an issue on the repository.

---

**Last Updated**: February 6, 2026
**Python Version**: 3.9+
**Status**: Active Development
