# Concrete Compressive Strength Prediction

A comprehensive machine learning project for predicting concrete compressive strength using multiple regression models with automated model selection, hyperparameter tuning, and detailed performance analysis.

## Overview

This project implements an end-to-end ML pipeline that:
- Loads and preprocesses concrete mixture data
- Performs domain-driven feature engineering
- Trains multiple regression models (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- Applies hyperparameter tuning with k-fold cross-validation
- Selects the best model based on multiple evaluation metrics
- Generates comprehensive visualizations and reports
- Provides a complete persistence and inference pipeline

## Key Features

- **Flexible Configuration**: Centralized config.py for all hyperparameters and settings
- **Automated Model Selection**: Benchmark multiple algorithms and automatically select the best one
- **K-Fold Cross-Validation**: Robust evaluation using stratified k-fold CV
- **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- **Feature Engineering**: Domain-driven features (ratios, interactions, binder content)
- **Comprehensive Evaluation**: R², RMSE, MAE, MAPE metrics with comparison charts
- **Feature Importance**: Visualization of most important features for tree-based models
- **Learning Curves**: Detect overfitting and underfitting patterns
- **Pipeline Persistence**: Save complete preprocessing + model pipeline for production use
- **CLI Arguments**: Configurable runs via command-line parameters
- **Detailed Logging**: Full execution logs for debugging and monitoring

## Project Structure

```
Concrete_CCS/
├── data/
│   ├── raw/
│   │   └── concrete.csv           # Original dataset
│   └── processed/
│       └── processed_concrete.csv  # Feature-engineered data
├── model/
│   ├── best_model.pkl            # Trained model
│   ├── preprocessing_pipeline.pkl # Preprocessing pipeline
│   └── model_comparison.csv       # Model benchmark results
├── src/
│   ├── main.py                   # Main execution pipeline
│   ├── config.py                 # Configuration and hyperparameters
│   ├── preprocessing.py          # Data loading and preprocessing
│   ├── features.py               # Feature engineering
│   ├── model.py                  # Model factory
│   ├── trainer.py                # Model training with CV and tuning
│   ├── evaluator.py              # Model evaluation metrics
│   ├── visualize.py              # Visualization utilities
│   ├── model_selector.py         # Advanced model selection
│   ├── utils.py                  # Utility functions
│   └── __pycache__/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Concrete_CCS
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Execution

Run the complete pipeline:
```bash
python src/main.py
```

### Command-Line Arguments

```bash
# Specify custom data path
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
