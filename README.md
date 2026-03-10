# Concrete Compressive Strength Prediction

A modular machine learning project for predicting concrete compressive strength based on ingredient ratios and curing time.

## Project Overview

This project implements a machine learning pipeline that:
1. Loads concrete mixture data (2,105 samples with 8 ingredient features)
2. Preprocesses data by normalizing features using StandardScaler
3. Trains two regression models: Random Forest and XGBoost
4. Evaluates both models on a held-out test set
5. Generates visualizations for model performance and feature importance
6. Provides interactive interface for training and making predictions

**Performance**: Random Forest R² = 0.965, XGBoost R² = 0.961

## Dataset

- **Total Samples**: 2,105 concrete mixtures
- **Features**: 8 ingredient measurements
  - Cement (kg/m³)
  - BlastFurnaceSlag (kg/m³)
  - FlyAsh (kg/m³)
  - Water (kg/m³)
  - Superplasticizer (kg/m³)
  - CoarseAggregate (kg/m³)
  - FineAggregate (kg/m³)
  - Age (days)
- **Target**: Compressive Strength in MPa (megapascals)
- **Train/Test Split**: 80% training, 20% testing

## Project Structure

```
Concrete_CCS/
├── main.py                    Interactive menu system
├── requirements.txt           Project dependencies
├── data/
│   └── concrete.csv          Dataset file
├── preprocessing/
│   └── preprocess.py         Data loading and preprocessing
├── random_forest/
│   └── random_forest.py      Random Forest model training
├── xg_boost/
│   └── xg_boost.py           XGBoost model training
├── evaluation/
│   └── evaluator.py          Model evaluation metrics
├── visualizations/
│   └── visual.py             Plotting and visualization
├── models/
│   └── model.py              Model persistence utilities
├── report/
│   └── report.py             Results report generation
├── predict/
│   └── predict.py            Prediction module
├── trainer/
│   └── trainer.py            Training pipeline orchestration
└── README.md                  This file
```

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

### Interactive Menu
```bash
python main.py
```

Menu options:
1. **Train new models** - Trains Random Forest and XGBoost models
2. **Make predictions** - Interactive single prediction input
3. **Batch predictions from CSV** - Predict for multiple samples
4. **Exit** - Quit the program

### Train Models Only
```bash
python main.py
```
Select option 1 to train models and generate visualizations.

### Make Single Predictions
```bash
python main.py
```
Select option 2 to input feature values and get predictions from both models.

### Batch Predictions
```bash
python main.py
```
Select option 3 to load a CSV file with multiple samples and generate predictions.

## Output Files

After training, you'll find:
- **results.txt** - Summary of model performance metrics
- **models/** - Saved trained models (.pkl files)
  - `random_forest.pkl`
  - `xgboost.pkl`
- **visualizations/** - Three PNG plots
  - `01_correlation_matrix.png` - Feature relationships heatmap
  - `02_predictions_vs_actual.png` - Model accuracy comparison
  - `03_feature_importance.png` - Feature importance for both models

## Evaluation Metrics

- **R² Score**: Proportion of variance explained by model (0-1, higher is better)
  - 0.965 means model explains 96.5% of strength variation

- **RMSE**: Root Mean Squared Error in MPa (lower is better)
  - 2.86 MPa means predictions are off by ~2.86 MPa on average

- **MAE**: Mean Absolute Error in MPa (lower is better)
  - 1.80 MPa means typical error is 1.80 MPa

## Model Comparison

| Model | R² Score | RMSE (MPa) | MAE (MPa) |
|-------|----------|-----------|-----------|
| Random Forest | 0.965 | 2.86 | 1.80 |
| XGBoost | 0.961 | 3.00 | 1.97 |

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine Learning models and preprocessing
- **XGBoost** - Gradient boosting library
- **Matplotlib & Seaborn** - Data visualization
- **pickle** - Model persistence

## Key Features

### Modular Architecture
- Separate modules for preprocessing, training, evaluation, and visualization
- Easy to extend and maintain
- Clean separation of concerns

### Training Pipeline
- Automated data loading and exploration
- Standardized feature scaling
- Parallel model training
- Comprehensive evaluation

### Prediction System
- Load pre-trained models
- Single sample prediction with user input
- Batch prediction from CSV files
- Auto-preprocessing of new data

### Visualizations
- Correlation matrix heatmap
- Actual vs Predicted scatter plots
- Feature importance comparison

## Dependencies

See `requirements.txt`:
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.1
- xgboost==2.0.2
- matplotlib==3.8.1
- seaborn==0.13.0

## Future Improvements

1. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
2. **Feature Engineering**: Create derived features (water-cement ratio, etc.)
3. **Cross-Validation**: K-fold CV for more robust evaluation
4. **Additional Models**: Gradient Boosting, SVR, Neural Networks
5. **Model Ensemble**: Combine predictions from multiple models
6. **Web Interface**: REST API for predictions

## Quick Start Example

```bash
# 1. Activate virtual environment (if needed)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the program
python main.py

# 4. Select option 1 to train models
# 5. Check models/ and visualizations/ directories for outputs
```

## Troubleshooting

**File not found errors**
- Ensure data/concrete.csv exists
- Make sure you're running from the project root directory

**ImportError for xgboost**
```bash
pip install xgboost
```

**Memory issues**
- Reduce data size in configuration
- Close other applications

**Prediction errors**
- Ensure all 8 feature values are numeric
- Verify feature order matches training data

## References

- Dataset: UCI Machine Learning Repository - Concrete Compressive Strength
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/

---

**Last Updated**: March 10, 2026
**Python Version**: 3.8+
**Status**: Active

