class Config:
    """Configuration for Concrete Compressive Strength ML Pipeline"""

    # -------------------------------------------------
    # General
    # -------------------------------------------------
    RANDOM_STATE: int = 42
    VERBOSE: bool = True
    LOG_FILE: str = "logs/concrete_ml.log"

    # -------------------------------------------------
    # Data Paths
    # -------------------------------------------------
    DATA_PATH: str = "data/raw/concrete.csv"
    PROCESSED_DATA_PATH: str = "data/processed/processed_concrete.csv"
    TARGET_COL: str = "Strength"

    # Raw feature columns
    CEMENT_COL: str = "Cement"
    SLAG_COL: str = "BlastFurnaceSlag"
    FLYASH_COL: str = "FlyAsh"
    WATER_COL: str = "Water"
    SUPERPLASTICIZER_COL: str = "Superplasticizer"
    COARSE_AGG_COL: str = "CoarseAggregate"
    FINE_AGG_COL: str = "FineAggregate"
    AGE_COL: str = "Age"

    # Magic numbers for feature engineering
    EPSILON: float = 1e-6  # Avoid division by zero
    
    NUMERIC_FEATURES: list = [
        CEMENT_COL,
        SLAG_COL,
        FLYASH_COL,
        WATER_COL,
        SUPERPLASTICIZER_COL,
        COARSE_AGG_COL,
        FINE_AGG_COL,
        AGE_COL,
    ]

    # -------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    SHUFFLE: bool = True

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------
    IMPUTATION_STRATEGY: str = "median"
    SCALER_TYPE: str = "standard"          # standard | robust | minmax | none
    DISTRIBUTION_TRANSFORM: str = "power"  # power | quantile | none

    # -------------------------------------------------
    # Models to Benchmark
    # -------------------------------------------------
    MODELS: list = [
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    ]

    # Model Selection
    SELECTION_METRIC: str = "R2"  # R2 | RMSE | MAE | MAPE
    USE_CV: bool = True
    ENABLE_HYPERPARAMETER_TUNING: bool = True

    # -------------------------------------------------
    # Hyperparameters (with Grid Search bounds)
    # -------------------------------------------------
    # Linear Models
    RIDGE_ALPHA: float = 1.0
    RIDGE_ALPHA_RANGE: list = [0.1, 1.0, 10.0]
    
    LASSO_ALPHA: float = 0.01
    LASSO_ALPHA_RANGE: list = [0.001, 0.01, 0.1]
    
    ELASTICNET_ALPHA: float = 0.1
    ELASTICNET_ALPHA_RANGE: list = [0.01, 0.1, 1.0]
    ELASTICNET_L1_RATIO: float = 0.5
    ELASTICNET_L1_RATIO_RANGE: list = [0.2, 0.5, 0.8]

    # Tree Models
    TREE_MAX_DEPTH: int = None
    TREE_MAX_DEPTH_RANGE: list = [5, 10, 15, None]

    # Random Forest
    RF_N_ESTIMATORS: int = 300
    RF_N_ESTIMATORS_RANGE: list = [100, 200, 300]
    RF_MAX_DEPTH: int = None
    RF_MAX_DEPTH_RANGE: list = [10, 20, None]
    RF_MIN_SAMPLES_SPLIT: int = 2
    RF_N_JOBS: int = -1

    # Gradient Boosting
    GB_N_ESTIMATORS: int = 200
    GB_N_ESTIMATORS_RANGE: list = [100, 150, 200]
    GB_LEARNING_RATE: float = 0.05
    GB_LEARNING_RATE_RANGE: list = [0.01, 0.05, 0.1]
    GB_MAX_DEPTH: int = 3
    GB_MAX_DEPTH_RANGE: list = [2, 3, 4]

    # XGBoost
    XGB_N_ESTIMATORS: int = 200
    XGB_N_ESTIMATORS_RANGE: list = [100, 150, 200]
    XGB_LEARNING_RATE: float = 0.05
    XGB_LEARNING_RATE_RANGE: list = [0.01, 0.05, 0.1]
    XGB_MAX_DEPTH: int = 5
    XGB_MAX_DEPTH_RANGE: list = [3, 5, 7]
    XGB_SUBSAMPLE: float = 0.8
    XGB_SUBSAMPLE_RANGE: list = [0.7, 0.8, 0.9]
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_COLSAMPLE_BYTREE_RANGE: list = [0.7, 0.8, 0.9]
    XGB_N_JOBS: int = -1

    # -------------------------------------------------
    # Visualization & Reporting
    # -------------------------------------------------
    PLOT_LEARNING_CURVES: bool = True
    PLOT_FEATURE_IMPORTANCE: bool = True
    PLOT_CV_RESULTS: bool = True
    ENABLE_VISUALIZATIONS: bool = True

    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------
    MODEL_SAVE_PATH: str = "model/best_model.pkl"
    PIPELINE_SAVE_PATH: str = "model/preprocessing_pipeline.pkl"
    RESULTS_SAVE_PATH: str = "model/model_comparison.csv"
    CV_RESULTS_SAVE_PATH: str = "model/cv_results.csv"
