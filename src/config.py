class Config:
    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    DATA_PATH = "data/raw/concrete.csv"
    TARGET_COL = "Strength"

    # Raw feature columns
    CEMENT_COL = "Cement"
    SLAG_COL = "BlastFurnaceSlag"
    FLYASH_COL = "FlyAsh"
    WATER_COL = "Water"
    SUPERPLASTICIZER_COL = "Superplasticizer"
    COARSE_AGG_COL = "CoarseAggregate"
    FINE_AGG_COL = "FineAggregate"
    AGE_COL = "Age"

    NUMERIC_FEATURES = [
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
    # Split
    # -------------------------------------------------
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------
    IMPUTATION_STRATEGY = "median"
    SCALER_TYPE = "standard"          # standard | robust | minmax | none
    DISTRIBUTION_TRANSFORM = "power"  # power | quantile | none

    # -------------------------------------------------
    # Models to Benchmark
    # -------------------------------------------------
    MODELS = [
        "linear",
        "ridge",
        "lasso",
        "elasticnet",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]

    # Hyperparameters
    RIDGE_ALPHA = 1.0
    LASSO_ALPHA = 0.01
    ELASTICNET_ALPHA = 0.1
    ELASTICNET_L1_RATIO = 0.5

    TREE_MAX_DEPTH = None

    RF_N_ESTIMATORS = 300
    RF_MAX_DEPTH = None

    GB_N_ESTIMATORS = 200
    GB_LEARNING_RATE = 0.05
    GB_MAX_DEPTH = 3

    # -------------------------------------------------
    # Persistence
    # -------------------------------------------------
    MODEL_SAVE_PATH = "model/best_model.pkl"
