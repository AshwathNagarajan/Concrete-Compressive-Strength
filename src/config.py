class Config:
    # ----------------------------
    # Data
    # ----------------------------
    DATA_PATH = "data/concrete.csv"
    TARGET_COL = "Strength"

    # ----------------------------
    # Split Parameters
    # ----------------------------
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ----------------------------
    # Preprocessing Parameters
    # ----------------------------
    IMPUTATION_STRATEGY = "median"      
    SCALER_TYPE = "standard"            
    DISTRIBUTION_TRANSFORM = "power"    

    # ----------------------------
    # Model Parameters
    # ----------------------------
    MODEL_TYPE = "ridge"                
    MODEL_PARAMS = {"alpha": 1.0}

    # ----------------------------
    # Training Parameters
    # ----------------------------
    CV_FOLDS = 5

    # ----------------------------
    # Paths
    # ----------------------------
    MODEL_SAVE_PATH = "models/best_model.pkl"
