import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
)

from config import Config

class Data_Preprocessor:

    def __init__(self):

        self.target_col = Config.TARGET_COL
        self.test_size = Config.TEST_SIZE
        self.random_state = Config.RANDOM_STATE
        self.imputation_strategy = Config.IMPUTATION_STRATEGY
        self.scaler_type = Config.SCALER_TYPE
        self.distribution_transform = Config.DISTRIBUTION_TRANSFORM

        self.pipeline = None
        self.feature_names = None

    def load_data(self, path):

        #------------LOAD DATA------------
        data = pd.read_csv(path)
        return data

    def split_feature_target(self, df):

        #------------SPLIT FEATURES AND TARGET------------
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        self.feature_names = X.columns.tolist()
        return X, y
    
    def preprocess_pipeline(self):

        #------------IMPUTER SELECTION------------
        imputer = SimpleImputer(strategy=self.imputation_strategy)
        
        if self.distribution_transform == "power":
            dist_transformer = PowerTransformer(method="yeo-johnson")
        elif self.distribution_transform == "quantile":
            dist_transformer = QuantileTransformer(
                output_distribution="normal",
                random_state=self.random_state
            )
        else:
            dist_transformer = "passthrough"   
        
        #------------SCALAR SELECTION------------
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "robust":
            scaler = RobustScaler()
        elif self.scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = "passthrough"

        numeric_pipline = Pipeline(steps = [

        ])
