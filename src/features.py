import pandas as pd
from config import Config


class FeatureEngineer:
    """
    Performs domain-driven feature engineering for concrete compressive strength prediction.
    """

    def __init__(self):
        self.cement_col = Config.CEMENT_COL
        self.slag_col = Config.SLAG_COL
        self.flyash_col = Config.FLYASH_COL
        self.water_col = Config.WATER_COL
        self.superplasticizer_col = Config.SUPERPLASTICIZER_COL
        self.coarse_agg_col = Config.COARSE_AGG_COL
        self.fine_agg_col = Config.FINE_AGG_COL
        self.age_col = Config.AGE_COL

    # -------------------------------------------------
    # Domain-Driven Ratios
    # -------------------------------------------------
    def add_water_cement_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Water_Cement_Ratio"] = df[self.water_col] / (df[self.cement_col] + 1e-6)
        return df

    def add_binder_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Binder_Content"] = (
            df[self.cement_col] +
            df[self.slag_col] +
            df[self.flyash_col]
        )
        df["Water_Binder_Ratio"] = df[self.water_col] / (df["Binder_Content"] + 1e-6)
        return df

    # -------------------------------------------------
    # Interaction Terms
    # -------------------------------------------------
    def add_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Cement_Water"] = df[self.cement_col] * df[self.water_col]
        df["Cement_Age"] = df[self.cement_col] * df[self.age_col]
        df["Water_Age"] = df[self.water_col] * df[self.age_col]
        df["Aggregate_Ratio"] = df[self.coarse_agg_col] / (df[self.fine_agg_col] + 1e-6)
        df["Superplasticizer_Water"] = df[self.superplasticizer_col] * df[self.water_col]

        return df

    # -------------------------------------------------
    # Unified Transformation
    # -------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_water_cement_ratio(df)
        df = self.add_binder_ratio(df)
        df = self.add_interaction_terms(df)
        return df
