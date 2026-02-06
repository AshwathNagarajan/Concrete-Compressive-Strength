import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import FeatureEngineer
from config import Config


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            Config.CEMENT_COL: [200, 250, 300],
            Config.SLAG_COL: [50, 60, 70],
            Config.FLYASH_COL: [20, 30, 40],
            Config.WATER_COL: [150, 160, 170],
            Config.SUPERPLASTICIZER_COL: [5, 6, 7],
            Config.COARSE_AGG_COL: [900, 920, 940],
            Config.FINE_AGG_COL: [600, 610, 620],
            Config.AGE_COL: [28, 28, 28],
            Config.TARGET_COL: [30.0, 35.0, 40.0]
        })
    
    def test_add_water_cement_ratio(self):
        """Test water-cement ratio calculation."""
        df = self.engineer.add_water_cement_ratio(self.sample_data)
        
        self.assertIn("Water_Cement_Ratio", df.columns)
        # Check calculation
        expected = 150 / (200 + Config.EPSILON)
        self.assertAlmostEqual(df.iloc[0]["Water_Cement_Ratio"], expected, places=5)
    
    def test_add_binder_ratio(self):
        """Test binder ratio calculation."""
        df = self.engineer.add_binder_ratio(self.sample_data)
        
        self.assertIn("Binder_Content", df.columns)
        self.assertIn("Water_Binder_Ratio", df.columns)
        
        # Check binder content calculation
        expected_binder = 200 + 50 + 20
        self.assertEqual(df.iloc[0]["Binder_Content"], expected_binder)
    
    def test_add_interaction_terms(self):
        """Test interaction terms creation."""
        df = self.engineer.add_interaction_terms(self.sample_data)
        
        expected_features = [
            "Cement_Water",
            "Cement_Age",
            "Water_Age",
            "Aggregate_Ratio",
            "Superplasticizer_Water"
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df.columns)
    
    def test_transform(self):
        """Test full transformation pipeline."""
        df = self.engineer.transform(self.sample_data)
        
        # Check that all new features are created
        original_cols = len(self.sample_data.columns)
        final_cols = len(df.columns)
        
        self.assertGreater(final_cols, original_cols)
        
        # Check specific features
        self.assertIn("Water_Cement_Ratio", df.columns)
        self.assertIn("Binder_Content", df.columns)
        self.assertIn("Cement_Water", df.columns)
    
    def test_no_nan_values(self):
        """Test that transformation doesn't create NaN values."""
        df = self.engineer.transform(self.sample_data)
        
        self.assertEqual(df.isnull().sum().sum(), 0, "Transformation created NaN values")


if __name__ == '__main__':
    unittest.main()
