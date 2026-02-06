import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import DataPreprocessor
from config import Config


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
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
    
    def test_load_data_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_data("nonexistent_file.csv")
    
    def test_split_feature_target(self):
        """Test splitting features and target."""
        X, y = self.preprocessor.split_feature_target(self.sample_data)
        
        # Check shapes
        self.assertEqual(X.shape[0], 3)
        self.assertEqual(len(y), 3)
        self.assertEqual(y.name, Config.TARGET_COL)
    
    def test_split_feature_target_missing_target(self):
        """Test error when target column missing."""
        bad_data = self.sample_data.drop(columns=[Config.TARGET_COL])
        
        with self.assertRaises(KeyError):
            self.preprocessor.split_feature_target(bad_data)
    
    def test_build_pipeline(self):
        """Test pipeline building."""
        X, y = self.preprocessor.split_feature_target(self.sample_data)
        self.preprocessor.build_pipeline()
        
        self.assertIsNotNone(self.preprocessor.pipeline)
    
    def test_build_pipeline_invalid_scaler(self):
        """Test error with invalid scaler type."""
        self.preprocessor.scaler_type = "invalid_scaler"
        X, y = self.preprocessor.split_feature_target(self.sample_data)
        
        with self.assertRaises(ValueError):
            self.preprocessor.build_pipeline()
    
    def test_train_test_split(self):
        """Test train-test split."""
        X, y = self.preprocessor.split_feature_target(self.sample_data)
        self.preprocessor.build_pipeline()
        X_processed = self.preprocessor.fit_transform(X)
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_train_test(X_processed, y)
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(X_processed))
        self.assertEqual(len(y_train) + len(y_test), len(y))


if __name__ == '__main__':
    unittest.main()
