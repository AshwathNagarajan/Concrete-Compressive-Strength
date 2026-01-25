from preprocessing import DataPreprocessor
from config import Config

if __name__ == "__main__":
    # Initialize the data preprocessor
    preprocessor = DataPreprocessor()

    # Load the dataset
    data = preprocessor.load_data(Config.DATA_PATH)

    # Split into features and target
    X, y = preprocessor.split_feature_target(data)

    # Build the preprocessing pipeline
    preprocessor.build_pipeline()

    # Fit and transform the features
    X_processed = preprocessor.fit_transform(X)

    # Now X_processed is ready for model training
    print("Preprocessing complete. Processed feature shape:", X_processed.shape)