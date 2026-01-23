from sklearn.preprocessing import *
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        
    def scalar_fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def scalar_transform(self, X):
        return self.scaler.transform(X)
    
    def encode_fit_transform(self, X):
        return self.encoder.fit_transform(X).toarray()
    
    def encode_transform(self, X):
        return self.encoder.transform(X).toarray()