import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('concrete.csv')
print(data.head())
print(data.info())
print(data.describe())

dup_sum = data.duplicated().sum()
print(f"Number of duplicate rows: {dup_sum}")
data = data.drop_duplicates()
data = data.dropna()
print(f"Data shape after cleaning: {data.shape}")