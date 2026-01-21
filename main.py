import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('concrete.csv')
print(data.head())
print(data.info())
print(data.describe())

print(data.keys())