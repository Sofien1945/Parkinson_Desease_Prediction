"""
Parkinson desease prediction
Part of https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/
machine learning course.
Date: 28.10.2021
"""
# Import Standard libraries
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset
file_path = 'C:/Users/Sofien/Desktop/parkinsons.data'
df = pd.read_csv(file_path)
print(df.head())

# Split the features and the labels:
features=df.loc[:,df.columns!='status'].values[:,1:]
y=df.loc[:,'status'].values
print(features.shape)
print(y.shape)

#Scale the features
scaler = MinMaxScaler((-1,1))
X = scaler.fit_transform(features)

# Split the data to test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Train the model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Testing the model
y_pred = xgb_model.predict(X_test)
score = accuracy_score(y_test,y_pred)
cf = confusion_matrix(y_test,y_pred)
print(f"Accuracy score: {score}")
plt.figure(figsize=(5,5))
sns.heatmap(cf, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()