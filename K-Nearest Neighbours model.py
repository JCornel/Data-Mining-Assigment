import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data
data = pd.read_csv('Covid Data.csv')

# Data Exploration
print(data.head())  # Display the first few rows of data
print(data.info())  # Display data information (e.g., data types, missing values)
print(data.describe())  # Display summary statistics

# This will simplify the classification_final to either Covid (1) or not covid (0)
data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].map(lambda x: 1 if x in [1, 2, 3] else 0)

# this replaces 97 and 99 with NaN in all columns except 'age', these values are missing values, and they will be excluded from the dataset.
columns_to_replace = data.columns.difference(['AGE'])  # Exclude 'age' column
data[columns_to_replace] = data[columns_to_replace].replace([97, 99], np.nan)
data.dropna(inplace=True)


# Assuming you want to predict 'CLASIFFICATION_FINAL' (change this if needed)
X = data.drop('CLASIFFICATION_FINAL', axis=1)  # Features (all columns except 'CLASIFFICATION_FINAL')
y = data['CLASIFFICATION_FINAL']  # Target variable ('CLASIFFICATION_FINAL')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and train the K-nearest neighbors (KNN) model
k = 800  # Number of neighbors (you can adjust this) 
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_mat)
print('Classification Report:')
print(classification_rep)
