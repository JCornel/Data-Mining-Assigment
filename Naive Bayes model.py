import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# This will load the dataset named Covid Data.csv. Make sure python looks in the correct directory
data = pd.read_csv("Covid Data.csv")

# this replaces 97 and 99 with NaN in all columns except 'age', these values are missing values, and they will be excluded from the dataset.
columns_to_replace = data.columns.difference(['AGE'])  # Exclude 'age' column
data[columns_to_replace] = data[columns_to_replace].replace([97, 99], np.nan)
data.dropna(inplace=True)

# This will simplify the classification_final to either Covid (1) or not covid (0)
data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].map(lambda x: 1 if x in [1, 2, 3] else 0)

# This will split the dataset into X (columns to use (except classification_final and date_died)), and y, the target column.
X = data.drop(columns=['CLASIFFICATION_FINAL', 'DATE_DIED'])
y = data['CLASIFFICATION_FINAL']

# this splits the data into training and testing sets. Where 40% of the dataset will be used for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Naive Bayes Model Training and Prediction
bayes = MultinomialNB()
bayes.fit(X_train, y_train)
y_pred = bayes.predict(X_test)

# This will print the accuracy, along with the classification report to the output.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# This will give a simple amount of correctly classified points, along with a percentage of that amount.
misclassified_count = (y_test != y_pred).sum()
percentage_misclassified = (misclassified_count / X_test.shape[0]) * 100
print("Number of mislabeled points out of a total %d points: %d" % (X_test.shape[0], misclassified_count))
print("Percentage of misclassified points: %.2f%%" % percentage_misclassified)
