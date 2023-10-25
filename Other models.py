import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Here the dataset will be loaded, and the numbers 97, and 99, which are missing values, will be removed
data = pd.read_csv("Covid Data.csv")

# Replace 97 and 99 with NaN in all columns except 'age'
columns_to_replace = data.columns.difference(['age'])  # Exclude 'age' column
data[columns_to_replace] = data[columns_to_replace].replace([97, 99], np.nan)
data.dropna(inplace=True)

# This will simplify the classification_final to either Covid (1) or not covid (0)
data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].map(lambda x: 1 if x in [1, 2, 3] else 0)

# Here the data is split, and columns classificatio_final, and date_died will be excluded from the models
X = data.drop(columns=['CLASIFFICATION_FINAL', 'DATE_DIED'])
y = data['CLASIFFICATION_FINAL']

# Here the data will be split into a testing set, and a training set, whhich is 40% - 60% respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# -- RANDOM FOREST MODEL ---------------------------------------------------------------------------------------------------------------------
# Random Forest Model Training and Prediction
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# this will print the accuracy of the model, along with the classification report of the Random Forest model.
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, zero_division=1)
print("\nRandom Forest Classifier Results:")
print("Accuracy:", accuracy_rf)
print("\nClassification Report:\n", report_rf)

# Here a simple calculation will be done to show the amount of misclassified points, along with a percentage
misclassified_count_rf = (y_test != y_pred_rf).sum()
percentage_misclassified_rf = (misclassified_count_rf / X_test.shape[0]) * 100
print("Number of mislabeled points out of a total %d points: %d" % (X_test.shape[0], misclassified_count_rf))
print("Percentage of misclassified points: %.2f%%" % percentage_misclassified_rf)



# -- LOGISTIC REGRESSION MODEL ---------------------------------------------------------------------------------------------------------------------
# Logistic Regression Model Training and Prediction
logistic_classifier = LogisticRegression(max_iter=20000)  # Increase max_iter to resolve the convergence warning
logistic_classifier.fit(X_train, y_train)
y_pred_logistic = logistic_classifier.predict(X_test)

# Here the same results will be printed, but this time for logistic regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
report_logistic = classification_report(y_test, y_pred_logistic, zero_division=1)
print("\nLogistic Regression Classifier Results:")
print("Accuracy:", accuracy_logistic)
print("\nClassification Report:\n", report_logistic)

# Here a simple calculation will be done again, to show the amount of misclassified points along with a percentage
misclassified_count_logistic = (y_test != y_pred_logistic).sum()
percentage_misclassified_logistic = (misclassified_count_logistic / X_test.shape[0]) * 100
print("Number of mislabeled points out of a total %d points: %d" % (X_test.shape[0], misclassified_count_logistic))
print("Percentage of misclassified points: %.2f%%" % percentage_misclassified_logistic)
