import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values

dataset1 = pd.read_csv('test.csv')
X_test = dataset1.iloc[:, :].values


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)

#Test Predictions
submission = pd.DataFrame({'Label': y_pred })
submission.to_csv("test_sub.csv", index=False)