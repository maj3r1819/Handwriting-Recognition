import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import cv2

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:21000, 1:].values
y_train = dataset.iloc[:21000, 0].values

X_test = dataset.iloc[21000:, 1:].values
y_test = dataset.iloc[21000:, 0].values


#Training the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
#print(cm)
print("Accuracy of the Random Forest Model :" ,accuracy_score(y_test, y_pred)*100 ,"%")



#Testing the test set!!!
"""
d =X_test[100]
d.shape = (28,28)
print(X_test[100])
print(d)
pt.imshow(d, cmap='gray')
print(classifier.predict([X_test[100]]))
"""


#Testing with 'img.png'

img = cv2.imread('img.png',)
cv2.imshow('image', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_arr = np.array(gray)
#print(img_arr)
flattened_array = img_arr.flatten()

#print("Flattened array is :" , flattened_array)
print("Number is :",classifier.predict([flattened_array]))
cv2.waitKey(0)
cv2.destroyAllWindows()