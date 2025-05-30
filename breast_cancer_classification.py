# -*- coding: utf-8 -*-
"""Breast Cancer Classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xj5yeVLCt1wgr44Ua97MCwxpcWNIIlys
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn.datasets

breast_cancer_dataset=sklearn.datasets.load_breast_cancer()

breast_cancer_dataset

data_set=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)

data_set.head()

data_set['label']=breast_cancer_dataset.target

data_set.head()

data_set.tail()

data_set.shape

data_set.info()

data_set.describe()

data_set.isnull().sum()

data_set.groupby("label").mean()

data_set.value_counts('label')

X=data_set.drop(columns='label',axis=1)
Y=data_set['label']

print(X)

print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

model=LogisticRegression()

model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)

print("The accuracy on training data is ",training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)

print("The accuracy on the testing data is ",test_data_accuracy)

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')

