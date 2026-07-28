# import the libraries
import numpy as np
import pandas as pd# import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')# get dummy variables
df_getdummy=pd.get_dummies(data=df, columns=['Gender'])# seperate X and y variables
X = df_getdummy.drop('Purchased',axis=1)
y = df_getdummy['Purchased']# split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)# fit Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)# predict the Test set results
y_pred = classifier.predict(X_test)# make the confusion matrix for evaluation
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)