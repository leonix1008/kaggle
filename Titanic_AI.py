#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer #Import SimpleImputer class
from sklearn.compose import ColumnTransformer #Import class to transform columns of dataset
from sklearn.preprocessing import OneHotEncoder #Import class to perform one-hot encoding
from sklearn.preprocessing import LabelEncoder #Import class to perform label encoding
from sklearn.model_selection import train_test_split #Import function to split data into training and test set
from sklearn.preprocessing import StandardScaler #Import class to perform feature scaling
from sklearn.linear_model import LogisticRegression #Import class to implement logistic regression
from sklearn.metrics import confusion_matrix, accuracy_score #Import methods to implement confusion matrix and calculate accuracy
from sklearn.neighbors import KNeighborsClassifier #Import class to implement kNN
from sklearn.svm import SVC #Import class to implement SVM
from sklearn.naive_bayes import GaussianNB #Import class to implement Naive Bayes
from sklearn.ensemble import RandomForestClassifier #Import class to implement Random Forest Classification

#Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Data Preprocessing
#Since there is 177 missing data in age, using mean of available data to replace missing data
missing_data = dataset.isnull().sum()
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [3]]) #Replace column 3 missing values
X[:, [3]] = imputer.transform(X[:, [3]])
for index, val in enumerate(X[:, 8]): #Replacing the two nan embarked values with assumed value Q
    if pd.isnull(X[index, 8]):
        X[index, 8] = 'Q'

#Encoding categorical data
for index, val in enumerate(X[:, 1]): #Only get the last names of the passengers before the ,
    X[index, 1] = val.split(',')[0]
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough', sparse_threshold = 0) #Apply one hot encoding to the last
#name column to make them into categories
X = np.array(ct.fit_transform(X)) #Use label encoding to categorize male and female in column 668
le = LabelEncoder()
X[:, 668] = le.fit_transform(X[:, [668]])
for index, val in enumerate(X[:, 672]): #Distinguish between tickets by seeing if there are only numbers or alphanumeric characters in the ticket
    if val.isnumeric():
        X[index, 672] = 'NumberTicket'
    else:
        X[index, 672] = 'AlphanumericTicket'
X[:, 672] = le.fit_transform(X[:, 672]) #Apply label encoding to the ticket column
ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [674])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct2.fit_transform(X)) #Apply one-hot encoding to column 674 to classify the embarked category

#Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Feature Scaling
sc = StandardScaler() #Applying feature scaling on columns 672 and 676 to feature scale age and fare
X_train[:, [672]] = sc.fit_transform(X_train[:, [672]])
X_train[:, [676]] = sc.fit_transform(X_train[:, [676]])
X_test[:, [672]] = sc.fit_transform(X_test[:, [672]])
X_test[:, [676]] = sc.fit_transform(X_test[:, [676]])

#Applying logistic regression
classifierlog = LogisticRegression()
classifierlog.fit(X, y)

#Predicting the test set results
y_pred = classifierlog.predict(X_test)

#Making the Confusion Matrix
cmlog = confusion_matrix(y_test, y_pred)
accuracylog = accuracy_score(y_test, y_pred)

#Applying kNN
classifierknn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierknn.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifierknn.predict(X_test)

#Making the Confusion Matrix
cmknn = confusion_matrix(y_test, y_pred)
accuracyknn = accuracy_score(y_test, y_pred)

#Applying SVM
classifiersvm = SVC(kernel = 'rbf')
classifiersvm.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifiersvm.predict(X_test)

#Making the Confusion Matrix
cmsvm = confusion_matrix(y_test, y_pred)
accuracysvm = accuracy_score(y_test, y_pred)

#Applying Naive Bayes
classifiernb = GaussianNB()
classifiernb.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifiernb.predict(X_test)

#Making the Confusion Matrix
cmnb = confusion_matrix(y_test, y_pred)
accuracynb = accuracy_score(y_test, y_pred)

#Applying random forest
classifierrf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
classifierrf.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifierrf.predict(X_test)

#Making the Confusion Matrix
cmrf = confusion_matrix(y_test, y_pred)
accuracyrf = accuracy_score(y_test, y_pred)
