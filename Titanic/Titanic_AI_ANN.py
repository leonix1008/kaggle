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
import tensorflow as tf #Import tensorflow to implement artificial neural networks

#Importing the dataset
dataset = pd.read_csv('train.csv')
#Ignoring the passenger id column since a simple numbering from 1 onwards is not helpful in predicting data
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
#Names of the passengers will not help in predictions hence removing that column
X = np.delete(X, 1, 1)

#Data Preprocessing
missing_data = dataset.isnull().sum() #Identify amount of missing data
print(missing_data)
#We see Age, Cabin and Embarked have missing data. Lets calculate the percentage of missing data in these categories
missingpercentage = (100 / len(X[:, 2])) * missing_data['Age']
missingpercentcabin = (100 / len(X[:, 7])) * missing_data['Cabin']
missingpercentembarked = (100 / len(X[:, 8])) * missing_data['Embarked']
print('Missing data in Age category is: ' + str(missingpercentage) + '%')
print('Missing data in Cabin category is: ' + str(missingpercentcabin) + '%')
print('Missing data in Embarked category is: ' + str(missingpercentembarked) + '%')
#Since there is around 20% missing data in age, using mean of available data to replace missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [2]]) #Replace column 3 missing values
X[:, [2]] = imputer.transform(X[:, [2]])
#Since age cannot be a decimal value, rounding off to the nearest integer
for index, val in enumerate(X[:, 2]):
    X[index, 2] = round(val)
#Since 77% of data is missing in cabin category, deleting that column
X = np.delete(X, 7, 1)
#Since just 0.2% data is missing, replace the embarked missing values with assumed value Q
for index, val in enumerate(X[:, 7]):
    if pd.isnull(X[index, 7]):
        X[index, 7] = 'Q'

#Encoding categorical data
#To distinguish between male and female, use label encoding to convert these into boolean 1 and 0
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, [1]])
#Distinguish between tickets by seeing if there are only numbers or alphanumeric characters in the ticket
for index, val in enumerate(X[:, 5]):
    if val.isnumeric():
        X[index, 5] = 'NumberTicket'
    else:
        X[index, 5] = 'AlphanumericTicket'
#Now apply label encoding to the ticket column
X[:, 5] = le.fit_transform(X[:, 5])
#Apply one-hot encoding to classify the embarked category
ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [7])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct2.fit_transform(X))

#Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Feature Scaling
sc = StandardScaler() #Applying feature scaling to feature scale age and fare
X_train[:, [5]] = sc.fit_transform(X_train[:, [5]])
X_train[:, [9]] = sc.fit_transform(X_train[:, [9]])
X_test[:, [5]] = sc.fit_transform(X_test[:, [5]])
X_test[:, [9]] = sc.fit_transform(X_test[:, [9]])
X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')

#Initializing the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 500)

#Predicting the Test set results
X_test = np.asarray(X_test).astype('float32')
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

#Training the ANN model on the whole dataset now to maximize learning
X[:, [5]] = sc.fit_transform(X[:, [5]])
X[:, [9]] = sc.fit_transform(X[:, [9]])
X = np.asarray(X).astype('float32')
ann.fit(X, y, batch_size = 32, epochs = 500)

#Importing the dataset
dataset = pd.read_csv('test.csv')
#Ignoring the passenger id column since a simple numbering from 892 onwards is not helpful in predicting data
X = dataset.iloc[:, 1:].values
#Names of the passengers will not help in predictions hence removing that column
X = np.delete(X, 1, 1)

#Data Preprocessing
missing_data = dataset.isnull().sum() #Identify amount of missing data
print(missing_data)
#We see Age, Fare and Cabin have missing data. Lets calculate the percentage of missing data in these categories
missingpercentage = (100 / len(X[:, 2])) * missing_data['Age']
missingpercentfare = (100 / len(X[:, 6])) * missing_data['Fare']
missingpercentcabin = (100 / len(X[:, 7])) * missing_data['Cabin']
print('Missing data in Age category is: ' + str(missingpercentage) + '%')
print('Missing data in Fare category is: ' + str(missingpercentfare) + '%')
print('Missing data in Cabin category is: ' + str(missingpercentcabin) + '%')
#Since there is around 20% missing data in age, using mean of available data to replace missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [2]]) #Replace column 3 missing values
X[:, [2]] = imputer.transform(X[:, [2]])
#Since age cannot be a decimal value, rounding off to the nearest integer
for index, val in enumerate(X[:, 2]):
    X[index, 2] = round(val)
#Since 78% of data is missing in cabin category, deleting that column
X = np.delete(X, 7, 1)
#Since just 0.2% data is missing, replace the fare missing values with assumed value 10
for index, val in enumerate(X[:, 6]):
    if pd.isnull(X[index, 6]):
        X[index, 6] = 10
        
#Encoding categorical data
#To distinguish between male and female, use label encoding to convert these into boolean 1 and 0
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, [1]])
#Distinguish between tickets by seeing if there are only numbers or alphanumeric characters in the ticket
for index, val in enumerate(X[:, 5]):
    if val.isnumeric():
        X[index, 5] = 'NumberTicket'
    else:
        X[index, 5] = 'AlphanumericTicket'
#Now apply label encoding to the ticket column
X[:, 5] = le.fit_transform(X[:, 5])
#Apply one-hot encoding to classify the embarked category
ct2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [7])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct2.fit_transform(X))

#Feature Scaling
sc = StandardScaler() #Applying feature scaling to feature scale age and fare
X_train[:, [5]] = sc.fit_transform(X_train[:, [5]])
X_train[:, [9]] = sc.fit_transform(X_train[:, [9]])
X_test[:, [5]] = sc.fit_transform(X_test[:, [5]])
X_test[:, [9]] = sc.fit_transform(X_test[:, [9]])
X = np.asarray(X).astype('float32')

#Applying ANN
y_predfinal = ann.predict(X)
y_predfinal = (y_predfinal > 0.5)

outputlist = []
for i in range(0, len(y_predfinal)):
    outputlist.append(int(y_predfinal[i][0]))

#Convert to csv
df = pd.DataFrame({"PassengerId" : outputlist, "Survived" : outputlist})
df.to_csv("TitanicSubmission.csv", index=False)