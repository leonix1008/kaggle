#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
#Ignoring the passenger id column since a simple numbering from 1 onwards is not helpful in predicting data
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
#Names of the passengers will not help in predictions hence removing that column
X = np.delete(X, 1, 1)

#Exploratory Data Analysis
dataset['Died'] = 1 - dataset['Survived']
#Checking what significance Pclass has on survival rate
dataset.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#From this bar plot, it is obvious that passengers with Pclass 1 are more likely to survive than 2 and then 3
#Checking what significance Sex has on survival rate
dataset.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#From this bar plot, it is obvious that more female passengers survived
#Checking what significance Age has on survival rate
plt.hist([dataset[dataset['Survived'] == 1]['Age']], stacked = True, bins = 50, label = 'Survived')
plt.hist([dataset[dataset['Survived'] == 0]['Age']], stacked = True, bins = 50, label = 'Dead')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
#From this plot, it is obvious that the youngest ones survive
#Checking what significance SibSp has on survival rate
dataset.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#From this bar plot, it is obvious higher SibSp means lower survival rate
#Checking what significance Parch has on survival rate
dataset.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#From this bar plot, it is obvious higher Parch means lower survival rate
#Checking what significance Fare has on survival rate
plt.hist([dataset[dataset['Survived'] == 1]['Fare']], stacked = True, bins = 50, label = 'Survived')
plt.hist([dataset[dataset['Survived'] == 0]['Fare']], stacked = True, bins = 50, label = 'Dead')
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
#From this, it is clear higher fare passengers are more likely to survive
#Checking what significance Embarked has on survival rate
dataset.groupby('Embarked').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#From this bar plot, it is obvious the most survival is with embarked C, then Q then S
#For the name column, each name has a title - Mr., Mrs., etc. Lets see how many such titles are there
titlelist = []
for name in dataset['Name']:
    titlelist.append(name.split(',')[1].split('.')[0].strip())
titlelistset = list(set(titlelist))
titlelistarray = np.asarray(titlelist)
#Seeing what significance these titles have on predictions
sns.barplot(x = titlelistarray, y = np.asarray(list(range(0, len(titlelistarray)))), hue = 'Survived', data = dataset)

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

#Identifying the model to use
print('The accuracy of KNN model is ' + str(accuracyknn) + '%')
print('The accuracy of logistic regression model is ' + str(accuracylog) + '%')
print('The accuracy of Naive Bayes model is ' + str(accuracynb) + '%')
print('The accuracy of Random Forest model is ' + str(accuracyrf) + '%')
print('The accuracy of SVM model is ' + str(accuracysvm) + '%')

#Since SVM gives the highest accuracy, we will use it to predict results in the test.csv file

#Training the SVM model on the whole dataset now to maximize learning
classifiersvmfinal = SVC(kernel = 'rbf')
classifiersvmfinal.fit(X, y)

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

#Applying SVM
y_predfinal = classifiersvmfinal.predict(X)

#Convert to csv
df = pd.DataFrame({"PassengerId" : y_predfinal, "Survived" : y_predfinal})
df.to_csv("TitanicSubmission.csv", index=False)