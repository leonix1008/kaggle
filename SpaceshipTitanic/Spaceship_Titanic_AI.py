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
from scipy import stats

#Importing the dataset
dataset = pd.read_csv('train.csv')
datasettest = pd.read_csv('test.csv')
#Ignoring the passenger id column since a simple numbering is not helpful in predicting data
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
Xtest = datasettest.iloc[:, 1:].values
#Converting False to 0 and True to 1 for easier exploratory data exploration
ylist = []
for val in y:
    if val == False:
        ylist.append(0)
    elif val == True:
        ylist.append(1)
y = np.asarray(ylist)
dataset['Transported'] = y

#Exploratory Data Analysis
dataset['NotTransported'] = 1 - dataset['Transported']
#Checking what significance HomePlanet has on outcome
dataset.groupby('HomePlanet').agg('mean')[['Transported', 'NotTransported']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#Based on this, can assign Europa as 3, Mars as 2, Earth as 1
#Checking what significance CryoSleep has on outcome
dataset.groupby('CryoSleep').agg('mean')[['Transported', 'NotTransported']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#Based on this, can assign TRUE as 1, FALSE as 0
#For cabin, port or starboard might be significant hence checking just the last character
pors = []
for val in dataset['Cabin']:
    if type(val) is str:
        pors.append(val[-1])
    else:
        pors.append('nan')
porsarr = np.asarray(pors)
sns.barplot(x = porsarr, y = np.asarray(list(range(0, len(porsarr)))), hue = 'Transported', data = dataset)
#Based on this, use one-hot encoding for port and starboard
#Checking what significance Destination has on outcome
dataset.groupby('Destination').agg('mean')[['Transported', 'NotTransported']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#Based on this, can assign Cancri e as 1 and the other 2 as 0
#Checking what significance Age has on outcome
plt.hist([dataset[dataset['Transported'] == 1]['Age']], stacked = True, bins = 50, label = 'Transported')
plt.hist([dataset[dataset['Transported'] == 0]['Age']], stacked = True, bins = 50, label = 'Not Transported')
#Checking what significance VIP has on outcome
dataset.groupby('VIP').agg('mean')[['Transported', 'NotTransported']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#Based on this, can assign FALSE as 1, TRUE as 0
#For the names, we will consider last names only as a criteria and check if it has any significance
namelist = []
for val in dataset['Name']:
    if type(val) is str:
        namelist.append(val.split(' ')[1])
    else:
        namelist.append('nan')
namelistarr = np.asarray(namelist)
#sns.barplot(x = namelistarr, y = np.asarray(list(range(0, len(namelistarr)))), hue = 'Transported', data = dataset)
#There are too many last names so name category is not useful in prediction. It will be removed

#Data Preprocessing
missing_data = dataset.isnull().sum() #Identify amount of missing data
#Handling the missing data through various techniques
print(stats.mode(X[:, 0]))
for index, val in enumerate(X[:, 0]):
    if pd.isnull(X[index, 0]):
        X[index, 0] = 'Earth'
print(stats.mode(X[:, 1]))
for index, val in enumerate(X[:, 1]):
    if pd.isnull(X[index, 1]):
        X[index, 1] = False
#For cabin, only port or starboard is important so using porsarr as variable
print(stats.mode(porsarr))
for index, val in enumerate(X[:, 2]):
    if pd.isnull(X[index, 2]):
        X[index, 2] = 'S'
print(stats.mode(X[:, 3]))
for index, val in enumerate(X[:, 3]):
    if pd.isnull(X[index, 3]):
        X[index, 3] = 'TRAPPIST-1e'
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [4]])
X[:, [4]] = imputer.transform(X[:, [4]])
for index, val in enumerate(X[:, 4]):
    X[index, 4] = round(val)
print(stats.mode(X[:, 5]))
for index, val in enumerate(X[:, 5]):
    if pd.isnull(X[index, 5]):
        X[index, 5] = False
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [6]])
X[:, [6]] = imputer.transform(X[:, [6]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [7]])
X[:, [7]] = imputer.transform(X[:, [7]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [8]])
X[:, [8]] = imputer.transform(X[:, [8]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [9]])
X[:, [9]] = imputer.transform(X[:, [9]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [10]])
X[:, [10]] = imputer.transform(X[:, [10]])
#Deleting name column as it was identified it wont be useful in prediction
X = np.delete(X, 11, 1)

#Encoding categorical data
#Encoding categorical data via identified techniques in EDA
planet = []
for val in X[:, 0]:
    if val == 'Europa':
        planet.append(3)
    elif val == 'Mars':
        planet.append(2)
    else:
        planet.append(1)
X[:, 0] = np.asarray(planet)
cryo = []
for val in X[:, 1]:
    if val == True:
        cryo.append(1)
    else:
        cryo.append(0)
X[:, 1] = np.asarray(cryo)
pors = []
for val in X[:, 2]:
    if type(val) is str:
        pors.append(val[-1])
    else:
        pors.append('nan')
X[:, 2] = np.asarray(pors)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [2])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
destination = []
for val in X[:, 4]:
    if val == '55 Cancri e':
        destination.append(1)
    else:
        destination.append(0)
X[:, 4] = np.asarray(destination)
vip = []
for val in X[:, 6]:
    if val == False:
        vip.append(1)
    else:
        vip.append(0)
X[:, 6] = np.asarray(vip)
#Verifying if we have any missing data still
print(np.isnan(np.min(X)))
#The answer is no

#Now making sure our test data is also in a proper format by following similar approaches
missing_data = datasettest.isnull().sum() #Identify amount of missing data
#Handling the missing data through various techniques
print(stats.mode(Xtest[:, 0]))
for index, val in enumerate(Xtest[:, 0]):
    if pd.isnull(Xtest[index, 0]):
        Xtest[index, 0] = 'Earth'
print(stats.mode(Xtest[:, 1]))
for index, val in enumerate(Xtest[:, 1]):
    if pd.isnull(Xtest[index, 1]):
        Xtest[index, 1] = False
#For cabin, only port or starboard is important so using porsarr as variable
print(stats.mode(porsarr))
for index, val in enumerate(Xtest[:, 2]):
    if pd.isnull(Xtest[index, 2]):
        Xtest[index, 2] = 'S'
print(stats.mode(Xtest[:, 3]))
for index, val in enumerate(Xtest[:, 3]):
    if pd.isnull(Xtest[index, 3]):
        Xtest[index, 3] = 'TRAPPIST-1e'
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [4]])
Xtest[:, [4]] = imputer.transform(Xtest[:, [4]])
for index, val in enumerate(Xtest[:, 4]):
    Xtest[index, 4] = round(val)
print(stats.mode(Xtest[:, 5]))
for index, val in enumerate(Xtest[:, 5]):
    if pd.isnull(Xtest[index, 5]):
        Xtest[index, 5] = False
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [6]])
Xtest[:, [6]] = imputer.transform(Xtest[:, [6]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [7]])
Xtest[:, [7]] = imputer.transform(Xtest[:, [7]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [8]])
Xtest[:, [8]] = imputer.transform(Xtest[:, [8]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [9]])
Xtest[:, [9]] = imputer.transform(Xtest[:, [9]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [10]])
Xtest[:, [10]] = imputer.transform(Xtest[:, [10]])
#Deleting name column as it was identified it wont be useful in prediction
Xtest = np.delete(Xtest, 11, 1)

#Encoding categorical data
#Encoding categorical data via identified techniques in EDA
planet = []
for val in Xtest[:, 0]:
    if val == 'Europa':
        planet.append(3)
    elif val == 'Mars':
        planet.append(2)
    else:
        planet.append(1)
Xtest[:, 0] = np.asarray(planet)
cryo = []
for val in Xtest[:, 1]:
    if val == True:
        cryo.append(1)
    else:
        cryo.append(0)
Xtest[:, 1] = np.asarray(cryo)
pors = []
for val in Xtest[:, 2]:
    if type(val) is str:
        pors.append(val[-1])
    else:
        pors.append('nan')
Xtest[:, 2] = np.asarray(pors)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [2])], remainder = 'passthrough', sparse_threshold = 0)
Xtest = np.array(ct.fit_transform(Xtest))
destination = []
for val in Xtest[:, 4]:
    if val == '55 Cancri e':
        destination.append(1)
    else:
        destination.append(0)
Xtest[:, 4] = np.asarray(destination)
vip = []
for val in Xtest[:, 6]:
    if val == False:
        vip.append(1)
    else:
        vip.append(0)
Xtest[:, 6] = np.asarray(vip)
#Verifying if we have any missing data still
print(np.isnan(np.min(Xtest)))
#The answer is no

#Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Feature Scaling
sc = StandardScaler() #Applying feature scaling
X_train[:, [2]] = sc.fit_transform(X_train[:, [2]])
X_train[:, [5]] = sc.fit_transform(X_train[:, [5]])
X_train[:, [7]] = sc.fit_transform(X_train[:, [7]])
X_train[:, [8]] = sc.fit_transform(X_train[:, [8]])
X_train[:, [9]] = sc.fit_transform(X_train[:, [9]])
X_train[:, [10]] = sc.fit_transform(X_train[:, [10]])
X_train[:, [11]] = sc.fit_transform(X_train[:, [11]])
X_test[:, [2]] = sc.transform(X_test[:, [2]])
X_test[:, [5]] = sc.transform(X_test[:, [5]])
X_test[:, [7]] = sc.transform(X_test[:, [7]])
X_test[:, [8]] = sc.transform(X_test[:, [8]])
X_test[:, [9]] = sc.transform(X_test[:, [9]])
X_test[:, [10]] = sc.transform(X_test[:, [10]])
X_test[:, [11]] = sc.transform(X_test[:, [11]])
Xtest[:, [2]] = sc.transform(Xtest[:, [2]])
Xtest[:, [5]] = sc.transform(Xtest[:, [5]])
Xtest[:, [7]] = sc.transform(Xtest[:, [7]])
Xtest[:, [8]] = sc.transform(Xtest[:, [8]])
Xtest[:, [9]] = sc.transform(Xtest[:, [9]])
Xtest[:, [10]] = sc.transform(Xtest[:, [10]])
Xtest[:, [11]] = sc.transform(Xtest[:, [11]])

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

#Since Random Forest gives the highest accuracy, we will use it to predict results in the test.csv file

#Training the Random Forest model on the whole dataset now to maximize learning
#Feature scaling the whole data
X[:, [2]] = sc.fit_transform(X[:, [2]])
X[:, [5]] = sc.fit_transform(X[:, [5]])
X[:, [7]] = sc.fit_transform(X[:, [7]])
X[:, [8]] = sc.fit_transform(X[:, [8]])
X[:, [9]] = sc.fit_transform(X[:, [9]])
X[:, [10]] = sc.fit_transform(X[:, [10]])
X[:, [11]] = sc.fit_transform(X[:, [11]])

classifierrffinal = SVC(kernel = 'rbf')
classifierrffinal.fit(X, y)

#Applying Random Forest
y_predfinal = classifierrffinal.predict(Xtest)
y_predfinal2 = []
for val in y_predfinal:
    if val == 1:
        y_predfinal2.append('True')
    else:
        y_predfinal2.append('False')
y_predfinal2arr = np.asarray(y_predfinal2)

#Convert to csv
df = pd.DataFrame({"PassengerId" : datasettest.PassengerId, "Transported" : y_predfinal2arr})
df.to_csv("SpaceshipSubmission.csv", index=False)