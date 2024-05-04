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
from scipy import stats #Import stats to help with data preprocessing

#Importing the dataset
dataset = pd.read_csv('train.csv')
datasettest = pd.read_csv('test.csv')
#Ignoring the passenger id column since a simple numbering from 1 onwards is not helpful in predicting data
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
Xtest = datasettest.iloc[:, 1:].values

#Exploratory Data Analysis
dataset['Died'] = 1 - dataset['Survived']
#Checking what significance Pclass has on survival rate
dataset.groupby('Pclass').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True,)
#From this bar plot, it is obvious that passengers with Pclass 1 are more likely to survive than 2 and then 3
#Checking if test data has a pclass value not present in the training set
print(set(dataset['Pclass']))
print(set(datasettest['Pclass']))
#The answer is no
#Checking what significance Sex has on survival rate
dataset.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True,)
#From this bar plot, it is obvious that more female passengers survived
#Checking if test data has a sex value not present in the training set
print(set(dataset['Sex']))
print(set(datasettest['Sex']))
#The answer is no
#Checking what significance Age has on survival rate
plt.hist([dataset[dataset['Survived'] == 1]['Age']], stacked = True, bins = 50, label = 'Survived')
plt.hist([dataset[dataset['Survived'] == 0]['Age']], stacked = True, bins = 50, label = 'Dead')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
#From this plot, it is obvious that the youngest ones between 0 to 5 years old survive the most
#Checking what significance SibSp has on survival rate
dataset.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True,)
#From this bar plot, it is obvious higher SibSp means lower survival rate
#Checking if test data has a sibsp value not present in the training set
print(set(dataset['SibSp']))
print(set(datasettest['SibSp']))
#The answer is no
#Checking what significance Parch has on survival rate
dataset.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True,)
#From this bar plot, it is obvious higher Parch means lower survival rate
#Checking if test data has a parch value not present in the training set
print(set(dataset['Parch']))
print(set(datasettest['Parch']))
#The answer is yes. There is one value of 9 in the test data not present in the training data
#However this is fine because a value of 6 means dead hence higher value of 9 will surely result
#in an accurate prediction of dead
#Checking what significance Fare has on survival rate
plt.hist([dataset[dataset['Survived'] == 1]['Fare']], stacked = True, bins = 50, label = 'Survived')
plt.hist([dataset[dataset['Survived'] == 0]['Fare']], stacked = True, bins = 50, label = 'Dead')
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
#From this, it is clear higher fare passengers are more likely to survive
#Checking what significance Embarked has on survival rate
dataset.groupby('Embarked').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), stacked=True,)
#From this bar plot, it is obvious the most survival is with embarked C, then Q then S. So C can be 3, Q 2, S 1
#Checking if test data has an embarked value not present in the training set
print(set(dataset['Embarked']))
print(set(datasettest['Embarked']))
#The answer is no (We dont count nan)
#For the name column, each name has a title - Mr., Mrs., etc. Lets see how many such titles are there
titlelist = []
for name in dataset['Name']:
    titlelist.append(name.split(',')[1].split('.')[0].strip())
titlelistset = list(set(titlelist))
print(set(titlelistset))
titlelistarray = np.asarray(titlelist)
#Seeing what significance these titles have on predictions
sns.barplot(x = titlelistarray, y = np.asarray(list(range(0, len(titlelistarray)))), hue = 'Survived', data = dataset)
#From this plot, its clear certain titles are least important, certain are more important and certain are in the middle
#For titles that have both survived and dead, we can quantify them as 2. For titles with survived only, these are the
#highest priority hence we give them 2. For titles that have all dead, we give them 1.
#Checking if test data has a title not present in the training set
print(set(titlelistset))
titlelisttest = []
for name in datasettest['Name']:
    titlelisttest.append(name.split(',')[1].split('.')[0].strip())
titlelisttestset = list(set(titlelisttest))
print(set(titlelisttestset))
#The answer is yes. Dona exists in the test set. However, this is simply the female version of Don hence we can consider
#Dona to hold the same value as Don
#For the ticket, lets classify them into tickets with just numbers and tickets without numbers and see if there is any
#correlation with who survives and who dies
ticketlist = []
for ticket in dataset['Ticket']:
    if ticket.isnumeric():
        ticketlist.append('Numbers Only Ticket')
    else:
        ticketlist.append('Letters and Numbers Ticket')
ticketlistarray = np.asarray(ticketlist)
sns.barplot(x = ticketlistarray, y = np.asarray(list(range(0, len(ticketlistarray)))), hue = 'Survived', data = dataset)
#We see there are more deaths than survivals when a passenger has a ticket with numbers only. Hence, we can assign the
#letters and numbers ticket to be 1 (higher) and numbers only ticket to be 0 (lower)
#Lastly, cabin. It has a lot of unknown data so it will be removed

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
imputer.fit(X[:, [3]]) #Replace column 3 missing values
X[:, [3]] = imputer.transform(X[:, [3]])
#Since age cannot be a decimal value, rounding off to the nearest integer
for index, val in enumerate(X[:, 3]):
    X[index, 3] = round(val)
#Since 77% of data is missing in cabin category, deleting that column
X = np.delete(X, 8, 1)
#Since just 0.2% data is missing, replace the embarked missing values with most occuring value
print(stats.mode(X[:, 8]))
#From this, we see S is the most occuring entry hence put S in missing entries
for index, val in enumerate(X[:, 8]):
    if pd.isnull(X[index, 8]):
        X[index, 8] = 'S'

#Encoding categorical data
#For the name, based on what we identified in exploratory data analysis, convert titles to 1, 2 and 3
titlelistnumber = []
for name in X[:, 1]:
    temp = name.split(',')[1].split('.')[0].strip()
    if temp == 'Don' or temp == 'Rev' or temp == 'Capt' or temp == 'Jonkheer':
        titlelistnumber.append(1)
    elif temp == 'Mme' or temp == 'Ms' or temp == 'Lady' or temp == 'Sir' or temp == 'Mlle' or temp == 'the Countess':
        titlelistnumber.append(3)
    else:
        titlelistnumber.append(2)
X[:, 1] = np.asarray(titlelistnumber)
#To distinguish between male and female, assign female as 1 and male as 0 since more females survived than males
sexnumber = []
for val in X[:, 2]:
    if val == 'male':
        sexnumber.append(0)
    else:
        sexnumber.append(1)
X[:, 2] = np.asarray(sexnumber)
#Distinguish between tickets by seeing if there are only numbers or alphanumeric characters in the ticket
#Based on exploratory data analysis, encode numbertickets as 0 and alphanumeric as 1
ticketlistnumber = []
for val in X[:, 6]:
    if val.isnumeric():
        ticketlistnumber.append(0)
    else:
        ticketlistnumber.append(1)
X[:, 6] = np.asarray(ticketlistnumber)
#For embarked, C is 3, Q is 2 and S is 1 based on exploratory data analysis
embarkednumber = []
for val in X[:, 8]:
    if val == 'C':
        embarkednumber.append(3)
    elif val == 'Q':
        embarkednumber.append(2)
    else:
        embarkednumber.append(1)
X[:, 8] = np.asarray(embarkednumber)
#Verifying if we have any missing data still
print(np.isnan(np.min(X)))
#The answer is no

#Now making sure our test data is also in a proper format by following similar approaches
missing_data = datasettest.isnull().sum() #Identify amount of missing data
print(missing_data)
#We see Age, Cabin and Fare have missing data. Lets calculate the percentage of missing data in these categories
missingpercentage = (100 / len(Xtest[:, 3])) * missing_data['Age']
missingpercentcabin = (100 / len(Xtest[:, 8])) * missing_data['Cabin']
missingpercentfare = (100 / len(Xtest[:, 7])) * missing_data['Fare']
print('Missing data in Age category is: ' + str(missingpercentage) + '%')
print('Missing data in Cabin category is: ' + str(missingpercentcabin) + '%')
print('Missing data in Fare category is: ' + str(missingpercentfare) + '%')
#Since there is around 20% missing data in age, using mean of available data to replace missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [3]]) #Replace column 3 missing values
Xtest[:, [3]] = imputer.transform(Xtest[:, [3]])
#Since age cannot be a decimal value, rounding off to the nearest integer
for index, val in enumerate(Xtest[:, 3]):
    Xtest[index, 3] = round(val)
#Since 77% of data is missing in cabin category, deleting that column
Xtest = np.delete(Xtest, 8, 1)
#Since just 0.2% data is missing, replace the fare missing values with average
imputer.fit(Xtest[:, [7]]) #Replace column 7 missing values
Xtest[:, [7]] = imputer.transform(Xtest[:, [7]])

#Encoding categorical data
#For the name, based on what we identified in exploratory data analysis, convert titles to 1, 2 and 3
titlelistnumber = []
for name in Xtest[:, 1]:
    temp = name.split(',')[1].split('.')[0].strip()
    if temp == 'Don' or temp == 'Rev' or temp == 'Capt' or temp == 'Jonkheer' or temp == 'Dona':
        titlelistnumber.append(1)
    elif temp == 'Mme' or temp == 'Ms' or temp == 'Lady' or temp == 'Sir' or temp == 'Mlle' or temp == 'the Countess':
        titlelistnumber.append(3)
    else:
        titlelistnumber.append(2)
Xtest[:, 1] = np.asarray(titlelistnumber)
#To distinguish between male and female, assign female as 1 and male as 0 since more females survived than males
sexnumber = []
for val in Xtest[:, 2]:
    if val == 'male':
        sexnumber.append(0)
    else:
        sexnumber.append(1)
Xtest[:, 2] = np.asarray(sexnumber)
#Distinguish between tickets by seeing if there are only numbers or alphanumeric characters in the ticket
#Based on exploratory data analysis, encode numbertickets as 0 and alphanumeric as 1
ticketlistnumber = []
for val in Xtest[:, 6]:
    if val.isnumeric():
        ticketlistnumber.append(0)
    else:
        ticketlistnumber.append(1)
Xtest[:, 6] = np.asarray(ticketlistnumber)
#For embarked, C is 3, Q is 2 and S is 1 based on exploratory data analysis
embarkednumber = []
for val in Xtest[:, 8]:
    if val == 'C':
        embarkednumber.append(3)
    elif val == 'Q':
        embarkednumber.append(2)
    else:
        embarkednumber.append(1)
Xtest[:, 8] = np.asarray(embarkednumber)
#Verifying if we have any missing data still
print(np.isnan(np.min(Xtest)))
#The answer is no

#Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Feature Scaling
sc = StandardScaler() #Applying feature scaling to feature scale age and fare
X_train[:, [3]] = sc.fit_transform(X_train[:, [3]])
X_train[:, [7]] = sc.fit_transform(X_train[:, [7]])
X_test[:, [3]] = sc.transform(X_test[:, [3]])
X_test[:, [7]] = sc.transform(X_test[:, [7]])
Xtest[:, [3]] = sc.transform(Xtest[:, [3]])
Xtest[:, [7]] = sc.transform(Xtest[:, [7]])

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
X[:, [3]] = sc.fit_transform(X[:, [3]])
X[:, [7]] = sc.fit_transform(X[:, [7]])
classifierrffinal = SVC(kernel = 'rbf')
classifierrffinal.fit(X, y)

#Applying Random Forest
y_predfinal = classifierrffinal.predict(Xtest)

#Convert to csv
df = pd.DataFrame({"PassengerId" : y_predfinal, "Survived" : y_predfinal})
df.to_csv("TitanicSubmission.csv", index=False)
