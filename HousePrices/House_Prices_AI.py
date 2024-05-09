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
#Ignoring the Id column since a simple numbering from 1 onwards is not helpful in predicting data
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, -1].values

#Exploratory Data Analysis
#Since there are a lot of features to explore one by one, we divide into numeric and categorical features. Starting with numeric
#Looking at the data
describe = dataset.describe()
#First, we tackle the data with high standard deviation since these are likely to have outliers. Lets only examine data with standard deviation
#of above 400
#Lot Area
plt.scatter(x = dataset['LotArea'], y = dataset['SalePrice'])
#We see most of the data points lie before 50000 and only a handful lie outside of it. Hence, those won't be helpful in prediction so removing
#data points above 50000 lot area
dataset = dataset[dataset['LotArea'] < 50000]
#Now, same thing with BsmtFinSF1
plt.scatter(x = dataset['BsmtFinSF1'], y = dataset['SalePrice'])
#In this case, we can remove the data above 2000 BsmtFinSF1
dataset = dataset[dataset['BsmtFinSF1'] < 2000]
#Now, same thing with TotalBsmtSF
plt.scatter(x = dataset['TotalBsmtSF'], y = dataset['SalePrice'])
#In this case, we can remove the data above 2500 TotalBsmtSF
dataset = dataset[dataset['TotalBsmtSF'] < 2500]
#Now, same thing with 2ndFlrSF
plt.scatter(x = dataset['2ndFlrSF'], y = dataset['SalePrice'])
#In this case, we can remove the data above 1500 2ndFlrSF
dataset = dataset[dataset['2ndFlrSF'] < 1500]
#Now, same thing with GrLivArea
plt.scatter(x = dataset['GrLivArea'], y = dataset['SalePrice'])
#In this case, we can remove the data above 4000 GrLivArea
dataset = dataset[dataset['GrLivArea'] < 4000]
#Now, same thing with MiscVal
plt.scatter(x = dataset['MiscVal'], y = dataset['SalePrice'])
#In this case, we can remove the data above 3000 MiscVal
dataset = dataset[dataset['MiscVal'] < 3000]
#Now lets examine the correlation between the numeric features and the final outcome
numeric_features = dataset.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])
print(corr['SalePrice'].sort_values(ascending=False))
#Checking via plots
sns.barplot(x = dataset['OverallQual'], y = dataset['SalePrice'], data = dataset)
sns.barplot(x = dataset['KitchenAbvGr'], y = dataset['SalePrice'], data = dataset)
plt.scatter(x = dataset['MiscVal'], y = dataset['SalePrice'])
#Now for categorical data
sns.barplot(x = dataset['MSZoning'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, we can label C (all) as 1, RM as 2, RH as 3, RL as 4, FV as 5
sns.barplot(x = dataset['Street'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Pave as 2 and Grvl as 1
sns.barplot(x = dataset['LotShape'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Reg as 1, IR3 as 2, IR1 as 3 and IR2 as 4
sns.barplot(x = dataset['LandContour'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Bnk as 1, Lvl and Low as 2, HLS as 3
sns.barplot(x = dataset['Utilities'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, AllPub as 2, NoSeWa as 1
sns.barplot(x = dataset['LotConfig'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Inside, FR2, Corner as 1, FR3 as 2, CulDSac as 3
#Since the data is not too large, we won't exclude any features. Reassigning X and y with the new dataset
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, -1].values

#Data Preprocessing
missing_data = dataset.isnull().sum() #Identify amount of missing data
print(missing_data)
#We see a bunch of missing data. Calculating the percentage in each category
missingpercentlotfrontage = (100 / len(X[:, 2])) * missing_data['LotFrontage']
missingpercentalley = (100 / len(X[:, 5])) * missing_data['Alley']
missingpercentmasvnrtype = (100 / len(X[:, 24])) * missing_data['MasVnrType']
missingpercentmasvnrarea = (100 / len(X[:, 25])) * missing_data['MasVnrArea']
missingpercentbsmtqual = (100 / len(X[:, 29])) * missing_data['BsmtQual']
missingpercentbsmtcond = (100 / len(X[:, 30])) * missing_data['BsmtCond']
missingpercentbsmtexposure = (100 / len(X[:, 31])) * missing_data['BsmtExposure']
missingpercentbsmtfintype1 = (100 / len(X[:, 32])) * missing_data['BsmtFinType1']
missingpercentbsmtfintype2 = (100 / len(X[:, 34])) * missing_data['BsmtFinType2']
missingpercentelectrical = (100 / len(X[:, 41])) * missing_data['Electrical']
missingpercentfireplacequ = (100 / len(X[:, 56])) * missing_data['FireplaceQu']
missingpercentgaragetype = (100 / len(X[:, 57])) * missing_data['GarageType']
missingpercentgarageyrblt = (100 / len(X[:, 58])) * missing_data['GarageYrBlt']
missingpercentgaragefinish = (100 / len(X[:, 59])) * missing_data['GarageFinish']
missingpercentgaragequal = (100 / len(X[:, 62])) * missing_data['GarageQual']
missingpercentgaragecond = (100 / len(X[:, 63])) * missing_data['GarageCond']
missingpercentpoolqc = (100 / len(X[:, 71])) * missing_data['PoolQC']
missingpercentfence = (100 / len(X[:, 72])) * missing_data['Fence']
missingpercentmiscfeature = (100 / len(X[:, 73])) * missing_data['MiscFeature']
print('Missing data in Lot Frontage category is: ' + str(missingpercentlotfrontage) + '%')
print('Missing data in Alley category is: ' + str(missingpercentalley) + '%')
print('Missing data in MasVnrType category is: ' + str(missingpercentmasvnrtype) + '%')
print('Missing data in MasVnrArea category is: ' + str(missingpercentmasvnrarea) + '%')
print('Missing data in BsmtQual category is: ' + str(missingpercentbsmtqual) + '%')
print('Missing data in BsmtCond category is: ' + str(missingpercentbsmtcond) + '%')
print('Missing data in BsmtExposure category is: ' + str(missingpercentbsmtexposure) + '%')
print('Missing data in BsmtFinType1 category is: ' + str(missingpercentbsmtfintype1) + '%')
print('Missing data in BsmtFinType2 category is: ' + str(missingpercentbsmtfintype2) + '%')
print('Missing data in Electrical category is: ' + str(missingpercentelectrical) + '%')
print('Missing data in FireplaceQu category is: ' + str(missingpercentfireplacequ) + '%')
print('Missing data in GarageType category is: ' + str(missingpercentgaragetype) + '%')
print('Missing data in GarageYrBlt category is: ' + str(missingpercentgarageyrblt) + '%')
print('Missing data in GarageFinish category is: ' + str(missingpercentgaragefinish) + '%')
print('Missing data in GarageQual category is: ' + str(missingpercentgaragequal) + '%')
print('Missing data in GarageCond category is: ' + str(missingpercentgaragecond) + '%')
print('Missing data in PoolQC category is: ' + str(missingpercentpoolqc) + '%')
print('Missing data in Fence category is: ' + str(missingpercentfence) + '%')
print('Missing data in MiscFeature category is: ' + str(missingpercentmiscfeature) + '%')
#Since there is around 17% missing data in lot frontage, using mean of available data to replace missing data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, [2]]) #Replace column 3 missing values
X[:, [2]] = imputer.transform(X[:, [2]])
#Since just a few data is missing in certain categories, replace the missing values with most occuring value
print(stats.mode(X[:, 24]))
for index, val in enumerate(X[:, 24]):
    if pd.isnull(X[index, 24]):
        X[index, 24] = 'None'
print(stats.mode(X[:, 25]))
for index, val in enumerate(X[:, 25]):
    if pd.isnull(X[index, 25]):
        X[index, 25] = 0
print(stats.mode(X[:, 29]))
for index, val in enumerate(X[:, 29]):
    if pd.isnull(X[index, 29]):
        X[index, 29] = 'TA'
print(stats.mode(X[:, 30]))
for index, val in enumerate(X[:, 30]):
    if pd.isnull(X[index, 30]):
        X[index, 30] = 'TA'
print(stats.mode(X[:, 31]))
for index, val in enumerate(X[:, 31]):
    if pd.isnull(X[index, 31]):
        X[index, 31] = 'No'
print(stats.mode(X[:, 32]))
for index, val in enumerate(X[:, 32]):
    if pd.isnull(X[index, 32]):
        X[index, 32] = 'Unf'
print(stats.mode(X[:, 34]))
for index, val in enumerate(X[:, 34]):
    if pd.isnull(X[index, 34]):
        X[index, 34] = 'Unf'
print(stats.mode(X[:, 41]))
for index, val in enumerate(X[:, 41]):
    if pd.isnull(X[index, 41]):
        X[index, 41] = 'SBrkr'
print(stats.mode(X[:, 57]))
for index, val in enumerate(X[:, 57]):
    if pd.isnull(X[index, 57]):
        X[index, 57] = 'Attchd'
print(stats.mode(X[:, 58]))
for index, val in enumerate(X[:, 58]):
    if pd.isnull(X[index, 58]):
        X[index, 58] = 2005
print(stats.mode(X[:, 59]))
for index, val in enumerate(X[:, 59]):
    if pd.isnull(X[index, 59]):
        X[index, 59] = 'Unf'
print(stats.mode(X[:, 62]))
for index, val in enumerate(X[:, 62]):
    if pd.isnull(X[index, 62]):
        X[index, 62] = 'TA'
print(stats.mode(X[:, 63]))
for index, val in enumerate(X[:, 63]):
    if pd.isnull(X[index, 63]):
        X[index, 63] = 'TA'
#Since a lot of data is missing in certain categories, deleting that column
X = np.delete(X, 5, 1) #Alley
X = np.delete(X, 55, 1) #Fireplace
X = np.delete(X, 69, 1) #PoolQC
X = np.delete(X, 69, 1) #Fence
X = np.delete(X, 69, 1) #MiscFeature

#Encoding categorical data
