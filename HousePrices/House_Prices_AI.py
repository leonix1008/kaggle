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
#Based on this plot, Pave as 1 and Grvl as 0
sns.barplot(x = dataset['LotShape'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Reg as 0, IR3 as 1, IR1 as 1 and IR2 as 1
sns.barplot(x = dataset['LandContour'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, one-hot encoding
sns.barplot(x = dataset['Utilities'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, AllPub as 1, NoSeWa as 0
sns.barplot(x = dataset['LotConfig'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, one-hot encoding
sns.barplot(x = dataset['LandSlope'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, one-hot encoding
sns.barplot(x = dataset['Neighborhood'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, NoRidge, NridgHt, StoneBr as 1, everything else 0
sns.barplot(x = dataset['Condition1'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, PosN, RRNn, PosA as 3, Norm, RRAn, RRNe as 2, Feedr, Artery, RRAe as 1
sns.barplot(x = dataset['Condition2'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, PosA, PosN as 1, everything else 0
sns.barplot(x = dataset['BldgType'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, 1Fam, TwnhsE as 1, everything else 0
sns.barplot(x = dataset['HouseStyle'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, 2Story as 1, everythinig else 0
sns.barplot(x = dataset['RoofStyle'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Shed as 1, everything else 0
sns.barplot(x = dataset['RoofMatl'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, WdShake, Membran as 1, everything else 0
sns.barplot(x = dataset['Exterior1st'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, VinylSd, CemntBd, Stone, ImStucc as 1, everything else 0
sns.barplot(x = dataset['Exterior2nd'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Other as 1, everything else 0
sns.barplot(x = dataset['MasVnrType'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Stone as 1, everything else 0
sns.barplot(x = dataset['ExterQual'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Ex as 1, everything else 0
sns.barplot(x = dataset['ExterCond'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Po, Fa as 0, everything else 1
sns.barplot(x = dataset['Foundation'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, PConc, Wood, Stone as 1, everything else 0
sns.barplot(x = dataset['BsmtQual'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Ex as 1, everything else 0
sns.barplot(x = dataset['BsmtCond'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, TA, Gd as 1, everything else 0
sns.barplot(x = dataset['BsmtExposure'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Gd as 1, everything else 0
sns.barplot(x = dataset['BsmtFinType1'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, GLQ as 1, everything else 0
sns.barplot(x = dataset['BsmtFinType2'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, one-hot encoding
sns.barplot(x = dataset['Heating'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, GasA, GasW as 1, everything else 0
sns.barplot(x = dataset['HeatingQC'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Ex as 1, everything else 0
sns.barplot(x = dataset['CentralAir'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Y as 1, N as 0
sns.barplot(x = dataset['Electrical'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, SBrkr as 1, everything else 0
sns.barplot(x = dataset['KitchenQual'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Ex as 1, everything else 0
sns.barplot(x = dataset['Functional'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Typ as 3, Maj2 as 1, everything else 2
sns.barplot(x = dataset['FireplaceQu'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Ex as 1, everything else 0
sns.barplot(x = dataset['GarageType'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, BuiltIn as 3, Attchd as 2, everything else 1
sns.barplot(x = dataset['GarageFinish'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Fin as 3, RFn as 2, Unf as 0
sns.barplot(x = dataset['GarageQual'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Gd as 3, TA as 2, everything else 1
sns.barplot(x = dataset['GarageCond'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, TA, Gd as 1, everything else 0
sns.barplot(x = dataset['PavedDrive'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Y as 1, everything else 0
sns.barplot(x = dataset['SaleType'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, New, Con as 3, ConLI, CWD as 2, everything else 1
sns.barplot(x = dataset['SaleCondition'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Abnorml, Partial as 1, everything else 0
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
#Encode data based on what we identified in exploratory data analysis
mszoning = []
for data in X[:, 1]:
    if data == 'FV':
        mszoning.append(5)
    elif data == 'RL':
        mszoning.append(4)
    elif data == 'RH':
        mszoning.append(3)
    elif data == 'RM':
        mszoning.append(2)
    else:
        mszoning.append(1)
X[:, 1] = np.asarray(mszoning)
street = []
for data in X[:, 4]:
    if data == 'Pave':
        street.append(1)
    else:
        street.append(0)
X[:, 4] = np.asarray(street)
lotshape = []
for data in X[:, 5]:
    if data == 'Reg':
        lotshape.append(0)
    else:
        lotshape.append(1)
X[:, 5] = np.asarray(lotshape)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
utilities = []
for data in X[:, 7]:
    if data == 'AllPub':
        utilities.append(1)
    else:
        utilities.append(0)
X[:, 7] = np.asarray(utilities)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [8])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [9])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
neighborhood = []
for data in X[:, 10]:
    if data == 'NoRidge' or data == 'NridgHt' or data == 'StoneBr':
        neighborhood.append(1)
    else:
        neighborhood.append(0)
X[:, 10] = np.asarray(neighborhood)
condition1 = []
for data in X[:, 11]:
    if data == 'PosN' or data == 'RRNn' or data == 'PosA':
        condition1.append(3)
    elif data == 'Norm' or data == 'RRAn' or data == 'RRNe':
        condition1.append(2)
    else:
        condition1.append(1)
X[:, 11] = np.asarray(condition1)
condition2 = []
for data in X[:, 12]:
    if data == 'PosA' or data == 'PosN':
        condition2.append(1)
    else:
        condition2.append(0)
X[:, 12] = np.asarray(condition2)
bldgtype = []
for data in X[:, 13]:
    if data == '1Fam' or data == 'TwnhsE':
        bldgtype.append(1)
    else:
        bldgtype.append(0)
X[:, 13] = np.asarray(bldgtype)
housestyle = []
for data in X[:, 14]:
    if data == '2Story':
        housestyle.append(1)
    else:
        housestyle.append(0)
X[:, 14] = np.asarray(housestyle)
