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
from sklearn.linear_model import LinearRegression #Import class to implement linear regression
from sklearn.ensemble import RandomForestRegressor #Import class to implement random forest regression
from sklearn.svm import SVC #Import class to implement SVM
from sklearn.naive_bayes import GaussianNB #Import class to implement Naive Bayes
from sklearn.ensemble import RandomForestClassifier #Import class to implement Random Forest Classification
from scipy import stats #Import stats to help with data preprocessing
from sklearn.metrics import r2_score

#Importing the dataset
dataset = pd.read_csv('train.csv')
datasettest = pd.read_csv('test.csv')
#Ignoring the Id column since a simple numbering from 1 onwards is not helpful in predicting data
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
Xtest = datasettest.iloc[:, 1:].values

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
sns.barplot(x = dataset['GarageType'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, BuiltIn as 3, Attchd as 2, everything else 1
sns.barplot(x = dataset['GarageFinish'], y = dataset['SalePrice'], data = dataset)
#Based on this plot, Fin as 3, RFn as 2, Unf as 1
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
X = dataset.iloc[:, 1:-1].values
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
utilities = []
for data in X[:, 10]:
    if data == 'AllPub':
        utilities.append(1)
    else:
        utilities.append(0)
X[:, 10] = np.asarray(utilities)
neighborhood = []
for data in X[:, 19]:
    if data == 'NoRidge' or data == 'NridgHt' or data == 'StoneBr':
        neighborhood.append(1)
    else:
        neighborhood.append(0)
X[:, 19] = np.asarray(neighborhood)
condition1 = []
for data in X[:, 20]:
    if data == 'PosN' or data == 'RRNn' or data == 'PosA':
        condition1.append(3)
    elif data == 'Norm' or data == 'RRAn' or data == 'RRNe':
        condition1.append(2)
    else:
        condition1.append(1)
X[:, 20] = np.asarray(condition1)
condition2 = []
for data in X[:, 21]:
    if data == 'PosA' or data == 'PosN':
        condition2.append(1)
    else:
        condition2.append(0)
X[:, 21] = np.asarray(condition2)
bldgtype = []
for data in X[:, 22]:
    if data == '1Fam' or data == 'TwnhsE':
        bldgtype.append(1)
    else:
        bldgtype.append(0)
X[:, 22] = np.asarray(bldgtype)
housestyle = []
for data in X[:, 23]:
    if data == '2Story':
        housestyle.append(1)
    else:
        housestyle.append(0)
X[:, 23] = np.asarray(housestyle)
roofstyle = []
for data in X[:, 28]:
    if data == 'Shed':
        roofstyle.append(1)
    else:
        roofstyle.append(0)
X[:, 28] = np.asarray(roofstyle)
roofmatl = []
for data in X[:, 29]:
    if data == 'WdShake' or data == 'Membran':
        roofmatl.append(1)
    else:
        roofmatl.append(0)
X[:, 29] = np.asarray(roofmatl)
exterior1st = []
for data in X[:, 30]:
    if data == 'VinylSd' or data == 'CemntBd' or data == 'Stone' or data == 'ImStucc':
        exterior1st.append(1)
    else:
        exterior1st.append(0)
X[:, 30] = np.asarray(exterior1st)
exterior2nd = []
for data in X[:, 31]:
    if data == 'Other':
        exterior2nd.append(1)
    else:
        exterior2nd.append(0)
X[:, 31] = np.asarray(exterior2nd)
masvnrtype = []
for data in X[:, 32]:
    if data == 'Stone':
        masvnrtype.append(1)
    else:
        masvnrtype.append(0)
X[:, 32] = np.asarray(masvnrtype)
exterqual = []
for data in X[:, 34]:
    if data == 'Ex':
        exterqual.append(1)
    else:
        exterqual.append(0)
X[:, 34] = np.asarray(exterqual)
extercond = []
for data in X[:, 35]:
    if data == 'Po' or data == 'Fa':
        extercond.append(0)
    else:
        extercond.append(1)
X[:, 35] = np.asarray(extercond)
foundation = []
for data in X[:, 36]:
    if data == 'PConc' or data == 'Wood' or data == 'Stone':
        foundation.append(1)
    else:
        foundation.append(0)
X[:, 36] = np.asarray(foundation)
bsmtqual = []
for data in X[:, 37]:
    if data == 'Ex':
        bsmtqual.append(1)
    else:
        bsmtqual.append(0)
X[:, 37] = np.asarray(bsmtqual)
bsmtcond = []
for data in X[:, 38]:
    if data == 'TA' or data == 'Gd':
        bsmtcond.append(1)
    else:
        bsmtcond.append(0)
X[:, 38] = np.asarray(bsmtcond)
bsmtexposure = []
for data in X[:, 39]:
    if data == 'Gd':
        bsmtexposure.append(1)
    else:
        bsmtexposure.append(0)
X[:, 39] = np.asarray(bsmtexposure)
bsmtfintype1 = []
for data in X[:, 40]:
    if data == 'GLQ':
        bsmtfintype1.append(1)
    else:
        bsmtfintype1.append(0)
X[:, 40] = np.asarray(bsmtfintype1)
heating = []
for data in X[:, 51]:
    if data == 'GasA' or data == 'GasW':
        heating.append(1)
    else:
        heating.append(0)
X[:, 51] = np.asarray(heating)
heatingqc = []
for data in X[:, 52]:
    if data == 'Ex':
        heatingqc.append(1)
    else:
        heatingqc.append(0)
X[:, 52] = np.asarray(heatingqc)
centralair = []
for data in X[:, 53]:
    if data == 'Y':
        centralair.append(1)
    else:
        centralair.append(0)
X[:, 53] = np.asarray(centralair)
electrical = []
for data in X[:, 54]:
    if data == 'SBrkr':
        electrical.append(1)
    else:
        electrical.append(0)
X[:, 54] = np.asarray(electrical)
kitchenqual = []
for data in X[:, 65]:
    if data == 'Ex':
        kitchenqual.append(1)
    else:
        kitchenqual.append(0)
X[:, 65] = np.asarray(kitchenqual)
functional = []
for data in X[:, 67]:
    if data == 'Typ':
        functional.append(3)
    elif data == 'Maj2':
        functional.append(1)
    else:
        functional.append(2)
X[:, 67] = np.asarray(functional)
garagetype = []
for data in X[:, 69]:
    if data == 'BuiltIn':
        garagetype.append(3)
    elif data == 'Attchd':
        garagetype.append(2)
    else:
        garagetype.append(1)
X[:, 69] = np.asarray(garagetype)
garagefinish = []
for data in X[:, 71]:
    if data == 'Fin':
        garagefinish.append(3)
    elif data == 'RFn':
        garagefinish.append(2)
    else:
        garagefinish.append(1)
X[:, 71] = np.asarray(garagefinish)
garagequal = []
for data in X[:, 74]:
    if data == 'Gd':
        garagequal.append(3)
    elif data == 'TA':
        garagequal.append(2)
    else:
        garagequal.append(1)
X[:, 74] = np.asarray(garagequal)
garagecond = []
for data in X[:, 75]:
    if data == 'Gd' or data == 'TA':
        garagecond.append(1)
    else:
        garagecond.append(0)
X[:, 75] = np.asarray(garagecond)
paveddrive = []
for data in X[:, 76]:
    if data == 'Y':
        paveddrive.append(1)
    else:
        paveddrive.append(0)
X[:, 76] = np.asarray(paveddrive)
saletype = []
for data in X[:, 86]:
    if data == 'Con' or data == 'New':
        saletype.append(3)
    elif data == 'ConLI' or data == 'CWD':
        saletype.append(2)
    else:
        saletype.append(1)
X[:, 86] = np.asarray(saletype)
salecond = []
for data in X[:, 87]:
    if data == 'Abnorml' or data == 'Partial':
        salecond.append(1)
    else:
        salecond.append(0)
X[:, 87] = np.asarray(salecond)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [11])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [16])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [42])], remainder = 'passthrough', sparse_threshold = 0)
X = np.array(ct.fit_transform(X))
#Verifying if we have any missing data still
print(np.isnan(np.min(X)))
#The answer is no

#Now making sure our test data is also in a proper format by following similar approaches
missing_data = datasettest.isnull().sum() #Identify amount of missing data
print(missing_data)
#Handling missing data in test.csv the same way
print(stats.mode(Xtest[:, 1]))
for index, val in enumerate(Xtest[:, 1]):
    if pd.isnull(Xtest[index, 1]):
        Xtest[index, 1] = 'RL'
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [2]])
Xtest[:, [2]] = imputer.transform(Xtest[:, [2]])
print(stats.mode(Xtest[:, 8]))
for index, val in enumerate(Xtest[:, 8]):
    if pd.isnull(Xtest[index, 8]):
        Xtest[index, 8] = 'AllPub'
print(stats.mode(Xtest[:, 22]))
for index, val in enumerate(Xtest[:, 22]):
    if pd.isnull(Xtest[index, 22]):
        Xtest[index, 22] = 'VinylSd'
print(stats.mode(Xtest[:, 23]))
for index, val in enumerate(Xtest[:, 23]):
    if pd.isnull(Xtest[index, 23]):
        Xtest[index, 23] = 'VinylSd'
print(stats.mode(Xtest[:, 24]))
for index, val in enumerate(Xtest[:, 24]):
    if pd.isnull(Xtest[index, 24]):
        Xtest[index, 24] = 'None'
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [25]])
Xtest[:, [25]] = imputer.transform(Xtest[:, [25]])
print(stats.mode(Xtest[:, 29]))
for index, val in enumerate(Xtest[:, 29]):
    if pd.isnull(Xtest[index, 29]):
        Xtest[index, 29] = 'TA'
print(stats.mode(Xtest[:, 30]))
for index, val in enumerate(Xtest[:, 30]):
    if pd.isnull(Xtest[index, 30]):
        Xtest[index, 30] = 'TA'
print(stats.mode(Xtest[:, 31]))
for index, val in enumerate(Xtest[:, 31]):
    if pd.isnull(Xtest[index, 31]):
        Xtest[index, 31] = 'No'
print(stats.mode(Xtest[:, 32]))
for index, val in enumerate(Xtest[:, 32]):
    if pd.isnull(Xtest[index, 32]):
        Xtest[index, 32] = 'GLQ'
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [33]])
Xtest[:, [33]] = imputer.transform(Xtest[:, [33]])
print(stats.mode(Xtest[:, 34]))
for index, val in enumerate(Xtest[:, 34]):
    if pd.isnull(Xtest[index, 34]):
        Xtest[index, 34] = 'Unf'
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [35]])
Xtest[:, [35]] = imputer.transform(Xtest[:, [35]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [36]])
Xtest[:, [36]] = imputer.transform(Xtest[:, [36]])
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [37]])
Xtest[:, [37]] = imputer.transform(Xtest[:, [37]])
print(stats.mode(Xtest[:, 46]))
for index, val in enumerate(Xtest[:, 46]):
    if pd.isnull(Xtest[index, 46]):
        Xtest[index, 46] = 0
print(stats.mode(Xtest[:, 47]))
for index, val in enumerate(Xtest[:, 47]):
    if pd.isnull(Xtest[index, 47]):
        Xtest[index, 47] = 0
print(stats.mode(Xtest[:, 52]))
for index, val in enumerate(Xtest[:, 52]):
    if pd.isnull(Xtest[index, 52]):
        Xtest[index, 52] = 'TA'
print(stats.mode(Xtest[:, 54]))
for index, val in enumerate(Xtest[:, 54]):
    if pd.isnull(Xtest[index, 54]):
        Xtest[index, 54] = 'Typ'
print(stats.mode(Xtest[:, 57]))
for index, val in enumerate(Xtest[:, 57]):
    if pd.isnull(Xtest[index, 57]):
        Xtest[index, 57] = 'Attchd'
print(stats.mode(Xtest[:, 58]))
for index, val in enumerate(Xtest[:, 58]):
    if pd.isnull(Xtest[index, 58]):
        Xtest[index, 58] = 2005
print(stats.mode(Xtest[:, 59]))
for index, val in enumerate(Xtest[:, 59]):
    if pd.isnull(Xtest[index, 59]):
        Xtest[index, 59] = 'Unf'
print(stats.mode(Xtest[:, 60]))
for index, val in enumerate(Xtest[:, 60]):
    if pd.isnull(Xtest[index, 60]):
        Xtest[index, 60] = 2
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(Xtest[:, [61]])
Xtest[:, [61]] = imputer.transform(Xtest[:, [61]])
print(stats.mode(Xtest[:, 62]))
for index, val in enumerate(Xtest[:, 62]):
    if pd.isnull(Xtest[index, 62]):
        Xtest[index, 62] = 'TA'
print(stats.mode(Xtest[:, 63]))
for index, val in enumerate(Xtest[:, 63]):
    if pd.isnull(Xtest[index, 63]):
        Xtest[index, 63] = 'TA'
print(stats.mode(Xtest[:, 77]))
for index, val in enumerate(Xtest[:, 77]):
    if pd.isnull(Xtest[index, 77]):
        Xtest[index, 77] = 'WD'
Xtest = np.delete(Xtest, 5, 1)
Xtest = np.delete(Xtest, 55, 1)
Xtest = np.delete(Xtest, 69, 1)
Xtest = np.delete(Xtest, 69, 1)
Xtest = np.delete(Xtest, 69, 1)

#Encoding categorical data
#Encode data based on what we identified in exploratory data analysis
mszoning = []
for data in Xtest[:, 1]:
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
Xtest[:, 1] = np.asarray(mszoning)
street = []
for data in Xtest[:, 4]:
    if data == 'Pave':
        street.append(1)
    else:
        street.append(0)
Xtest[:, 4] = np.asarray(street)
lotshape = []
for data in Xtest[:, 5]:
    if data == 'Reg':
        lotshape.append(0)
    else:
        lotshape.append(1)
Xtest[:, 5] = np.asarray(lotshape)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6])], remainder = 'passthrough', sparse_threshold = 0)
Xtest = np.array(ct.fit_transform(Xtest))
utilities = []
for data in Xtest[:, 10]:
    if data == 'AllPub':
        utilities.append(1)
    else:
        utilities.append(0)
Xtest[:, 10] = np.asarray(utilities)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [11])], remainder = 'passthrough', sparse_threshold = 0)
Xtest = np.array(ct.fit_transform(Xtest))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [16])], remainder = 'passthrough', sparse_threshold = 0)
Xtest = np.array(ct.fit_transform(Xtest))
neighborhood = []
for data in Xtest[:, 19]:
    if data == 'NoRidge' or data == 'NridgHt' or data == 'StoneBr':
        neighborhood.append(1)
    else:
        neighborhood.append(0)
Xtest[:, 19] = np.asarray(neighborhood)
condition1 = []
for data in Xtest[:, 20]:
    if data == 'PosN' or data == 'RRNn' or data == 'PosA':
        condition1.append(3)
    elif data == 'Norm' or data == 'RRAn' or data == 'RRNe':
        condition1.append(2)
    else:
        condition1.append(1)
Xtest[:, 20] = np.asarray(condition1)
condition2 = []
for data in Xtest[:, 21]:
    if data == 'PosA' or data == 'PosN':
        condition2.append(1)
    else:
        condition2.append(0)
Xtest[:, 21] = np.asarray(condition2)
bldgtype = []
for data in Xtest[:, 22]:
    if data == '1Fam' or data == 'TwnhsE':
        bldgtype.append(1)
    else:
        bldgtype.append(0)
Xtest[:, 22] = np.asarray(bldgtype)
housestyle = []
for data in Xtest[:, 23]:
    if data == '2Story':
        housestyle.append(1)
    else:
        housestyle.append(0)
Xtest[:, 23] = np.asarray(housestyle)
roofstyle = []
for data in Xtest[:, 28]:
    if data == 'Shed':
        roofstyle.append(1)
    else:
        roofstyle.append(0)
Xtest[:, 28] = np.asarray(roofstyle)
roofmatl = []
for data in Xtest[:, 29]:
    if data == 'WdShake' or data == 'Membran':
        roofmatl.append(1)
    else:
        roofmatl.append(0)
Xtest[:, 29] = np.asarray(roofmatl)
exterior1st = []
for data in Xtest[:, 30]:
    if data == 'VinylSd' or data == 'CemntBd' or data == 'Stone' or data == 'ImStucc':
        exterior1st.append(1)
    else:
        exterior1st.append(0)
Xtest[:, 30] = np.asarray(exterior1st)
#Too many features. Reached my limit. Using OHE from here on out
exterior2nd = []
for data in Xtest[:, 31]:
    if data == 'Other':
        exterior2nd.append(1)
    else:
        exterior2nd.append(0)
Xtest[:, 31] = np.asarray(exterior2nd)
masvnrtype = []
for data in Xtest[:, 32]:
    if data == 'Stone':
        masvnrtype.append(1)
    else:
        masvnrtype.append(0)
Xtest[:, 32] = np.asarray(masvnrtype)
exterqual = []
for data in Xtest[:, 34]:
    if data == 'Ex':
        exterqual.append(1)
    else:
        exterqual.append(0)
Xtest[:, 34] = np.asarray(exterqual)
extercond = []
for data in Xtest[:, 35]:
    if data == 'Po' or data == 'Fa':
        extercond.append(0)
    else:
        extercond.append(1)
Xtest[:, 35] = np.asarray(extercond)
foundation = []
for data in Xtest[:, 36]:
    if data == 'PConc' or data == 'Wood' or data == 'Stone':
        foundation.append(1)
    else:
        foundation.append(0)
Xtest[:, 36] = np.asarray(foundation)
bsmtqual = []
for data in Xtest[:, 37]:
    if data == 'Ex':
        bsmtqual.append(1)
    else:
        bsmtqual.append(0)
Xtest[:, 37] = np.asarray(bsmtqual)
bsmtcond = []
for data in Xtest[:, 38]:
    if data == 'TA' or data == 'Gd':
        bsmtcond.append(1)
    else:
        bsmtcond.append(0)
Xtest[:, 38] = np.asarray(bsmtcond)
bsmtexposure = []
for data in Xtest[:, 39]:
    if data == 'Gd':
        bsmtexposure.append(1)
    else:
        bsmtexposure.append(0)
Xtest[:, 39] = np.asarray(bsmtexposure)
bsmtfintype1 = []
for data in Xtest[:, 40]:
    if data == 'GLQ':
        bsmtfintype1.append(1)
    else:
        bsmtfintype1.append(0)
Xtest[:, 40] = np.asarray(bsmtfintype1)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [42])], remainder = 'passthrough', sparse_threshold = 0)
Xtest = np.array(ct.fit_transform(Xtest))
heating = []
for data in Xtest[:, 51]:
    if data == 'GasA' or data == 'GasW':
        heating.append(1)
    else:
        heating.append(0)
Xtest[:, 51] = np.asarray(heating)
heatingqc = []
for data in Xtest[:, 52]:
    if data == 'Ex':
        heatingqc.append(1)
    else:
        heatingqc.append(0)
Xtest[:, 52] = np.asarray(heatingqc)
centralair = []
for data in Xtest[:, 53]:
    if data == 'Y':
        centralair.append(1)
    else:
        centralair.append(0)
Xtest[:, 53] = np.asarray(centralair)
electrical = []
for data in Xtest[:, 54]:
    if data == 'SBrkr':
        electrical.append(1)
    else:
        electrical.append(0)
Xtest[:, 54] = np.asarray(electrical)
kitchenqual = []
for data in Xtest[:, 65]:
    if data == 'Ex':
        kitchenqual.append(1)
    else:
        kitchenqual.append(0)
Xtest[:, 65] = np.asarray(kitchenqual)
functional = []
for data in Xtest[:, 67]:
    if data == 'Typ':
        functional.append(3)
    elif data == 'Maj2':
        functional.append(1)
    else:
        functional.append(2)
Xtest[:, 67] = np.asarray(functional)
garagetype = []
for data in Xtest[:, 69]:
    if data == 'BuiltIn':
        garagetype.append(3)
    elif data == 'Attchd':
        garagetype.append(2)
    else:
        garagetype.append(1)
Xtest[:, 69] = np.asarray(garagetype)
garagefinish = []
for data in Xtest[:, 71]:
    if data == 'Fin':
        garagefinish.append(3)
    elif data == 'RFn':
        garagefinish.append(2)
    else:
        garagefinish.append(1)
Xtest[:, 71] = np.asarray(garagefinish)
garagequal = []
for data in Xtest[:, 74]:
    if data == 'Gd':
        garagequal.append(3)
    elif data == 'TA':
        garagequal.append(2)
    else:
        garagequal.append(1)
Xtest[:, 74] = np.asarray(garagequal)
garagecond = []
for data in Xtest[:, 75]:
    if data == 'Gd' or data == 'TA':
        garagecond.append(1)
    else:
        garagecond.append(0)
Xtest[:, 75] = np.asarray(garagecond)
paveddrive = []
for data in Xtest[:, 76]:
    if data == 'Y':
        paveddrive.append(1)
    else:
        paveddrive.append(0)
Xtest[:, 76] = np.asarray(paveddrive)
saletype = []
for data in Xtest[:, 86]:
    if data == 'Con' or data == 'New':
        saletype.append(3)
    elif data == 'ConLI' or data == 'CWD':
        saletype.append(2)
    else:
        saletype.append(1)
Xtest[:, 86] = np.asarray(saletype)
salecond = []
for data in Xtest[:, 87]:
    if data == 'Abnorml' or data == 'Partial':
        salecond.append(1)
    else:
        salecond.append(0)
Xtest[:, 87] = np.asarray(salecond)
#Verifying if we have any missing data still
print(np.isnan(np.min(Xtest)))
#The answer is no

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

# Evaluating the Model Performance
r2scorelinreg = r2_score(y_test, y_pred)

regressor = RandomForestRegressor(n_estimators = 500)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2scorerf = r2_score(y_test, y_pred)

#Since linear regression gives the highest accuracy, we will use it to predict results in the test.csv file

#Training the linear regression model on the whole dataset now to maximize learning
regressorlinreg= LinearRegression()
regressorlinreg.fit(X, y)

#Applying Random Forest
y_predfinal = regressorlinreg.predict(Xtest)

#Convert to csv
df = pd.DataFrame({"Id" : y_predfinal, "SalePrice" : y_predfinal})
df.to_csv("HousePricesSubmission.csv", index=False)
