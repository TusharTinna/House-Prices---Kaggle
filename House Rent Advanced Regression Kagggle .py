#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# In[ ]:


train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")


# In[ ]:


train.shape,test.shape


# In[ ]:


testid=test['Id']


# In[ ]:


null=train.isnull().sum().sort_values(ascending=False)
null[null>0]


# In[ ]:


# Percentage of missing values in train data
total_rows=1460
miss_per=pd.DataFrame(null[null>0])
miss_per.columns=['Missing']
miss_per['Missing%']=((miss_per['Missing']/total_rows)*100).round(2)
miss_per


# In[ ]:


# Null values in test dataset
null2=test.isnull().sum().sort_values(ascending=False)
null2[null2>0]


# In[ ]:


# Percentage of missing values in Test Data
total_rows=1460
miss_per=pd.DataFrame(null2[null2>0])
miss_per.columns=['Missing']
miss_per['Missing%']=((miss_per['Missing']/total_rows)*100).round(2)
miss_per


# In[ ]:


# dropping columns which has around 50 percent missing values
train.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],inplace=True)
test.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],inplace=True)


# ### inputing missing values

# In[ ]:


# LotFrontage


# In[ ]:


#Let's see what the 'LotFrontage' variable looks like in general.
mean=train["LotFrontage"].mean()
median=train["LotFrontage"].median()
a = train["LotFrontage"].hist(bins=100,density=True  )
train["LotFrontage"].plot(kind='density')
a.set(xlabel='LotFrontage')
plt.axvline(x=mean, color='r', linestyle='-')
plt.axvline(x=median, color='y', linestyle='-')
plt.show()


# In[ ]:


print(median)
print(mean)


# In[ ]:


#Let's see what the 'LotFrontage' variable looks like in general.
meant=test["LotFrontage"].mean()
mediant=test["LotFrontage"].median()
a = test["LotFrontage"].hist(bins=100,density=True  )
test["LotFrontage"].plot(kind='density')
a.set(xlabel='LotFrontage')
plt.axvline(x=mean, color='r', linestyle='-')
plt.axvline(x=median, color='y', linestyle='-')
plt.show()


# In[ ]:


print(mediant)
print(meant)


# In[ ]:


## as median is closer to peak i fill missing values with the median


# In[ ]:


train['LotFrontage'].fillna(69,inplace=True)
test['LotFrontage'].fillna(67,inplace=True)


# In[ ]:


# GarageType


# In[ ]:


train['GarageType'].value_counts()


# In[ ]:


# I will fill it with forward fill method
train['GarageType'].fillna(method='ffill',inplace=True)
test['GarageType'].fillna(method='ffill',inplace=True)


# In[ ]:


# GarageCond
train['GarageCond'].describe()


# In[ ]:


# In data, Most frequent value is TA, count of all other values is negligible in comparison to TA
# Its probably the best choice to choose to fill missing values in this GarageCond column.

train['GarageCond'].fillna('TA',inplace=True)
test['GarageCond'].fillna('TA',inplace=True)


# In[ ]:


# GarageYrBlt


# In[ ]:


# Plotting univariate plot for train 
plt.figure(figsize=(11,3))
plt.subplot(1,2,1)
sns.distplot(train['GarageYrBlt'])
plt.ylabel('Count')


# Data is scattered vastly, it is not normally distributed we can't use mean value to fill missing records. We need to look at the data more closely.

# In[ ]:


# Chekcing which column has most correlation with GarageYrBlt column
train.corr()['GarageYrBlt'].nlargest(3)


# In[ ]:


# we can see that GarageYrBlt is mostly related to YearBuilt column, lets print these two columns separately
train[['GarageYrBlt','YearBuilt']][0:50]


# In[ ]:


# we can see these columns are exactly same so we drop GarageYrBlt as it has more null values
train.drop('GarageYrBlt',axis=1,inplace=True)
# repeat the same operation on test data
test.drop('GarageYrBlt',axis=1,inplace=True)


# In[ ]:


# GarageFinish
train['GarageFinish'].value_counts(ascending=False)


# In[ ]:


test['GarageFinish'].value_counts(ascending=False)


# In[ ]:


train['GarageFinish'].isnull().sum(), test['GarageFinish'].isnull().sum()


# In[ ]:


# Filling 51 values with 'Unf' and 30 values using 'RFn'
train['GarageFinish'].fillna('Unf',limit=51,inplace=True)
train['GarageFinish'].fillna('Rfn',limit=30,inplace=True)

# Filling 50 values with 'Unf' and 38 values using 'RFn'
test['GarageFinish'].fillna('Unf',limit=50,inplace=True)
test['GarageFinish'].fillna('RFn',limit=38,inplace=True)


# In[ ]:


# GarageQual
train['GarageQual'].describe()


# In[ ]:


# In data, Most frequent value is TA, count of all other values is negligible in comparison to TA
# Its probably the best choice to choose to fill missing values in this GarageCond column.

train['GarageQual'].fillna('TA',inplace=True)
test['GarageQual'].fillna('TA',inplace=True)


# In[ ]:


# BsmtFinType2
train['BsmtFinType2'].describe()


# In[ ]:


train['BsmtFinType2'].fillna('Unf',inplace=True)
test['BsmtFinType2'].fillna('Unf',inplace=True)


# In[ ]:


# BsmtExposure
train['BsmtExposure'].describe()


# In[ ]:


# Filling with most frequent value 'No'
train['BsmtExposure'].fillna('No',inplace=True)
test['BsmtExposure'].fillna('No',inplace=True)


# In[ ]:


# BsmtFinType1
train['BsmtFinType1'].value_counts(ascending=False)


# In[ ]:


test['BsmtFinType1'].value_counts(ascending=False)


# In[ ]:


train['BsmtFinType1'].isnull().sum(),   test['BsmtFinType1'].isnull().sum()


# In[ ]:


train['BsmtFinType1'].fillna('GLQ',limit=19,inplace=True)
train['BsmtFinType1'].fillna('Unf',limit=18,inplace=True)

test['BsmtFinType1'].fillna('GLQ',limit=22,inplace=True)
test['BsmtFinType1'].fillna('Unf',limit=20,inplace=True)


# In[ ]:


# BsmtCond
train['BsmtCond'].describe()


# In[ ]:


train['BsmtCond'].fillna('TA',inplace=True)
test['BsmtCond'].fillna('TA',inplace=True)  


# In[ ]:


# BsmtQual
train['BsmtQual'].describe()


# In[ ]:


plt.figure(figsize=(11,3))
plt.subplot(1,2,1)
sns.countplot(train['BsmtQual'])
plt.ylabel('COUNT')

plt.subplot(1,2,2)
sns.countplot(test['BsmtQual'])
plt.ylabel('COUNT')
plt.show()


# In[ ]:


train['BsmtQual'].isnull().sum(), test['BsmtQual'].isnull().sum()


# In[ ]:


train['BsmtQual'].fillna('TA',limit=20,inplace=True)
train['BsmtQual'].fillna('Gd',limit=17,inplace=True)

test['BsmtQual'].fillna('TA',limit=24,inplace=True)
test['BsmtQual'].fillna('Gd',limit=20,inplace=True)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


# MasVnrArea
train['MasVnrArea'].describe()


# In[ ]:


plt.figure(figsize=(13,4))
plt.subplot(1,2,1)
sns.distplot(train['MasVnrArea'])
plt.title('Train Distribution')
plt.ylabel('COUNT')


# In[ ]:


#Most value is zero, and since this column has just 8 missed record,we will fill all values with 0, As distribution is not like Gaussion distribution, we can't use mean to fill missing values
# filling with 0(zero)
train['MasVnrArea'].fillna(0.0,inplace=True)
test['MasVnrArea'].fillna(0.0,inplace=True)


# In[ ]:


# MasVnrType

train['MasVnrType'].value_counts()


# In[ ]:


train['MasVnrType'].fillna('None',inplace=True)
test['MasVnrType'].fillna('None',inplace=True)


# In[ ]:


# checking train data again for missed values
train.isnull().sum().sum()


# In[ ]:


# checking train data again for missed values
test.isnull().sum().sum()


# In[ ]:


# we have very few missing values so i am removing the rows with null values


# In[ ]:


print('The number of samples into the train data is {}.'.format(train.shape[0]))


# In[ ]:


train = train.dropna(subset=['Electrical'])


# In[ ]:


print('The number of samples into the train data is {}.'.format(train.shape[0]))


# In[ ]:


print('The number of samples into the test data is {}.'.format(test.shape[0]))


# WE Still HAVE A VERY FEW MISSING VALUES IN TEST DATASET, WE CAN SIMPLY DROP THESE ROWS. But since submission requires all rows for prediction, we will try to fill these values also

# In[ ]:


# Printing columns with null values greater than zero
null=test.isnull().sum().sort_values(ascending=False)
null[null>0]


# In[ ]:


# filling MSZoing column
test['MSZoning'].describe()


# In[ ]:


# filling with most frequent value 'RL'
test['MSZoning'].fillna('RL',inplace=True)


# In[ ]:


# BsmtFullBath
test['BsmtFullBath'].describe()


# In[ ]:


#It is better to fill this column with mean value, as std is not very wide so missing values must be lying near to mean values
mean=test['BsmtFullBath'].mean()
test['BsmtFullBath'].fillna(mean,inplace=True)


# In[ ]:


# Functional
test['Functional'].describe()


# In[ ]:


# filling with most frequent values
test['Functional'].fillna('top',inplace=True)


# In[ ]:


# BsmtHalfBath
test['BsmtHalfBath'].describe()


# In[ ]:


mean=test['BsmtHalfBath'].mean()
test['BsmtHalfBath'].fillna(mean,inplace=True)


# In[ ]:


# Utilities
test['Utilities'].describe()


# In[ ]:


test['Utilities'].fillna('AllPub',inplace=True)


# In[ ]:


# BsmtFinSF1
test['BsmtFinSF1'].describe()


# In[ ]:


mean=test['BsmtFinSF1'].mean()
test['BsmtFinSF1'].fillna(mean,inplace=True)


# In[ ]:


# KitchenQual
test['KitchenQual'].describe()


# In[ ]:



test['KitchenQual'].fillna('TA',inplace=True)


# In[ ]:


# GarageCars
test['GarageCars'].describe()


# In[ ]:


mean=test['GarageCars'].mean()
test['GarageCars'].fillna(mean,inplace=True)


# In[ ]:


# GarageArea
test['GarageArea'].describe()


# In[ ]:


mean=test['GarageArea'].mean()
test['GarageArea'].fillna(mean,inplace=True)


# In[ ]:


# Exterior2nd
test['Exterior2nd'].describe()


# In[ ]:


test['Exterior2nd'].fillna('VinylSd',inplace=True)


# In[ ]:


# TotalBsmtSF
test['TotalBsmtSF'].describe()


# In[ ]:



mean=test['TotalBsmtSF'].mean()
test['TotalBsmtSF'].fillna(mean,inplace=True)


# In[ ]:


# BsmtFinSF2
test['BsmtFinSF2'].describe()


# In[ ]:


# lets plot univariate plot for this column
sns.distplot(test['BsmtFinSF2']) 


# In[ ]:


test['BsmtFinSF2'].fillna(0.0,inplace=True)


# In[ ]:


# BsmtUnfSF
test['BsmtUnfSF'].describe()


# In[ ]:


# lets plot univariate plot for this column
sns.distplot(test['BsmtUnfSF'])


# In[ ]:


mean=test['BsmtUnfSF'].mean()
test['BsmtUnfSF'].fillna(mean,inplace=True)


# In[ ]:


# SaleType
test['SaleType'].describe()


# In[ ]:


test['SaleType'].fillna('WD',inplace=True)


# In[ ]:


# Exterior1st
test['Exterior1st'].describe()


# In[ ]:


test['Exterior1st'].fillna('VinylSd',inplace=True)


# In[ ]:


test.isnull().sum().sum()


# In[ ]:


# checking shape of train and test dataset
train.shape, test.shape


# In[ ]:


# Selecting depending and indepent variable
x=train.drop('SalePrice',axis=1)
y=train['SalePrice']

# Now we will drop this 'Id' column from both of dataset as it doesn't possess much meaning
x.drop('Id',axis=1,inplace=True)
test.drop('Id',axis=1,inplace=True)

x.shape,test.shape


# In[ ]:


# PREPROCESSING
from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
ss=StandardScaler()


# In[ ]:


# selectring categorical features for label encoding
cat_train= x.select_dtypes(include='object')
cat_train.columns


# In[ ]:


# Label encoding of input features of train data
x['MSZoning']=le.fit_transform(x['MSZoning'])
x['Street']=le.fit_transform(x['Street'])
x['LotShape']=le.fit_transform(x['LotShape'])
x['LandContour']=le.fit_transform(x['LandContour'])
x['Utilities']=le.fit_transform(x['Utilities'])
x['LotConfig']=le.fit_transform(x['LotConfig'])
x['LandSlope']=le.fit_transform(x['LandSlope'])
x['Neighborhood']=le.fit_transform(x['Neighborhood'])
x['Condition1']=le.fit_transform(x['Condition1'])
x['Condition2']=le.fit_transform(x['Condition2'])
x['BldgType']=le.fit_transform(x['BldgType'])
x['HouseStyle']=le.fit_transform(x['HouseStyle'])
x['RoofStyle']=le.fit_transform(x['RoofStyle'])
x['RoofMatl']=le.fit_transform(x['RoofMatl'])
x['Exterior1st']=le.fit_transform(x['Exterior1st'])
x['Exterior2nd']=le.fit_transform(x['Exterior2nd'])
x['MasVnrType']=le.fit_transform(x['MasVnrType'])
x['ExterQual']=le.fit_transform(x['ExterQual'])
x['ExterCond']=le.fit_transform(x['ExterCond'])
x['Foundation']=le.fit_transform(x['Foundation'])
x['BsmtQual']=le.fit_transform(x['BsmtQual'])
x['BsmtCond']=le.fit_transform(x['BsmtCond'])
x['BsmtExposure']=le.fit_transform(x['BsmtExposure'])
x['BsmtFinType1']=le.fit_transform(x['BsmtFinType1'])
x['BsmtFinType2']=le.fit_transform(x['BsmtFinType2'])
x['Heating']=le.fit_transform(x['Heating'])
x['HeatingQC']=le.fit_transform(x['HeatingQC'])
x['CentralAir']=le.fit_transform(x['CentralAir'])
x['Electrical']=le.fit_transform(x['Electrical'])
x['KitchenQual']=le.fit_transform(x['KitchenQual'])
x['Functional']=le.fit_transform(x['Functional'])
x['GarageType']=le.fit_transform(x['GarageType'])
x['GarageFinish']=le.fit_transform(x['GarageFinish'])
x['GarageQual']=le.fit_transform(x['GarageQual'])
x['GarageCond']=le.fit_transform(x['GarageCond'])
x['PavedDrive']=le.fit_transform(x['PavedDrive'])
x['SaleType']=le.fit_transform(x['SaleType'])
x['SaleCondition']=le.fit_transform(x['SaleCondition'])

# Label encoding of Test Dataset
test['MSZoning']=le.fit_transform(test['MSZoning'])
test['Street']=le.fit_transform(test['Street'])
test['LotShape']=le.fit_transform(test['LotShape'])
test['LandContour']=le.fit_transform(test['LandContour'])
test['Utilities']=le.fit_transform(test['Utilities'])
test['LotConfig']=le.fit_transform(test['LotConfig'])
test['LandSlope']=le.fit_transform(test['LandSlope'])
test['Neighborhood']=le.fit_transform(test['Neighborhood'])
test['Condition1']=le.fit_transform(test['Condition1'])
test['Condition2']=le.fit_transform(test['Condition2'])
test['BldgType']=le.fit_transform(test['BldgType'])
test['HouseStyle']=le.fit_transform(test['HouseStyle'])
test['RoofStyle']=le.fit_transform(test['RoofStyle'])
test['RoofMatl']=le.fit_transform(test['RoofMatl'])
test['Exterior1st']=le.fit_transform(test['Exterior1st'])
test['Exterior2nd']=le.fit_transform(test['Exterior2nd'])
test['MasVnrType']=le.fit_transform(test['MasVnrType'])
test['ExterQual']=le.fit_transform(test['ExterQual'])
test['ExterCond']=le.fit_transform(test['ExterCond'])
test['Foundation']=le.fit_transform(test['Foundation'])
test['BsmtQual']=le.fit_transform(test['BsmtQual'])
test['BsmtCond']=le.fit_transform(test['BsmtCond'])
test['BsmtExposure']=le.fit_transform(test['BsmtExposure'])
test['BsmtFinType1']=le.fit_transform(test['BsmtFinType1'])
test['BsmtFinType2']=le.fit_transform(test['BsmtFinType2'])
test['Heating']=le.fit_transform(test['Heating'])
test['HeatingQC']=le.fit_transform(test['HeatingQC'])
test['CentralAir']=le.fit_transform(test['CentralAir'])
test['Electrical']=le.fit_transform(test['Electrical'])
test['KitchenQual']=le.fit_transform(test['KitchenQual'])
test['Functional']=le.fit_transform(test['Functional'])
test['GarageType']=le.fit_transform(test['GarageType'])
test['GarageFinish']=le.fit_transform(test['GarageFinish'])
test['GarageQual']=le.fit_transform(test['GarageQual'])
test['GarageCond']=le.fit_transform(test['GarageCond'])
test['PavedDrive']=le.fit_transform(test['PavedDrive'])
test['SaleType']=le.fit_transform(test['SaleType'])
test['SaleCondition']=le.fit_transform(test['SaleCondition'])


# In[ ]:


# Scaling input and test data using StandardScaler module
x=ss.fit_transform(x)
test=ss.fit_transform(test)


# #### Model Development

# In[ ]:


x.shape,y.shape,test.shape


# In[ ]:


# Model Building
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,VotingRegressor

# Metrics to evaluate the model
from sklearn.metrics import mean_squared_error as mse


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[ ]:


from sklearn import metrics
reg = LogisticRegression()


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


reg.score(X_train,y_train)


# In[ ]:


reg.score(X_test,y_test)


# In[ ]:


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model_dt=DecisionTreeRegressor(criterion='mse')
model_dt.fit(X_train,y_train)


# In[ ]:


model_dt.score(X_train,y_train)


# In[ ]:


model_dt.score(X_test,y_test)


# In[ ]:


# Random forest model
model_rf=RandomForestRegressor(n_estimators=500)
model_rf.fit(X_train,y_train)


# In[ ]:


model_rf.score(X_train,y_train)


# In[ ]:


model_rf.score(X_test,y_test)


# In[ ]:


# Linear Regression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_train,y_train)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


# AdaAdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
model_adb=AdaBoostRegressor(n_estimators=250)
model_adb.fit(X_train,y_train)


# In[ ]:


model_adb.score(X_train,y_train)


# In[ ]:


model_adb.score(X_test,y_test)


# In[ ]:


ypred_rf=model_rf.predict(test)


# In[ ]:


#We have found that Random Forest is the best model, with best score.


# In[ ]:


submission=pd.DataFrame()
submission['ID']=testid
submission['SalePrice']=ypred_rf
submission.head()


# In[ ]:


submission.to_csv("submissionrf.csv", index=False)


# In[ ]:




