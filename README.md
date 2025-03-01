# Housing-Price-Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('/Churn_Modelling.csv')
dataset.shape
dataset.dtypes
#check for missing value
dataset.isnull().sum()
dataset.dropna(inplace=True)
#viewing the data statistics
dataset.describe()
#finding out the correlation between multiple columns
corr=dataset.corr()
corr.shape
#plotting the heatmap of correlation between feature
plt.figure(figsize=(20,20))
sns.heatmap(corr,cbar=True ,square=True, fmt='.1f',annot=True,annot_kws={'size':15},cmap='Blues')
#spliting target variable and independent variable
x=dataset[['RM']]
y=dataset[['MEDV']]
x
y
#splitting to training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
#import library for linear Regression
from sklearn.linear_model import LinearRegression
#create a linear regressor
lm=LinearRegression()
#train the model using the training sets
lm.fit(x_train,y_train)
#value of y intercept
lm.intercept_
#converting the cofficient value to a dataframe
coeffcients=pd.DataFrame([x_train.columns,lm.coef_]).T
coeffcients=coeffcients.rename(columns={0:'Attribute',1:'Coeffcients'})
coeffcients
y_pred=lm.predict(x_train)
from sklearn import metrics
#model evaluiation on train data
print('MAE:',metrics.mean_absolute_error(y_train,y_pred))
print('MSE:',metrics.mean_squared_error(y_train,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train,y_pred)))
plt.scatter(y_train,y_pred)
sns.regplot(x=y_train,y=y_pred)
plt.xlabel('Prices')
plt.ylabel('predicted prices')
plt.title('prices vs Predicted Prices')
plt.show()
plt.scatter(y_train,y_pred)
plt.xlabel('Prices')
plt.ylabel('predicted prices')
plt.title('prices vs Predicted Prices')
plt.show()
sns.distplot(y_train-y_pred)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
