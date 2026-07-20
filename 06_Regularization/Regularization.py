# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing


data = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Regularization\synthetic_auto_dataset_records.csv")

data



data.head()
data.tail()
data.columns
data.describe()
data.info          
data.info()
data.isna().any()          # Dataset contains some '?' but it will not consider as null values
data.isna().sum()

data = data.drop(['car_name'], axis = 1)     # no longer required hence drop

data = data.replace('?',np.nan)       # we'll replace the '?' with null values
data = data.apply(pd.to_numeric, errors = 'coerce')    # will convert other data type values (like '234' is string but it'll conveted to 234 int ) and also ignores the Erros.


data.isna().sum()       # now it will show the null values

data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != 'object' else x, axis = 0 )    # replaces the null values with median

data['origin'] = data['origin'].replace({1:'america',2:'europe',3:'asia'})       # in the origin column we'll assign 1 to america and 2 to europe and 3 to asia
data  = pd.get_dummies(data, columns = ['origin'],dtype = int)                 # and Transformation

y = data.iloc[:,0]
x = data.iloc[:,1:12]


#############################    PREPROCESSING   ##############################



x_pre = preprocessing.scale(x)          # we will scale down the data here. it will give values between -1 to 1, which helps to improve the accuracy
x_pre


y_pre = preprocessing.scale(y)
y_pre


######################    TRAINING - TESTING, COEF AND INTERCEPT   #########################



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_pre,y_pre, test_size=0.3, random_state=0)


from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
regression_model.fit(x_train,y_train)


for idx, col_name in enumerate(x.columns):
    print('The coef for {} is {}'.format(col_name, regression_model.coef_[idx]))       # getting coef of every column ( coef = m )
    
print("_____________________________")


intercept = regression_model.intercept_
print("The intercept is {}".format(intercept))


print("_____________________________")



############################# RIDGE AND LASSO  ##############################


from sklearn.linear_model import Ridge, Lasso      

ridge_model = Ridge(alpha=0.4)
ridge_model.fit(x_train,y_train)



print(f'Ridge model Coef: {ridge_model.coef_}')

lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(x_train,y_train)

print(f'Lasso model Coef: {lasso_model.coef_}')


print(regression_model.score(x_train,y_train))
print(regression_model.score(x_test,y_test))

print("^^^")

print(ridge_model.score(x_train,y_train))
print(ridge_model.score(x_test,y_test))

print("^^^")

print(lasso_model.score(x_train,y_train))
print(lasso_model.score(x_test,y_test))
      

