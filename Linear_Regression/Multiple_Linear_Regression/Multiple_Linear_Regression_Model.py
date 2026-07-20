# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Simple_Linear_Regression\Multiple_Linear_Regression\marketing_dataset_100_records.csv")

df

df.columns

# Dependant Variable = y = Profit
# Independant Variable = x = Other variable

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x = pd.get_dummies(x, columns=['State'], drop_first=True, dtype=int)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)


x = x.values

x = np.append(arr=np.full((100,1), -3383).astype(int), values = x, axis = 1)

import statsmodels.api as sm
x_opt = x[:,[1,2,3,4]]

regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()  
regressor_OLS.summary()


# after running above 4 lines of code we get highest p value of attribute 4, i,e.. (x[4]). Hence we need to eliminate it using "RSE (Recursive Feature Elimination)"


x_opt = x[:,[1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()  
regressor_OLS.summary()


# now we have got p value of attributes x2 and x3 as 0
# Hence we'll consider both attributes here
# x1 attribute's  p value is also 0 but that attribute (Column) contains only "Constant" (c) values. Hence we'll not consider it.
