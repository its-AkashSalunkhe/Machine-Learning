# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

dataset = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\employee_salary_data.csv")

dataset

# Here Salary is Dependant Variable (y) and Experience_Years is independant(x)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, train_size=0.8, random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Here, regressor is "Model" and LinearRegression is "Algorithm"

regressor.fit(x_train, y_train)       # Model will train on this data

y_pred = regressor.predict(x_test)

comparision = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparision)


plt.figure(figsize=(5,3))
plt.scatter(x_test, y_test, color='Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary of emp based on exp")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()



#==========================================================================================================


m_slope = regressor.coef_              # calculating the value of slope (m)
print(m_slope)

c_intercept = regressor.intercept_     # calculating the value of c (constant)
print(c_intercept)
 
y12_year_exp = m_slope*12+ c_intercept      # predicting the salary for 12 year exp employee
print(y12_year_exp)

y20_year_exp = m_slope*20+ c_intercept        # predicting the salary for 20 year exp employee
print(y20_year_exp)

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)


# after calculating the "Bias and Varinace" we can see both are high (high bias and high variance)
# Hence these is the best fit for model


#============================================== Implemting Statistics ============================================================

print(dataset.mean())

print(dataset['Salary'].mean())

print(dataset.median())
print(dataset['Salary'].mode() )


print(dataset.std())
print(dataset.var())


from scipy.stats import variation

print(variation(dataset.values))   # this will give coeff of var of entire dataframe.
print(variation(dataset['Salary']))         # this will give coeff of var of perticular Column.

# Correlation

dataset.corr()
dataset['Salary'].corr(dataset['Experience_Years'])


# Skewness

dataset.skew()     # this will give skewness of entire dataframe.

# Standard Error

dataset.sem()      # this will give Standard Error of entire dataframe.
dataset['Salary'].sem()


# Z - Score

import scipy.stats as stats

dataset.apply(stats.zscore)       # this will give Z - Score of entire dataframe.
stats.zscore(dataset["Salary"])




#============================================== ANOVA ============================================================


# SSR

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

r_square = 1 - (SSR - SST)
r_square


print(r_square)
print(bias)
print(variance)


import pickle
filename = 'Linear_Regression_Model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as Linear_Regression_Model.pkl") 

import os
print(os.getcwd())   




  