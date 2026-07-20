# In this dataset, there are salaries that are distributed among the employee according to there levels.
# We have to predict that if a new employee comes with level of 6.5 how much salary we should give.
# By looking at dataset, we can offer salary between range 1,50,00 to 2,00,000


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Non_Linear_Regression\Random_Forest\employee_salary_data.csv")

df
df.head(2)
df.columns

y = df.iloc[:,2].values
x = df.iloc[:,1:2].values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.figure(figsize=(4, 2))
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title("Linear Regression Graph")
plt.xlabel("Position & Level")
plt.ylabel("Salary")
plt.show()


lin_reg_pred = lin_reg.predict([[6.5]])
print(lin_reg_pred)                                 # This gives wrong prediction as we are making the predictions using Simple Linear Regression





################################   Random Forest     ################################




from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=25, 
                               criterion='poisson', 
                               random_state=0,
                               min_samples_split=3)          
rf_reg.fit(x, y)


rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)
 




###########################       Conclusion    #############################



#  Case 1 (using default parameters):  n_estimators=100, criterion="squared_error", splitter="best", max_depth=None,   --> 168700
#  Case 2: n_estimators=30,random_state=0 --> 164333
#  Case 3: n_estimators=25,random_state=0, max_depth=2 --> 155750
#  Case 4: n_estimators=25, criterion='poisson', random_state=0, min_samples_split=3 --> 175166


# Here by looking at the Test Cases we can see that -> Test case 6 gives the more accurate prediction which is 175166.
# The HR or Manager will offer salary around 175166.
