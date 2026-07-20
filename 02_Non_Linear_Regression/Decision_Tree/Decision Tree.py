# In this dataset, there are salaries that are distributed among the employee according to there levels.
# We have to predict that if a new employee comes with level of 6.5 how much salary we should give.
# By looking at dataset, we can offer salary between range 1,50,00 to 2,00,000


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Non_Linear_Regression\Decision_Tree\employee_salary_data.csv")

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





################################   Decision Tree    ################################




from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(criterion='poisson', splitter='random', random_state=12, max_depth=2)            
dt_reg.fit(x, y)


dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)
 




###########################       Conclusion    #############################



#  Case 1 (using default parameters):  criterion="squared_error", splitter="best", max_depth=None,   --> 150000
#  Case 2: criterion='absolute_error', splitter='random', random_state=0, max_depth=3 --> 400000
#  Case 3: criterion='absolute_error', splitter='best', random_state=0, max_depth=2 --> 70000
#  Case 4: criterion='poisson', splitter='best', random_state=0, max_depth=2 --> 216666
#  Case 5: criterion='poisson', splitter='random', random_state=11, max_depth=2 --> 168000
#  Case 6: criterion='poisson', splitter='random', random_state=12, max_depth=2 --> 175000

# Here by looking at the Test Cases we can see that -> Test case 6 gives the more accurate prediction which is  175000.
# The HR or Manager will offer salary between 175000.



