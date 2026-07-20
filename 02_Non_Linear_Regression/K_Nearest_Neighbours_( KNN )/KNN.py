# In this dataset, there are salaries that are distributed among the employee according to there levels.
# We have to predict that if a new employee comes with level of 6.5 how much salary we should give.
# By looking at dataset, we can offer salary between range 1,50,00 to 2,00,000


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Non_Linear_Regression\Support_Vector_Regression_( SVR )\employee_salary_data.csv")

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





########################   K-Nearest Neighbours ( KNN )    ######################




from sklearn.neighbors import KNeighborsRegressor 

knn_reg = KNeighborsRegressor(n_neighbors = 2)              # By default it Considers:  neighbours = 5, Euclidean Distance = 2, Manhatten Distance = 1.
knn_reg.fit(x, y)


knn_pred = knn_reg.predict([[6.5]])
print(knn_pred)
 




###########################       Conclusion    #############################



#  Case 1 (using default parameters): n_neighbours = 5, Pd = 2, Ed = 1   --> 168000
#  Case 2: n_neighbours = 3 --> 153333
#  Case 3: n_neighbours = 4 --> 190000
#  Case 4: n_neighbours = 2 --> 175000



# Here by looking at the Test Cases we can see that -> Test case 4 gives the more accurate prediction which is  175000.
# The HR or Manager will offer salary between 175000.



