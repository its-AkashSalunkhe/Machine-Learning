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





##########################   Support Vector Regression    #########################


from sklearn.svm import SVR 

#svr_reg = SVR()
svr_reg = SVR(kernel='linear', degree=4, gamma='scale')
svr_reg.fit(x,y)

svr_pred = svr_reg.predict([[6.5]])
print(svr_pred)





###########################       Conclusion    #############################



#  Case 1 (using default parameters):  kernal -rbf, degree-3, gamma-scale --> 130001
#  Case 2: (kernel='sigmoid', degree=4, gamma='auto') --> 129999
#  Case 3: (kernel='poly', degree=4, gamma='auto') --> 175707
#  Case 4: (kernel='linear', degree=4, gamma='scale') --> 130025



# Here by looking at the Test Cases we can see that -> Test case 3 gives the more accurate prediction which is 175707.
# The HR or Manager will offer salary between 175707.



