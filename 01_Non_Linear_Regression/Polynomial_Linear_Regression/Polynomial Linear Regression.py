# In this dataset, there are salaries that are distributed among the employee according to there levels.
# We have to predict that if a new employee comes with level of 6.5 how much salary we should give.
# By looking at dataset, we can offer salary between range 1,50,00 to 2,00,000

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Non_Linear_Regression\Polynomial_Linear_Regression\employee_salary_data.csv")

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
lin_reg_pred                                 # This gives wrong prediction as we are making the predictions using Simple Linear Regression





##############################   Polynomial Model    ############################





from sklearn.preprocessing import PolynomialFeatures

# poly_reg = PolynomialFeatures()
# poly_reg = PolynomialFeatures(degree = 3)                            # Here By defalut system considers degree of 2
# poly_reg = PolynomialFeatures(degree = 4) 
# poly_reg = PolynomialFeatures(degree = 5) 
poly_reg = PolynomialFeatures(degree = 6) 


x_poly = poly_reg.fit_transform(x)
#poly_reg.fit(x_poly, y)                #    this line is not necessay at all

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)



plt.figure(figsize=(4, 2))
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title("Linear Regression Graph")
plt.xlabel("Position & Level")
plt.ylabel("Salary")
plt.show()


poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)





###########################       Conclusion    #############################



# Here by looking at the Graphs we can see that

#  Prediction for Degree 2 --> 189498
#  Prediction for Degree 3 --> 133259
#  Prediction for Degree 4 --> 158862
#  Prediction for Degree 5 --> 170445
#  Prediction for Degree 6 --> 174271


# By analyzing the Graphs Degree 5 and 6 gives the most accurate Predictions.
# The HR or Manager will offer salary between 170445 to 174271



