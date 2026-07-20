

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Akash\Downloads\Churn_Modelling - Churn_Modelling.csv")

df
# Checking if data has null values or missing data

df.isna().any()         # Data is cleaned


# Dividing the data into x (Independant Variables) and y (Dependant Variable)
x = df.iloc[:,3:-1].values
y = df.iloc[:,13].values



# Now Encoding Categorical Data
# Label Encoding the "Gender" column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])


# In a dataset when you have more than 2 categorical variables
# Since these data more than two columns with "Categorical Values" we need to use ColumnTransformer
from sklearn.compose import ColumnTransformer       
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', 
                                      OneHotEncoder(), 
                                      [1])],
                                      remainder="passthrough")

x = np.array(ct.fit_transform(x))



# Splitting the data into Traning and Testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=0)


# Data Processing is done up to these



# Now Training the Model XGBoost Algorithm

from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
   

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr) 


bias = classifier.score(x_train, y_train)
print(bias)

variance =  classifier.score(x_test,y_test)
print(variance)



import pickle 

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
    

with open('gender_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)
    
#############################      CONCLUSION    ##############################


# Test Case 1: Accuracy : 85%, Bias: 95% and Variance: 85%   (train_size = 0.20)
 
 
