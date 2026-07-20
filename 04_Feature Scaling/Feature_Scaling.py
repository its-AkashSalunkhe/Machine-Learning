# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cust_data = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\ML Model - Training and Testing\customer_data_from_image.csv")

cust_data

# here Y means dependant variable is "Purchased"
# and all the other are independant variables (x)

# x = cust_data.iloc[:,:-1].values
x = cust_data.iloc[:,:3].values

# y = cust_data.iloc[:,-1]
y = cust_data.iloc[:,3]

cust_data.isna().max()   

from sklearn.impute import SimpleImputer

# imputer = SimpleImputer()   # By default considers the mean values
imputer = SimpleImputer(strategy='median') 
# imputer = SimpleImputer(strategy='most_frequent')

imputer =  imputer.fit(x[:,1:3])   # Understands the data and calculates the mean, std. But don't apply the values

x[:,1:3] = imputer.transform(x[:,1:3])   # Applies the values



from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()
label_encoder_x.fit_transform(x[:,0])
x[:,0] = label_encoder_x.fit_transform(x[:,0])

label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)





##############################  Train and Test Data   ############################




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, train_size=0.7, random_state=0)

#x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=0)

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)




##############################  FEATURE SCALING  ############################


# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)


