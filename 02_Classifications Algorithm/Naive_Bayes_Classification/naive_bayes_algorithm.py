# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Classifications Algorithm\Logistic_Regression\Social_Network_Ads.csv")

df

x = df.iloc[:,2:4]
y = df.iloc[:,4]

# Splitting the data into Traning and Testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=41)


# Scanling the data using Starderdization

from sklearn.preprocessing import StandardScaler     # Standerdization = Z-score ( Range is -3 to 3 )
#from sklearn.preprocessing import Normalizer

std = StandardScaler()
#nl = Normalizer()



x_train = std.fit_transform(x_train)
#x_train = nl.fit_transform(x_train)
x_test = std.transform(x_test)
#x_test = nl.transform(x_test)

# Data Processing is done up to these



# Now Training the logistics Regression Model on the Training set

from sklearn.naive_bayes import GaussianNB  
# from sklearn.naive_bayes import BernoulliNB  
# from sklearn.naive_bayes import MultinomialNB 

classifier =  GaussianNB()
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
print("bais:",bias)

variance =  classifier.score(x_test,y_test)
print("variance:", variance)





#############################      CONCLUSION    ##############################



#              TYPE 1 ---->>> GAUSIAN NB (Doesn't require Feature Scaling)


# for 25% data splitting and with Standarization Scaling technique

# Test Case 1: Accuracy : 93%, Bias: 87% and Variance: 93%    (with Scaling)
# Test Case 1: Accuracy : 93%, Bias: 87% and Variance: 93%    (without Scaling)



# for 25% data splitting and with Normalization Scaling technique
# Test Case 3: Accuracy : 51%, Bias: 53% and Variance: 51%



#                      TYPE 2 ---->>> BERNOULLI NB


# Test Case 4: Accuracy : 83%, Bias: 78% and Variance: 83%     (with Scaling)
# Test Case 5: Accuracy : 56%, Bias: 56% and Variance: 56%     (without Scaling)




#                      TYPE 3 ---->>> MULTINOMIAL NB


# Multinomial only work data doens't contain negative values after scaling

# Test Case 6: Accuracy : 56%, Bias: 56% and Variance: 56%      (with Scaling)
# Test Case 6: Accuracy : 53%, Bias: 59% and Variance: 53%      (without Scaling)




