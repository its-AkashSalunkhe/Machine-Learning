

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

from sklearn.neighbors import KNeighborsClassifier   # By default KNN Considers 5 Neighbours ( k = 5 )

classifier =  KNeighborsClassifier()
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





