

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

from sklearn.linear_model import LogisticRegression

classifier =  LogisticRegression()
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





from sklearn.metrics import roc_auc_score, roc_curve

y_pred_prob = classifier.predict_proba(x_test)[:,1]

auc_score = roc_auc_score(y_test, y_pred_prob)
auc_score

fpr,tpr, thresholds = roc_curve(y_test, y_pred_prob)


plt.figure(figsize = (6,4))
plt.plot(fpr,tpr, label = f'Logistic Regression ( AUC = {auc_score:.2f})')
plt.plot([0,1],[0,1], 'k--')         # Random Classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.show()





############################       CONCLUSION     ############################


# Test size = 25% and Scaling Technique = StandardScalar()
# Test Case 1 (random state = 0): bias = 88%, variance = 87%, accuracy = 87%
# Test Case 2 (random state = 100): bias = 90%, variance = 83%, accuracy = 83%
# Test Case 3 (random state = 51): bias = 88%, variance = 91%, accuracy = 91%
# Test Case 4 (random state = 41): bias = 87%, variance = 92%, accuracy = 92%


# Now Scanling the data using Normalizatoin (min-max Scalar)

# Test Case 5 (random state = 0): bias = 54%, variance = 52%, accuracy = 52%  (Very Poor Result)



# Test size = 20% and Scaling Technique = StandardScalar()

# Test Case 6 (random state = 0): bias = 88%, variance = 86%, accuracy = 86%
# Test Case 7 (random state = 100): bias = 89%, variance = 85%, accuracy = 85%
