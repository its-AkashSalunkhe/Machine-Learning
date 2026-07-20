

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\Classifications Algorithm\Support_Vector_Machine\Social_Network_Ads.csv")

df

x = df.iloc[:,2:4]
y = df.iloc[:,4]

# Splitting the data into Traning and Testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=41)


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



# Now Training the Support Vector Machine on the Training set

from sklearn.svm import SVC

sv_model = SVC(probability=True, random_state=0)
sv_model.fit(x_train,y_train)
y_pred = sv_model.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
   

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr) 


bias = sv_model.score(x_train, y_train)
print(bias)

variance =  sv_model.score(x_test,y_test)
print(variance)



############################## K - FOlD CROSS VALIDATION #######################


#  Cross validation technique is always used at the end of the code
#  IT is used only when the model is overfitted 


from sklearn.model_selection import cross_val_score

accuracies  = cross_val_score(estimator = sv_model, X = x_train, y = y_train, cv = 12)
print("Accuracy: {:.2f}%".format(accuracies.mean()*100)) 
print("Standard Deviation: {:.2f}%".format(accuracies.std()*100)) 

'''
from sklearn.metrics import roc_auc_score, roc_curve

y_pred_prob = sv_model.predict_proba(x_test)[:,1]

auc_score = roc_auc_score(y_test, y_pred_prob)
auc_score

fpr,tpr, thresholds = roc_curve(y_test, y_pred_prob)


plt.figure(figsize = (4,2))
plt.plot(fpr,tpr, label = f'Support Vector Machine ( AUC = {auc_score:.2f})')
plt.plot([0,1],[0,1], 'k--')         # Random Classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid()
plt.show()

'''


############################       CONCLUSION     ############################


# Test size = 25% and Scaling Technique = StandardScalar()
# Test Case 1 (random state = 0): bias = 87%, variance = 92%, accuracy = 92%

