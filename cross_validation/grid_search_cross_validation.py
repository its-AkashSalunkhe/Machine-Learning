


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



########################### GRID- SEARCH CROSS VALIDATION ########################


#  Cross validation technique is always used at the end of the code
#  IT is used only when the model is overfitted 


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10]}
]

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=sv_model,
    param_grid=parameters,
    scoring='accuracy',
    cv=10
)

# Fit the model to the training data
grid_search = grid_search.fit(x_train, y_train)

# Get the best score and the best parameters
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Print the results
print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
print("Best Parameters:", best_parameters)
