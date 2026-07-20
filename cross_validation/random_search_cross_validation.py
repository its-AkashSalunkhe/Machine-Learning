
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



########################### RANDOM - SEARCH CROSS VALIDATION ########################




# Import necessary libraries
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter distributions
parameters = {
    'C': uniform(1, 1000),             # Random values between 1 and 1001
    'kernel': ['linear', 'rbf'],       # Selection of kernels
    'gamma': uniform(0.1, 0.9)         # Random values between 0.1 and 1.0
}

# Apply Randomized Search
random_search = RandomizedSearchCV(
    estimator=sv_model,
    param_distributions=parameters,
    n_iter=20,                         # Number of random combinations to try
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=0,                         # Use all available CPU cores
)

# Fit the model
random_search.fit(x_train, y_train)

# Get the best results
best_accuracy = random_search.best_score_
best_parameters = random_search.best_params_

# Print the results
print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
print("Best Parameters:", best_parameters)