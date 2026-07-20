import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (Lasso, Ridge, LinearRegression, SGDRegressor, HuberRegressor, ElasticNet)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


data = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\Project using ML Libraries\USA_Housing_5000Rows_7Columns.csv")


print(data)

#Preprocessing
x = data.drop(['Price','Address'], axis=1)
y = data['Price']

# Split the Data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "RandamForestReg": RandomForestRegressor(),
    "SVM": SVR(),
    "ElasticNet": ElasticNet(),
    "PolynomailReg": Pipeline([
        ("Poly", PolynomialFeatures(degree=4)),
        ("Linear", LinearRegression())
    ]),
    "knnRegressor": KNeighborsRegressor(),
    "RobustRegression": HuberRegressor(),
    "SDGRegressor": SGDRegressor(),
    "ANN": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    "LGBM":lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor()
}


# Train and Evalute Models

results = []

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model' : name,
        "MAE" : mae,
        "MSE": mse,
        "R2": r2
    })

    with open(f'{name}.pkl','wb') as f:
        pickle.dump(model, f)

results_df = pd.DataFrame(results)
results_df.to_csv('Model_Evaluation_Result.csv', index=False)

print("Models has been trained and saved as Pickle Files. Evaluation results has been saved to model.")