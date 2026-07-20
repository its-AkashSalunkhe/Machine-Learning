from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open("RandomForestReg.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    income = float(request.form['income'])
    age = float(request.form['age'])
    rooms = float(request.form['rooms'])
    bedrooms = float(request.form['bedrooms'])
    population = float(request.form['population'])

    features = np.array([[income, age, rooms, bedrooms, population]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ${prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)