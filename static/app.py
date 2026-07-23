from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask application
app = Flask(__name__)

# Load the trained pipeline model
model = joblib.load("model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from HTML form
        age = int(request.form["Age"])
        gender = request.form["Gender"]
        region = request.form["Region"]
        occupation = request.form["Occupation"]
        income = float(request.form["Income"])

        # Create DataFrame
        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Region": [region],
            "Occupation": [occupation],
            "Income": [income]
        })

        # Prediction
        prediction = model.predict(data)[0]

        return render_template(
            "index.html",
            prediction_text=f"Prediction : {prediction.upper()}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error : {e}"
        )


# Run Application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
