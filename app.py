from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask App
app = Flask(__name__)

# Load the trained Pipeline model
model = joblib.load("model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from HTML form
        age = int(request.form["Age"])
        gender = request.form["Gender"]
        region = request.form["Region"]
        occupation = request.form["Occupation"]
        income = float(request.form["Income"])

        # Create DataFrame with the same column names used during training
        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Region": [region],
            "Occupation": [occupation],
            "Income": [income]
        })

        # Make prediction
        prediction = model.predict(data)[0]

        # Convert prediction into readable text
        result = "YES" if prediction == 1 else "NO"

        # Show prediction on webpage
        return render_template(
            "index.html",
            prediction_text=f"Prediction : {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error : {str(e)}"
        )


# Run Flask Application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
