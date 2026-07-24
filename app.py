from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read user inputs
        age = int(request.form["Age"])
        gender = request.form["Gender"]
        region = request.form["Region"]
        occupation = request.form["Occupation"]
        income = float(request.form["Income"])

        # Create dataframe
        data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Region": [region],
            "Occupation": [occupation],
            "Income": [income]
        })

        # Predict
        prediction = model.predict(data)[0]

        # Convert prediction to readable text
        prediction_text = (
            "✅ Person is likely to own a Laptop"
            if prediction == 1
            else "❌ Person is NOT likely to own a Laptop"
        )

        return render_template(
            "index.html",
            prediction=prediction_text,
            prediction_value=int(prediction)
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"⚠️ Error: {str(e)}",
            prediction_value=-1
        )


if __name__ == "__main__":
    app.run(debug=True)
