from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# -------- Correct Model Path --------
MODEL_PATH = "C:/Users/giriy/Downloads/motor_model.pkl"

# -------- Load Model --------
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found! Please check file location.")
    
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# -------- Home Route --------
@app.route('/')
def home():
    return render_template("index.html")

# -------- About Route --------
@app.route('/about')
def about():
    return render_template("about.html")

# -------- Prediction Route --------
@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return render_template("index.html", prediction="Model not loaded!")

    try:
        voltage = float(request.form['voltage'])
        current = float(request.form['current'])
        speed = float(request.form['speed'])
        torque = float(request.form['torque'])

        # Create DataFrame with correct column names
        input_data = pd.DataFrame(
            [[voltage, current, speed, torque]],
            columns=['voltage', 'current', 'speed', 'torque']
        )

        prediction = model.predict(input_data)
        result = round(prediction[0], 2)

        return render_template("index.html", prediction=f"Predicted Value: {result}")

    except ValueError:
        return render_template("index.html", prediction="Please enter valid numbers!")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

# -------- Run App --------
if __name__ == "__main__":
    app.run(debug=True)