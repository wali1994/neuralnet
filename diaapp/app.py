from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import os
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODELS_DIR = "models"

model = load_model(os.path.join(MODELS_DIR, "diabetes_model.h5"))
scaler = load(os.path.join(MODELS_DIR, "scaler.pkl"))

with open(os.path.join(MODELS_DIR, "feature_columns.json"), "r") as f:
    FEATURE_COLUMNS = json.load(f)

CATEGORICAL_COLS = ["gender", "smoking_history"]

@app.route("/")
def index():
    return render_template("index.html")

def preprocess_input(data):
    # data is a dict from form or JSON
    input_df = pd.DataFrame([data])

    # Convert numeric fields
    numeric_fields = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ]
    for col in numeric_fields:
        input_df[col] = pd.to_numeric(input_df[col])

    input_df["hypertension"] = input_df["hypertension"].astype(int)
    input_df["heart_disease"] = input_df["heart_disease"].astype(int)

    # One hot encoding
    input_encoded = pd.get_dummies(
        input_df,
        columns=CATEGORICAL_COLS,
        drop_first=True
    )

    # Align columns with training
    input_aligned = input_encoded.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    # Scale
    scaled = scaler.transform(input_aligned)

    return scaled

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Example expected keys in data:
    # gender, age, hypertension, heart_disease,
    # smoking_history, bmi, HbA1c_level, blood_glucose_level

    processed = preprocess_input(data)
    prob = float(model.predict(processed)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"

    if label == "Positive":
        comment = "You may be at higher risk of diabetes. Please consult a doctor."
    else:
        comment = "Your predicted diabetes risk is low based on this model. This is not a medical diagnosis."

    return jsonify({
        "result": f"Diabetes Risk: {label}",
        "probability": prob,
        "comment": comment
    })

if __name__ == "__main__":
    app.run(debug=True)

