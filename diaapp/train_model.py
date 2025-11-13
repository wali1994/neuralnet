# train_model.py
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump
from tensorflow.keras import models, layers

DATA_PATH = "data/diabetes_raw_cleaned_25k.csv"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # One hot encoding for gender and smoking_history
    df_model = pd.get_dummies(
        df,
        columns=["gender", "smoking_history"],
        drop_first=True
    )

    y = df_model["diabetes"]
    X = df_model.drop("diabetes", axis=1)

    feature_columns = X.columns.tolist()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Save feature column names
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_model(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(32, activation="relu", input_dim=input_dim))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Class weights
    classes = np.array([0, 1])
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {
        0: float(class_weights_values[0]),
        1: float(class_weights_values[1])
    }
    print("Class weights:", class_weight_dict)

    model = build_model(X_train.shape[1])

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluation
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_prob)
    print("ROC AUC:", auc)

    # Save model
    model.save(os.path.join(MODELS_DIR, "diabetes_model.h5"))
    print("Model saved to models/diabetes_model.h5")

if __name__ == "__main__":
    main()

