import joblib
import numpy as np

from tensorflow.keras.models import load_model


dnn = load_model("fraud_dnn.keras", compile=False)
autoencoder = load_model("autoencoder.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# Fraud Prediction Agent

def predict(features):

    # Check feature count
    if len(features) != 11:
        raise ValueError(
            f"Model expects 11 features, but received {len(features)}."
        )

    X = np.array(features).reshape(1, -1)

    X = scaler.transform(X)

    # DNN Prediction
    fraud_probability = float(
        dnn.predict(X, verbose=0)[0][0]
    )

    # Autoencoder Anomaly Detection
    reconstructed = autoencoder.predict(
        X,
        verbose=0
    )

    anomaly_score = float(
        np.mean((X - reconstructed) ** 2)
    )

    return fraud_probability, anomaly_score