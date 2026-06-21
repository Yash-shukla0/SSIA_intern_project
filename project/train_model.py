import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("insurance_claims.csv")

# =====================================
# CLEANING
# =====================================

if "_c39" in df.columns:
    df.drop(columns=["_c39"], inplace=True)

df.replace("?", np.nan, inplace=True)

# =====================================
# DATE FEATURES
# =====================================

df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
df["incident_date"] = pd.to_datetime(df["incident_date"])

df["policy_age_days"] = (
    df["incident_date"] -
    df["policy_bind_date"]
).dt.days

df.drop(
    columns=[
        "policy_bind_date",
        "incident_date"
    ],
    inplace=True
)

# =====================================
# MISSING VALUES
# =====================================

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    else:

        df[col] = df[col].fillna(
            df[col].median()
        )

# =====================================
# TARGET ENCODING
# =====================================

df["fraud_reported"] = (
    df["fraud_reported"]
    .map({
        "N": 0,
        "Y": 1
    })
)

# =====================================
# FEATURE ENGINEERING
# =====================================

df["claim_to_premium_ratio"] = (
    df["total_claim_amount"] /
    (df["policy_annual_premium"] + 1)
)

df["high_claim"] = (
    df["total_claim_amount"] > 50000
).astype(int)

# =====================================
# LABEL ENCODING
# =====================================

encoders = {}

for col in df.select_dtypes(
    include="object"
).columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le

# =====================================
# FEATURES / TARGET
# =====================================

X = df.drop(
    "fraud_reported",
    axis=1
)

y = df["fraud_reported"]

feature_columns = X.columns.tolist()

# =====================================
# SCALING
# =====================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# =====================================
# SMOTE
# =====================================

smote = SMOTE(
    random_state=42
)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

# =====================================
# DNN MODEL
# =====================================

dnn = Sequential([

    Dense(
        256,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

dnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

dnn.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# =====================================
# AUTOENCODER
# =====================================

input_dim = X_train.shape[1]

input_layer = Input(
    shape=(input_dim,)
)

encoded = Dense(
    64,
    activation="relu"
)(input_layer)

encoded = Dense(
    32,
    activation="relu"
)(encoded)

decoded = Dense(
    64,
    activation="relu"
)(encoded)

decoded = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    input_layer,
    decoded
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=32
)

# =====================================
# SAVE EVERYTHING
# =====================================

dnn.save(
    "fraud_dnn.keras"
)

autoencoder.save(
    "autoencoder.keras"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    encoders,
    "encoders.pkl"
)

joblib.dump(
    feature_columns,
    "feature_columns.pkl"
)

print("Training Complete")