import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# LOAD DATA
# ==========================================

df=pd.read_csv("insurance_claims.csv")

# ==========================================
# REMOVE UNUSED COLUMN
# ==========================================

if "_c39" in df.columns:
    df.drop(columns=["_c39"],inplace=True)

df.replace("?",np.nan,inplace=True)

# ==========================================
# DATE FEATURES
# ==========================================

df["policy_bind_date"]=pd.to_datetime(df["policy_bind_date"])
df["incident_date"]=pd.to_datetime(df["incident_date"])

df["policy_age_days"]=(
    df["incident_date"]-
    df["policy_bind_date"]
).dt.days

df.drop(
    columns=[
        "policy_bind_date",
        "incident_date"
    ],
    inplace=True
)

# ==========================================
# MISSING VALUES
# ==========================================

for col in df.columns:

    if df[col].dtype=="object":

        df[col]=df[col].fillna(
            df[col].mode()[0]
        )

    else:

        df[col]=df[col].fillna(
            df[col].median()
        )

# ==========================================
# TARGET
# ==========================================

df["fraud_reported"]=df["fraud_reported"].map(
    {
        "N":0,
        "Y":1
    }
)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["claim_to_premium_ratio"]=(
    df["total_claim_amount"]/
    (df["policy_annual_premium"]+1)
)

df["high_claim"]=(
    df["total_claim_amount"]>50000
).astype(int)

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

# ==========================================
# SELECT FEATURES USED IN STREAMLIT
# ==========================================

selected_features = [

    "age",

    "policy_annual_premium",

    "total_claim_amount",

    "injury_claim",

    "property_claim",

    "vehicle_claim",

    "incident_severity",

    "police_report_available",

    "witnesses",

    "number_of_vehicles_involved",

    "bodily_injuries"

]

X = df[selected_features].copy()

y = df["fraud_reported"]

# ==========================================
# LABEL ENCODING
# ==========================================

encoders = {}

categorical_columns = [

    "incident_severity",

    "police_report_available"

]

for col in categorical_columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = le

# ==========================================
# SAVE FEATURE NAMES
# ==========================================

feature_columns = X.columns.tolist()

# ==========================================
# SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ==========================================
# HANDLE CLASS IMBALANCE
# ==========================================

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("Training Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# ==========================================
# DNN MODEL
# ==========================================

dnn = Sequential([

    Input(shape=(len(selected_features),)),

    Dense(128, activation="relu"),

    BatchNormalization(),

    Dropout(0.30),

    Dense(64, activation="relu"),

    BatchNormalization(),

    Dropout(0.25),

    Dense(32, activation="relu"),

    Dropout(0.20),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")

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

history = dnn.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=50,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1

)

# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = (dnn.predict(X_test) > 0.5).astype(int)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

# ==========================================
# AUTOENCODER
# ==========================================

input_dim = len(selected_features)

input_layer = Input(shape=(input_dim,))

encoded = Dense(16, activation="relu")(input_layer)

encoded = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)

decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(

    inputs=input_layer,

    outputs=decoded

)

autoencoder.compile(

    optimizer="adam",

    loss="mse"

)

autoencoder.fit(

    X_train,

    X_train,

    epochs=30,

    batch_size=32,

    validation_split=0.20,

    verbose=1

)

# ==========================================
# SAVE EVERYTHING
# ==========================================

dnn.save("fraud_dnn.keras")

autoencoder.save("autoencoder.keras")

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

print("\n================================")
print("PROJECT TRAINING COMPLETED")
print("================================")

print("Saved Files:")
print("✔ fraud_dnn.keras")
print("✔ autoencoder.keras")
print("✔ scaler.pkl")
print("✔ encoders.pkl")
print("✔ feature_columns.pkl")

print("\nSelected Features Used:")

for f in feature_columns:
    print("•", f)