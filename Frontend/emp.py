import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# === Function to convert time string to minutes ===
def time_to_minutes(t):
    return int(datetime.strptime(t, "%H:%M").hour) * 60 + int(datetime.strptime(t, "%H:%M").minute)
# === Load and prepare training data ===
@st.cache_data
def train_model():
    df_train = pd.read_csv("employee_fraud_risk_dataset.csv")
    df_train["LoginMinutes"] = df_train["LoginTime"].apply(time_to_minutes)
    df_train["LogoutMinutes"] = df_train["LogoutTime"].apply(time_to_minutes)
    df_train["WorkDuration"] = df_train["LogoutMinutes"] - df_train["LoginMinutes"]
    df_train.drop(columns=["LoginTime", "LogoutTime"], inplace=True)
    categorical_features = ["EmployeeRole"]
    numeric_features = df_train.drop(columns=["FraudRiskScore"] + categorical_features).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", SimpleImputer(strategy="mean"), numeric_features)
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])
    X_train = df_train.drop(columns=["FraudRiskScore"])
    y_train = df_train["FraudRiskScore"]
    model.fit(X_train, y_train)
    return model
# === Streamlit UI ===
st.title("Employee Fraud Risk Predictor")
uploaded_file = st.file_uploader("Add the file whose data you want to predict", type="csv")
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    try:
        df_new["LoginMinutes"] = df_new["LoginTime"].apply(time_to_minutes)
        df_new["LogoutMinutes"] = df_new["LogoutTime"].apply(time_to_minutes)
        df_new["WorkDuration"] = df_new["LogoutMinutes"] - df_new["LoginMinutes"]
        df_new.drop(columns=["LoginTime", "LogoutTime"], inplace=True)
        model = train_model()
        predictions = model.predict(df_new)
        df_new["PredictedFraudRiskScore"] = predictions
        st.success("Predictions completed successfully!")
        st.dataframe(df_new)
        csv = df_new.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted Results", csv, "predicted_fraud_risks.csv", "text/csv")
    except Exception as e:
        st.error(f"Error while processing file: {e}")