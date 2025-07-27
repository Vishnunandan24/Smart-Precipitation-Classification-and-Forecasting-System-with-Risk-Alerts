# weatherPrecipitation.py

"""
Smart Precipitation Classification and Forecasting System with Risk Alerts
---------------------------------------------------------------------------
This script performs:
- Data cleaning and feature engineering
- Training 4 ML models (KNN, Naive Bayes, Decision Tree, SVM)
- Saving trained models and scaler
- AI-based signal generation (risk alerts) with logic
- Time series forecasting using ARIMA for precipitation proxy (Humidity)
"""

import pandas as pd
import numpy as np
import logging
import joblib
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info(" Starting: Smart Precipitation Classification System")

# ------------------------- Load Dataset -------------------------
url = "https://raw.githubusercontent.com/Vishnunandan24/Weather-Precipitation-Classification/main/weatherHistory.csv"
weather_data = pd.read_csv(url)
weather_data = weather_data[["Precip Type", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]]

# Clean and fill missing values
weather_data["Precip Type"] = weather_data["Precip Type"].replace(["null", "NULL", "Null", ""], "No Precipitation")
weather_data.fillna("No Precipitation", inplace=True)
weather_data.drop_duplicates(inplace=True)
weather_data.dropna(inplace=True)

# ---------------------- Feature Engineering ----------------------
logging.info("Performing feature engineering...")

weather_data["temp_diff"] = abs(weather_data["Temperature (C)"] - weather_data["Temperature (C)"].rolling(window=3).mean())
weather_data["humidity_index"] = weather_data["Humidity"] * weather_data["Wind Speed (km/h)"]
weather_data["rolling_precip_3"] = weather_data["Humidity"].rolling(window=3).mean()
weather_data.fillna(method='bfill', inplace=True)

features = ["Temperature (C)", "Humidity", "Wind Speed (km/h)", "temp_diff", "humidity_index", "rolling_precip_3"]
target = "Precip Type"

# ---------------------- Scaling & Splitting ----------------------
logging.info("Scaling features and splitting dataset...")
scaler = StandardScaler()
weather_data[features] = scaler.fit_transform(weather_data[features])

X = weather_data[features]
y = weather_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Save scaler for Streamlit app input
joblib.dump(scaler, "scaler.pkl")
logging.info("Scaler saved as scaler.pkl")

# ---------------------- Model Training ----------------------
def train_and_save(model, name):
    logging.info(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    logging.info(f"{name} Accuracy: {acc * 100:.2f}%")
    print(f"\n{name} Classification Report:\n", classification_report(y_test, predictions, zero_division=1))
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")
    logging.info(f"{name} model saved as {name.lower().replace(' ', '_')}_model.pkl")

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

for name, model in models.items():
    train_and_save(model, name)

# ---------------------- Real-Time Simulation ----------------------
def sensor_data_stream(df):
    for i in range(len(df)):
        yield df.iloc[i:i+1]

def generate_ai_signal(prediction, confidence):
    if prediction == "Rain" and confidence > 0.85:
        return "High Risk: Rainfall Alert - Take Precaution"
    elif prediction == "Snow":
        return "Medium Risk: Snow Expected"
    elif prediction == "No Precipitation":
        return "Safe: Clear Weather"
    else:
        return "Low Risk: Uncertain Condition"

logging.info("Simulating Real-Time Prediction Samples using SVM...")
stream = sensor_data_stream(X_test)
model = models["SVM"]

for i, row in enumerate(stream):
    prediction = model.predict(row)[0]
    confidence = np.max(model.predict_proba(row))
    signal = generate_ai_signal(prediction, confidence)
    print(f"Sample {i+1} | Prediction: {prediction} | Confidence: {confidence:.2f} | Signal: {signal}")
    if i == 4:
        break

# ---------------------- ARIMA Forecasting ----------------------
logging.info("Performing 3-day forecast using ARIMA on Humidity...")

ts_data = pd.read_csv(url)[['Humidity']].dropna().head(200)
model_arima = ARIMA(ts_data, order=(3, 1, 0))
model_fit = model_arima.fit()
forecast = model_fit.forecast(steps=3)

print("\n3-Day Forecast (Humidity as Proxy for Precipitation):")
for i, val in enumerate(forecast, 1):
    print(f"Day {i}: Predicted Humidity = {val:.4f}")

logging.info("Model training and saving complete. Ready for Streamlit dashboard.")
