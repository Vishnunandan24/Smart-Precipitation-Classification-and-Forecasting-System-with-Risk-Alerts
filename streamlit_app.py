# streamlit_app.py

"""
🌦 Smart Precipitation Classification and Forecasting System with Risk Alerts
------------------------------------------------------------
Built using:
- ML models: KNN, Naive Bayes, Decision Tree, SVM
- Advanced feature engineering
- Real-time prediction with AI-based risk alerts
- Time series forecasting using ARIMA
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

from statsmodels.tsa.arima.model import ARIMA

# Page settings
st.set_page_config(page_title="Weather Precipitation Forecasting Dashboard", layout="wide", page_icon="🌧️")
st.title("🌧️ Smart Precipitation Classification and Forecasting System with Risk Alerts")
st.markdown("Powered by **KNN**, **Naive Bayes**, **Decision Tree**, and **SVM** | With Simulated Real-Time Predictions + Forecasting")

# Load & preprocess data
weather_data = pd.read_csv("https://raw.githubusercontent.com/Vishnunandan24/Weather-Precipitation-Classification/main/weatherHistory.csv")
weather_data = weather_data[["Precip Type", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
weather_data["Precip Type"] = weather_data["Precip Type"].replace(["null", "NULL", "Null", ""], "No Precipitation").fillna("No Precipitation")
weather_data.drop_duplicates(inplace=True)
weather_data.dropna(inplace=True)

# Advanced feature engineering
weather_data["temp_diff"] = abs(weather_data["Temperature (C)"] - weather_data["Temperature (C)"].rolling(window=3).mean())
weather_data["humidity_index"] = weather_data["Humidity"] * weather_data["Wind Speed (km/h)"]
weather_data["rolling_precip_3"] = weather_data["Humidity"].rolling(window=3).mean()
weather_data.fillna(method='bfill', inplace=True)

features = ["Temperature (C)", "Humidity", "Wind Speed (km/h)", "temp_diff", "humidity_index", "rolling_precip_3"]
scaler = StandardScaler()
weather_data[features] = scaler.fit_transform(weather_data[features])

X = weather_data[features]
y = weather_data["Precip Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Sidebar Controls
st.sidebar.header("⚙ Controls")
model_choice = st.sidebar.selectbox("Choose ML Model", ("KNN", "Naive Bayes", "Decision Tree", "SVM"))
show_eda = st.sidebar.checkbox("Show EDA Plots", value=True)
show_conf_matrix = st.sidebar.checkbox("Show Confusion Matrix")
show_tree = st.sidebar.checkbox("Show Decision Tree (DT only)")
show_forecast = st.sidebar.checkbox("Show 3-Day Humidity Forecast")

# Load trained models
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "SVM": joblib.load("svm_model.pkl")
}
selected_model = models[model_choice]
predictions = selected_model.predict(X_test)
acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# ----------- Section 1: EDA -----------
if show_eda:
    st.subheader("📊 Exploratory Data Analysis")
    with st.expander("Distributions and Boxplots"):
        for col in features:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(weather_data[col], kde=True, ax=axs[0])
            axs[0].set_title(f"Distribution of {col}")
            sns.boxplot(x=weather_data[col], ax=axs[1])
            axs[1].set_title(f"Outlier Detection in {col}")
            st.pyplot(fig)

    with st.expander("Correlation Heatmap"):
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(weather_data[features].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

# ----------- Section 2: Model Performance -----------
st.subheader(f"🧠 {model_choice} Model Performance")
st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")
st.code(classification_report(y_test, predictions), language="text")

# ----------- Section 3: Confusion Matrix -----------
if show_conf_matrix:
    st.subheader("🧾 Confusion Matrix")
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=y.unique(), yticklabels=y.unique())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# ----------- Section 4: Decision Tree -----------
if show_tree and model_choice == "Decision Tree":
    st.subheader("🌲 Visualize Decision Tree Structure")
    fig = plt.figure(figsize=(12, 8))
    plot_tree(selected_model, feature_names=X.columns, class_names=selected_model.classes_, filled=True)
    st.pyplot(fig)

# ----------- Section 5: Real-Time AI Signal Generator -----------
st.subheader("⏱ Simulated Real-Time Precipitation Prediction with AI-Based Risk Alert System (First 5 Samples)")

def sensor_data_stream(df):
    for i in range(len(df)):
        yield df.iloc[i:i+1]

def generate_ai_signal(pred, prob):
    if pred == "rain" and prob > 0.85:
        return "🚨 High Risk: Rainfall Alert - Stay Safe"
    elif pred == "snow":
        return "⚠️ Medium Risk: Snowfall Expected"
    elif pred == "No Precipitation":
        return "✅ Safe Weather: No Precipitation Expected"
    else:
        return "🔍 Uncertain: Monitor Closely"

stream = sensor_data_stream(X_test)
for i, row in enumerate(stream):
    pred = selected_model.predict(row)[0]
    if hasattr(selected_model, "predict_proba"):
        prob = np.max(selected_model.predict_proba(row))
    else:
        prob = 0.75  # fallback default
    signal = generate_ai_signal(pred, prob)
    st.success(f"Sample {i+1} → Prediction: `{pred}` | Confidence: `{prob:.2f}` | AI Signal: **{signal}**")
    if i == 4: break

# ----------- Section 6: Time Series Forecasting -----------
if show_forecast:
    st.subheader("📈 3-Day Precipitation Proxy Forecast (Humidity)")

    ts_data = pd.read_csv("https://raw.githubusercontent.com/Vishnunandan24/Weather-Precipitation-Classification/main/weatherHistory.csv")
    ts_data = ts_data[['Humidity']].dropna().head(200)

    model_arima = ARIMA(ts_data, order=(3, 1, 0))
    model_fit = model_arima.fit()
    forecast = model_fit.forecast(steps=3)

    st.write("### Forecasted Humidity Levels:")
    for i, val in enumerate(forecast, start=1):
        st.write(f"**Day {i}**: `{val:.4f}`")

    fig, ax = plt.subplots()
    ts_data.plot(ax=ax, legend=False, title="Humidity Trend (ARIMA Input)")
    st.pyplot(fig)
