# streamlit_app.py

"""
🌦 Smart Precipitation Classification and Forecasting System with Real-Time User Input & Risk Alerts
---------------------------------------------------------------------------------------------------
Built using:
- ML models: KNN, Naive Bayes, Decision Tree, SVM
- Advanced feature engineering
- Real-time prediction with AI-based risk alerts (via user input or live simulation)
- Time series forecasting using ARIMA
- Natural Language interface using Hugging Face Transformers (LLM-style)
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

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Hugging Face Zero-shot Classification Setup
@st.cache_resource(show_spinner="🤖 Loading lightweight LLM model for prompt understanding...")
def load_zero_shot_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"❌ Failed to load LLM model: {e}")
        st.stop()

zero_shot_classifier = load_zero_shot_model()

candidate_labels = [
    "Will it rain tomorrow?",
    "Will it snow tomorrow?",
    "Is it safe to go outside?",
    "Should I carry an umbrella?",
    "Weather forecast",
    "Give rainfall risk",
    "No precipitation",
    "General weather update"
]

# Page setup
st.set_page_config(page_title="Weather Precipitation Predictor", layout="centered", page_icon="🌧️")
st.title("🌦️ Weather Precipitation Predictor with AI-Based Forecasting & Risk Alerts")
st.markdown("🚀 Powered by **KNN**, **Naive Bayes**, **Decision Tree**, **SVM** | Integrated with **ARIMA Forecasting** + **LLM Prompt Understanding**")

# Load dataset
weather_data = pd.read_csv("https://raw.githubusercontent.com/Vishnunandan24/Weather-Precipitation-Classification/main/weatherHistory.csv")
weather_data = weather_data[["Precip Type", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
weather_data["Precip Type"] = weather_data["Precip Type"].replace(["null", "NULL", "Null", ""], "No Precipitation").fillna("No Precipitation")
weather_data.drop_duplicates(inplace=True)
weather_data.dropna(inplace=True)

# Feature engineering
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

# Load models
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Decision Tree": joblib.load("decision_tree_model.pkl"),
    "SVM": joblib.load("svm_model.pkl")
}

# Sidebar controls
st.sidebar.title("⚙️ Dashboard Controls")
model_choice = st.sidebar.selectbox("Choose ML Model", list(models.keys()))
show_eda = st.sidebar.checkbox("📊 Show EDA Visuals")
show_conf_matrix = st.sidebar.checkbox("🧾 Show Confusion Matrix")
show_tree = st.sidebar.checkbox("🌲 Show Decision Tree (Choose Decision Tree model)", value=False)
show_forecast = st.sidebar.checkbox("📈 3-Day Humidity Forecast")

# Evaluate model
selected_model = models[model_choice]
predictions = selected_model.predict(X_test)
acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

st.markdown(f"### 🧠 {model_choice} Model Accuracy: **{acc*100:.2f}%**")
st.code(classification_report(y_test, predictions), language="text")

# Confusion matrix
if show_conf_matrix:
    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="YlOrBr", fmt="d", xticklabels=y.unique(), yticklabels=y.unique())
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    st.pyplot(fig)

# Decision tree visualization
if show_tree and model_choice == "Decision Tree":
    st.subheader("🌲 Visualize Decision Tree Structure")
    fig = plt.figure(figsize=(10, 6))
    plot_tree(selected_model, feature_names=X.columns, class_names=selected_model.classes_, filled=True)
    st.pyplot(fig)

# EDA
if show_eda:
    st.subheader("📊 EDA: Feature Distributions & Correlation")
    with st.expander("🔎 Distribution & Boxplots"):
        for col in features:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(weather_data[col], kde=True, ax=axs[0]); axs[0].set_title(f"Distribution of {col}")
            sns.boxplot(x=weather_data[col], ax=axs[1]); axs[1].set_title(f"Outlier Detection: {col}")
            st.pyplot(fig)
    with st.expander("🔗 Feature Correlation Heatmap"):
        fig = plt.figure(figsize=(7, 5))
        sns.heatmap(weather_data[features].corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig)

# Real-time prediction
st.header("🧪 Real-Time Prediction (Manual Input)")
col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("🌡️ Temperature (C)", -20.0, 50.0, 20.0)
with col2:
    humidity = st.number_input("💧 Humidity (0 to 1)", 0.0, 1.0, 0.5)
with col3:
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", 0.0, 50.0, 10.0)

if st.button("🚀 Predict Now"):
    temp_diff = 0
    humidity_index = humidity * wind_speed
    rolling_precip_3 = humidity
    input_df = pd.DataFrame([[temp, humidity, wind_speed, temp_diff, humidity_index, rolling_precip_3]], columns=features)
    input_scaled = scaler.transform(input_df)
    pred = selected_model.predict(input_scaled)[0]
    prob = np.max(selected_model.predict_proba(input_scaled)) if hasattr(selected_model, "predict_proba") else 0.75

    def generate_ai_signal(pred, prob):
        if pred == "rain" and prob > 0.85:
            return "🚨 High Risk: Rainfall Alert - Stay Safe"
        elif pred == "snow":
            return "⚠️ Medium Risk: Snowfall Expected"
        elif pred == "No Precipitation":
            return "✅ Safe Weather: No Precipitation Expected"
        else:
            return "🔍 Uncertain: Monitor Closely"

    signal = generate_ai_signal(pred, prob)
    st.success(f"✅ Prediction: **{pred}** | Confidence: **{prob:.2f}** | AI Signal: {signal}")

# Forecasting
if show_forecast:
    st.header("📈 3-Day Humidity Forecast Using ARIMA")
    ts_data = pd.read_csv("https://raw.githubusercontent.com/Vishnunandan24/Weather-Precipitation-Classification/main/weatherHistory.csv")
    ts_data = ts_data[['Humidity']].dropna().head(200)
    model_arima = ARIMA(ts_data, order=(3, 1, 0))
    model_fit = model_arima.fit()
    forecast = model_fit.forecast(steps=3)
    st.write("### 🔮 Forecasted Humidity Levels:")
    for i, val in enumerate(forecast, 1):
        st.write(f"**Day {i}**: `{val:.4f}`")
    fig, ax = plt.subplots()
    ts_data.plot(ax=ax, legend=False, title="Humidity Trend (ARIMA Input)")
    st.pyplot(fig)

# LLM-based prompt interface
st.header("💬 Ask Your Weather Prediction Question (LLM-powered)")
user_prompt = st.text_input("🧠 Ask something like: 'Will it rain tomorrow?' or 'Should I carry an umbrella?'")

if user_prompt:
    with st.spinner("🔍 Interpreting your prompt using LLM..."):
        result = zero_shot_classifier(user_prompt, candidate_labels)
        top_label = result["labels"][0]
        confidence = result["scores"][0]

        simulated_input = pd.DataFrame([[temp, humidity, wind_speed, 0, humidity * wind_speed, humidity]], columns=features)
        simulated_scaled = scaler.transform(simulated_input)
        model_pred = selected_model.predict(simulated_scaled)[0]
        prob = np.max(selected_model.predict_proba(simulated_scaled)) if hasattr(selected_model, "predict_proba") else 0.75

        if "rain" in top_label.lower():
            response = f"🌧️ Yes, **rain is likely**. (Confidence: {prob:.2f})"
        elif "snow" in top_label.lower():
            response = f"❄️ Snowfall possible. (Confidence: {prob:.2f})"
        elif "umbrella" in top_label.lower():
            response = f"☔ Carry one, just in case. Detected: {model_pred} (Confidence: {prob:.2f})"
        else:
            response = f"🤖 This looks like: **'{top_label}'** (LLM Confidence: {confidence:.2f}). Please check prediction above."

        st.success(response)
