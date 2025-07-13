# streamlit_app.py

"""
Weather Precipitation Classification Dashboard using Streamlit
------------------------------------------------
Interactive visualization for:
- EDA
- ML model performance
- Confusion matrices
- Decision Tree
- Real-time simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# Page settings
st.set_page_config(page_title="Weather Precipitation ML Dashboard", layout="wide", page_icon="🌦")

st.title("🌧 Weather Precipitation Classification Dashboard")
st.markdown("Built using **KNN**, **Naive Bayes**, **Decision Tree**, and **SVM** models.")

# Load & preprocess data
# weather_data = pd.read_csv("C:/Users/vishn/Desktop/weatherHistory.csv")
weather_data = pd.read_csv("https://raw.githubusercontent.com/Vishnunandan24/Weather-Precipitation-Classification/main/weatherHistory.csv")
weather_data = weather_data[["Precip Type", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
weather_data["Precip Type"] = weather_data["Precip Type"].replace(["null", "NULL", "Null", ""], "No Precipitation").fillna("No Precipitation")
weather_data.drop_duplicates(inplace=True)
weather_data.dropna(inplace=True)

features = ["Temperature (C)", "Humidity", "Wind Speed (km/h)"]
scaler = StandardScaler()
weather_data[features] = scaler.fit_transform(weather_data[features])

X = weather_data.drop("Precip Type", axis=1)
y = weather_data["Precip Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Sidebar Controls
st.sidebar.header("⚙ Controls")
model_choice = st.sidebar.selectbox("Choose ML Model", ("KNN", "Naive Bayes", "Decision Tree", "SVM"))
show_eda = st.sidebar.checkbox("Show EDA Plots", value=True)
show_conf_matrix = st.sidebar.checkbox("Show Confusion Matrix")
show_tree = st.sidebar.checkbox("Show Decision Tree (DT only)")

# Load model
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

# Section 1: EDA
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

# Section 2: Model Performance
st.subheader(f"🧠 {model_choice} Model Performance")
st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")
st.code(classification_report(y_test, predictions), language="text")

# Section 3: Confusion Matrix
if show_conf_matrix:
    st.subheader("🧾 Confusion Matrix")
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=y.unique(), yticklabels=y.unique())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# Section 4: Visualize Decision Tree
if show_tree and model_choice == "Decision Tree":
    st.subheader("🌲 Visualize Decision Tree Structure")
    fig = plt.figure(figsize=(12, 8))
    plot_tree(selected_model, feature_names=X.columns, class_names=selected_model.classes_, filled=True)
    st.pyplot(fig)

# Section 5: Simulate Real-Time Sensor Stream
st.subheader("⏱ Real-Time Sensor Prediction Simulation (First 5 Samples)")

def sensor_data_stream(df):
    for i in range(len(df)):
        yield df.iloc[i:i+1]

stream = sensor_data_stream(X_test)
for i, row in enumerate(stream):
    pred = selected_model.predict(row)
    st.success(f"Sample {i+1} → *Predicted:* {pred[0]}")
    if i == 4: break