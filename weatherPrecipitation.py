# weatherPrecipitation.py

"""
Weather Precipitation Classification - Model Training Script
------------------------------------------------------------
This script:
- Loads & cleans data
- Trains 4 ML models
- Saves trained models as .pkl
- Prints accuracy and classification report in terminal
Note: All visualizations are handled in Streamlit (no plt.show())
"""

import pandas as pd
import numpy as np
import logging
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info("🚀 Starting Weather Precipitation ML training...")

# Load and preprocess dataset
weather_data = pd.read_csv("C:/Users/vishn/Desktop/weatherHistory.csv")
weather_data = weather_data[["Precip Type", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]]
weather_data["Precip Type"] = weather_data["Precip Type"].replace(["null", "NULL", "Null", ""], "No Precipitation").fillna("No Precipitation")
weather_data.drop_duplicates(inplace=True)
weather_data.dropna(inplace=True)

# Feature scaling
features = ["Temperature (C)", "Humidity", "Wind Speed (km/h)"]
scaler = StandardScaler()
weather_data[features] = scaler.fit_transform(weather_data[features])

# Prepare inputs and outputs
X = weather_data.drop("Precip Type", axis=1)
y = weather_data["Precip Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model training & saving
def train_and_save(model, name):
    logging.info(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    logging.info(f"{name} Accuracy: {acc * 100:.2f}%")
    print(f"\n{name} Classification Report:\n", classification_report(y_test, predictions,zero_division=1))
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

for name, model in models.items():
    train_and_save(model, name)

# Optional: Simulate prediction stream
def sensor_data_stream(df):
    for i in range(len(df)):
        yield df.iloc[i:i+1]

logging.info("⏱️ Simulating Real-Time Predictions (First 5 Samples)...")
stream = sensor_data_stream(X_test)
model = models["KNN"]  # Pick any trained model

for i, row in enumerate(stream):
    pred = model.predict(row)
    print(f"Sample {i+1} - Predicted: {pred[0]}")
    if i == 4:
        break

logging.info("✅ Training and saving complete. Streamlit dashboard will handle visualizations.")
