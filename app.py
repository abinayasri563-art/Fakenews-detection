import os
import pandas as pd
import string
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize app
app = Flask(__name__)

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -----------------------------
# LOAD DATA (only once)
# -----------------------------
data = pd.read_csv("news.csv")

# Clean text
data['clean_text'] = data['text'].apply(clean_text)

# Features and labels
X = data['clean_text']
y = data['label']

# TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

print("✅ Model trained successfully")

# -----------------------------
# HOME ROUTE (IMPORTANT)
# -----------------------------
@app.route('/')
def home():
    return "🚀 Fake News Detection is Live!"

# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get("text")

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prediction = model.predict(vec)[0]

    return jsonify({"prediction": prediction})

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)