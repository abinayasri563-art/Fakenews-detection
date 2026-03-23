from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Fake News Detection Working!"

@app.route('/predict', methods=['POST'])
def predict():
    return "Prediction: FAKE"