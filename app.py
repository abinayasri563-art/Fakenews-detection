from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model (if available)
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except:
    model = None
    vectorizer = None

@app.route('/')
def home():
    return '''
    <h2>Fake News Detection</h2>
    <form method="POST" action="/predict">
        <textarea name="news" rows="5" cols="40"></textarea><br><br>
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']

    if model and vectorizer:
        data = vectorizer.transform([text])
        result = model.predict(data)[0]
    else:
        # fallback (for testing)
        result = "FAKE" if "fake" in text.lower() else "REAL"

    return f"<h3>Prediction: {result}</h3><a href='/'>Go Back</a>"

if __name__ == "__main__":
    app.run(debug=True)