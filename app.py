# Simple Flask UI to load the trained model and score input email text.
from flask import Flask, request, render_template, redirect, url_for, flash
import joblib
import os

MODEL_PATH = os.environ.get('SPAM_MODEL_PATH', 'models/spam_pipe.joblib')

app = Flask(__name__)
app.secret_key = 'dev-key-change-me'

# Load model lazily
model = None
def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
        model = joblib.load(MODEL_PATH)
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    prob = None
    label = None
    text = ''
    if request.method == 'POST':
        text = request.form.get('email_text','').strip()
        if not text:
            flash('Please paste email text to score.', 'warning')
            return redirect(url_for('index'))
        pipe = get_model()
        p = float(pipe.predict_proba([text])[0][1])
        prob = p
        label = 'SPAM' if p >= 0.7 else 'SUSPICIOUS' if p >= 0.5 else 'HAM'
    return render_template('index.html', prob=prob, label=label, text=text)

if __name__ == '__main__':
    # Use PORT env var if provided (helpful for deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
