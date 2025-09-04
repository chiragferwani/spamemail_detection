# Email Spam Detector - Demo (Python + Flask)

## What this project contains
- `train.py` - Train a TF-IDF + LogisticRegression pipeline on your CSV dataset.
  - Expects a CSV with columns: `text` and `label` (label: 1 for spam, 0 for ham)
  - Saves pipeline to `models/spam_pipe.joblib` by default.
- `app.py` - Simple Flask UI to paste a single email and get a spam probability.
- `templates/index.html` - Minimal UI.
- `requirements.txt` - Python dependencies.

## Quick start
1. Create a virtual environment and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train:
   ```bash
   python train.py --data path/to/your_emails.csv --outdir models
   ```
   CSV must have `text` and `label` columns.
3. Run the UI:
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:5000 and paste an email to score.

## Notes & next steps
- Tweak `TfidfVectorizer` settings in `train.py` (max_features, ngram_range) for better performance.
- To support on-device/in-browser inference later, you can:
  - Convert a Keras model to TFJS, or
  - Export vectorizer (vocab + idf) and weights separately and implement vectorization in JS.
- For production: add input sanitization, rate-limiting, and a privacy policy if sending data to a server.
