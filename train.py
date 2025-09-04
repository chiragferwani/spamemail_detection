import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import os

def load_data(path):
    df = pd.read_csv(path)
    # Map Category → label
    if "Category" in df.columns and "Message" in df.columns:
        df = df.rename(columns={"Category":"label", "Message":"text"})
        df["label"] = df["label"].map({"ham":0, "spam":1})
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    df = df.dropna(subset=['text','label'])
    return df['text'].astype(str).tolist(), df['label'].astype(int).tolist()


def build_and_train(X_train, y_train):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=400, solver='liblinear'))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def main():
    parser = argparse.ArgumentParser(description='Train spam detector')
    parser.add_argument('--data', required=True, help='Path to CSV file with columns text,label')
    parser.add_argument('--outdir', default='models', help='Directory to save model artifacts')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size fraction')
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model = build_and_train(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print("=== Evaluation on test set ===")
    print(classification_report(y_test, preds, digits=4))
    try:
        print("ROC AUC: %.4f" % roc_auc_score(y_test, probs))
    except Exception:
        pass
    print("Accuracy: %.4f" % accuracy_score(y_test, preds))

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(model, os.path.join(args.outdir, "spam_pipe.joblib"))
    print(f"Saved pipeline to {os.path.join(args.outdir, 'spam_pipe.joblib')}")
    print("Model artifacts saved. You can now run the Flask UI (app.py) to test single emails.")

if __name__ == '__main__':
    main()
