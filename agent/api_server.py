"""
Lightweight HTTP API exposing the trained SMS spam classifier.
Consumed by the n8n orchestration workflow (action/perception layer).
"""
import html
import re
import string

import joblib
from flask import Flask, request, jsonify

MODEL_PATH = "spam_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def clean_text(text):
    text = text.replace("&lt;#&gt;", " ")
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Missing 'text' field"}), 400

    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    spam_probability = float(proba[1])
    label = "spam" if spam_probability >= 0.5 else "ham"

    return jsonify({
        "text": text,
        "label": label,
        "spam_probability": round(spam_probability, 4),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
