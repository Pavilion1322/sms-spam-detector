"""
Trains the final tuned pipeline (TF-IDF + Logistic Regression) with the
hyperparameters selected by GridSearchCV in SMS_Spam_Detector_Final.ipynb
(data without duplicates), and saves it to disk so the API server can load
it without retraining.
"""
import html
import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "SMSSpamCollection"
MODEL_PATH = "spam_model.joblib"
RANDOM_STATE = 42


def load_data(path):
    labels, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, text = line.split("\t", 1)
            labels.append(1 if label == "spam" else 0)
            texts.append(text)
    return pd.DataFrame({"label": labels, "text": texts})


def clean_text(text):
    text = text.replace("&lt;#&gt;", " ")
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = load_data(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df.drop_duplicates("clean_text").reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=3)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=50, random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("Sanity check on held-out test set:")
    print(f"  accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  f1:        {f1_score(y_test, y_pred):.4f}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
