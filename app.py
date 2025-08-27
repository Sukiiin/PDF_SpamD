# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import joblib
import math
import re

try:
    # PyPDF2 v3+
    from PyPDF2 import PdfReader
except Exception:
    # fallback
    from PyPDF2 import PdfFileReader as PdfReader

import pickle

app = Flask(__name__)
CORS(app)

# ---------- Config ----------
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
BLACKLIST_PATH = "blacklist.pkl"
# ----------------------------

# Load model/vectorizer if available
def try_load_pickle(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

model = try_load_pickle(MODEL_PATH)
vectorizer = try_load_pickle(VECTORIZER_PATH)

blacklist = try_load_pickle(BLACKLIST_PATH)
if blacklist is None:
    blacklist = [
        "lottery", "winner", "prize", "urgent", "click here", "free", "buy now",
        "limited time", "act now", "account suspended", "password", "verify"
    ]

# ---------- PDF text extraction ----------
def extract_text_from_pdf(file_stream):
    try:
        if hasattr(file_stream, "read"):
            file_bytes = io.BytesIO(file_stream.read())
        else:
            file_bytes = io.BytesIO(file_stream)

        reader = PdfReader(file_bytes)
        text_parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
            except Exception:
                txt = None
            if txt:
                text_parts.append(txt)
        return "\n".join(text_parts).strip()
    except Exception:
        return ""

# ---------- Simple blacklist-based classifier ----------
def blacklist_classifier(text):
    text_l = text.lower()
    matches = [word for word in blacklist if word.lower() in text_l]
    if not matches:
        return False, 0.05, []
    conf = len(matches) / max(len(blacklist), 1)
    conf = 0.5 + 0.49 * min(1.0, conf * 2)
    return True, round(conf, 2), matches

# ---------- Utility: classification using ML model if available ----------
def predict_with_model(text):
    if not model or not vectorizer:
        return blacklist_classifier(text)

    try:
        X = vectorizer.transform([text])
    except Exception:
        return blacklist_classifier(text)

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
            spam_idx = classes.index(1) if 1 in classes else len(classes) - 1

            spam_prob = float(probs[spam_idx])
            prediction = model.predict(X)[0]
            is_spam = bool(prediction == 1 or str(prediction).lower() == "spam")
        else:
            if hasattr(model, "decision_function"):
                score = model.decision_function(X)[0]
                spam_prob = 1 / (1 + math.exp(-score))
            else:
                spam_prob = 0.75
            prediction = model.predict(X)[0]
            is_spam = bool(prediction == 1 or str(prediction).lower() == "spam")

        suspicious_terms = []
        if is_spam:
            text_l = text.lower()
            suspicious_terms = [w for w in blacklist if w.lower() in text_l]

        return is_spam, round(spam_prob, 2), suspicious_terms

    except Exception:
        return blacklist_classifier(text)

# ---------- Endpoint ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No selected files"}), 400

    results = []
    any_spam = False

    for idx, file in enumerate(uploaded_files):
        extracted_text = extract_text_from_pdf(file.stream)
        if not extracted_text:
            results.append({
                "id": idx + 1,
                "filename": file.filename,
                "subject": "Unknown",
                "content": "Could not extract text from PDF",
                "is_spam": False,
                "confidence": 0.0,
                "suspicious_terms": []
            })
            continue

        subj_match = re.search(r"Subject\s*:\s*(.+)", extracted_text, re.IGNORECASE)
        if subj_match:
            subject = subj_match.group(1).strip().splitlines()[0]
        else:
            first_line = next((ln.strip() for ln in extracted_text.splitlines() if ln.strip()), "")
            subject = first_line[:80] if first_line else "Document"

        is_spam, confidence, suspicious_terms = predict_with_model(extracted_text)
        if is_spam:
            any_spam = True

        results.append({
            "id": idx + 1,
            "filename": file.filename,
            "subject": subject,
            "content": extracted_text[:1000],
            "is_spam": bool(is_spam),
            "confidence": float(confidence),
            "suspicious_terms": suspicious_terms
        })

    response = {
        "overall": "SPAM" if any_spam else "Non-SPAM",
        "emails": results
    }

    return jsonify(response), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(model and vectorizer),
        "blacklist_loaded": bool(blacklist)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
