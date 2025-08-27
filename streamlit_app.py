import streamlit as st
import joblib
import os
import io
import math
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import pickle

# --- MUST be first Streamlit command ---
st.set_page_config(page_title="PDF Spam Classifier", layout="centered")

# ---------- Config ----------
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
BLACKLIST_PATH = "blacklist.pkl"
# ----------------------------

# Load model/vectorizer if available
@st.cache_resource
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
def clean_extracted_text(text: str) -> str:
    """Fix line breaks after every word while preserving paragraphs."""
    text = re.sub(r"-\n", "", text)      # handle hyphenated breaks
    text = re.sub(r"\n+", " ", text)     # convert multiple newlines to space
    text = re.sub(r"\s+", " ", text)     # normalize whitespace
    return text.strip()

def extract_text_from_pdf(file_stream):
    try:
        reader = PdfReader(file_stream)
        text_parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text()
            except Exception:
                txt = None
            if txt:
                text_parts.append(txt)
        raw_text = "\n".join(text_parts).strip()
        return clean_extracted_text(raw_text)
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

# ---------- Streamlit UI ----------
st.title("📄 PDF Spam Classifier")
st.markdown("Upload one or more PDF emails and detect if they're **Spam or Not Spam**")

uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    spam_count = 0
    ham_count = 0
    file_spam_counts = {}

    for idx, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if not text:
            st.warning(f"❌ Could not extract text from **{file.name}**")
            continue

        subj_match = re.search(r"Subject\s*:\s*(.+)", text, re.IGNORECASE)
        if subj_match:
            subject = subj_match.group(1).strip().splitlines()[0]
        else:
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            subject = first_line[:80] if first_line else "Document"

        is_spam, confidence, suspicious_terms = predict_with_model(text)

        if is_spam:
            spam_count += 1
            file_spam_counts[file.name] = file_spam_counts.get(file.name, 0) + 1
        else:
            ham_count += 1
            file_spam_counts[file.name] = file_spam_counts.get(file.name, 0)

        results.append({
            "id": idx + 1,
            "filename": file.name,
            "subject": subject,
            "content": text[:800],
            "is_spam": is_spam,
            "confidence": confidence,
            "suspicious_terms": suspicious_terms
        })

    # ---- Display Summary ----
    st.subheader("📊 Batch Summary")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(3, 3))  # compact pie chart
        ax.pie([spam_count, ham_count],
               labels=["Spam", "Not Spam"],
               autopct="%1.1f%%",
               colors=["#ef4444", "#10b981"])
        ax.set_title("Spam vs Not Spam")
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # compact bar chart
        sns.barplot(x=list(file_spam_counts.keys()), y=list(file_spam_counts.values()), ax=ax, palette="pastel")
        ax.set_ylabel("Spam Count")
        ax.set_xlabel("File Name")
        ax.set_title("Per-file Spam Count")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig, use_container_width=True)

    # ---- Display Per-file Results ----
    st.subheader("📧 Classification Results")
    for email in results:
        with st.expander(f"📄 {email['filename']} — {'🚨 SPAM' if email['is_spam'] else '✅ Not Spam'} ({int(email['confidence']*100)}%)"):
            st.markdown(f"**Subject:** {email['subject']}")

            # Highlight suspicious terms
            highlighted_content = email["content"]
            if email["suspicious_terms"]:
                for term in email["suspicious_terms"]:
                    regex = re.compile(f"({re.escape(term)})", re.IGNORECASE)
                    highlighted_content = regex.sub(r"**⚠️ \1**", highlighted_content)

            # Display cleaned, continuous text
            st.markdown(f"<div style='white-space: pre-wrap; text-align: justify;'>{highlighted_content}</div>", unsafe_allow_html=True)

            if email["suspicious_terms"]:
                st.markdown("**Suspicious terms:** " + ", ".join(email["suspicious_terms"]))
