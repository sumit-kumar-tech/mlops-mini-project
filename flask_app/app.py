import pickle
from pathlib import Path
import re
import string

import numpy as np
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with",
    }
LEMMATIZER = WordNetLemmatizer()

def lemmatization(text):
    """Lemmatize the text."""
    text = text.split()
    try:
        text = [LEMMATIZER.lemmatize(word) for word in text]
    except LookupError:
        pass
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    text = [word for word in str(text).split() if word not in STOP_WORDS]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


app = Flask(__name__)

with open(MODEL_DIR / "model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(MODEL_DIR / "vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template("index.html", result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    features_array = features.toarray()
    result = int(model.predict(features_array)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features_array)[0]) * 100)

    message = None
    if features.nnz == 0:
        message = "No known words found in the model vocabulary. Try a longer sentence."

    # show
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        cleaned_text=text,
        matched_words=features.nnz,
        message=message,
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", use_reloader=False,port=5001)
