import streamlit as st
import nltk
import re
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# NLTK RESOURCE DOWNLOADS

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# PREPROCESSING FUNCTIONS

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(sentence):
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.lower()

def preprocess_sentences(sentences):
    processed = []
    for sentence in sentences:
        sentence = clean_text(sentence)
        words = word_tokenize(sentence)
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        processed.append(" ".join(words))
    return processed


# SUMMARIZATION LOGIC

def summarize_text(text, summary_ratio):
    sentences = sent_tokenize(text)

    if len(sentences) < 2:
        return text

    processed_sentences = preprocess_sentences(sentences)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    num_sentences = max(1, int(len(sentences) * summary_ratio))
    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices = sorted(top_indices)

    summary = " ".join([sentences[i] for i in top_indices])
    return summary


# STREAMLIT UI

st.set_page_config(page_title="Text Summarization System", layout="wide")

st.title("NLP-Based Text Summarization System")
st.write("Extractive text summarization using NLP and TF-IDF")

input_method = st.radio(
    "Choose Input Method:",
    ("Paste Text", "Upload Text File")
)

text = ""

if input_method == "Paste Text":
    text = st.text_area("Enter text to summarize:", height=250)

elif input_method == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

summary_ratio = st.slider(
    "Summary Length (% of original text):",
    min_value=10,
    max_value=70,
    value=30
) / 100

if st.button("Generate Summary"):
    if text.strip() == "":
        st.error("Please provide input text.")
    else:
        summary = summarize_text(text, summary_ratio)
        st.subheader("Summary Output")
        st.success(summary)
