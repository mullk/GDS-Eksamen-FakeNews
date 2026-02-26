import re
import csv
from pathlib import Path

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def clean_text(text):
    # lowercase
    text = str(text).lower()

    # Dato -> <DATE>
    text = re.sub(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?!\.\w)"
        r"(?:\.(?=\s|$|\d))?"
        r"(?:\s+(?:0?[1-9]|[12][0-9]|3[01])"
        r"(?:\s*,?\s*(?:\d{2}|\d{4}))?"
        r")?"
        r"(?=\W|$)",
        "<DATE>",
        text
    )

    # Num -> <NUM>
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", "<NUM>", text)

    # URL -> <URL>
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"www\.\S+", "<URL>", text)
    text = re.sub(r"\S+\.com\S*", "<URL>", text)

    # normaliser whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text



tokenizer = RegexpTokenizer(r"<URL>|<NUM>|<DATE>|[a-z]+(?:'[a-z]+)?")

def tokenize_nltk(cleaned_text: str):
    return tokenizer.tokenize(cleaned_text)

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]



stemmer = SnowballStemmer("english")

def stem_tokens(tokens):
    out = []
    for t in tokens:
        if t in ("<URL>", "<NUM>", "<DATE>"):
            out.append(t)
        else:
            out.append(stemmer.stem(t))
    return out



def vocab_size(tokens):
    return len(set(tokens))

def reduction_rate(before, after):
    # reduction rate = (før - efter) / før
    # fx før=1000, efter=600 => (1000-600)/1000 = 0.4 = 40%
    if before == 0:
        return 0.0
    return (before - after) / before


#file path
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "Data" / "news_sample.csv"

print("Using file:", file_path)
print("Exists?", file_path.exists())

if not file_path.exists():
    raise FileNotFoundError(
        f"Kan ikke finde filen: {file_path}\n"
        "Tjek at mappen 'Data' ligger ved siden af scriptet, og at filnavnet er 'news_sample.csv'."
    )

# n styrrer hvor mange linjer vi prøver på
N = None  # sample størrelse. None for hele filen

all_tokens_raw = []
all_tokens_no_stop = []
all_tokens_stemmed = []

with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    docs_read = 0
    for i, row in enumerate(reader):
        if N is not None and i >= N:
            break
        docs_read += 1
        
        raw_content = row.get("content", "") or ""
        cleaned = clean_text(raw_content)

        tokens = tokenize_nltk(cleaned)
        tokens_no_stop = remove_stopwords(tokens)
        tokens_stem = stem_tokens(tokens_no_stop)

        all_tokens_raw.extend(tokens)
        all_tokens_no_stop.extend(tokens_no_stop)
        all_tokens_stemmed.extend(tokens_stem)

#printeren
V_raw = vocab_size(all_tokens_raw)
V_no_stop = vocab_size(all_tokens_no_stop)
V_stem = vocab_size(all_tokens_stemmed)

rr_stop = reduction_rate(V_raw, V_no_stop)
rr_stem = reduction_rate(V_no_stop, V_stem)

print(f"\nSample size (documents): {docs_read}")
print(f"Vocabulary size (tokenized): {V_raw}")
print(f"Vocabulary size (after stopwords): {V_no_stop}")
print(f"Reduction rate (stopwords): {rr_stop:.4f}  = {rr_stop*100:.2f}%")

print(f"Vocabulary size (after stemming): {V_stem}")
print(f"Reduction rate (stemming): {rr_stem:.4f}  = {rr_stem*100:.2f}%")

# (valgfrit) hvis du også vil se stemming-reduktion ift. originalt vocab:
rr_stem_from_raw = reduction_rate(V_raw, V_stem)
print(f"Reduction rate (stemming vs raw): {rr_stem_from_raw:.4f}  = {rr_stem_from_raw*100:.2f}%")



total_raw = len(all_tokens_raw)
total_no_stop = len(all_tokens_no_stop)
token_rr_stop = (total_raw - total_no_stop) / total_raw if total_raw else 0

print(f"Total tokens (raw): {total_raw}")
print(f"Total tokens (after stopwords): {total_no_stop}")
print(f"Token reduction (stopwords): {token_rr_stop:.4f} = {token_rr_stop*100:.2f}%")