import re
import csv
from pathlib import Path
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

csv.field_size_limit(2**31-1)

tokenizer = RegexpTokenizer(r"<URL>|<NUM>|<DATE>|[a-z]+(?:'[a-z]+)?")
stemmer = SnowballStemmer("english")

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

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


def tokenize_nltk(cleaned_text: str):
    return tokenizer.tokenize(cleaned_text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def stem_tokens(tokens):
    out = []
    for t in tokens:
        if t in ("<URL>", "<NUM>", "<DATE>"):
            out.append(t)
        else:
            out.append(stemmer.stem(t))
    return out

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize_nltk(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    tokens_stem = stem_tokens(tokens_no_stop)
    return " ".join(tokens_stem)


def reduction_rate(before, after):
    # reduction rate = (før - efter) / før
    # fx før=1000, efter=600 => (1000-600)/1000 = 0.4 = 40%
    if before == 0:
        return 0.0
    return (before - after) / before

def main():
    #file path
    BASE_DIR = Path(__file__).resolve().parent
    if len(sys.argv) < 2:
        print("Usage: python Data_Processing.py <path_to_csv> <optional row limit>" )
        sys.exit(1)

    limit_rows = int(sys.argv[2]) if len(sys.argv) >= 3 else None

    file_path = Path(sys.argv[1])
    out_path = file_path.with_name("processed_" + file_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
        

    print("Using file:", file_path)
    print("Exists?", file_path.exists())
    if not file_path.exists():
        raise FileNotFoundError(f"Kan ikke finde filen: {file_path}\n")

    vocab_raw = set()
    vocab_no_stop = set()
    vocab_stemmed = set()

    total_raw = 0
    total_no_stop = 0

    docs_read = 0

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f_in, \
        open(out_path, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["type", "processed_content"])
        writer.writeheader()


        for row in reader:
            if limit_rows is not None and docs_read >= limit_rows:
                break
            docs_read += 1

            raw_content = row.get("content", "") or ""
            cleaned = clean_text(raw_content)

            tokens = tokenize_nltk(cleaned)
            tokens_no_stop = remove_stopwords(tokens)
            tokens_stem = stem_tokens(tokens_no_stop)

            total_raw += len(tokens)
            total_no_stop += len(tokens_no_stop)

            vocab_raw.update(tokens)
            vocab_no_stop.update(tokens_no_stop)
            vocab_stemmed.update(tokens_stem)

            writer.writerow({"type": row.get("type"), "processed_content": " ".join(tokens_stem)})

            if docs_read % 50_000 == 0:
                print(f"Processed {docs_read} documents...")

    print(f"Processed data saved to: {out_path}")

    #printeren
    V_raw = len(vocab_raw)
    V_no_stop = len(vocab_no_stop)
    V_stem = len(vocab_stemmed)

    rr_stop = reduction_rate(V_raw, V_no_stop)
    rr_stem = reduction_rate(V_no_stop, V_stem)
    rr_stem_from_raw = reduction_rate(V_raw, V_stem)

    token_rr_stop = (total_raw - total_no_stop) / total_raw if total_raw else 0

    print(f"\nSample size (documents): {docs_read:,}")
    print(f"Vocabulary size (tokenized): {V_raw:,}")
    print(f"Vocabulary size (after stopwords): {V_no_stop:,}")
    print(f"Reduction rate (stopwords): {rr_stop:.4f} = {rr_stop*100:.2f}%")
    print(f"Vocabulary size (after stemming): {V_stem:,}")
    print(f"Reduction rate (stemming): {rr_stem:.4f} = {rr_stem*100:.2f}%")
    print(f"Reduction rate (stemming vs raw): {rr_stem_from_raw:.4f} = {rr_stem_from_raw*100:.2f}%")
    print(f"Total tokens (raw): {total_raw:,}")
    print(f"Total tokens (after stopwords): {total_no_stop:,}")
    print(f"Token reduction (stopwords): {token_rr_stop:.4f} = {token_rr_stop*100:.2f}%")

if __name__ == "__main__":
    main()