import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from pathlib import Path

splits_dir = Path("Data/splits")

# Indlæser filer
train = pd.read_csv(splits_dir / "train.csv")
val = pd.read_csv(splits_dir / "val.csv")
test = pd.read_csv(splits_dir / "test.csv")

# Mapping af labels:
labels = {
    'reliable': 'reliable', 'political': 'reliable',
    'fake': 'fake', 'unreliable': 'fake', 'conspiracy': 'fake',
    'bias': 'fake', 'hate': 'fake', 'clickbait': 'fake'
}

for df in [train, val, test]:
    df["binary_label"] = df["type"].map(labels)

train = train.dropna(subset=['binary_label', "processed_content"])
val = val.dropna(subset=['binary_label', "processed_content"])
test = test.dropna(subset=['binary_label', "processed_content"])


# Sets up model candidates, however, since we have strict requirements in part 2 only one is used
candidates = [
    {
        "name": "CountVectorizer_10k_C1",
        "vectorizer": CountVectorizer(max_features=10000),
        "model": LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    },
    # {
    #     "name": "CountVectorizer_20k_C1",
    #     "vectorizer": CountVectorizer(max_features=20000),
    #     "model": LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0) 
    # }
]

best_name = None
best_vectorizer = None
best_model = None
best_val_f1 = -1


# Finds best model from val data
for candidate in candidates:
    vectorizer = candidate["vectorizer"]
    model = candidate["model"]

    X_train = vectorizer.fit_transform(train["processed_content"])
    X_val = vectorizer.transform(val["processed_content"])

    model.fit(X_train, train["binary_label"])
    val_preds = model.predict(X_val)

    val_f1 = f1_score(val["binary_label"], val_preds, pos_label="fake")

    print(f"\nValidation results for {candidate['name']}:")
    print(classification_report(val["binary_label"], val_preds, zero_division=0))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_name = candidate["name"]
        best_vectorizer = vectorizer
        best_model = model

print(f"\nBest model chosen on validation set: {best_name}")
print(f"Best validation F1 (fake): {best_val_f1:.4f}")


# Runs the best model on the test data
X_test = best_vectorizer.transform(test['processed_content'])
test_preds = best_model.predict(X_test)

print("Final results on test set:")
print(classification_report(test['binary_label'], test_preds, zero_division=0))