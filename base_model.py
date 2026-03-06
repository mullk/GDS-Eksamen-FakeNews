import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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

train['binary_label'] = train['type'].map(labels)
val['binary_label'] = val['type'].map(labels)
test['binary_label'] = test['type'].map(labels)

train = train.dropna(subset=['binary_label'])
val = val.dropna(subset=['binary_label'])
test = test.dropna(subset=['binary_label'])

vectorizer = CountVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train['processed_content'].fillna(''))
X_val = vectorizer.transform(val['processed_content'].fillna(''))
X_test = vectorizer.transform(test['processed_content'].fillna(''))

# Træner modellen med vægtning for at håndtere ubalance:
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, train['binary_label'])

print("Resultater på Validation Set:")
print(classification_report(val['binary_label'], model.predict(X_val), zero_division=0))

print("Resultater på Test Set:")
print(classification_report(test['binary_label'], model.predict(X_test), zero_division=0))