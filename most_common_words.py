from collections import Counter
import csv

freq = Counter()

with open("data/processed_995,000_rows.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        freq.update(row["processed_content"].split())

print(freq.most_common(50))