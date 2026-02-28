import csv
import random
import sys
from pathlib import Path
from collections import Counter

def compute_targets(label_counts: Counter, train=0.8, val=0.1, test=0.1):
    targets = {}
    for label, n in label_counts.items():
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        n_test = n - n_train - n_val
        targets[label] = {"train": n_train, "val": n_val, "test": n_test}
    return targets

def choose_split(remaining: dict, rng: random.Random):
    total = remaining["train"] + remaining["val"] + remaining["test"]
    if total <= 0:
        return None
    r = rng.randrange(total)
    if r < remaining["train"]:
        return "train"
    r -= remaining["train"]
    if r < remaining["val"]:
        return "val"
    return "test"

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_data.py <processed_csv> [seed]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 42
    out_dir = in_path.parent / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    # Pass 1: count labels
    label_counts = Counter()
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = row["type"]
            label_counts[label] += 1

    targets = compute_targets(label_counts, train=0.8, val=0.1, test=0.1)

    print("Label counts:", dict(label_counts))
    print("Targets per label:", targets)

    # Pass 2: stream rows to split files
    rng = random.Random(seed)

    with open(in_path, "r", encoding="utf-8", newline="") as f_in, \
         open(train_path, "w", encoding="utf-8", newline="") as f_train, \
         open(val_path, "w", encoding="utf-8", newline="") as f_val, \
         open(test_path, "w", encoding="utf-8", newline="") as f_test:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        w_train = csv.DictWriter(f_train, fieldnames=fieldnames)
        w_val = csv.DictWriter(f_val, fieldnames=fieldnames)
        w_test = csv.DictWriter(f_test, fieldnames=fieldnames)

        w_train.writeheader()
        w_val.writeheader()
        w_test.writeheader()

        written = Counter()

        for row in reader:
            label = row["type"]
            remaining = targets[label]

            split = choose_split(remaining, rng)
            if split is None:
                # should not happen, but safety fallback
                split = "train"

            if split == "train":
                w_train.writerow(row)
            elif split == "val":
                w_val.writerow(row)
            else:
                w_test.writerow(row)

            targets[label][split] -= 1
            written[split] += 1

    print("Wrote:", dict(written))
    print("Output directory:", out_dir)

if __name__ == "__main__":
    main()