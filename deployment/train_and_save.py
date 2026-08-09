"""
Run this script on Kaggle after training to save model artifacts.
Upload the generated artifacts/ folder to your HuggingFace Space.

Usage (in Kaggle notebook, add a cell at the end):
    !python train_and_save.py
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

os.makedirs("artifacts", exist_ok=True)

OPTION_COLS = ["A", "B", "C", "D", "E"]

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv("/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv").fillna("")


# ── Pairwise features ────────────────────────────────────────────────────────
def make_pairwise(df, is_train=True):
    rows = []
    for _, r in df.iterrows():
        qid    = r["id"]
        prompt = str(r["prompt"])
        p_words = set(re.findall(r"\w+", prompt.lower()))
        ans    = r.get("answer", None)
        for c in OPTION_COLS:
            opt_str = str(r[c])
            o_words = set(re.findall(r"\w+", opt_str.lower()))
            overlap   = len(p_words & o_words)
            jaccard   = overlap / max(len(p_words | o_words), 1)
            len_ratio = len(opt_str) / max(len(prompt), 1)
            row = {
                "qid": qid, "option": c,
                "text": f"{prompt} [SEP] {opt_str}",
                "overlap": overlap, "jaccard": jaccard, "len_ratio": len_ratio,
            }
            if is_train:
                row["label"] = 1 if ans == c else 0
            rows.append(row)
    return pd.DataFrame(rows)

pw_train = make_pairwise(train, is_train=True)

# ── TF-IDF ───────────────────────────────────────────────────────────────────
print("Fitting TF-IDF vectorizers on full train corpus...")
tfidf_word = TfidfVectorizer(max_features=80000, ngram_range=(1,3), sublinear_tf=True, analyzer="word")
tfidf_char = TfidfVectorizer(max_features=50000, ngram_range=(2,5), sublinear_tf=True, analyzer="char_wb")

tfidf_word.fit(pw_train["text"])
tfidf_char.fit(pw_train["text"])

Xw = tfidf_word.transform(pw_train["text"])
Xc = tfidf_char.transform(pw_train["text"])
Xn = csr_matrix(pw_train[["overlap","jaccard","len_ratio"]].values)
X  = hstack([Xw, Xc, Xn])
y  = pw_train["label"].values

# ── Train final models on FULL train data ────────────────────────────────────
print("Training LR on full data...")
lr = LogisticRegression(C=6.0, max_iter=2000)
lr.fit(X, y)

print("Training Ridge on full data...")
ridge = RidgeClassifier(alpha=1.0)
ridge.fit(X, y)

print("Training SGD on full data...")
sgd = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
sgd.fit(X, y)

# ── Save artifacts ───────────────────────────────────────────────────────────
with open("artifacts/tfidf_word.pkl", "wb") as f:
    pickle.dump(tfidf_word, f)

with open("artifacts/tfidf_char.pkl", "wb") as f:
    pickle.dump(tfidf_char, f)

with open("artifacts/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

with open("artifacts/ridge_model.pkl", "wb") as f:
    pickle.dump(ridge, f)

with open("artifacts/sgd_model.pkl", "wb") as f:
    pickle.dump(sgd, f)

print("\nAll artifacts saved to artifacts/ folder:")
for fname in os.listdir("artifacts"):
    size = os.path.getsize(f"artifacts/{fname}") / 1e6
    print(f"  {fname}: {size:.1f} MB")

print("\nDone! Upload the artifacts/ folder to your HuggingFace Space.")
