"""Train Naive Bayes + logistic regression sentiment models on NLTK movie_reviews and export for the browser.

    pip install nltk scikit-learn numpy
    python -c "import nltk; nltk.download('movie_reviews'); nltk.download('vader_lexicon')"
    python training/sentiment_train.py lab/sentiment/data

Writes <out>/models.json (shared vocab, NB log-probabilities, LR weights, held-out metrics)
and <out>/vader.json (VADER lexicon: word -> mean valence).
The browser tokenizer must match TOKEN_RE below.
"""
import json, os, re, sys
import numpy as np
from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

OUT = sys.argv[1]
TOKEN_RE = r"[a-z]+(?:'[a-z]+)?"

docs, labels = [], []
for fid in movie_reviews.fileids():
    docs.append(" ".join(movie_reviews.words(fid)).lower().replace(" ' ", "'"))
    labels.append(1 if movie_reviews.categories(fid)[0] == "pos" else 0)
y = np.array(labels)
Xtr_t, Xte_t, ytr, yte = train_test_split(docs, y, test_size=0.2, random_state=42, stratify=y)

vec = CountVectorizer(token_pattern=TOKEN_RE, min_df=5, max_features=8000)
Xtr = vec.fit_transform(Xtr_t); Xte = vec.transform(Xte_t)
vocab = vec.get_feature_names_out().tolist()

nb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
nb_acc = float((nb.predict(Xte) == yte).mean())

Xtr_b = (Xtr > 0).astype(np.float64); Xte_b = (Xte > 0).astype(np.float64)
best = None
for C in [0.03, 0.1, 0.3, 1.0]:  # pick C on a slice of the training set, not the test set
    a, b, ya, yb = train_test_split(Xtr_b, ytr, test_size=0.2, random_state=0, stratify=ytr)
    acc = (LogisticRegression(C=C, max_iter=2000).fit(a, ya).predict(b) == yb).mean()
    if best is None or acc > best[1]: best = (C, acc)
lr = LogisticRegression(C=best[0], max_iter=2000).fit(Xtr_b, ytr)
lr_acc = float((lr.predict(Xte_b) == yte).mean())

sia = SentimentIntensityAnalyzer()
vader_acc = float(np.mean([(sia.polarity_scores(t)["compound"] >= 0) == bool(l) for t, l in zip(Xte_t, yte)]))

# class order in sklearn: [neg(0), pos(1)]
lp = nb.feature_log_prob_
models = {
    "token_re": TOKEN_RE,
    "vocab": vocab,
    "nb": {"prior": [round(float(v), 5) for v in nb.class_log_prior_],
           "logp_neg": [round(float(v), 4) for v in lp[0]], "logp_pos": [round(float(v), 4) for v in lp[1]]},
    "lr": {"bias": round(float(lr.intercept_[0]), 5), "w": [round(float(v), 4) for v in lr.coef_[0]], "C": best[0]},
    "metrics": {"train_docs": int(len(ytr)), "test_docs": int(len(yte)), "nb_acc": round(nb_acc, 4),
                "lr_acc": round(lr_acc, 4), "vader_acc": round(vader_acc, 4), "vocab_size": len(vocab)},
}
os.makedirs(OUT, exist_ok=True)
json.dump(models, open(os.path.join(OUT, "models.json"), "w"), separators=(",", ":"))
json.dump({w: round(v, 2) for w, v in sia.lexicon.items()}, open(os.path.join(OUT, "vader.json"), "w"), separators=(",", ":"))
print(models["metrics"], "C=", best[0])
