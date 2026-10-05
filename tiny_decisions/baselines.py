"""TF-IDF + logistic regression, trained on the same k examples as the fine-tuned models (same split, draws and test
split): the classic text classifier a Decisions Model has to beat.

It only knows the answers it saw among the k examples; a Decisions Model can also pick an answer it never saw, by
reading it.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from .config import FINETUNE_TASKS, KS, RUNS, SEEDS
from .decisions import draw, eval_items, split

C = 10                  # inverse regularization strength
OUT = RUNS / "tfidf.json"


def label(x: dict) -> str:
    return x["options"][x["gold"]]


def classifier():
    """Word 1-2-grams and character 2-5-grams, TF-IDF weighted, into logistic regression."""
    words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
    return make_pipeline(make_union(words, chars), LogisticRegression(C=C, max_iter=5000))


def correct(train: list[dict], test: list[dict]) -> np.ndarray:
    """Whether the classifier trained on `train` picks each test decision's label among the answers it offers."""
    y = [label(x) for x in train]
    if len(set(y)) == 1:  # every example has the same answer: all a classifier can do is repeat it
        pred = [y[0] if y[0] in x["options"] else x["options"][0] for x in test]
    else:
        clf = classifier().fit([x["context"] for x in train], y)
        proba = clf.predict_proba([x["context"] for x in test])
        col = {c: i for i, c in enumerate(clf.classes_)}
        pred = [max(x["options"], key=lambda o: proba[j, col[o]] if o in col else -1.0) for j, x in enumerate(test)]
    return np.array([p == label(x) for p, x in zip(pred, test)])


def run(out: Path = OUT) -> dict:
    """Test accuracy per task, k and seed → runs/tfidf.json."""
    items, res = eval_items(), {}
    for task in FINETUNE_TASKS:
        train, test = split(items[task], task)
        res[task] = {"n_test": len(test), "k": {}}
        for k in [k for k in KS if 0 < k <= len(train)]:
            accs = [float(correct([train[i] for i in draw(len(train), k, seed)], test).mean()) for seed in SEEDS]
            res[task]["k"][str(k)] = accs
            print(f"{task:9} k={k:4}  acc {np.mean(accs):.3f}  (seeds: {[round(a, 3) for a in accs]})", flush=True)
    out.write_text(json.dumps(res, indent=1))
    return res
