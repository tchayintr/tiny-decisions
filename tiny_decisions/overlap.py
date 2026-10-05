"""Learning or memorising? The same splits, draws and seeds as finetune.py, scored three ways:

  examples                  accuracy on the k examples the model was fine-tuned on
  test                      accuracy on the test split (must reproduce finetune.py's result, to one test item)
  test_no_copy_in_examples  ... on test items with no near-copy among the k examples
  test_no_copy_in_train     ... on test items with no near-copy anywhere in the train split (SMS Spam repeats templates)

TF-IDF + logistic regression on the same examples for comparison.
"""
import json
import re
from pathlib import Path

import numpy as np

from . import baselines, checkpoints, finetune
from .config import DEVICE, FINETUNE_TASKS, KS, RUNS, SEEDS
from .decisions import draw, encode_items, eval_items, split
from .evaluate import predict

SUBSETS = ("test", "test_no_copy_in_examples", "test_no_copy_in_train")
OUT = RUNS / "overlap"


def copy_key(text: str) -> str:
    """Messages that differ only in case, digits or punctuation count as copies (spam templates)."""
    return re.sub(r"\W+", " ", re.sub(r"\d", "#", text.lower())).strip()


def masks(train: list[dict], test: list[dict], picked: list[int]) -> dict[str, np.ndarray]:
    """Which test items count in each subset."""
    in_train = {copy_key(x["context"]) for x in train}
    in_examples = {copy_key(train[i]["context"]) for i in picked}
    keys = [copy_key(x["context"]) for x in test]
    return {"test": np.ones(len(keys), bool),
            "test_no_copy_in_examples": np.array([k not in in_examples for k in keys]),
            "test_no_copy_in_train": np.array([k not in in_train for k in keys])}


def record(res: dict, method: str, k: int, examples_ok: np.ndarray, test_ok: np.ndarray, m: dict) -> None:
    r = res.setdefault(method, {}).setdefault(str(k), {"examples": [], **{s: [] for s in SUBSETS}})
    r["examples"].append(float(examples_ok.mean()))
    for s in SUBSETS:
        r[s].append(float(test_ok[m[s]].mean()))


def run(model: str, out: Path = OUT) -> dict:
    """Both methods on every task, k and seed → runs/overlap/<model>.json."""
    tok, sp = checkpoints.tokenizer(model)
    state, lr = checkpoints.weights(model, DEVICE), finetune.learning_rate(model)
    stored = json.loads((finetune.OUT / f"{model}.json").read_text())
    items, res = eval_items(), {}
    for task in FINETUNE_TASKS:
        train_raw, test_raw = split(items[task], task)
        train, test = encode_items(train_raw, tok, sp), encode_items(test_raw, tok, sp)
        n_clean = int(masks(train_raw, test_raw, [])["test_no_copy_in_train"].sum())
        r = {"n_test": len(test), "n_test_no_copy_in_train": n_clean, "copies_in_examples": {}}
        for k in [k for k in KS if k]:
            for seed in SEEDS:
                picked = draw(len(train), k, seed)
                m = masks(train_raw, test_raw, picked)
                r["copies_in_examples"].setdefault(str(k), []).append(int((~m["test_no_copy_in_examples"]).sum()))
                net = checkpoints.fresh(model, state, DEVICE)
                finetune.adapt(net, [train[i] for i in picked], DEVICE, seed, lr)
                test_ok = predict(net, test, DEVICE)[0]
                before = stored[task]["k"][str(k)][seed]   # bf16 GPU kernels are not bit-reproducible: allow one item
                assert abs(test_ok.mean() - before) <= 1 / len(test) + 1e-9, (task, k, seed, test_ok.mean(), before)
                record(r, "model", k, predict(net, [train[i] for i in picked], DEVICE)[0], test_ok, m)
                examples = [train_raw[i] for i in picked]
                record(r, "tfidf", k, baselines.correct(examples, examples), baselines.correct(examples, test_raw), m)
            for method in ("model", "tfidf"):
                v = r[method][str(k)]
                print(f"{task:9} k={k:4} {method:6} examples {np.mean(v['examples']):.3f} | test {np.mean(v['test']):.3f}"
                      f"  no copy in examples {np.mean(v['test_no_copy_in_examples']):.3f}"
                      f"  no copy in train split {np.mean(v['test_no_copy_in_train']):.3f} (n={n_clean})", flush=True)
        res[task] = r
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model}.json").write_text(json.dumps(res, indent=1))
    return res
