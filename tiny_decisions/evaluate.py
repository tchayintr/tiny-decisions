"""Accuracy of a Decisions Model on the held-out tasks."""
import numpy as np
import torch

from .config import HELD_OUT
from .decisions import OURS, Specials, collate, encode_items, eval_items

BATCH = 64


@torch.no_grad()
def predict(model, examples: list, dev, batch: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """For (ids, markers, gold) examples: whether each choice was right, and its probability."""
    model.eval()
    correct, conf = [], []
    for i in range(0, len(examples), batch):
        x, mk, tgt = collate(examples[i:i + batch], dev, model.pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.decide(x, mk).softmax(-1)
        correct += (p.argmax(-1) == tgt).tolist()
        conf += p.max(-1).values.tolist()
    model.train()
    return np.array(correct), np.array(conf)


def accuracy(model, examples: list, dev) -> float:
    return float(predict(model, examples, dev, batch=128)[0].mean())


def evaluate(model, tok, dev, sp: Specials = OURS) -> dict[str, dict]:
    """Accuracy and chance level on every held-out task, with every answer offered."""
    items, res = eval_items(), {}
    for name in HELD_OUT:
        xs = items[name]
        correct, _ = predict(model, encode_items(xs, tok, sp), dev)
        options = len(xs[0]["options"])
        res[name] = {"acc": float(correct.mean()), "n": len(xs), "options": options, "chance": 1 / options}
        print(f"  {name:10} acc {res[name]['acc']:.3f}  (chance {1 / options:.3f})", flush=True)
    return res
