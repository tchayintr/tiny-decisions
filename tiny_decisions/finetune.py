"""Fine-tuning on k examples of a held-out task, then accuracy on that task's test split.

Each held-out task is split once in half (decisions.split): the k examples are drawn from the train split, the test
split stays fixed. Every k > 0 runs with three draws of examples (seeds 0, 1, 2); k = 0 is the model as decision
training left it (zero-shot).
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import checkpoints
from .config import DEVICE, ETTIN, FINETUNE_TASKS, KS, RUNS, SEEDS
from .decisions import collate, draw, encode_items, eval_items, split
from .evaluate import accuracy

LR_SCRATCH = 3e-4   # our from-scratch models
LR_ETTIN = 5e-5     # Ettin encoders: the learning rate of their decision training
EPOCHS = 10         # passes over the k examples
BATCH = 16
OUT = RUNS / "finetune"


def learning_rate(model: str) -> float:
    return LR_ETTIN if model in ETTIN else LR_SCRATCH


def adapt(model, train: list, dev, seed: int, lr: float, epochs: int = EPOCHS) -> None:
    """Fine-tune on the k (ids, markers, gold) examples."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bs = min(BATCH, len(train))
    model.train()
    for _ in range(math.ceil(epochs * len(train) / bs)):
        x, mk, tgt = collate(rng.sample(train, bs), dev, model.pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model.decide(x, mk), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def run(model: str, tasks: tuple[str, ...] = FINETUNE_TASKS, ks: tuple[int, ...] = KS, out: Path = OUT) -> dict:
    """Test accuracy after fine-tuning on k examples, per task, k and seed → <out>/<model>.json."""
    tok, sp = checkpoints.tokenizer(model)
    state = checkpoints.weights(model, DEVICE)
    items, res = eval_items(), {}
    for task in tasks:
        train, test = (encode_items(part, tok, sp) for part in split(items[task], task))
        res[task] = {"n_test": len(test), "k": {}}
        for k in [k for k in ks if k <= len(train)]:
            accs = []
            for seed in SEEDS if k else (0,):
                net = checkpoints.fresh(model, state, DEVICE)
                if k:
                    adapt(net, [train[i] for i in draw(len(train), k, seed)], DEVICE, seed, learning_rate(model))
                accs.append(accuracy(net, test, DEVICE))
            res[task]["k"][str(k)] = accs
            print(f"{task:9} k={k:4}  acc {np.mean(accs):.3f}  (seeds: {[round(a, 3) for a in accs]})", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model}.json").write_text(json.dumps(res, indent=1))
    return res
