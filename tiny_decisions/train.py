"""Training: masked-word pretraining of the from-scratch sizes on web text, decision training of every model, then
zero-shot accuracy on the held-out tasks.

Decision training runs the same loop for the from-scratch models (train_scratch) and for the models that start from
an Ettin encoder (train_ettin), on the same 466,344 decisions (23 public tasks + synthetic).
"""
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import checkpoints
from .config import DEVICE, ETTIN, RUNS
from .decisions import TrainDecisions, collate
from .evaluate import evaluate
from .model import build, n_params
from .pretrained import PretrainedDecider, download
from .pretrained import load_tokenizer as ettin_tokenizer
from .text import MASK, SPECIALS, VOCAB, WEB_TOKENS, load_tokenizer

PRETRAIN_STEPS, PRETRAIN_BATCH, PRETRAIN_LEN = 15_000, 256, 256    # 15,000 steps of 65,536 tokens: about 1B tokens
PRETRAIN_LR, PRETRAIN_WARMUP = 1e-3, 1000
MASK_RATE = 0.15                                                   # BERT's share of masked tokens
SCRATCH_EPOCHS, SCRATCH_BATCH, SCRATCH_LR = 2, 128, 3e-4
ETTIN_EPOCHS, ETTIN_BATCH, ETTIN_LR, ETTIN_HEAD_LR = 1, 64, 5e-5, 1e-3   # the new scoring head learns faster
ETTIN_MICRO = 32             # batches of 64 run as two halves of 32: the same update, less memory on a shared GPU
DECISION_WARMUP = 300
WEIGHT_DECAY = 0.01
LOG_EVERY = 500
PRETRAIN_LOGS, ZERO_SHOT = RUNS / "pretrain", RUNS / "zero_shot"


def schedule(opt, step: int, total: int, lr: float, warmup: int) -> None:
    """Linear warmup, then cosine decay to 10% of the peak learning rate (a group's own "peak_lr" if it has one)."""
    f = step / warmup if step < warmup else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    for g in opt.param_groups:
        g["lr"] = g.get("peak_lr", lr) * f


def optimizer_step(model, opt, loss) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def mask_words(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT-style masking: pick 15% of real tokens; 80% -> [MASK], 10% -> random token, 10% unchanged."""
    sel = (torch.rand(x.shape, device=x.device) < MASK_RATE) & (x >= len(SPECIALS))
    y = torch.where(sel, x, -100)
    r = torch.rand(x.shape, device=x.device)
    x = torch.where(sel & (r < 0.8), MASK, x)
    x = torch.where(sel & (r >= 0.8) & (r < 0.9), torch.randint_like(x, len(SPECIALS), VOCAB), x)
    return x, y


def pretrain(size: str, steps: int = PRETRAIN_STEPS, bs: int = PRETRAIN_BATCH, T: int = PRETRAIN_LEN,
             lr: float = PRETRAIN_LR) -> None:
    """Masked-word pretraining of a from-scratch size on the tokenized web text."""
    torch.manual_seed(0)
    tokens = np.memmap(WEB_TOKENS, dtype=np.uint16, mode="r")
    model = build(size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)
    print(f"pretrain {size}: {n_params(model):,} params, {steps} steps x {bs * T:,} tokens", flush=True)
    t0, log = time.time(), []
    for step in range(1, steps + 1):
        starts = np.random.randint(0, len(tokens) - T, bs)
        x = torch.from_numpy(np.stack([tokens[s:s + T] for s in starts]).astype(np.int64)).to(DEVICE, non_blocking=True)
        x, y = mask_words(x)
        schedule(opt, step, steps, lr, warmup=PRETRAIN_WARMUP)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.mlm_loss(x, y)
        optimizer_step(model, opt, loss)
        if step % LOG_EVERY == 0 or step == steps:
            el = time.time() - t0
            log.append({"step": step, "loss": round(loss.item(), 4), "min": round(el / 60, 1)})
            print(f"  {size} step {step}/{steps} loss {loss.item():.3f}  {step * bs * T / el / 1e6:.2f}M tok/s  {el / 60:.1f} min", flush=True)
    checkpoints.save(model, size, "pretrain")
    PRETRAIN_LOGS.mkdir(parents=True, exist_ok=True)
    (PRETRAIN_LOGS / f"{size}.json").write_text(json.dumps(log))


def decision_training(model, data: TrainDecisions, opt, epochs: int, bs: int, lr: float, name: str,
                      warmup: int = DECISION_WARMUP, micro: int = 0) -> float:
    """Practice decisions in shuffled order, each drawn with a fresh answer subset; returns the minutes it took.

    micro > 0 splits every batch into pieces of that size and adds up their gradients before the step: the same
    update as the whole batch, with less GPU memory.
    """
    rng = random.Random(0)
    steps = epochs * len(data.rows) // bs
    piece = micro or bs
    print(f"decision training {name}: {n_params(model):,} params, {steps} steps", flush=True)
    t0, order, pos = time.time(), [], 0
    for step in range(1, steps + 1):
        if pos + bs > len(order):
            order, pos = rng.sample(range(len(data.rows)), len(data.rows)), 0
        batch = [data.example(i, rng) for i in order[pos:pos + bs]]
        pos += bs
        schedule(opt, step, steps, lr, warmup)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for j in range(0, bs, piece):
            x, mk, tgt = collate(batch[j:j + piece], DEVICE, model.pad_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                part = F.cross_entropy(model.decide(x, mk), tgt) * len(tgt) / bs
            part.backward()
            loss += part.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % LOG_EVERY == 0 or step == steps:
            print(f"  {name} step {step}/{steps} loss {loss:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    return round((time.time() - t0) / 60, 1)


def finish(model, name: str, minutes: float, scores: dict) -> None:
    """Save the trained model and its zero-shot accuracy (runs/zero_shot/<name>.json)."""
    checkpoints.save(model, name)
    ZERO_SHOT.mkdir(parents=True, exist_ok=True)
    res = {"model": name, "params": n_params(model), "train_minutes": minutes, "eval": scores}
    (ZERO_SHOT / f"{name}.json").write_text(json.dumps(res, indent=1))


def train_scratch(size: str, epochs: int = SCRATCH_EPOCHS, bs: int = SCRATCH_BATCH, lr: float = SCRATCH_LR) -> None:
    """Decision training of a from-scratch size, starting from its pretraining checkpoint."""
    torch.manual_seed(0)
    tok = load_tokenizer()
    data = TrainDecisions(tok)
    model = build(size).to(DEVICE)
    model.load_state_dict(checkpoints.weights(size, DEVICE, "pretrain"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    minutes = decision_training(model, data, opt, epochs, bs, lr, size)
    finish(model, size, minutes, evaluate(model, tok, DEVICE))


def train_ettin(base: str, epochs: int = ETTIN_EPOCHS, bs: int = ETTIN_BATCH, lr: float = ETTIN_LR,
                head_lr: float = ETTIN_HEAD_LR, micro: int = ETTIN_MICRO) -> None:
    """The same decision training, starting from a pretrained Ettin encoder with a new scoring head."""
    torch.manual_seed(0)
    download(base)
    tok, sp = ettin_tokenizer(base)
    data = TrainDecisions(tok, sp=sp)
    model = PretrainedDecider(base, sp.pad).to(DEVICE)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "peak_lr": lr},
                             {"params": model.score.parameters(), "peak_lr": head_lr}], lr=lr, weight_decay=WEIGHT_DECAY)
    minutes = decision_training(model, data, opt, epochs, bs, lr, base, micro=micro)
    finish(model, base, minutes, evaluate(model, tok, DEVICE, sp))


def train(model: str) -> None:
    """Decision training of one model, from scratch or from Ettin, then zero-shot on the held-out tasks."""
    if model in ETTIN:
        train_ettin(model)
    else:
        train_scratch(model)
