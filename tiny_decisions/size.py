"""How small a model can be stored: every weight matrix rounded to 8 bits (per-row symmetric; norms and biases stay
full precision), the size that gives, and zero-shot accuracy after rounding (to compare with runs/zero_shot/)."""
import copy
import json
from pathlib import Path

import torch.nn as nn

from . import checkpoints
from .config import DEVICE, MODEL_NAMES, RUNS
from .evaluate import evaluate
from .model import n_params

BITS = 8
OUT = RUNS / "size.json"


def quantize_(model: nn.Module, bits: int) -> None:
    """Weight-only, per-row symmetric rounding of every matrix, in place."""
    q = 2 ** (bits - 1) - 1
    for p in model.parameters():
        if p.dim() == 2:
            s = p.data.abs().amax(1, keepdim=True).clamp(min=1e-8) / q
            p.data = (p.data / s).round().clamp(-q, q) * s


def size_mb(n: int, bits: int) -> float:
    return round(n * bits / 8 / 1e6, 2)


def run(models: tuple[str, ...] = MODEL_NAMES, out: Path = OUT) -> dict:
    """Parameters, 8-bit size and 8-bit zero-shot accuracy of every trained model → runs/size.json."""
    res = json.loads(out.read_text()) if out.exists() else {}
    for model in models:
        if not checkpoints.path(model).exists():
            continue
        net, tok, sp = checkpoints.load(model, DEVICE)
        n = n_params(net)
        rounded = copy.deepcopy(net)
        quantize_(rounded, BITS)
        print(f"{model} at {BITS} bits:", flush=True)
        res[model] = {"params": n, "mb_8bit": size_mb(n, BITS),
                      "acc_8bit": {t: v["acc"] for t, v in evaluate(rounded, tok, DEVICE, sp).items()}}
        out.write_text(json.dumps(res, indent=1))
    return res
