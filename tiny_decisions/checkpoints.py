"""Saving and loading trained Decisions Models, from scratch or from an Ettin encoder, with their tokenizer."""
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from .config import CHECKPOINTS, ETTIN
from .decisions import OURS, Specials
from .model import build
from .pretrained import PretrainedDecider
from .pretrained import load_tokenizer as ettin_tokenizer
from .text import load_tokenizer


def path(model: str, name: str = "decider") -> Path:
    """runs/checkpoints/<model>/<name>.pt: 'pretrain' after masked-word pretraining, 'decider' after decision training."""
    return CHECKPOINTS / model / f"{name}.pt"


def save(net: nn.Module, model: str, name: str = "decider") -> None:
    p = path(model, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"size": model, "state": net.state_dict()}, p)


def weights(model: str, device, name: str = "decider") -> dict:
    return torch.load(path(model, name), map_location=device)["state"]


def tokenizer(model: str) -> tuple[Tokenizer, Specials]:
    """Our tokenizer for a from-scratch model, the encoder's own for an Ettin model."""
    return ettin_tokenizer(model) if model in ETTIN else (load_tokenizer(), OURS)


def fresh(model: str, state: dict, device) -> nn.Module:
    """The model's architecture on the device, with the given weights."""
    net = PretrainedDecider(model, tokenizer(model)[1].pad, pretrained=False) if model in ETTIN else build(model)
    net = net.to(device)
    net.load_state_dict(state)
    return net


def load(model: str, device) -> tuple[nn.Module, Tokenizer, Specials]:
    """A trained Decisions Model with its tokenizer and special tokens."""
    tok, sp = tokenizer(model)
    return fresh(model, weights(model, device), device), tok, sp
