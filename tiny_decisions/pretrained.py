"""Decisions Models that start from a pretrained open encoder instead of from scratch.

Ettin (JHU, 2025): ModernBERT-style encoders from 17M to 400M parameters, each pretrained on about 2T tokens of open
data, about 2,000x the web text our from-scratch models read. Same layout and scoring as model.Decider:
  [CLS] question [SEP] [MASK] answer 1 [MASK] answer 2 ... [SEP] context [SEP]
Each answer is scored at its own [MASK], the token the encoder was pretrained to fill in; softmax over the answers.
"""
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer, pre_tokenizers
from transformers import AutoConfig, AutoModel

from .config import ETTIN, MODELS
from .decisions import Specials

BASES = {name: f"jhu-clsp/ettin-encoder-{name.removeprefix('ettin-')}" for name in ETTIN}
FILES = ["*.json", "pytorch_model.bin"]


def model_dir(base: str) -> Path:
    return MODELS / base


def download(base: str) -> Path:
    return Path(snapshot_download(BASES[base], local_dir=model_dir(base), allow_patterns=FILES))


def load_tokenizer(base: str) -> tuple[Tokenizer, Specials]:
    """The encoder's own tokenizer and its special tokens; answers are marked with its [MASK].

    Every piece (question, each answer, context) is encoded as it would appear mid-text, with a leading space, so an
    answer like "spam" becomes the word the encoder knows ("Ġspam") instead of "sp" + "am".
    """
    tok = Tokenizer.from_file(str(model_dir(base) / "tokenizer.json"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
    tid = tok.token_to_id
    return tok, Specials(cls=tid("[CLS]"), sep=tid("[SEP]"), marker=tid("[MASK]"), pad=tid("[PAD]"))


class PretrainedDecider(nn.Module):
    """A pretrained encoder plus a small scoring head on each answer's marker."""

    def __init__(self, base: str, pad_id: int, pretrained: bool = True):
        super().__init__()
        self.base, self.pad_id = base, pad_id
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_dir(base), attn_implementation="sdpa")
        else:  # the architecture only: the weights come from a saved checkpoint
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir(base)), attn_implementation="sdpa")
        d = self.encoder.config.hidden_size
        self.score = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def decide(self, ids: torch.Tensor, markers: torch.Tensor) -> torch.Tensor:
        """(batch, T) logits; only answer-marker positions are finite."""
        h = self.encoder(input_ids=ids, attention_mask=(ids != self.pad_id).long()).last_hidden_state
        return self.score(h).squeeze(-1).float().masked_fill(~markers, -1e4)
