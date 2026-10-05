"""Decisions as model input, and the fixed train/test split of every held-out task.

Layout:  [CLS] question [SEP] [A] answer 1 [A] answer 2 ... [SEP] context [SEP]
Training shows a random subset of 2-60 answers in random order, so the model must read the answers.
An Ettin encoder brings its own tokenizer: the same layout with its special tokens (Specials), [MASK] as marker.
"""
import json
import random
from dataclasses import dataclass

import torch

from .config import DATA
from .model import MAX_LEN
from .text import ANS, CLS, PAD, SEP, encoder

MAX_CTX, MAX_OPTIONS = 400, 60
DECISIONS = DATA / "decisions"


@dataclass(frozen=True)
class Specials:
    """The token ids a tokenizer uses in the decision layout."""
    cls: int
    sep: int
    marker: int  # in front of every allowed answer: [A] in our tokenizer, [MASK] in an Ettin encoder's
    pad: int


OURS = Specials(cls=CLS, sep=SEP, marker=ANS, pad=PAD)


def assemble(q: list[int], options: list[list[int]], ctx: list[int], sp: Specials = OURS) -> tuple[list[int], list[int]]:
    """Token ids + answer-marker positions for one decision."""
    ids, markers = [sp.cls] + q + [sp.sep], []
    for a in options:
        markers.append(len(ids))
        ids += [sp.marker] + a
    ids.append(sp.sep)
    ids += ctx[:max(0, MAX_LEN - len(ids) - 1)] + [sp.sep]
    return ids[:MAX_LEN], markers


def collate(examples: list[tuple[list[int], list[int], int]], dev, pad: int = PAD):
    """Pad (ids, markers, gold) examples into tensors: ids, marker mask, target position."""
    T = max(len(ids) for ids, _, _ in examples)
    x = torch.full((len(examples), T), pad, dtype=torch.long)
    mk = torch.zeros((len(examples), T), dtype=torch.bool)
    tgt = []
    for i, (ids, markers, gold) in enumerate(examples):
        x[i, :len(ids)] = torch.tensor(ids)
        mk[i, markers] = True
        tgt.append(markers[gold])
    return x.to(dev), mk.to(dev), torch.tensor(tgt, device=dev)


def load_tasks(role: str) -> dict:
    """Tasks with role 'train' or 'eval' from data/decisions (written by the data step)."""
    meta = json.loads((DECISIONS / "tasks.json").read_text())
    tasks = {}
    for t, m in meta.items():
        if m["role"] == role:
            tasks[t] = {**m, "rows": [json.loads(line) for line in open(DECISIONS / f"{t}.jsonl")]}
    return tasks


def eval_items(n_options: int | None = None) -> dict[str, list[dict]]:
    """Held-out decisions with a fixed, shuffled answer order per example (the same for every model and baseline).

    n_options=k keeps the gold answer plus k-1 random others (only for tasks with more than k answers).
    """
    items = {}
    for t, m in load_tasks("eval").items():
        if n_options and len(m["answers"]) <= n_options:
            continue
        name = t if not n_options else f"{t}_{n_options}opt"
        rng, answers, out = random.Random(f"{name}-order"), m["answers"], []
        for r in m["rows"]:
            opts = list(answers)
            if n_options:
                opts = [r["label"]] + rng.sample([a for a in answers if a != r["label"]], n_options - 1)
            rng.shuffle(opts)
            out.append({"context": r["context"], "question": m["questions"][0], "options": opts, "gold": opts.index(r["label"])})
        items[name] = out
    return items


def split(items: list[dict], task: str) -> tuple[list[dict], list[dict]]:
    """A held-out task's fixed half/half split: the train split the k examples are drawn from, and the test split."""
    items = list(items)
    random.Random(f"{task}-split").shuffle(items)
    half = len(items) // 2
    return items[:half], items[half:]


def draw(n: int, k: int, seed: int) -> list[int]:
    """Positions in the train split of the k examples a run with this seed learns from."""
    return random.Random(seed).sample(range(n), k)


def encode_items(items: list[dict], tok, sp: Specials = OURS) -> list[tuple[list[int], list[int], int]]:
    """eval_items entries -> (ids, markers, gold) examples."""
    enc = encoder(tok)
    qs, cs = enc([x["question"] for x in items]), enc([x["context"] for x in items])
    return [(*assemble(q, enc(x["options"]), c[:MAX_CTX], sp), x["gold"]) for x, q, c in zip(items, qs, cs)]


class TrainDecisions:
    """All training decisions (public tasks + synthetic), tokenized once; example() draws a fresh answer subset each
    time."""

    def __init__(self, tok, sp: Specials = OURS):
        enc = encoder(tok)
        self.sp = sp
        self.rows, self.answers, self.questions = [], {}, {}
        for t, m in load_tasks("train").items():
            self.answers[t] = enc(m["answers"])
            self.questions[t] = enc(m["questions"]) if m["questions"] else None
            lab = {a: i for i, a in enumerate(m["answers"])}
            ctx = enc([r["context"] for r in m["rows"]])
            own_q = enc([r["question"] for r in m["rows"]]) if "question" in m["rows"][0] else [None] * len(ctx)
            self.rows += [(t, c[:MAX_CTX], lab[r["label"]], q, None) for r, c, q in zip(m["rows"], ctx, own_q)]
        # every synthetic decision brings its own question and answer list
        rows = [json.loads(line) for line in open(DECISIONS / "synthetic.jsonl")]
        strings = sorted({s for r in rows for s in [r["question"], *r["answers"]]})
        ids = dict(zip(strings, enc(strings)))
        ctx = enc([r["context"] for r in rows])
        self.rows += [("synthetic", c[:MAX_CTX], r["answers"].index(r["label"]), ids[r["question"]],
                       [ids[a] for a in r["answers"]]) for r, c in zip(rows, ctx)]
        print(f"decision training rows: {len(self.rows):,} from {len(self.answers)} tasks + synthetic", flush=True)

    def example(self, i: int, rng: random.Random) -> tuple[list[int], list[int], int]:
        t, ctx, gold, q, own = self.rows[i]
        if own:
            opts = list(range(len(own)))
            rng.shuffle(opts)
            ids, markers = assemble(q, [own[a] for a in opts], ctx, self.sp)
            return ids, markers, opts.index(gold)
        n = len(self.answers[t])
        k = min(n, MAX_OPTIONS) if (n <= 8 or rng.random() < 0.3) else rng.randint(2, min(n, MAX_OPTIONS))
        opts = [gold] + rng.sample([a for a in range(n) if a != gold], k - 1)
        rng.shuffle(opts)
        q = q or rng.choice(self.questions[t])
        ids, markers = assemble(q, [self.answers[t][a] for a in opts], ctx, self.sp)
        return ids, markers, opts.index(gold)
