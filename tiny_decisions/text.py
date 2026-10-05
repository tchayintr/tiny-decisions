"""Our tokenizer (8K byte-level BPE, trained from scratch) and the web text used for pretraining."""
import time
from collections.abc import Callable, Iterator

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import DATA, FINEWEB

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[A]"]
PAD, UNK, CLS, SEP, MASK, ANS = range(len(SPECIALS))  # [A] marks each allowed answer
VOCAB = 8192
TOKENIZER = DATA / "tokenizer.json"
WEB_TOKENS = DATA / "fineweb.u16"
TOKENIZER_DOCS = 200_000                     # web documents the tokenizer is trained on
WEB_TOKEN_LIMIT = 1_000_000_000              # tokens of web text kept for pretraining


def fineweb_batches(batch: int = 20_000) -> Iterator[list[str]]:
    """The FineWeb-Edu shard's documents, in batches."""
    for b in pq.ParquetFile(FINEWEB).iter_batches(batch_size=batch, columns=["text"]):
        yield b.column(0).to_pylist()


def train_tokenizer(n_docs: int = TOKENIZER_DOCS) -> None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def docs() -> Iterator[str]:
        n = 0
        for texts in fineweb_batches():
            yield from texts
            n += len(texts)
            if n >= n_docs:
                return

    tok.train_from_iterator(docs(), trainer=trainer)
    tok.save(str(TOKENIZER))
    print("tokenizer:", tok.get_vocab_size(), "tokens;", tok.encode("I was charged twice for my coffee").tokens)


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER))


def encoder(tok: Tokenizer) -> Callable[[list[str]], list[list[int]]]:
    """A batch encoder: list of strings -> list of token-id lists, without special tokens (the decision layout
    adds its own; ours has none to add, a pretrained encoder's tokenizer would wrap every piece in [CLS] ... [SEP])."""
    return lambda texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def pretokenize(max_tokens: int = WEB_TOKEN_LIMIT) -> None:
    """Tokenize the web text once into a flat uint16 file (documents separated by [SEP])."""
    tok, n, t0 = load_tokenizer(), 0, time.time()
    with open(WEB_TOKENS, "wb") as f:
        for texts in fineweb_batches():
            arr = np.concatenate([np.array(e.ids + [SEP], dtype=np.uint16) for e in tok.encode_batch(texts)])
            arr.tofile(f)
            n += len(arr)
            print(f"  {n / 1e6:.0f}M tokens  {time.time() - t0:.0f}s", flush=True)
            if n >= max_tokens:
                break
