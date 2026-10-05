"""Bekko System One v0 17M (hotchpotch, 30 September 2026): another 17M Decisions Model on Ettin, started from the
Ettin reranker and trained on 8.4M judgments from 153 datasets. It runs through its own inference code
(inference_v0.py, from the pinned download in its model folder; Sentence Transformers needs trust_remote_code=True to
run it) on our test splits with no examples, and is timed per decision with speed.py's protocol on one CPU thread
(its own card reports its GPU times).

Its training data includes SMS spam, Banking77, AG News and emotion datasets but not TREC, so TREC is the held-out
test for it; the SMS spam score is kept and marked as seen in training.
"""
import json
import os
import sys
from pathlib import Path

from .config import MODELS, ROOT, RUNS

# Transformers copies a model's own code into a modules cache; keep it in the project, not under the home folder.
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".tmp" / "hf_modules"))

import torch  # noqa: E402  (after the cache location is set)
from huggingface_hub import snapshot_download  # noqa: E402

from . import speed  # noqa: E402
from .decisions import eval_items, split  # noqa: E402

REPO = "hotchpotch/bekko-system-one-v0-17m"
REVISION = "b886a1f9b91f4e8368d7830080d52d955e2c8dfa"
BEKKO = MODELS / "bekko-system-one-v0-17m"
OUT = RUNS / "bekko.json"
TASKS = {"trec": "task not in its training data", "sms_spam": "task in its training data"}


def download() -> None:
    snapshot_download(REPO, revision=REVISION, local_dir=BEKKO, ignore_patterns=["onnx_browser/*"])


def load(device: str):
    """Bekko through its own inference class, imported from its model folder; SDPA attention."""
    sys.path.insert(0, str(BEKKO))
    from inference_v0 import BekkoSentenceTransformer
    return BekkoSentenceTransformer(str(BEKKO), device=device, trust_remote_code=True, attn_implementation="sdpa")


def request(x: dict) -> dict:
    """One of our decisions in Bekko's native input, laid out as its own choice data is: the text as the state, our
    question as the instructions, the answers as candidates."""
    criteria = [{"id": f"o{i}", "description_json": json.dumps(o), "value": None} for i, o in enumerate(x["options"])]
    return {"state_json": json.dumps({"text": x["context"]}),
            "decisions": [{"id": "q", "kind": "judgment", "type": "choice", "instructions_json": json.dumps(x["question"]),
                           "system_prompt": "", "criteria": criteria, "documents": [], "scoring": None}]}


def choose(model, x: dict) -> int:
    """The index of the answer Bekko picks for one decision."""
    out = model.predict(request(x), batch_size=1, show_progress_bar=False)
    return int(out["q"]["selected_id"][1:])


def timed_ms(device: str) -> float:
    """Median time per decision on `device`, with the protocol and decisions of speed.py."""
    model, items = load(device), eval_items(n_options=5)[speed.ITEMS]
    n = speed.OURS_MODEL                                    # same warm-up and timed counts as our 17M
    return speed.median_ms(lambda x: choose(model, x), items, speed.WARMUP[n], speed.TIMED[n], speed.DIGITS[n])


def run(out: Path = OUT) -> dict:
    """Zero-shot accuracy on the test splits and CPU time per decision (fp32, one thread) → runs/bekko.json."""
    torch.set_num_threads(1)
    items, model = eval_items(), load("cpu")
    res = {"model": REPO, "revision": REVISION}
    for task, note in TASKS.items():
        _, test = split(items[task], task)
        right = sum(choose(model, x) == x["gold"] for x in test)
        res[task] = {"n_test": len(test), "zero_shot": right / len(test), "note": note}
        print(f"{task:9} zero-shot acc {right / len(test):.3f} (n={len(test)}; {note})", flush=True)
    res["cpu"], res["cpu_ms"] = speed.hardware("cpu")["cpu"], timed_ms("cpu")
    print("cpu", res["cpu_ms"], "ms", flush=True)
    out.write_text(json.dumps(res, indent=1))
    return res
