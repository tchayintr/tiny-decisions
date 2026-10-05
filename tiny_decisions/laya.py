"""Laya (Convai's open Decisions Model, English checkpoint, 421M parameters) zero-shot on the same test splits, through
its own inference code (rl_agent_api.py in its model folder)."""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import FINETUNE_TASKS, MODELS, RUNS
from .decisions import eval_items, split

REPO = "convaiinnovations/laya"
LAYA = MODELS / "laya"
OUT = RUNS / "laya.json"


def download() -> None:
    snapshot_download(REPO, local_dir=LAYA)


def agent(device: str):
    """Laya's own agent, loaded from its model folder."""
    sys.path.insert(0, str(LAYA))
    from rl_agent_api import RLAgent
    return RLAgent(str(LAYA), device=device)


def decide(laya, x: dict) -> str:
    """One call in Laya's (and Jev's) shape: a choice question about one text; returns the chosen answer."""
    return laya.system_one(x["context"], {"q": {"type": "choice", "instructions": x["question"],
                                               "criteria": x["options"]}})["answers"]["q"]["choice"]


def zero_shot(out: Path = OUT) -> dict:
    """Accuracy on each test split → runs/laya.json."""
    laya, items, res = agent("cuda"), eval_items(), {"model": f"{REPO} (English)"}
    for task in FINETUNE_TASKS:
        _, test = split(items[task], task)
        right = sum(decide(laya, x) == x["options"][x["gold"]] for x in test)
        res[task] = {"n_test": len(test), "zero_shot": right / len(test)}
        print(f"{task:9} zero-shot acc {right / len(test):.3f} (n={len(test)})", flush=True)
    out.write_text(json.dumps(res, indent=1))
    return res
