"""Qwen3.5 on the same decisions, forced (vLLM structured output) to answer with exactly one allowed answer.

  zero_shot    every held-out task, no examples
  in_context   the k examples the small models are fine-tuned on, as earlier turns of the conversation (no training),
               then each test decision; same split, draws and test split as finetune.py
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .config import FINETUNE_TASKS, HELD_OUT, MODELS, RUNS, SEEDS
from .decisions import draw, eval_items, split

PROMPT = ("Context:\n{context}\n\nQuestion: {question}\n\nAllowed answers:\n{options}\n\n"
          "Answer with exactly one of the allowed answers.")
IN_CONTEXT_KS = (0, 8, 32)       # 128 SMS messages as examples would not fit the context
MAX_LEN, MAX_LEN_IN_CONTEXT = 4096, 12288
MAX_TOKENS = 48
CHAT = {"use_tqdm": False, "chat_template_kwargs": {"enable_thinking": False}}
ZERO_SHOT, IN_CONTEXT = RUNS / "llm_zero_shot", RUNS / "llm_in_context"


def name(hub_id: str) -> str:
    """Qwen/Qwen3.5-0.8B -> Qwen3.5-0.8B, the result file's name."""
    return hub_id.split("/")[-1]


def local_dir(hub_id: str) -> Path:
    return MODELS / name(hub_id).lower()


def download(hub_id: str) -> None:
    snapshot_download(hub_id, local_dir=local_dir(hub_id))


def source(hub_id: str) -> str:
    """The downloaded copy in models/ if there is one, else the Hub id (read from the Hugging Face cache)."""
    return str(local_dir(hub_id)) if (local_dir(hub_id) / "config.json").exists() else hub_id


def turn(x: dict) -> str:
    """One decision as a user turn."""
    return PROMPT.format(context=x["context"], question=x["question"], options="\n".join(f"- {o}" for o in x["options"]))


def conversation(examples: list[dict], x: dict) -> list[dict]:
    """The k examples as earlier user/assistant turns (each answered with its label), then the test decision."""
    msgs = []
    for e in examples:
        msgs += [{"role": "user", "content": turn(e)}, {"role": "assistant", "content": e["options"][e["gold"]]}]
    return msgs + [{"role": "user", "content": turn(x)}]


def engine(hub_id: str, mem: float, max_len: int, **kw):
    """A vLLM engine for the model, without the vision encoder of multimodal checkpoints."""
    from vllm import LLM
    path = source(hub_id)
    config = Path(path) / "config.json" if Path(path).exists() else Path(hf_hub_download(hub_id, "config.json"))
    arch = json.loads(config.read_text())["architectures"][0]
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if "ConditionalGeneration" in arch else {}
    return LLM(model=path, max_model_len=max_len, gpu_memory_utilization=mem, seed=0, **mm, **kw)


def accuracy(llm, conversations: list[list[dict]], xs: list[dict]) -> float:
    """Share of decisions answered with the label; each reply is restricted to the decision's allowed answers."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    sps = [SamplingParams(temperature=0, max_tokens=MAX_TOKENS, structured_outputs=StructuredOutputsParams(choice=x["options"]))
           for x in xs]
    outs = llm.chat(conversations, sps, **CHAT)
    return sum(o.outputs[0].text.strip() == x["options"][x["gold"]] for o, x in zip(outs, xs)) / len(xs)


def zero_shot(hub_id: str, mem: float = 0.55, out: Path = ZERO_SHOT) -> dict:
    """Accuracy on every held-out task with no examples → runs/llm_zero_shot/<name>.json."""
    llm, items, res = engine(hub_id, mem, MAX_LEN), eval_items(), {"model": hub_id, "eval": {}}
    for task in HELD_OUT:
        xs = items[task]
        acc = accuracy(llm, [[{"role": "user", "content": turn(x)}] for x in xs], xs)
        res["eval"][task] = {"acc": acc, "n": len(xs), "options": len(xs[0]["options"])}
        print(f"{task:10} acc {acc:.3f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name(hub_id)}.json").write_text(json.dumps(res, indent=1))
    return res


def in_context(hub_id: str, mem: float = 0.25, out: Path = IN_CONTEXT) -> dict:
    """Accuracy on each test split with k examples in the prompt → runs/llm_in_context/<name>.json."""
    llm = engine(hub_id, mem, MAX_LEN_IN_CONTEXT, enable_prefix_caching=True)
    items, res = eval_items(), {"model": hub_id}
    for task in FINETUNE_TASKS:
        train, test = split(items[task], task)
        res[task] = {"n_test": len(test), "k": {}}
        for k in IN_CONTEXT_KS:
            accs = []
            for seed in SEEDS if k else (0,):
                examples = [train[i] for i in draw(len(train), k, seed)]
                accs.append(accuracy(llm, [conversation(examples, x) for x in test], test))
                print(f"{task:9} k={k:3} seed {seed}  acc {accs[-1]:.3f}", flush=True)
            res[task]["k"][str(k)] = accs
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name(hub_id)}.json").write_text(json.dumps(res, indent=1))
    return res
