"""Command line: `uv run python -m tiny_decisions <step> ...`. GPU steps use the GPU in CUDA_VISIBLE_DEVICES; scripts/
runs the steps in order.

  data          public tasks recast as decisions, and the held-out test sets → data/decisions/
  tokenizer     our 8K BPE tokenizer, trained on FineWeb-Edu → data/tokenizer.json
  pretok        the web text, tokenized once → data/fineweb.u16
  models        download the Ettin encoders, Laya, Bekko and Qwen3.5-0.8B → models/
  synth         synthetic decisions written by Qwen3.5-9B → data/decisions/synthetic.jsonl (GPU, vLLM)
  pretrain      masked-word pretraining of one from-scratch size (GPU)
  train         decision training of one model, then zero-shot on the held-out tasks → runs/zero_shot/ (GPU)
  finetune      fine-tuning on k examples of each held-out task → runs/finetune/ (GPU)
  tfidf         TF-IDF + logistic regression on the same examples → runs/tfidf.json
  llm           an LLM zero-shot, forced to one allowed answer → runs/llm_zero_shot/ (GPU, vLLM)
  llm-examples  the same LLM with the k examples in its prompt → runs/llm_in_context/ (GPU, vLLM)
  laya          Laya zero-shot on the test splits → runs/laya.json (GPU)
  bekko         Bekko System One v0 17M zero-shot, and its time per decision on one CPU thread → runs/bekko.json
  overlap       accuracy on its own examples vs the test split, and near-copies → runs/overlap/ (GPU)
  size          weights rounded to 8 bits: size and zero-shot accuracy → runs/size.json (GPU)
  speed         time per decision: ours, Laya, Qwen3.5-0.8B on one CPU thread → runs/cpu_latency.json
                (--device cuda: on one GPU → runs/gpu_latency.json)
  report        the result tables → runs/results.md
"""
import argparse
from importlib import import_module

from .config import ETTIN, FINETUNE_TASKS, KS, MODEL_NAMES, SCRATCH, SMALL_LLM
from .synth import N_DECISIONS

LLM_MEMORY = {"llm": 0.55, "llm-examples": 0.25}   # vLLM's share of GPU memory (the GPU is shared)
SYNTH_MEMORY = 0.4


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tiny_decisions", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("data", "tokenizer", "pretok", "models", "tfidf", "laya", "bekko", "size", "report"):
        sub.add_parser(name)
    p = sub.add_parser("synth")
    p.add_argument("--n", type=int, default=N_DECISIONS, help="decisions to write")
    p.add_argument("--mem", type=float, default=SYNTH_MEMORY, help="vLLM's share of GPU memory")
    p = sub.add_parser("pretrain")
    p.add_argument("--size", required=True, choices=SCRATCH)
    p = sub.add_parser("train")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p = sub.add_parser("finetune")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--tasks", nargs="+", default=list(FINETUNE_TASKS))
    p.add_argument("--ks", nargs="+", type=int, default=list(KS))
    for name, mem in LLM_MEMORY.items():
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help=f"Hub id, e.g. {SMALL_LLM} (a copy in models/ is used if there)")
        p.add_argument("--mem", type=float, default=mem, help="vLLM's share of GPU memory")
    p = sub.add_parser("overlap")
    p.add_argument("--model", default="ettin-17m", choices=MODEL_NAMES)
    p = sub.add_parser("speed")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="one CPU thread, or one GPU")
    return ap


def download_models() -> None:
    """The Ettin encoders, Laya, Bekko and Qwen3.5-0.8B into models/ (Qwen3.5-9B is read from the Hugging Face
    cache)."""
    pretrained, laya, bekko, llm = (import_module(f".{m}", __package__) for m in ("pretrained", "laya", "bekko", "llm"))
    for base in ETTIN:
        pretrained.download(base)
    laya.download()
    bekko.download()
    llm.download(SMALL_LLM)


def main() -> None:
    a = parser().parse_args()
    mod = lambda name: import_module(f".{name}", __package__)   # imported when its step runs: torch and vLLM load slowly
    steps = {
        "data": lambda: mod("data").build(),
        "tokenizer": lambda: mod("text").train_tokenizer(),
        "pretok": lambda: mod("text").pretokenize(),
        "models": download_models,
        "synth": lambda: mod("synth").generate(a.n, a.mem),
        "pretrain": lambda: mod("train").pretrain(a.size),
        "train": lambda: mod("train").train(a.model),
        "finetune": lambda: mod("finetune").run(a.model, tuple(a.tasks), tuple(a.ks)),
        "tfidf": lambda: mod("baselines").run(),
        "llm": lambda: mod("llm").zero_shot(a.model, a.mem),
        "llm-examples": lambda: mod("llm").in_context(a.model, a.mem),
        "laya": lambda: mod("laya").zero_shot(),
        "bekko": lambda: mod("bekko").run(),
        "overlap": lambda: mod("overlap").run(a.model),
        "size": lambda: mod("size").run(),
        "speed": lambda: mod("speed").run(a.device),
        "report": lambda: mod("report").write(),
    }
    steps[a.step]()


if __name__ == "__main__":
    main()
