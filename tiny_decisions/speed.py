"""Time per decision for our model and the models it is compared with, on the same machine and the same decisions:
batch 1, tokenization included, the answer read back to the CPU, median after a warm-up.

  cpu    one CPU thread → runs/cpu_latency.json
  cuda   one GPU (the first in CUDA_VISIBLE_DEVICES) → runs/gpu_latency.json

  ettin-17m      our Decisions Model, fp32, eager PyTorch
  laya           Convai's Laya, English checkpoint (ModernBERT-large + its decision head), its own code (eager PyTorch):
                 fp32 on a CPU, bf16 autocast on a GPU
  qwen3.5-0.8b   Qwen3.5-0.8B zero-shot, forced to one allowed answer. On a CPU: transformers in fp32, greedy decoding
                 where each step may only continue one of the answer strings. On a GPU: served by vLLM as llm.py scores
                 it (bf16, CUDA graphs, structured output); plain transformers there would mostly time Python and
                 kernel-launch overhead that a serving engine removes

Jev is a closed API and cannot be timed here; runs/external.json holds its published third-party figure.
"""
import gc
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import checkpoints, laya, llm
from .config import RUNS, SMALL_LLM
from .decisions import MAX_CTX, assemble, eval_items
from .text import encoder

OURS_MODEL = "ettin-17m"
ITEMS = "banking77_5opt"                   # Banking77 decisions with 5 of its 77 answers offered
WARMUP = {OURS_MODEL: 20, "laya": 10, "qwen3.5-0.8b": 3}
TIMED = {OURS_MODEL: 180, "laya": 60, "qwen3.5-0.8b": 25}
DIGITS = {OURS_MODEL: 2, "laya": 1, "qwen3.5-0.8b": 1}   # decimals of the median in ms
VLLM_MEMORY = 0.15                         # vLLM's share of GPU memory when timing Qwen3.5-0.8B (the GPU is shared)
OUT = {"cpu": RUNS / "cpu_latency.json", "cuda": RUNS / "gpu_latency.json"}


def median_ms(fn: Callable[[dict], object], items: list[dict], warmup: int, timed: int, digits: int = 1) -> float:
    """Median wall time of fn(item) in ms over `timed` items after `warmup` items."""
    times = []
    for x in items[:warmup + timed]:
        t = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t)
    return round(statistics.median(times[warmup:]) * 1000, digits)


def ours_decide(device: str) -> Callable[[dict], torch.Tensor]:
    """Our Decisions Model: the probabilities of the offered answers."""
    model, tok, sp = checkpoints.load(OURS_MODEL, device)
    model, enc = model.float().eval(), encoder(tok)

    @torch.inference_mode()
    def decide(x: dict) -> torch.Tensor:
        q, c = enc([x["question"], x["context"]])
        ids, markers = assemble(q, enc(x["options"]), c[:MAX_CTX], sp)
        mk = torch.zeros(1, len(ids), dtype=torch.bool)
        mk[0, markers] = True
        return model.decide(torch.tensor([ids], device=device), mk.to(device)).softmax(-1).cpu()
    return decide


def laya_decide(device: str) -> Callable[[dict], str]:
    """Laya through its own code: the chosen answer."""
    agent = laya.agent(device)
    return lambda x: laya.decide(agent, x)


def qwen_decide(device: str) -> Callable[[dict], object]:
    """Qwen3.5-0.8B answering with exactly one allowed answer: served by vLLM on a GPU, transformers on a CPU."""
    return qwen_vllm() if device == "cuda" else qwen_transformers(device)


def qwen_vllm() -> Callable[[dict], str]:
    """Qwen3.5-0.8B in vLLM, restricted to the allowed answers by structured output: the chosen answer."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    engine = llm.engine(SMALL_LLM, VLLM_MEMORY, llm.MAX_LEN)

    def decide(x: dict) -> str:
        sp = SamplingParams(temperature=0, max_tokens=llm.MAX_TOKENS,
                            structured_outputs=StructuredOutputsParams(choice=x["options"]))
        return engine.chat([[{"role": "user", "content": llm.turn(x)}]], [sp], **llm.CHAT)[0].outputs[0].text
    return decide


def qwen_transformers(device: str) -> Callable[[dict], list[int]]:
    """Qwen3.5-0.8B in transformers, each greedy step allowed only to continue an answer (then its end-of-turn token):
    the answer's token ids."""
    path = llm.source(SMALL_LLM)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")

    def decide(x: dict) -> list[int]:
        msgs = [{"role": "user", "content": llm.turn(x)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                      return_dict=True)["input_ids"].to(device)
        answers = [tok(o, add_special_tokens=False)["input_ids"] + [end] for o in x["options"]]
        start = ids.shape[1]

        def allowed(_, sent):
            done = sent[start:].tolist()
            nxt = {a[len(done)] for a in answers if a[:len(done)] == done and len(a) > len(done)}
            return sorted(nxt) or [end]

        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=llm.MAX_TOKENS, do_sample=False, prefix_allowed_tokens_fn=allowed,
                                 eos_token_id=end, pad_token_id=end)
        return out[0, start:].tolist()
    return decide


def hardware(device: str) -> dict:
    """What the times are measured on."""
    if device == "cuda":
        return {"gpu": torch.cuda.get_device_name(0)}
    return {"cpu": Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0].strip(": \t")}


def run(device: str = "cpu") -> dict:
    """Time per decision for each model on one CPU thread or one GPU → runs/cpu_latency.json or runs/gpu_latency.json."""
    torch.set_num_threads(1)
    items = eval_items(n_options=5)[ITEMS]
    res = {**hardware(device), "threads": 1, "items": ITEMS, "ms": {}}
    for name, make in ((OURS_MODEL, ours_decide), ("laya", laya_decide), ("qwen3.5-0.8b", qwen_decide)):
        res["ms"][name] = median_ms(make(device), items, WARMUP[name], TIMED[name], DIGITS[name])
        print(name, res["ms"][name], "ms", flush=True)
        if device == "cuda":                   # the GPU is shared: hand each model's memory back
            gc.collect()
            torch.cuda.empty_cache()
    OUT[device].write_text(json.dumps(res, indent=1))
    return res
