"""The result tables (runs/results.md), from the saved results in runs/; no GPU needed."""
import json
import re
from pathlib import Path

from .config import ETTIN, FINETUNE_TASKS, HELD_OUT, KS, MODEL_NAMES, RUNS

OUT = RUNS / "results.md"
SHOWN = "ettin-17m"            # the smallest model that works after fine-tuning: overlap check and speed
TASK_NAMES = {"banking77": "Banking77", "trec": "TREC", "sms_spam": "SMS spam", "emotion": "Emotion", "ag_news": "AG News"}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def human(n: float) -> str:
    """Parameter count as 0.6M / 17M / 396M."""
    return f"{n / 1e6:.1f}M" if n < 1e7 else f"{n / 1e6:.0f}M"


def label(model: str) -> str:
    return f"{'Ettin start' if model in ETTIN else 'from scratch'}, {model}"


def zero_shot_table() -> str:
    """Every model and LLM on the 5 held-out tasks with no examples."""
    zs = {m: read(RUNS / "zero_shot" / f"{m}.json") for m in MODEL_NAMES}
    llms = {p.stem: read(p) for p in sorted((RUNS / "llm_zero_shot").glob("*.json"))}

    def row(name: str, params: str, acc: dict[str, float]) -> str:
        cells = " | ".join(pct(acc[t]) for t in HELD_OUT)
        return f"| {name} | {params} | {cells} | **{pct(mean([acc[t] for t in HELD_OUT]))}** |"

    lines = ["| model | params | " + " | ".join(TASK_NAMES[t] for t in HELD_OUT) + " | average |",
             "|---|---:|" + "---:|" * (len(HELD_OUT) + 1),
             row("random guess", "–", {t: zs[SHOWN]["eval"][t]["chance"] for t in HELD_OUT})]
    lines += [row(label(m), human(r["params"]), {t: v["acc"] for t, v in r["eval"].items()}) for m, r in zs.items()]
    lines += [row(f"{name}, zero-shot", re.search(r"[\d.]+B$", name).group(0), {t: v["acc"] for t, v in r["eval"].items()})
              for name, r in llms.items()]
    return "\n".join(lines)


def finetune_table(task: str) -> str:
    """One held-out task's test split after fine-tuning on k examples, with the baselines on the same examples."""
    cells = lambda by_k: " | ".join(pct(mean(by_k[str(k)])) if str(k) in by_k else "–" for k in KS)
    lines = ["| model | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---:|" * len(KS)]
    for m in MODEL_NAMES:
        lines.append(f"| {label(m)} | {cells(read(RUNS / 'finetune' / f'{m}.json')[task]['k'])} |")
    lines.append(f"| TF-IDF + logistic regression, same examples | {cells(read(RUNS / 'tfidf.json')[task]['k'])} |")
    for p in sorted((RUNS / "llm_in_context").glob("*.json")):
        lines.append(f"| {p.stem}, same examples in its prompt | {cells(read(p)[task]['k'])} |")
    lines.append(f"| Laya, English checkpoint | {cells({'0': [read(RUNS / 'laya.json')[task]['zero_shot']]})} |")
    bekko = read(RUNS / "bekko.json")
    lines.append(f"| Bekko System One v0 17M ({bekko[task]['note']}) | {cells({'0': [bekko[task]['zero_shot']]})} |")
    return "\n".join(lines)


def size_table() -> str:
    """Storage at 8 bits, and whether rounding the weights changes zero-shot accuracy."""
    size = read(RUNS / "size.json")
    lines = ["| model | params | MB at 8 bits | zero-shot average | after rounding to 8 bits |", "|---|---:|---:|---:|---:|"]
    for m in MODEL_NAMES:
        acc = read(RUNS / "zero_shot" / f"{m}.json")["eval"]
        lines.append(f"| {label(m)} | {human(size[m]['params'])} | {size[m]['mb_8bit']:.1f} | "
                     f"{pct(mean([acc[t]['acc'] for t in HELD_OUT]))} | {pct(mean([size[m]['acc_8bit'][t] for t in HELD_OUT]))} |")
    return "\n".join(lines)


def speed_table() -> str:
    """Time per decision on one CPU thread and on one GPU, with Jev's published figure."""
    cpu, gpu = read(RUNS / "cpu_latency.json"), read(RUNS / "gpu_latency.json")
    jev = read(RUNS / "external.json")["jev"]
    assert cpu["items"] == gpu["items"], "the CPU and GPU times are not on the same decisions"
    names = {SHOWN: f"Decisions Model, {SHOWN}", "laya": "Laya, English checkpoint", "qwen3.5-0.8b": "Qwen3.5-0.8B, zero-shot"}
    task, options = re.fullmatch(r"(\w+?)_(\d+)opt", cpu["items"]).groups()
    lines = [f"The same {TASK_NAMES[task]} decisions with {options} answers offered, batch 1, tokenization included, the "
             "answer read back; median ms per decision. On the CPU every model runs in fp32. On the GPU, ours and Laya "
             "run in eager PyTorch (Laya through its own code, in bf16) and Qwen3.5-0.8B in vLLM (bf16, CUDA graphs, "
             "structured output), as in the accuracy runs.", "",
             f"| model | {cpu['cpu']}, {cpu['threads']} thread | {gpu['gpu']} |", "|---|---:|---:|"]
    lines += [f"| {names[m]} | {ms:,} | {gpu['ms'][m]:,} |" for m, ms in cpu["ms"].items()]
    bekko = read(RUNS / "bekko.json")
    assert bekko["cpu"] == cpu["cpu"], "Bekko was timed on another CPU"
    lines.append(f"| Bekko System One v0 17M, its own code (timed in a separate run) | {bekko['cpu_ms']:,} | – |")
    lines += ["", f"{jev['name']} is a closed API and is not timed here: third parties report {jev['p50_ms'][0]}–"
                  f"{jev['p50_ms'][1]} ms p50 per call, network included. Bekko's GPU time is not measured here (the "
                  "shared GPU was busy); its model card reports 4.6 ms eager on an RTX 5090."]
    return "\n".join(lines)


def overlap_table() -> str:
    """The shown model on its own examples vs the test split, with and without near-copies of the train split."""
    lines = ["| task | k | on its k examples | test split | test items with no near-copy in the train split "
             "| TF-IDF on its k examples | TF-IDF, test split |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task, r in read(RUNS / "overlap" / f"{SHOWN}.json").items():
        for k, v in r["model"].items():
            t = r["tfidf"][k]
            lines.append(f"| {TASK_NAMES[task]} | {k} | {pct(mean(v['examples']))} | {pct(mean(v['test']))} (n={r['n_test']}) | "
                         f"{pct(mean(v['test_no_copy_in_train']))} (n={r['n_test_no_copy_in_train']}) | "
                         f"{pct(mean(t['examples']))} | {pct(mean(t['test']))} |")
    return "\n".join(lines)


def write(out: Path = OUT) -> str:
    """All tables → runs/results.md."""
    n_test = {t: read(RUNS / "finetune" / f"{SHOWN}.json")[t]["n_test"] for t in FINETUNE_TASKS}
    parts = ["# Results", "Accuracy in %. Every number is read from the JSON files next to this one.",
             "## Zero-shot: after decision training, no examples of the held-out tasks", zero_shot_table()]
    for task in FINETUNE_TASKS:
        parts += [f"## {TASK_NAMES[task]}: fine-tuning on k examples from the train split, accuracy on the "
                  f"{n_test[task]}-item test split (mean of 3 draws)", finetune_table(task)]
    parts += ["## Size: weights rounded to 8 bits", size_table(), "## Speed: time per decision on a CPU and a GPU", speed_table(),
              f"## Overlap check: {SHOWN} after fine-tuning (mean of 3 draws)", overlap_table()]
    md = "\n\n".join(parts) + "\n"
    out.write_text(md)
    print(md)
    return md
