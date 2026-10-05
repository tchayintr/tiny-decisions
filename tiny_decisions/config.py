"""Paths and constants shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"                     # downloaded weights stay in the project folder
RUNS = ROOT / "runs"
CHECKPOINTS = RUNS / "checkpoints"           # trained weights (not in the repository)
FINEWEB = DATA / "raw" / "HuggingFaceFW__fineweb-edu" / "sample" / "10BT" / "000_00000.parquet"

SCRATCH = ("xxs", "xs", "s", "m", "l")                                      # from scratch, 0.6M to 29M parameters
ETTIN = ("ettin-17m", "ettin-32m", "ettin-68m", "ettin-150m", "ettin-400m")  # from JHU's Ettin encoders, 17M to 400M
MODEL_NAMES = SCRATCH + ETTIN
BIG_LLM, SMALL_LLM = "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-0.8B"   # the 9B also writes the synthetic decisions

HELD_OUT = ("banking77", "trec", "sms_spam", "emotion", "ag_news")  # never trained on; every answer offered
FINETUNE_TASKS = ("sms_spam", "trec")        # held-out tasks with fine-tuning on k examples
KS = (0, 8, 32, 128)                         # examples drawn from a task's train split (0: zero-shot)
SEEDS = (0, 1, 2)                            # three draws of the k examples
DEVICE = "cuda"                              # the GPU in CUDA_VISIBLE_DEVICES (see scripts/)
