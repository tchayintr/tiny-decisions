#!/usr/bin/env bash
# Data and models, into data/ and models/: the public tasks recast as decisions and the held-out test sets, our
# tokenizer (kept in the repository) and the tokenized web text, then the Ettin encoders, Laya, Bekko and Qwen3.5-0.8B.
# Resumable: a step is skipped when its output exists.   bash scripts/get_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f ../env.sh ]; then source ../env.sh; fi   # our machine's cache and temp locations; not needed elsewhere

step() { uv run --no-sync python -m tiny_decisions "$@"; }

[ -s data/decisions/tasks.json ] || step data
[ -s data/tokenizer.json ] || step tokenizer
[ -s data/fineweb.u16 ] || step pretok
step models
