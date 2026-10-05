#!/usr/bin/env bash
# The baselines on the same decisions: TF-IDF + logistic regression on the same examples (CPU), Qwen3.5-9B and
# Qwen3.5-0.8B zero-shot, Qwen3.5-0.8B with the same examples in its prompt, Laya zero-shot, and Bekko System One v0
# 17M zero-shot with its CPU time per decision.
# Resumable: a step is skipped when its output exists.   bash scripts/run_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f ../env.sh ]; then source ../env.sh; fi   # our machine's cache and temp locations; not needed elsewhere
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
mkdir -p runs/logs

step() { local log=$1; shift; uv run --no-sync python -m tiny_decisions "$@" 2>&1 | tee -a "runs/logs/$log.log"; }

[ -s runs/tfidf.json ] || step tfidf tfidf
[ -s runs/llm_zero_shot/Qwen3.5-9B.json ] || step llm_9b llm --model Qwen/Qwen3.5-9B
[ -s runs/llm_zero_shot/Qwen3.5-0.8B.json ] || step llm_0.8b llm --model Qwen/Qwen3.5-0.8B --mem 0.2
[ -s runs/llm_in_context/Qwen3.5-0.8B.json ] || step llm_examples_0.8b llm-examples --model Qwen/Qwen3.5-0.8B
[ -s runs/laya.json ] || step laya laya
[ -s runs/bekko.json ] || step bekko bekko
