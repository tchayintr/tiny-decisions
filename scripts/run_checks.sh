#!/usr/bin/env bash
# Checks on the results, then the tables: Ettin-17M on its own fine-tuning examples vs the test split (with near-copies
# counted), the storage size at 8 bits, and time per decision for Ettin-17M, Laya and Qwen3.5-0.8B on one CPU thread
# and on one GPU.
# Resumable: a step is skipped when its output exists.   bash scripts/run_checks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f ../env.sh ]; then source ../env.sh; fi   # our machine's cache and temp locations; not needed elsewhere
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
mkdir -p runs/logs

step() { local log=$1; shift; uv run --no-sync python -m tiny_decisions "$@" 2>&1 | tee -a "runs/logs/$log.log"; }

[ -s runs/overlap/ettin-17m.json ] || step overlap overlap --model ettin-17m
[ -s runs/size.json ] || step size size
[ -s runs/cpu_latency.json ] || step speed speed
[ -s runs/gpu_latency.json ] || step speed speed --device cuda
uv run --no-sync python -m tiny_decisions report > /dev/null
echo "tables in runs/results.md"
