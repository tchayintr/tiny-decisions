#!/usr/bin/env bash
# The five Ettin starts (17M to 400M) on one GPU: the same decision training as the from-scratch sizes, then
# fine-tuning on k examples. Needs the synthetic decisions from scripts/run_scratch.sh.
# Resumable: a step is skipped when its output exists.   bash scripts/run_ettin.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f ../env.sh ]; then source ../env.sh; fi   # our machine's cache and temp locations; not needed elsewhere
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASES="ettin-17m ettin-32m ettin-68m ettin-150m ettin-400m"
mkdir -p runs/logs

step() { local log=$1; shift; uv run --no-sync python -m tiny_decisions "$@" 2>&1 | tee -a "runs/logs/$log.log"; }

for base in $BASES; do
  [ -s "runs/zero_shot/$base.json" ] || step "train_$base" train --model "$base"
  [ -s "runs/finetune/$base.json" ] || step "finetune_$base" finetune --model "$base"
done
