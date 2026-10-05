#!/usr/bin/env bash
# The five from-scratch sizes (0.6M to 29M) on one GPU: Qwen3.5-9B writes the synthetic decisions while each size is
# pretrained on the web text, then every size gets decision training and is fine-tuned on k examples.
# Resumable: a step is skipped when its output exists.   bash scripts/run_scratch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f ../env.sh ]; then source ../env.sh; fi   # our machine's cache and temp locations; not needed elsewhere
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
SIZES="xxs xs s m l"
mkdir -p runs/logs

step() { local log=$1; shift; uv run --no-sync python -m tiny_decisions "$@" 2>&1 | tee -a "runs/logs/$log.log"; }

[ -s data/decisions/synthetic.jsonl ] || step synth synth &
(for size in $SIZES; do [ -s "runs/checkpoints/$size/pretrain.pt" ] || step "pretrain_$size" pretrain --size "$size"; done) &
wait
[ -s data/decisions/synthetic.jsonl ] || { echo "no synthetic decisions: see runs/logs/synth.log"; exit 1; }
for size in $SIZES; do
  [ -s "runs/zero_shot/$size.json" ] || step "train_$size" train --model "$size"
  [ -s "runs/finetune/$size.json" ] || step "finetune_$size" finetune --model "$size"
done
