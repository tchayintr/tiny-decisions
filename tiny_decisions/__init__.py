"""How small can a Decisions Model be?

A Decisions Model reads a question, its allowed answers and a text in one forward pass and gives one logit per answer,
at the marker in front of it (Laya's design); softmax over the markers gives the choice and its probability. Ten
sizes get the same decision training: five from scratch (0.6M to 29M parameters, own tokenizer and pretraining) and
five from JHU's pretrained Ettin encoders (17M to 400M). They are tested on held-out tasks with no examples and after
fine-tuning on a few examples, against TF-IDF, Qwen3.5 and Laya, plus CPU time per decision. Run
`python -m tiny_decisions --help` for the steps.

Modules
  config       paths and shared constants
  data         public classification sets recast as decisions; the held-out test sets
  synth        synthetic decisions written by Qwen3.5-9B
  text         our 8K BPE tokenizer and the web text for pretraining
  decisions    a decision as model input, and the fixed train/test split of each held-out task
  model        the from-scratch encoder with its masked-word and decision heads
  pretrained   an Ettin encoder with a decision head
  checkpoints  saving and loading trained models
  train        masked-word pretraining and decision training
  evaluate     accuracy on the held-out tasks
  finetune     fine-tuning on k examples of a held-out task
  baselines    TF-IDF + logistic regression on the same examples
  llm          Qwen3.5 zero-shot and with the same examples in its prompt
  laya         Convai's Laya, zero-shot
  overlap      accuracy on the fine-tuning examples vs the test split, and near-copies between them
  size         weights rounded to 8 bits: storage size and accuracy
  speed        time per decision on one CPU thread or one GPU
  report       the result tables
"""
