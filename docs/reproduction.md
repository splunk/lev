# Reproducing Lev-3B Training

Lev-3B uses three training stages: initial decision training, a short decision
continuation, and joint log-task adaptation. The decision stages build on
[Jared Palmer's KEV decision-training recipe](https://github.com/jaredpalmer/kev/blob/e943f21e40574d99cefb2d292089333bcda9047c/README.md#training),
pinned to `e943f21e40574d99cefb2d292089333bcda9047c`, with the Granite backbone
and settings below. Use LEV's patched KEV dependency, not unmodified upstream
KEV.

This guide records the commands and settings used for the released model.
**The prepared joint log-training data is not included.** You must supply the
KEV JSONLs described below; matching their record counts alone does not
reproduce the original examples, prompts, labels, or ordering. This is a
training-recipe guide, not a self-contained reproduction of the released
checkpoint or a guarantee of bitwise-identical weights.

## Setup and checkpoint locations

Follow the [README installation instructions](../README.md#requirements-and-installation),
including patch-based KEV source setup when needed, and select the CUDA
12.8/PyTorch 2.8 environment. Run the following commands from the LEV repository
root in that environment, in order and in the same shell. They use one CUDA
GPU, BF16 frozen backbone weights and activations, FP32 LoRA/pointer-head
parameters, and gradient checkpointing.

Replace the output root with a new local location. Each checkpoint destination
must not already exist. The GPU ordinal and output paths are caller-selected.

```bash
export CUDA_VISIBLE_DEVICES=0
BASE_MODEL=ibm-granite/granite-4.1-3b-base
BASE_REVISION=dacb9cb9157bec98e99b09f285c92a4d58405c96
TRAINING_ROOT=/absolute/path/to/new-lev-training
DECISION_CHECKPOINT="$TRAINING_ROOT/decision/checkpoint"
CONTINUATION_CHECKPOINT="$TRAINING_ROOT/decision-continuation/checkpoint"
LEV_CHECKPOINT="$TRAINING_ROOT/joint-log/checkpoint"
```

The first two commands use KEV's underscore-style arguments, including
`--base_revision` and `--init_from`. The joint stage uses LEV's hyphen-style
arguments, including `--base-revision` and `--init-from`.

## 1. Initial decision training

Start from the pinned Granite base with fresh LoRA parameters and pointer
head; there is no checkpoint warm start in this stage. Use KEV's frozen
`decision-v7` training partition: 12,576 records containing 15,576 questions
(10,000 public-dataset examples, 896 generated policy examples, and 1,680
compositional policy examples).

The suite metadata and continuation data path below belong to the pinned KEV
checkout reconstructed by source setup. If the suite's training partition is
not present, KEV fetches it from its pinned public dataset mirror and verifies
the manifest's hash and count. This requires network access or a populated
Hugging Face cache; no dataset regeneration is needed.

```bash
uv run python -m kev.train \
  --suite third_party/kev/evals/v7/decision-v7 \
  --base "$BASE_MODEL" --base_revision "$BASE_REVISION" \
  --epochs 2 --seed 2 \
  --lr 1e-4 --head_lr 0 --weight_decay 0.01 \
  --lora 16 --lora_targets all --head_dim 256 \
  --batch 8 --accum 1 \
  --device cuda --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
  --max_state 384 --max_branch 1024 --max_packed 2048 \
  --perm_kl 0 --perm_frac 0.3 --ord_w 0 \
  --p_none 0.1 --p_none_distract 0.12 --p_distract 0.15 --p_none_pair 0.25 \
  --synthetic_repeat 1 --public_frac 1 \
  --out "$DECISION_CHECKPOINT"
```

## 2. Decision continuation

Warm-start from stage 1. Train for one epoch on 1,425 date/unknowability
examples plus 2,000 replay examples sampled from the same decision training
partition with seed 1: 3,425 records total. The added examples include explicit
date-policy decisions and cases with missing deciding evidence and soft
targets. The learning rate decreases to `4e-5`; the other decision-stage
settings remain the same.

```bash
uv run python -m kev.train \
  --suite third_party/kev/evals/v7/decision-v7 \
  --data third_party/kev/evals/night2/dates_unknowable.jsonl \
  --replay 2000 --init_from "$DECISION_CHECKPOINT" \
  --base "$BASE_MODEL" --base_revision "$BASE_REVISION" \
  --epochs 1 --seed 1 \
  --lr 4e-5 --head_lr 0 --weight_decay 0.01 \
  --lora 16 --lora_targets all --head_dim 256 \
  --batch 8 --accum 1 \
  --device cuda --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
  --max_state 384 --max_branch 1024 --max_packed 2048 \
  --perm_kl 0 --perm_frac 0.3 --ord_w 0 \
  --p_none 0.1 --p_none_distract 0.12 --p_distract 0.15 --p_none_pair 0.25 \
  --synthetic_repeat 1 --public_frac 1 \
  --out "$CONTINUATION_CHECKPOINT"
```

`--init_from` restores the trained adapter and pointer head; this is a new
training run with a new optimizer/schedule, not an optimizer-state resume.

## 3. Joint log adaptation

Supply three labelled KEV JSONLs in the [README's record format](../README.md#supplied-record-format).
The released model used these training populations:

| Input variable | Records | Required population |
|---|---:|---|
| `ANOMALY_TRAIN_JSONL` | 7,566 | BGL, Spirit, and Thunderbird anomaly decisions: 2,522 unique sessions per source, with 1,261 anomalous and 1,261 normal sessions each. JSON-list log states and neutral anomaly prompts. HDFS is excluded from this anomaly-supervision input. |
| `GENERAL_LOG_TRAIN_JSONL` | 28,740 | The general-log auxiliary mixture with corrected LogGLUE summary labels and its direct four-anomaly-dataset component removed. It includes log understanding, summaries, semantic relations, and fault-related tasks; it is not an anomaly-free or strictly HDFS-blind population. |
| `ATTACK_TECHNIQUE_TRAIN_JSONL` | 3,945 | ATT&CK technique classification with 130 alternatives: a 1,315-record training population (544 original and 771 synthesized examples), repeated three times. |

These inputs are already prepared: do not split, sample, rebalance, or regenerate
them inside the training command. The ATT&CK input must already contain its
three repetitions; do not repeat it again through mixture weights. The
proportional mixture uses all supplied records once per epoch, in the input
order shown below before KEV's seeded training shuffle: **40,251 records** in
total. Replacing these populations with different examples is a new training
experiment, even if the counts are unchanged.

```bash
ANOMALY_TRAIN_JSONL=/absolute/path/to/anomaly-train.jsonl
GENERAL_LOG_TRAIN_JSONL=/absolute/path/to/general-log-auxiliary-train.jsonl
ATTACK_TECHNIQUE_TRAIN_JSONL=/absolute/path/to/attack-technique-train-x3.jsonl

uv run lev-train \
  --train-input "anomaly=$ANOMALY_TRAIN_JSONL" \
  --train-input "auxiliary=$GENERAL_LOG_TRAIN_JSONL" \
  --train-input "attack_technique=$ATTACK_TECHNIQUE_TRAIN_JSONL" \
  --mixture proportional \
  --out "$LEV_CHECKPOINT" \
  --base "$BASE_MODEL" --base-revision "$BASE_REVISION" \
  --init-from "$CONTINUATION_CHECKPOINT" \
  --epochs 1 --seed 1 \
  --lr 2e-5 --head-lr 0 --weight-decay 0.01 \
  --lora 16 --lora-targets all --head-dim 256 \
  --batch 1 --accum 8 \
  --device cuda --dtype bf16 --weights-dtype bf16 --checkpointing \
  --max-state 32768 --max-branch 65536 --max-packed 65536 \
  --perm-kl 0 --perm-frac 0.3 --ord-w 0 \
  --p-none 0.1 --p-none-distract 0.12 --p-distract 0.15 --p-none-pair 0
```

This stage starts from the **continued decision checkpoint**, not directly
from the Granite base or stage 1. It changes the context limits and uses
physical batch 1 with accumulation 8, retaining an effective batch of 8.
The three augmentation probabilities remain `0.1 / 0.12 / 0.15`; only the
additional none-pair augmentation is disabled.

Across all three stages, `head_lr=0` means the pointer head uses the main
learning rate, **not** that it is frozen. The optimizer is AdamW with a
OneCycleLR schedule. Anchoring, permutation-KL, ordinal penalties, option
isolation, and the optional special-embedding flag are disabled. Required
added structural-token rows are still trained and saved by patched KEV.

## Expected training outputs

Each stage writes an adapter, pointer head, tokenizer, `training_config.json`,
and `training_metrics.json` into its checkpoint directory. The two warm-start
handoffs load those model parameters, not optimizer state. The joint LEV run
also writes `run.json`, `train.log`, and `train-mixture.jsonl` beside its
checkpoint, as described in the README.

The retained runs completed these training budgets without rejected or
truncated records:

| Stage | Epochs | Records before augmentation per epoch | Optimizer updates |
|---|---:|---:|---:|
| Initial decision training | 2 | 12,576 | 3,144 |
| Decision continuation | 1 | 3,425 | 429 |
| Joint log adaptation | 1 | 40,251 | 5,032 |

Decision-stage none-pair augmentation adds variants, so their recorded
`records_seen` exceeds the unaugmented training budget; it does not add
optimizer updates. For calibration and evaluation after training, follow the
[README workflow](../README.md#train-calibrate-and-evaluate) using separate
caller-supplied calibration and evaluation records.
