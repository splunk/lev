# LEV: A System-1 Decision Model for Log Intelligence

LEV is a research implementation of a small, typed decision model for system
logs. Starting from supplied KEV labelled-request JSONLs, LEV adapts a supported
causal-decoder backbone with LoRA and uses KEV's pointer head to return
probabilities over a caller-defined, closed answer set.

LEV does **not** autoregressively generate an answer, explanation, or JSON
object. Its Kev-based decision path bypasses the language-model vocabulary
head; application code maps the pointer-head distribution back to the
requested labels.

LEV is an independent project built from [Kev](https://github.com/jaredpalmer/kev).
It is not affiliated with TypeSafe AI and does not claim to reproduce Jev or
TypeSafe's proprietary RLCD training method.

## Implemented architecture

```text
supplied KEV records: state + typed questions + labels
    -> supported causal-decoder backbone + LoRA
    -> KEV pointer head
    -> per-question probabilities over the supplied alternatives
```

The workflow is **supplied KEV records → train → calibrate → evaluate**.
Callers provide separate labelled training, calibration, and evaluation files;
LEV does not select examples or create those splits. Source JSONLs remain
unchanged. During fine-tuning, LEV trains the LoRA parameters and pointer head
with cross-entropy and can warm-start from a compatible Kev checkpoint.

For the Lev-3B training recipe used in the paper, see
[Reproducing Lev-3B Training](docs/reproduction.md).

### Supplied-record format

Each nonempty JSONL line contains one object with a shared `state` and one or
more named `questions`. Every question needs a `label` for training and the
evaluation commands below. For readability, this generic example is formatted
across multiple lines:

```json
{
  "state": "Job started. Job completed successfully.",
  "questions": {
    "outcome": {
      "type": "choice",
      "instructions": "Which outcome is reported?",
      "criteria": {
        "success": "The job completed successfully",
        "failure": "The job failed"
      },
      "label": "success"
    }
  }
}
```

| Question type | Alternatives | Label | Probability order |
|---|---|---|---|
| `noul` | Binary, with optional `criteria` descriptions for `false` and `true` | JSON Boolean | `false`, then `true` |
| `choice` | 1–255 named alternatives in a `criteria` object | An alternative's key | Dictionary insertion order |
| `score` | 2–255 ordered descriptions in a `criteria` list | Zero-based level index | List order |

The state can be text or structured JSON; KEV renders structured values as
text. Question names and alternatives are caller-defined. Optional `_meta`
and per-question `src` fields can carry producer metadata; they are not
required in the supplied example.

Evaluation produces one prediction row per question, including `option_keys`,
`probabilities`, logits, the gold index, and the predicted index/option. Each
probability vector is normalized over that question's alternatives. For
`score`, the evaluator also records the expected zero-based level.

## Requirements and installation

- Linux on x86-64
- Python 3.12
- Git
- [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA GPU and one of the supported CUDA/PyTorch environments below for
  model evaluation and GPU training

If your checkout does **not** contain `third_party/kev/`, prepare the KEV source
from the repository root before installing dependencies:

```bash
python3 scripts/setup_kev.py
```

Checkouts that already contain `third_party/kev/` skip this source-setup step.
The script refuses existing destinations without changing them; it is not an
update command. Source setup requires access to GitHub and fetches the revision
pinned in
[`patches/kev/upstream-revision.txt`](patches/kev/upstream-revision.txt), applies
[`patches/kev/changes.patch`](patches/kev/changes.patch), and creates
`third_party/kev/`.

Choose exactly one locked environment. The CUDA 12.4/PyTorch 2.6 environment
is the baseline stack. The CUDA 12.8/PyTorch 2.8 environment additionally
installs the Triton, Flash Linear Attention, and causal-convolution versions
used by the H100 training path.

For CUDA 12.4 with PyTorch 2.6:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/lev-cuda124-torch26
uv sync --frozen --extra cuda124-torch26 --default-index https://pypi.org/simple
```

For CUDA 12.8 with PyTorch 2.8 and the optimized kernels:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/lev-cuda128-torch28
uv sync --frozen --extra cuda128-torch28 --default-index https://pypi.org/simple
```

The environment path is local to the machine, while the same repository can
be shared. The lockfile resolves general packages from public PyPI and each
PyTorch build from its matching public CUDA index. The two extras are mutually
exclusive.

CPU-only tests do not require GPU execution, but they still require one of the
two dependency environments above because LEV imports KEV and PyTorch.

Run the test suite:

```bash
uv run pytest -q
```

## Train, calibrate, and evaluate

Run these commands from the repository root after installing the selected
environment. Replace the example absolute paths with your supplied JSONLs and
a new run location. Training uses CUDA by default, and the model-evaluation
CLI requires CUDA. Calibration and reporting operate on saved predictions
without loading a model.

### 1. Train

```bash
uv run lev-train --train-input examples=/absolute/path/to/train.jsonl \
  --out /absolute/path/to/run/checkpoint
```

`--train-input NAME=PATH` is repeatable. The default proportional mixture uses
all supplied records; `equal_dataset` and explicit weighted mixtures are also
supported. Training validates the records and context limits before invoking
KEV. Use `lev-train --help` for backbone, revision, warm-start, context,
precision, and optimization options.

The run directory contains `train-mixture.jsonl`, `run.json`, `train.log`, and
the checkpoint directory. The checkpoint includes the LoRA weights, pointer
head, tokenizer, training configuration, and training metrics.

### 2. Generate calibration predictions and fit temperature

First run the trained checkpoint on the supplied calibration records:

```bash
uv run lev-evaluate --checkpoint /absolute/path/to/run/checkpoint \
  --records examples=/absolute/path/to/calibration.jsonl \
  --run-dir /absolute/path/to/run/calibration --role calibration
```

`lev-evaluate` writes `predictions.jsonl`, raw `metrics.json`, and a `run.json`
manifest. Predictions contain raw logits and probabilities at temperature 1,
regardless of any saved checkpoint temperature.

Fit a scalar temperature using only those calibration predictions:

```bash
uv run python scripts/calibrate_temperature.py fit --objective pooled \
  --predictions examples=/absolute/path/to/run/calibration/predictions.jsonl \
  --output /absolute/path/to/run/temperature.json
```

The output JSON records the fitted `temperature`, input hashes, fitting
objective, and raw/calibrated NLL. `pooled` weights questions equally;
`equal-group` gives equal weight to each named prediction group supplied by
repeating `--predictions NAME=PATH`. Keep the fitted artifact fixed for
evaluation; this command does not update the checkpoint.

### 3. Evaluate and report with the frozen temperature

Run model inference on the separate evaluation records:

```bash
uv run lev-evaluate --checkpoint /absolute/path/to/run/checkpoint \
  --records examples=/absolute/path/to/evaluation.jsonl \
  --run-dir /absolute/path/to/run/evaluation --role validation
```

The evaluator currently accepts `calibration` and `validation` roles. It
again writes raw predictions, raw metrics, and a run manifest. There is no
temperature argument to `lev-evaluate`; applying the fitted temperature is a
separate post-processing step:

```bash
uv run python scripts/report_probability_quality.py \
  --predictions examples=/absolute/path/to/run/evaluation/predictions.jsonl \
  --temperature /absolute/path/to/run/temperature.json \
  --output /absolute/path/to/run/probability-quality
```

This reads the frozen fit artifact without refitting and writes `summary.json`
and `REPORT.md`, comparing raw and temperature-scaled NLL, Brier score, and
ECE, with reliability bins in the summary. It does not rerun inference or
rewrite the checkpoint or saved predictions. A positive scalar temperature
leaves the argmax decisions unchanged.

Probability-quality reporting supports `noul` and `choice`; each supplied
prediction file must contain a single question type. It uses binary
Brier/positive-class ECE for `noul` and categorical Brier/top-label ECE for
`choice`. Training and model evaluation also support `score` questions.

Outputs are caller-selected, and existing run/report artifacts are not
overwritten. Inspect the interfaces with `lev-train --help`,
`lev-evaluate --help`, `python scripts/calibrate_temperature.py --help`, and
`python scripts/report_probability_quality.py --help` in the installed
environment.

## Repository layout

```text
src/lev/prepared.py                   supplied-record validation and mixtures
src/lev/train.py                      training orchestration and KEV delegation
src/lev/evaluate.py                   checkpoint inference and raw metrics
scripts/calibrate_temperature.py      temperature fitting and NLL evaluation
scripts/report_probability_quality.py report raw/frozen-temperature probability quality
scripts/setup_kev.py                  dependency source setup
patches/kev/                          pinned revision, integration patch, upstream license
third_party/kev/                      Kev source used by the editable installation
tests/                               training, evaluation, calibration, and setup tests
```

Run directories carry their source hashes and configuration. Keep large
checkpoints, raw predictions, logs, and downloaded datasets outside version
control.

## Attribution and terms

- LEV builds on [Jared Palmer's Kev](https://github.com/jaredpalmer/kev/tree/e943f21e40574d99cefb2d292089333bcda9047c),
  pinned to commit `e943f21e40574d99cefb2d292089333bcda9047c` with LEV integration
  changes distributed in [`patches/kev/changes.patch`](patches/kev/changes.patch).
  Kev's Apache-2.0 license is preserved in
  [`patches/kev/LICENSE`](patches/kev/LICENSE).
- The patched KEV runtime supports compatible Qwen3, Qwen3.5, Gemma 4, and
  Granite 4.1 causal-decoder backbones. Individual models remain subject to
  their own licenses and terms.
- HDFS_v1 and BGL come from [LogHub](https://github.com/logpai/loghub) and
  remain subject to their dataset terms. Cite the
  [LogHub paper](https://arxiv.org/abs/2008.06448), the
  [HDFS source paper](https://people.eecs.berkeley.edu/~jordan/papers/xu-etal-sosp09.pdf),
  and the [BGL source paper](https://ieeexplore.ieee.org/document/4273008) as
  appropriate.

LEV source code is licensed under [Apache-2.0](LICENSE). The licenses and
terms of Kev, the Qwen, Gemma, and Granite model backbones,
LogHub, HDFS, and BGL continue to apply to their respective components and
data.

## Research status

LEV is research software, not a production detector or autonomous remediation
system. Public benchmark labels are imperfect proxies for operational
incidents, and closed-set probabilities can still be wrong or miscalibrated
under distribution shift. Any deployment requires local validation,
calibration, drift monitoring, access controls, and human-reviewed action
thresholds.
