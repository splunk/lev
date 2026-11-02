# Copyright 2026 Splunk Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import sys

import pytest
import torch

from lev.evaluate import (
    _prediction_rows,
    aggregate_predictions,
    main,
    resolve_evaluation_dtype,
)
from lev.prepared import NamedInput


def test_generic_metrics_cover_noul_choice_and_score():
    rows = [
        {"input": "a", "question_type": "noul", "gold_index": 1, "predicted_index": 1, "probabilities": [0.1, 0.9]},
        {"input": "a", "question_type": "noul", "gold_index": 0, "predicted_index": 1, "probabilities": [0.4, 0.6]},
        {"input": "a", "question_type": "choice", "option_keys": ["a", "b"], "gold_index": 0, "predicted_index": 0, "probabilities": [0.8, 0.2]},
        {"input": "a", "question_type": "score", "gold_index": 1, "predicted_index": 1, "probabilities": [0.2, 0.7, 0.1]},
    ]
    metrics = aggregate_predictions(rows)
    noul = metrics["a"]["noul"]
    assert noul["count"] == 2
    assert noul["accuracy"] == 0.5
    assert noul["precision"] == 0.5
    assert noul["recall"] == 1.0
    assert noul["positive_f1"] == pytest.approx(2 / 3)
    assert noul["macro_f1"] == pytest.approx(1 / 3)
    assert noul["brier"] == pytest.approx((0.1**2 + 0.6**2) / 2)
    assert metrics["a"]["choice"]["accuracy"] == 1.0
    assert metrics["a"]["score"]["accuracy"] == 1.0


def test_choice_macro_f1_averages_gold_present_classes():
    rows = [
        {"input": "a", "question_type": "choice", "option_keys": ["a", "b", "c"], "gold_index": 0, "predicted_index": 0, "probabilities": [0.8, 0.1, 0.1]},
        {"input": "a", "question_type": "choice", "option_keys": ["a", "b", "c"], "gold_index": 0, "predicted_index": 2, "probabilities": [0.1, 0.2, 0.7]},
        {"input": "a", "question_type": "choice", "option_keys": ["a", "b", "c"], "gold_index": 1, "predicted_index": 2, "probabilities": [0.1, 0.2, 0.7]},
    ]

    metrics = aggregate_predictions(rows)["a"]["choice"]

    assert metrics["macro_f1_gold_present"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("keys", "indices", "expected"),
    [
        # a -> a, b -> a: F1(a) = 2/3, F1(b) = 0, despite reversed order.
        ([["a", "b"], ["b", "a"]], [(0, 0), (0, 1)], 1 / 3),
        # a -> c, b -> b: average only gold a/b, not predicted-only c.
        ([["a", "c"], ["b", "c"]], [(0, 1), (0, 0)], 1 / 2),
    ],
)
def test_choice_macro_f1_uses_each_records_option_keys(keys, indices, expected):
    rows = [
        {"input": "a", "question_type": "choice", "option_keys": options,
         "gold_index": gold, "predicted_index": predicted,
         "probabilities": [0.8, 0.2] if predicted == 0 else [0.2, 0.8]}
        for options, (gold, predicted) in zip(keys, indices)
    ]

    assert aggregate_predictions(rows)["a"]["choice"]["macro_f1_gold_present"] == pytest.approx(expected)


def test_average_precision_is_invariant_to_tied_input_order():
    forward = [
        {"input": "a", "question_type": "noul", "gold_index": 1, "predicted_index": 1, "probabilities": [0.5, 0.5]},
        {"input": "a", "question_type": "noul", "gold_index": 0, "predicted_index": 1, "probabilities": [0.5, 0.5]},
    ]
    reverse = list(reversed(forward))
    assert aggregate_predictions(forward)["a"]["noul"]["auprc"] == 0.5
    assert aggregate_predictions(reverse)["a"]["noul"]["auprc"] == 0.5


def test_bf16_checkpoint_rejects_false_fp32_override():
    assert resolve_evaluation_dtype("bf16", None) == "bf16"
    assert resolve_evaluation_dtype("fp32", "bf16") == "bf16"
    with pytest.raises(ValueError, match="BF16-trained"):
        resolve_evaluation_dtype("bf16", "fp32")


def test_prediction_rows_cover_every_question_type(monkeypatch, tmp_path):
    request = {
        "state": "one prepared state",
        "questions": {
            "binary": {"type": "noul", "instructions": "Binary?", "label": True, "src": "fixture"},
            "choice": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "first", "b": "second"},
                "label": "a",
                "src": "fixture",
            },
            "score": {
                "type": "score",
                "instructions": "How much?",
                "criteria": ["low", "middle", "high"],
                "label": 1,
                "src": "fixture",
            },
        },
        "_meta": {
            "id": "fixture/0",
            "source": "fixture",
            "split": "validation",
            "phrasing": {"binary": {"index": 2, "is_ood": True}},
        },
    }

    class FakeModel:
        def __init__(self):
            self.head = type("Head", (), {"temperature": 7.0})()

        def encode(self, tokenizer, record, strict):
            assert strict is True
            return record

        def forward_batch(self, encoded):
            return [[torch.tensor([0.0, 1.0]), torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])]]

    fake_model = FakeModel()
    monkeypatch.setattr("kev.evaluate.load", lambda *args, **kwargs: (object(), fake_model))
    named = NamedInput("fixture", tmp_path / "unused.jsonl", (request,), (request,), "hash")
    rows = list(
        _prediction_rows(
            [named],
            tmp_path / "checkpoint",
            1,
            {"max_state": 10, "max_branch": 20, "max_packed": 30},
            "fp32",
        )
    )
    assert [row["question_type"] for row in rows] == ["noul", "choice", "score"]
    assert {row["row_index"] for row in rows} == {0}
    assert rows[1]["option_keys"] == ["a", "b"]
    assert rows[2]["expected_level"] == pytest.approx(1.0)
    assert rows[0]["phrasing"] == {"index": 2, "is_ood": True}
    assert fake_model.head.temperature == 1.0

    request["_meta"]["phrasing"] = "producer-defined"
    opaque_rows = list(
        _prediction_rows(
            [named],
            tmp_path / "checkpoint",
            1,
            {"max_state": 10, "max_branch": 20, "max_packed": 30},
            "fp32",
        )
    )
    assert all("phrasing" not in row for row in opaque_rows)


def test_main_forwards_and_records_checkpoint_dtype(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        {"weights_dtype": "bf16", "max_state": 10, "max_branch": 20, "max_packed": 30},
        checkpoint / "head.pt",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "adapter_config.json").write_text("{}")
    records = tmp_path / "validation.jsonl"
    request_without_meta = {
        "state": "state without producer metadata",
        "questions": {"q": {"type": "noul", "instructions": "Anomaly?", "label": False}},
    }
    request_with_meta = {
        "state": "state",
        "questions": {"q": {"type": "noul", "instructions": "Anomaly?", "label": False}},
        "_meta": {"id": "fixture/0", "source": "fixture", "split": "producer-defined"},
    }
    records.write_text(json.dumps(request_without_meta) + "\n" + json.dumps(request_with_meta) + "\n")
    captured = {}

    class FakeModel:
        def __init__(self):
            self.head = type("Head", (), {"temperature": 2.0})()

        def encode(self, tokenizer, record, strict):
            return record

        def forward_batch(self, encoded):
            return [[torch.tensor([1.0, 0.0])]]

    def fake_load(*args, **kwargs):
        captured["dtype"] = kwargs["dtype"]
        return object(), FakeModel()

    monkeypatch.setattr("kev.evaluate.load", fake_load)
    monkeypatch.setattr("lev.evaluate._runtime", lambda: {"gpu_name": "fixture"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    run_dir = tmp_path / "run"
    command = [
        "lev-evaluate",
        "--checkpoint", str(checkpoint),
        "--records", f"fixture={records}",
        "--role", "validation",
        "--run-dir", str(run_dir),
        "--batch-size", "1",
    ]
    for flag in ("--checkpoint", "--records", "--run-dir"):
        index = command.index(flag)
        monkeypatch.setattr(sys, "argv", command[:index] + command[index + 2:])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
        assert flag in capsys.readouterr().err
        assert captured == {}
        assert not run_dir.exists()

    monkeypatch.setattr(sys, "argv", command)
    main()
    manifest = json.loads((run_dir / "run.json").read_text())
    assert captured["dtype"] is torch.bfloat16
    assert manifest["role"] == "validation"
    assert manifest["status"] == "completed"
    assert manifest["weights_dtype"] == "bf16"
    assert manifest["limits"] == {"max_state": 10, "max_branch": 20, "max_packed": 30}
    assert manifest["checkpoint_sha256"]["head.pt"] == hashlib.sha256((checkpoint / "head.pt").read_bytes()).hexdigest()
    assert manifest["inputs"]["fixture"]["sha256"] == hashlib.sha256(records.read_bytes()).hexdigest()
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert [row["row_index"] for row in predictions] == [0, 1]

    with pytest.raises(FileExistsError):
        main()
    rejected_dir = tmp_path / "rejected"
    rejected_command = list(command)
    rejected_command[command.index("--run-dir") + 1] = str(rejected_dir)
    monkeypatch.setattr(sys, "argv", rejected_command + ["--weights-dtype", "fp32"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "BF16-trained" in capsys.readouterr().err
    assert not rejected_dir.exists()
