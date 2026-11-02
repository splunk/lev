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

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "calibrate_temperature.py"


def load_script():
    spec = importlib.util.spec_from_file_location("calibrate_temperature", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction(logits: list[float], gold_index: int, row_index: int = 0) -> dict:
    shifted = [value - max(logits) for value in logits]
    exponentials = [math.exp(value) for value in shifted]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    return {
        "input": "fixture",
        "row_index": row_index,
        "record_id": f"record-{row_index}",
        "question_id": "is_anomalous",
        "question_type": "noul",
        "option_keys": ["false", "true"],
        "gold_index": gold_index,
        "predicted_index": max(range(len(logits)), key=logits.__getitem__),
        "predicted_option": ["false", "true"][max(range(len(logits)), key=logits.__getitem__)],
        "logits": logits,
        "probabilities": probabilities,
    }


def write_predictions(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_script(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "third_party" / "kev"), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


def test_fit_supports_explicit_equal_group_and_pooled_objectives(tmp_path: Path) -> None:
    small = tmp_path / "small.jsonl"
    large = tmp_path / "large.jsonl"
    small_hash = write_predictions(small, [prediction([4.0, 0.0], 1)])
    large_hash = write_predictions(
        large,
        [prediction([4.0, 0.0], 0, row_index=index) for index in range(10)],
    )

    outputs = {}
    for objective in ("equal-group", "pooled"):
        output = tmp_path / f"{objective}.json"
        run_script(
            "fit",
            "--objective",
            objective,
            "--predictions",
            f"small={small}",
            "--predictions",
            f"large={large}",
            "--output",
            str(output),
        )
        outputs[objective] = json.loads(output.read_text())

    equal_group = outputs["equal-group"]
    pooled = outputs["pooled"]
    assert equal_group["temperature"] == pytest.approx(4.0)
    assert equal_group["boundary_hit"] == "maximum"
    assert pooled["temperature"] != equal_group["temperature"]
    assert equal_group["grid"] == {"maximum": 4.0, "minimum": 0.25, "points": 121}
    assert equal_group["inputs"]["small"] == {
        "path": str(small.resolve()),
        "rows": 1,
        "sha256": small_hash,
    }
    assert equal_group["inputs"]["large"] == {
        "path": str(large.resolve()),
        "rows": 10,
        "sha256": large_hash,
    }
    assert set(equal_group["nll"]["groups"]) == {"small", "large"}
    assert equal_group["nll"]["objective"]["calibrated"] <= equal_group["nll"]["objective"]["raw"]
    assert equal_group["method"]["kev_metrics"]["commit"] == "1f04cfa28e6a708485b28a9585010b9e436b776e"
    assert equal_group["method"]["kev_metrics"]["upstream_sha256"] == "167a586334cfcced4adcc197758535c3bcda832b855c151095ef2d1968cf8983"
    assert equal_group["method"]["kev_metrics"]["sha256"] == "92c04571a920d3ae07f93ac8407e45628c6dcb8bc96c3987d869233b4ceaa24a"
    assert hashlib.sha256(small.read_bytes()).hexdigest() == small_hash
    assert hashlib.sha256(large.read_bytes()).hexdigest() == large_hash


def test_fit_passes_explicit_121_point_grid_and_objective_mapping(monkeypatch) -> None:
    module = load_script()
    adapted = module.adapt_prediction_row(
        prediction([1.0, 0.0], 0), "source", Path("fixture.jsonl"), 1
    )
    calls = []

    def capture_fit(rows, **kwargs):
        calls.append(kwargs)
        return 1.0

    monkeypatch.setattr(module.kev_metrics, "fit_temperature", capture_fit)
    for objective in ("equal-group", "pooled"):
        module.fit_result(
            {"source": [adapted]},
            {"source": {"path": "fixture.jsonl", "rows": 1, "sha256": "fixture"}},
            objective,
        )

    assert calls == [
        {"aggregation": "macro", "points": 121},
        {"aggregation": "micro", "points": 121},
    ]


def test_evaluate_applies_supplied_temperature_without_refitting(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    original_hash = write_predictions(
        predictions,
        [prediction([2.0, 0.0], 0), prediction([0.0, 2.0], 1, row_index=1)],
    )
    output = tmp_path / "evaluation.json"

    run_script(
        "evaluate",
        "--temperature",
        "2.0",
        "--predictions",
        f"unknowable={predictions}",
        "--output",
        str(output),
    )

    result = json.loads(output.read_text())
    assert result["operation"] == "evaluate"
    assert result["temperature"] == 2.0
    assert "objective" not in result
    assert result["nll"]["groups"]["unknowable"]["calibrated"] > result["nll"]["groups"]["unknowable"]["raw"]
    assert set(result["nll"]["aggregates"]) == {"equal-group", "pooled"}
    assert hashlib.sha256(predictions.read_bytes()).hexdigest() == original_hash


def test_adapter_preserves_option_order_logits_and_gold_index() -> None:
    module = load_script()
    row = prediction([0.25, -1.5], 1)

    adapted = module.adapt_prediction_row(row, "arbitrary-group", Path("fixture.jsonl"), 3)

    assert adapted["keys"] == row["option_keys"]
    assert adapted["logits"] == row["logits"]
    assert adapted["label"] == row["gold_index"]
    assert adapted["task"] == "arbitrary-group"
    assert adapted["source"] != "arbitrary-group"


def test_evaluate_restores_rows_with_a_recorded_inference_temperature(tmp_path: Path) -> None:
    predictions = tmp_path / "served.jsonl"
    row = prediction([1.0, 0.0], 0)
    row["inference_temperature"] = 2.0
    write_predictions(predictions, [row])
    output = tmp_path / "evaluation.json"

    run_script(
        "evaluate",
        "--temperature",
        "1.0",
        "--predictions",
        f"source={predictions}",
        "--output",
        str(output),
    )

    result = json.loads(output.read_text())
    expected_raw_nll = math.log1p(math.exp(-2.0))
    assert result["nll"]["groups"]["source"]["raw"] == pytest.approx(expected_raw_nll)
    assert result["nll"]["groups"]["source"]["calibrated"] == pytest.approx(expected_raw_nll)


def test_fit_accepts_multiclass_choice_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "choice.jsonl"
    logits = [0.25, -0.5, 1.5]
    shifted = [value - max(logits) for value in logits]
    exponentials = [math.exp(value) for value in shifted]
    total = sum(exponentials)
    row = prediction([0.0, 0.0], 0)
    row.update(
        question_type="choice",
        option_keys=["a", "b", "c"],
        gold_index=2,
        logits=logits,
        probabilities=[value / total for value in exponentials],
    )
    write_predictions(predictions, [row])
    output = tmp_path / "temperature.json"

    run_script(
        "fit",
        "--objective",
        "pooled",
        "--predictions",
        f"choice={predictions}",
        "--output",
        str(output),
    )

    result = json.loads(output.read_text())
    assert result["inputs"]["choice"]["rows"] == 1
    assert math.isfinite(result["temperature"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.pop("logits"), "missing logits"),
        (lambda row: row.__setitem__("logits", [float("nan"), 0.0]), "finite logits"),
        (lambda row: row.__setitem__("gold_index", 2), "gold_index"),
        (lambda row: row.__setitem__("probabilities", [1.0]), "same length"),
    ],
)
def test_fit_rejects_invalid_prediction_rows(tmp_path: Path, mutate, message: str) -> None:
    row = prediction([1.0, 0.0], 0)
    mutate(row)
    predictions = tmp_path / "invalid.jsonl"
    write_predictions(predictions, [row])

    completed = run_script(
        "fit",
        "--objective",
        "pooled",
        "--predictions",
        f"source={predictions}",
        "--output",
        str(tmp_path / "output.json"),
        check=False,
    )

    assert completed.returncode != 0
    assert message in completed.stderr


def test_fit_rejects_duplicate_or_empty_named_inputs(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(predictions, [prediction([1.0, 0.0], 0)])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    duplicate = run_script(
        "fit",
        "--objective",
        "pooled",
        "--predictions",
        f"source={predictions}",
        "--predictions",
        f"source={predictions}",
        "--output",
        str(tmp_path / "duplicate.json"),
        check=False,
    )
    assert duplicate.returncode != 0
    assert "duplicate prediction group" in duplicate.stderr

    no_rows = run_script(
        "fit",
        "--objective",
        "pooled",
        "--predictions",
        f"source={empty}",
        "--output",
        str(tmp_path / "empty.json"),
        check=False,
    )
    assert no_rows.returncode != 0
    assert "empty prediction input" in no_rows.stderr
