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
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "report_probability_quality.py"


def probabilities(logits: list[float]) -> list[float]:
    shifted = [value - max(logits) for value in logits]
    values = [math.exp(value) for value in shifted]
    return [value / sum(values) for value in values]


def prediction(
    logits: list[float],
    gold_index: int,
    row_index: int,
    *,
    question_type: str = "noul",
    option_keys: list[str] | None = None,
) -> dict:
    option_keys = option_keys or ["false", "true"]
    return {
        "input": "fixture",
        "row_index": row_index,
        "record_id": f"record-{row_index}",
        "question_id": "is_anomalous" if question_type == "noul" else "attack_technique",
        "question_type": question_type,
        "option_keys": option_keys,
        "gold_index": gold_index,
        "predicted_index": max(range(len(logits)), key=logits.__getitem__),
        "predicted_option": option_keys[max(range(len(logits)), key=logits.__getitem__)],
        "logits": logits,
        "probabilities": probabilities(logits),
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_temperature(path: Path, temperature: float = 2.0) -> str:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "fit",
                "temperature": temperature,
                "objective": "equal-group",
                "grid": {"minimum": 0.25, "maximum": 4.0, "points": 121},
                "inputs": {"calibration": {"path": "/calibration/predictions.jsonl", "rows": 2, "sha256": "a" * 64}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
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


def test_reports_weighted_binary_raw_and_fitted_probability_quality(tmp_path: Path) -> None:
    predictions = tmp_path / "anomaly.jsonl"
    prediction_hash = write_jsonl(
        predictions,
        [prediction([-2.197224577, 0.0], 1, 0), prediction([0.0, -2.197224577], 0, 1)],
    )
    population = tmp_path / "population.json"
    population.write_text(
        json.dumps(
            {
                "output_counts": {"normal": 1, "anomalous": 1},
                "source_counts": {"normal": 9, "anomalous": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    temperature = tmp_path / "temperature.json"
    temperature_hash = write_temperature(temperature)
    output = tmp_path / "report"

    run_script(
        "--predictions",
        f"hdfs={predictions}",
        "--population",
        f"hdfs={population}",
        "--temperature",
        str(temperature),
        "--output",
        str(output),
    )

    summary = json.loads((output / "summary.json").read_text())
    result = summary["inputs"]["hdfs"]
    assert result["predictions"] == {
        "path": str(predictions.resolve()),
        "rows": 2,
        "sha256": prediction_hash,
    }
    assert result["question_type"] == "noul"
    assert result["metric_conventions"] == {
        "brier": "binary",
        "ece": "positive",
        "bins": 10,
    }
    assert result["population"]["weights"] == {"normal": 9.0, "anomalous": 1.0}
    assert result["raw"]["nll"] == pytest.approx(-math.log(0.9))
    assert result["raw"]["brier"] == pytest.approx(0.01)
    assert result["raw"]["ece"] == pytest.approx(0.1)
    assert result["fitted"]["nll"] != result["raw"]["nll"]
    assert sum(item["count"] for item in result["raw"]["reliability_bins"]) == 2
    assert sum(item["weight"] for item in result["raw"]["reliability_bins"]) == 10.0
    assert summary["temperature"]["sha256"] == temperature_hash
    assert summary["temperature"]["value"] == 2.0
    assert hashlib.sha256(predictions.read_bytes()).hexdigest() == prediction_hash
    assert hashlib.sha256(temperature.read_bytes()).hexdigest() == temperature_hash
    assert "HDFS" not in (output / "REPORT.md").read_text()


def test_reports_unweighted_choice_with_categorical_metrics(tmp_path: Path) -> None:
    predictions = tmp_path / "attack.jsonl"
    write_jsonl(
        predictions,
        [prediction([math.log(0.1), math.log(0.2), math.log(0.7)], 2, 0, question_type="choice", option_keys=["a", "b", "c"])],
    )
    temperature = tmp_path / "temperature.json"
    write_temperature(temperature)
    output = tmp_path / "report"

    run_script(
        "--predictions",
        f"represented={predictions}",
        "--temperature",
        str(temperature),
        "--output",
        str(output),
    )

    result = json.loads((output / "summary.json").read_text())["inputs"]["represented"]
    assert result["question_type"] == "choice"
    assert result["metric_conventions"] == {
        "brier": "categorical",
        "ece": "top-label",
        "bins": 10,
    }
    assert result["population"] == {"kind": "observed", "weight": 1.0}
    assert result["raw"]["nll"] == pytest.approx(-math.log(0.7))
    assert result["raw"]["brier"] == pytest.approx(0.14)
    assert result["raw"]["ece"] == pytest.approx(0.3)


def test_rejects_mixed_question_types_and_choice_population_weights(tmp_path: Path) -> None:
    predictions = tmp_path / "mixed.jsonl"
    write_jsonl(
        predictions,
        [
            prediction([0.0, 1.0], 1, 0),
            prediction([0.0, 1.0, 2.0], 2, 1, question_type="choice", option_keys=["a", "b", "c"]),
        ],
    )
    temperature = tmp_path / "temperature.json"
    write_temperature(temperature)

    mixed = run_script(
        "--predictions",
        f"mixed={predictions}",
        "--temperature",
        str(temperature),
        "--output",
        str(tmp_path / "mixed-output"),
        check=False,
    )
    assert mixed.returncode != 0
    assert "one question type" in mixed.stderr

    choice = tmp_path / "choice.jsonl"
    write_jsonl(
        choice,
        [prediction([0.0, 1.0, 2.0], 2, 0, question_type="choice", option_keys=["a", "b", "c"])],
    )
    population = tmp_path / "population.json"
    population.write_text(
        json.dumps({"output_counts": {"normal": 1, "anomalous": 1}, "source_counts": {"normal": 1, "anomalous": 1}}),
        encoding="utf-8",
    )
    weighted_choice = run_script(
        "--predictions",
        f"choice={choice}",
        "--population",
        f"choice={population}",
        "--temperature",
        str(temperature),
        "--output",
        str(tmp_path / "choice-output"),
        check=False,
    )
    assert weighted_choice.returncode != 0
    assert "only supported for Noul" in weighted_choice.stderr


def test_refuses_unknown_population_and_existing_output(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [prediction([0.0, 1.0], 1, 0)])
    population = tmp_path / "population.json"
    population.write_text("{}\n", encoding="utf-8")
    temperature = tmp_path / "temperature.json"
    write_temperature(temperature)

    unknown = run_script(
        "--predictions",
        f"source={predictions}",
        "--population",
        f"other={population}",
        "--temperature",
        str(temperature),
        "--output",
        str(tmp_path / "unknown"),
        check=False,
    )
    assert unknown.returncode != 0
    assert "no matching predictions" in unknown.stderr

    output = tmp_path / "existing"
    output.mkdir()
    existing = run_script(
        "--predictions",
        f"source={predictions}",
        "--temperature",
        str(temperature),
        "--output",
        str(output),
        check=False,
    )
    assert existing.returncode != 0
    assert "refusing to overwrite" in existing.stderr
