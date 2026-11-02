#!/usr/bin/env python3
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

"""Fit or evaluate scalar temperature calibration on saved LEV predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import kev.metrics as kev_metrics


TEMPERATURE_MINIMUM = 0.25
TEMPERATURE_MAXIMUM = 4.0
TEMPERATURE_POINTS = 121
KEV_METRICS_REPOSITORY = "https://github.com/jaredpalmer/kev.git"
KEV_METRICS_COMMIT = "1f04cfa28e6a708485b28a9585010b9e436b776e"
KEV_METRICS_UPSTREAM_SHA256 = "167a586334cfcced4adcc197758535c3bcda832b855c151095ef2d1968cf8983"
KEV_METRICS_SHA256 = "92c04571a920d3ae07f93ac8407e45628c6dcb8bc96c3987d869233b4ceaa24a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_predictions(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        if name in seen:
            raise ValueError(f"duplicate prediction group: {name}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        parsed.append((name, path))
    return parsed


def _finite_numbers(value, field: str, location: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{location}: {field} must contain only numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{location}: expected finite {field}")
    return result


def adapt_prediction_row(row: dict, group: str, path: Path, line_number: int) -> dict:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{location}: prediction row must be an object")
    if "logits" not in row:
        raise ValueError(f"{location}: missing logits")
    logits = _finite_numbers(row["logits"], "logits", location)
    probabilities = _finite_numbers(row.get("probabilities"), "probabilities", location)
    option_keys = row.get("option_keys")
    if not isinstance(option_keys, list) or not option_keys:
        raise ValueError(f"{location}: option_keys must be a non-empty list")
    if len(logits) != len(probabilities) or len(logits) != len(option_keys):
        raise ValueError(f"{location}: option_keys, logits, and probabilities must have the same length")
    if any(value < 0 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError(f"{location}: probabilities must be non-negative and sum to one")
    gold_index = row.get("gold_index")
    if isinstance(gold_index, bool) or not isinstance(gold_index, int) or not 0 <= gold_index < len(option_keys):
        raise ValueError(f"{location}: gold_index must identify an option")
    question_type = row.get("question_type")
    if not isinstance(question_type, str) or not question_type:
        raise ValueError(f"{location}: question_type must be a non-empty string")
    inference_temperature = row.get("inference_temperature", 1.0)
    if isinstance(inference_temperature, bool) or not isinstance(inference_temperature, (int, float)):
        raise ValueError(f"{location}: inference_temperature must be numeric")
    inference_temperature = float(inference_temperature)
    if not math.isfinite(inference_temperature) or inference_temperature <= 0:
        raise ValueError(f"{location}: inference_temperature must be finite and positive")
    return {
        "p": probabilities,
        "logits": logits,
        "label": gold_index,
        "type": question_type,
        "keys": option_keys,
        "task": group,
        "source": "lev_prediction",
        "variant": "clean",
        "inference_temperature": inference_temperature,
    }


def load_prediction_groups(named: Sequence[tuple[str, Path]]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    groups: dict[str, list[dict]] = {}
    inputs: dict[str, dict] = {}
    for name, path in named:
        rows: list[dict] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                rows.append(adapt_prediction_row(row, name, path, line_number))
        if not rows:
            raise ValueError(f"empty prediction input: {path}")
        groups[name] = rows
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(rows)}
    return groups, inputs


def _flatten(groups: dict[str, list[dict]]) -> list[dict]:
    return [row for rows in groups.values() for row in rows]


def _nll_by_group(groups: dict[str, list[dict]]) -> dict[str, float]:
    return {name: float(kev_metrics.metrics(rows)["nll"]) for name, rows in groups.items()}


def _aggregate_nll(group_nll: dict[str, float], groups: dict[str, list[dict]], objective: str) -> float:
    if objective == "equal-group":
        return float(np.mean(list(group_nll.values())))
    return float(np.average(list(group_nll.values()), weights=[len(groups[name]) for name in group_nll]))


def _nll_diagnostics(
    raw_groups: dict[str, list[dict]], calibrated_groups: dict[str, list[dict]]
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    raw_nll = _nll_by_group(raw_groups)
    calibrated_nll = _nll_by_group(calibrated_groups)
    groups = {
        name: {"raw": raw_nll[name], "calibrated": calibrated_nll[name]}
        for name in raw_groups
    }
    aggregates = {
        objective: {
            "raw": _aggregate_nll(raw_nll, raw_groups, objective),
            "calibrated": _aggregate_nll(calibrated_nll, calibrated_groups, objective),
        }
        for objective in ("equal-group", "pooled")
    }
    return groups, aggregates


def _method_identity() -> dict:
    metrics_path = Path(kev_metrics.__file__).resolve()
    metrics_hash = sha256_file(metrics_path)
    if metrics_hash != KEV_METRICS_SHA256:
        raise ValueError(
            f"kev.metrics hash mismatch: expected {KEV_METRICS_SHA256}, got {metrics_hash}"
        )
    return {
        "name": "kev.metrics scalar temperature calibration",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "kev_metrics": {
            "repository": KEV_METRICS_REPOSITORY,
            "commit": KEV_METRICS_COMMIT,
            "path": "kev/metrics.py",
            "upstream_sha256": KEV_METRICS_UPSTREAM_SHA256,
            "sha256": metrics_hash,
        },
    }


def _base_result(operation: str, temperature: float, inputs: dict[str, dict]) -> dict:
    return {
        "schema_version": 1,
        "operation": operation,
        "temperature": temperature,
        "inputs": inputs,
        "method": _method_identity(),
    }


def fit_result(groups: dict[str, list[dict]], inputs: dict[str, dict], objective: str) -> dict:
    raw_groups = {name: kev_metrics.served_at(rows, 1.0) for name, rows in groups.items()}
    aggregation = {"equal-group": "macro", "pooled": "micro"}[objective]
    temperature = kev_metrics.fit_temperature(
        _flatten(raw_groups), aggregation=aggregation, points=TEMPERATURE_POINTS
    )
    calibrated_groups = {
        name: kev_metrics.served_at(rows, temperature) for name, rows in raw_groups.items()
    }
    group_nll, aggregates = _nll_diagnostics(raw_groups, calibrated_groups)
    result = _base_result("fit", temperature, inputs)
    result.update(
        {
            "objective": objective,
            "grid": {
                "minimum": TEMPERATURE_MINIMUM,
                "maximum": TEMPERATURE_MAXIMUM,
                "points": TEMPERATURE_POINTS,
            },
            "boundary_hit": (
                "minimum"
                if temperature == TEMPERATURE_MINIMUM
                else "maximum" if temperature == TEMPERATURE_MAXIMUM else None
            ),
            "nll": {"groups": group_nll, "objective": aggregates[objective]},
        }
    )
    return result


def evaluation_result(groups: dict[str, list[dict]], inputs: dict[str, dict], temperature: float) -> dict:
    raw_groups = {name: kev_metrics.served_at(rows, 1.0) for name, rows in groups.items()}
    calibrated_groups = {
        name: kev_metrics.served_at(rows, temperature) for name, rows in raw_groups.items()
    }
    group_nll, aggregates = _nll_diagnostics(raw_groups, calibrated_groups)
    result = _base_result("evaluate", temperature, inputs)
    result["nll"] = {"groups": group_nll, "aggregates": aggregates}
    return result


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit or evaluate scalar temperature calibration on saved LEV predictions."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    fit = subparsers.add_parser("fit", help="fit a temperature on the supplied predictions")
    fit.add_argument("--objective", choices=("equal-group", "pooled"), required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a supplied frozen temperature without refitting"
    )
    evaluate.add_argument("--temperature", type=float, required=True)

    for command in (fit, evaluate):
        command.add_argument(
            "--predictions", action="append", required=True, metavar="NAME=PATH"
        )
        command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        named = parse_named_predictions(args.predictions)
        groups, inputs = load_prediction_groups(named)
        if args.operation == "fit":
            result = fit_result(groups, inputs, args.objective)
        else:
            result = evaluation_result(groups, inputs, args.temperature)
        write_json_exclusive(args.output, result)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
