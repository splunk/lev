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

"""Report raw and temperature-scaled probability quality from saved LEV predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import kev.metrics as kev_metrics

from calibrate_temperature import load_prediction_groups, parse_named_predictions, sha256_file


METRIC_BINS = 10


def load_temperature(path: Path) -> tuple[float, dict]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("operation") != "fit":
            raise ValueError("temperature artifact must record a fit operation")
        temperature = artifact["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be numeric")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid temperature artifact: {error}") from error
    return temperature, {
        "path": str(path),
        "sha256": sha256_file(path),
        "value": temperature,
        "artifact": artifact,
    }


def _binary_labels(rows: list[dict]) -> np.ndarray:
    labels = []
    for row in rows:
        if row["type"] != "noul" or len(row["p"]) != 2 or set(row["keys"]) != {"false", "true"}:
            raise ValueError("population weighting is only supported for Noul false/true rows")
        labels.append(int(row["label"] == row["keys"].index("true")))
    return np.asarray(labels, dtype=int)


def load_population(path: Path, rows: list[dict]) -> tuple[np.ndarray, dict]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        output_counts = {
            "normal": int(artifact["output_counts"]["normal"]),
            "anomalous": int(artifact["output_counts"]["anomalous"]),
        }
        source_counts = {
            "normal": int(artifact["source_counts"]["normal"]),
            "anomalous": int(artifact["source_counts"]["anomalous"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid population manifest: {error}") from error
    if min(output_counts.values()) <= 0 or min(source_counts.values()) <= 0:
        raise ValueError(f"{path}: population counts must be positive")
    labels = _binary_labels(rows)
    observed = {
        "normal": int(np.sum(labels == 0)),
        "anomalous": int(np.sum(labels == 1)),
    }
    if observed != output_counts:
        raise ValueError(
            f"{path}: manifest output counts do not match predictions: "
            f"expected {output_counts}, observed {observed}"
        )
    weights = {
        name: source_counts[name] / output_counts[name]
        for name in ("normal", "anomalous")
    }
    row_weights = np.where(labels == 1, weights["anomalous"], weights["normal"])
    return row_weights, {
        "kind": "projected",
        "path": str(path),
        "sha256": sha256_file(path),
        "output_counts": output_counts,
        "source_counts": source_counts,
        "weights": weights,
    }


def metric_conventions(question_type: str) -> dict:
    if question_type == "noul":
        return {"brier": "binary", "ece": "positive", "bins": METRIC_BINS}
    if question_type == "choice":
        return {"brier": "categorical", "ece": "top-label", "bins": METRIC_BINS}
    raise ValueError(f"unsupported question type for probability-quality reporting: {question_type!r}")


def probability_metrics(rows: list[dict], weights: np.ndarray, conventions: dict) -> dict:
    report = kev_metrics.metrics(
        rows,
        weights=weights,
        brier=conventions["brier"],
        ece_target=conventions["ece"],
        bins=conventions["bins"],
        include_reliability_bins=True,
    )
    return {
        "nll": report["nll"],
        "brier": report["brier"],
        "ece": report["ece"],
        "reliability_bins": report["reliability_bins"],
    }


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# Probability quality",
        "",
        "| Input | Type | Probability version | NLL | Brier | ECE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, item in summary["inputs"].items():
        for version in ("raw", "fitted"):
            metrics = item[version]
            lines.append(
                f"| {name} | {item['question_type']} | {version} | "
                f"{metrics['nll']:.6f} | {metrics['brier']:.6f} | {metrics['ece']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Temperature artifact: `{summary['temperature']['path']}`",
            f"- Temperature SHA-256: `{summary['temperature']['sha256']}`",
            "- Predictions are read-only inputs and were not rewritten.",
            "- Machine-readable reliability bins and population weights are in `summary.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compute(
    *,
    predictions: list[tuple[str, Path]],
    populations: list[tuple[str, Path]],
    temperature_path: Path,
    output: Path,
) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    groups, inputs = load_prediction_groups(predictions)
    population_paths = dict(populations)
    unknown = set(population_paths) - set(groups)
    if unknown:
        raise ValueError(f"population inputs have no matching predictions: {sorted(unknown)}")
    temperature, temperature_artifact = load_temperature(temperature_path)
    summary = {
        "schema_version": 1,
        "temperature": temperature_artifact,
        "inputs": {},
        "method": {
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "kev_metrics": {"path": str(Path(kev_metrics.__file__).resolve()), "sha256": sha256_file(Path(kev_metrics.__file__).resolve())},
        },
    }
    for name, rows in groups.items():
        question_types = {row["type"] for row in rows}
        if len(question_types) != 1:
            raise ValueError(f"prediction input {name!r} must contain exactly one question type")
        question_type = next(iter(question_types))
        conventions = metric_conventions(question_type)
        if name in population_paths:
            weights, population = load_population(population_paths[name], rows)
        else:
            weights = np.ones(len(rows), dtype=float)
            population = {"kind": "observed", "weight": 1.0}
        raw = kev_metrics.served_at(rows, 1.0)
        fitted = kev_metrics.served_at(rows, temperature)
        summary["inputs"][name] = {
            "predictions": inputs[name],
            "question_type": question_type,
            "metric_conventions": conventions,
            "population": population,
            "raw": probability_metrics(raw, weights, conventions),
            "fitted": probability_metrics(fitted, weights, conventions),
        }
    json.dumps(summary, allow_nan=False)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(summary, output / "REPORT.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument(
        "--population",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="optional population manifest for a matching Noul prediction input",
    )
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = compute(
            predictions=parse_named_predictions(args.predictions),
            populations=parse_named_predictions(args.population),
            temperature_path=args.temperature,
            output=args.output,
        )
        print(json.dumps({"output": str(args.output.resolve()), "inputs": len(summary["inputs"])}, indent=2))
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
