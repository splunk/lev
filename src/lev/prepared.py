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

"""Small prepared-JSONL helpers for dataset-agnostic LEV experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from kev.data import load_records, materialize


@dataclass(frozen=True)
class NamedInput:
    name: str
    path: Path
    records: tuple[dict, ...]
    validated: tuple[dict, ...]
    sha256: str


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError(f"expected non-empty NAME=PATH, got {value!r}")
    return name, Path(raw_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_named_inputs(values: list[str]) -> list[NamedInput]:
    if not values:
        raise ValueError("at least one named input is required")
    result: list[NamedInput] = []
    names: set[str] = set()
    for value in values:
        name, path = parse_named_path(value)
        if name in names:
            raise ValueError(f"duplicate input name: {name}")
        names.add(name)
        if not path.is_file():
            raise ValueError(f"input {name!r} does not exist: {path}")
        raw: list[dict] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not raw:
            raise ValueError(f"input {name!r} is empty: {path}")
        try:
            validated = load_records(path, source=name)
            for record in validated:
                materialize(record)
        except Exception as error:
            raise ValueError(f"input {name!r} is not valid KEV JSONL: {error}") from error
        result.append(
            NamedInput(
                name=name,
                path=path,
                records=tuple(raw),
                validated=tuple(validated),
                sha256=sha256_file(path),
            )
        )
    return result


def parse_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        name, raw = value.split("=", 1) if "=" in value else ("", "")
        if not name or name in weights:
            raise ValueError(f"invalid or duplicate weight {value!r}; expected NAME=POSITIVE_NUMBER")
        try:
            weight = float(raw)
        except ValueError as error:
            raise ValueError(f"invalid weight {value!r}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight must be positive and finite: {value!r}")
        weights[name] = weight
    return weights


def _explicit_counts(inputs: list[NamedInput], weights: dict[str, float]) -> dict[str, int]:
    """Repeat, but never discard, supplied records to approximate explicit weights."""
    scale = max(len(item.records) / weights[item.name] for item in inputs)
    return {item.name: max(len(item.records), math.ceil(scale * weights[item.name])) for item in inputs}


def materialize_mixture(inputs: list[NamedInput], output: Path, *, policy: str, weights: dict[str, float]) -> dict:
    by_name = {item.name: item for item in inputs}
    if len(by_name) != len(inputs):
        raise ValueError("duplicate input name")
    unknown = set(weights) - set(by_name)
    if unknown:
        raise ValueError(f"weights name unknown inputs: {sorted(unknown)}")
    if policy == "proportional":
        targets = {item.name: len(item.records) for item in inputs}
    elif policy == "equal_dataset":
        target = max(len(item.records) for item in inputs)
        targets = {item.name: target for item in inputs}
    elif policy == "explicit":
        if set(weights) != set(by_name):
            missing = set(by_name) - set(weights)
            raise ValueError(f"explicit weights missing inputs: {sorted(missing)}")
        targets = _explicit_counts(inputs, weights)
    else:
        raise ValueError(f"unknown mixture policy: {policy}")

    records: list[dict] = []
    if policy == "proportional":
        records.extend(record for item in inputs for record in item.records)
    else:
        for index in range(max(targets.values(), default=0)):
            for item in inputs:
                if index < targets[item.name]:
                    records.append(item.records[index % len(item.records)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    total = len(records)
    return {
        "policy": policy,
        "requested_weights": dict(sorted(weights.items())),
        "realized_counts": dict(sorted(targets.items())),
        "realized_shares": {name: count / total for name, count in sorted(targets.items())},
        "records": total,
        "sha256": sha256_file(output),
    }
