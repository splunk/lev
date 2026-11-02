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

"""Shared construction of KEV-labelled anomaly-decision requests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


ANOMALY_CRITERIA = {
    "false": "Routine or successful normal system operation",
    "true": "Failure, error, injected fault, or behavior inconsistent with normal operation",
}


def anomaly_request(
    *,
    state: str,
    label: bool,
    dataset: str,
    record_id: str,
    group_id: str,
    row: int,
    metadata: dict | None = None,
) -> dict:
    """Return KEV's existing labelled-request shape plus non-model metadata."""
    state = state.strip()
    if not state:
        raise ValueError("an anomaly request needs non-empty state")
    source = f"lev_{dataset.lower()}"
    return {
        "state": state,
        "questions": {
            "is_anomalous": {
                "type": "noul",
                "instructions": "Is this log session anomalous?",
                "criteria": dict(ANOMALY_CRITERIA),
                "label": bool(label),
                "src": "log_anomaly_classification",
            }
        },
        "_meta": {
            "source": source,
            "dataset": dataset,
            "id": record_id,
            "group_id": group_id,
            "row": row,
            "split": "unassigned",
            "variant": "clean",
            **(metadata or {}),
        },
    }


def deterministic_subset(records: list[dict], maximum: int | None, seed: int) -> list[dict]:
    """Select at most ``maximum`` records reproducibly while retaining source order."""
    if maximum is None or maximum >= len(records):
        return list(records)
    if maximum < 1:
        raise ValueError("maximum must be positive")
    indices = sorted(random.Random(seed).sample(range(len(records)), maximum))
    return [records[index] for index in indices]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
