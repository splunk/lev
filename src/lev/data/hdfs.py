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

"""HDFS raw-log adapter: events grouped into block sessions."""

from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path

from .records import anomaly_request


BLOCK_ID = re.compile(r"\bblk_-?\d+\b")


def _load_labels(path: str | Path) -> dict[str, bool]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"BlockId", "Label"}.issubset(reader.fieldnames):
            raise ValueError("HDFS labels need BlockId and Label columns")
        labels = {}
        for row_number, row in enumerate(reader, start=2):
            block_id = row["BlockId"].strip()
            value = row["Label"].strip().casefold()
            if value not in {"normal", "anomaly"}:
                raise ValueError(f"invalid HDFS label at row {row_number}: {row['Label']!r}")
            labels[block_id] = value == "anomaly"
    return labels


def load_hdfs_session_states(log_path: str | Path) -> OrderedDict[str, list[str]]:
    """Group raw HDFS lines by block ID without opening or reading the label file."""
    sessions: OrderedDict[str, list[str]] = OrderedDict()
    with Path(log_path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            event = line.rstrip("\r\n")
            for block_id in dict.fromkeys(BLOCK_ID.findall(event)):
                sessions.setdefault(block_id, []).append(event)
    if not sessions:
        raise ValueError(f"no HDFS block IDs found in {log_path}")
    return sessions


def load_hdfs_sessions(log_path: str | Path, labels_path: str | Path) -> list[dict]:
    """Group raw HDFS lines by block ID and map the separate ground-truth labels."""
    sessions = load_hdfs_session_states(log_path)

    labels = _load_labels(labels_path)
    missing = sorted(set(sessions) - set(labels))
    if missing:
        raise ValueError(f"missing labels for {len(missing)} HDFS sessions: {missing[:3]}")

    records = []
    for row, (block_id, events) in enumerate(sessions.items()):
        records.append(
            anomaly_request(
                state="\n".join(events),
                label=labels[block_id],
                dataset="HDFS",
                record_id=f"hdfs/{block_id}",
                group_id=f"hdfs/{block_id}",
                row=row,
                metadata={"block_id": block_id, "event_count": len(events)},
            )
        )
    return records
