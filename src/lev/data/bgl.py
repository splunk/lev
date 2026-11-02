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

"""BGL raw-log adapter: chronological, non-overlapping fixed-event windows."""

from __future__ import annotations

from pathlib import Path

from .records import anomaly_request


def _event(line: str, row: int) -> tuple[str, bool]:
    fields = line.rstrip("\r\n").split(maxsplit=1)
    if len(fields) != 2:
        raise ValueError(f"malformed BGL row {row}: expected label token and event")
    label, event = fields
    return event, label != "-"


def load_bgl_windows(log_path: str | Path, window_size: int = 50) -> list[dict]:
    """Create chronological non-overlapping windows; an incomplete tail is excluded."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    events = []
    with Path(log_path).open(encoding="utf-8", errors="replace") as stream:
        for row, line in enumerate(stream, start=1):
            if line.strip():
                events.append(_event(line, row))

    records = []
    for start in range(0, len(events) - window_size + 1, window_size):
        window = events[start : start + window_size]
        end = start + window_size
        records.append(
            anomaly_request(
                state="\n".join(event for event, _ in window),
                label=any(label for _, label in window),
                dataset="BGL",
                record_id=f"bgl/{start:09d}-{end:09d}",
                group_id=f"bgl/{start:09d}-{end:09d}",
                row=len(records),
                metadata={"event_start": start, "event_end": end, "event_count": window_size},
            )
        )
    if not records:
        raise ValueError(f"BGL input has fewer than {window_size} events")
    return records
