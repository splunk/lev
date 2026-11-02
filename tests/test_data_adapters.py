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

import json
from pathlib import Path

import pytest

from kev.data import materialize
from lev.data.bgl import load_bgl_windows
from lev.data.hdfs import load_hdfs_sessions
from lev.data.records import deterministic_subset, write_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


def test_hdfs_groups_by_block_and_emits_kev_requests(tmp_path):
    records = load_hdfs_sessions(FIXTURES / "hdfs.log", FIXTURES / "hdfs_labels.csv")

    assert [record["_meta"]["block_id"] for record in records] == ["blk_-1", "blk_2"]
    assert [len(record["state"].splitlines()) for record in records] == [2, 2]
    assert [record["questions"]["is_anomalous"]["label"] for record in records] == [False, True]
    assert all(record["_meta"]["dataset"] == "HDFS" for record in records)
    assert all("Normal" not in record["state"] and "Anomaly" not in record["state"] for record in records)
    assert materialize(records[0])["questions"][0]["qtype"] == "noul"

    output = tmp_path / "hdfs.jsonl"
    write_jsonl(output, records)
    assert len([json.loads(line) for line in output.read_text().splitlines()]) == 2


def test_hdfs_rejects_missing_labels(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("BlockId,Label\nblk_-1,Normal\n")
    with pytest.raises(ValueError, match="missing labels"):
        load_hdfs_sessions(FIXTURES / "hdfs.log", labels)


def test_bgl_builds_chronological_non_overlapping_windows_without_label_tokens():
    records = load_bgl_windows(FIXTURES / "bgl.log", window_size=2)

    assert len(records) == 2
    assert [record["_meta"]["event_start"] for record in records] == [0, 2]
    assert [record["_meta"]["event_end"] for record in records] == [2, 4]
    assert [record["questions"]["is_anomalous"]["label"] for record in records] == [True, False]
    assert "KERNDTLB" not in records[0]["state"]
    assert not records[0]["state"].splitlines()[0].startswith("-")
    assert materialize(records[0])["questions"][0]["qtype"] == "noul"


def test_bgl_drops_incomplete_tail_by_default():
    records = load_bgl_windows(FIXTURES / "bgl.log", window_size=3)
    assert len(records) == 1
    assert records[0]["_meta"]["event_end"] == 3


def test_bgl_default_is_non_overlapping_50_event_windows(tmp_path):
    log = tmp_path / "bgl.log"
    rows = [f"- {index:010d} normal event {index}" for index in range(101)]
    log.write_text("\n".join(rows) + "\n")

    records = load_bgl_windows(log)

    assert len(records) == 2
    assert [record["_meta"]["event_start"] for record in records] == [0, 50]
    assert [record["_meta"]["event_end"] for record in records] == [50, 100]
    assert all(len(record["state"].splitlines()) == 50 for record in records)


def test_subset_is_deterministic_and_preserves_records():
    records = load_bgl_windows(FIXTURES / "bgl.log", window_size=1)
    first = deterministic_subset(records, maximum=2, seed=7)
    second = deterministic_subset(records, maximum=2, seed=7)

    assert first == second
    assert len(first) == 2
    assert all(record in records for record in first)
    assert deterministic_subset(records, maximum=None, seed=7) == records
