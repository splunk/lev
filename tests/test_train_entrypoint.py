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
from pathlib import Path

import pytest

from lev.prepared import load_named_inputs, materialize_mixture, parse_named_path
from lev.train import (
    checkpoint_identity,
    kev_train_command,
    main as train_main,
    run_and_tee,
    validate_training_context,
    write_run_manifest,
)


def test_kev_command_is_released_08b_delta_run(tmp_path):
    command = kev_train_command(
        data=tmp_path / "records.jsonl",
        output=tmp_path / "run",
        seed=5,
        max_state=1024,
        max_branch=1280,
        max_packed=1536,
        device="cpu",
        dtype="bf16",
        weights_dtype="bf16",
        checkpointing=True,
        batch=8,
        accum=1,
        base="Qwen/Qwen3.5-0.8B-Base",
        base_revision="revision",
        init_from="checkpoint",
        epochs=3,
        lr=1e-5,
        head_lr=2e-5,
        weight_decay=0.02,
        lora=8,
        lora_targets="dense",
        head_dim=128,
        option_isolation=True,
        special_embeddings=False,
        ord_w=0.1,
        perm_kl=0.2,
        perm_frac=0.4,
        anchor="anchors.json",
        anchor_w=0.3,
        anchor_sources="HDFS",
        p_none=0.0,
        p_none_distract=0.0,
        p_distract=0.0,
        p_none_pair=0.0,
    )
    joined = " ".join(command)
    assert "Qwen/Qwen3.5-0.8B-Base" in joined
    assert "--base_revision revision" in joined
    assert "--init_from checkpoint" in joined
    assert "--epochs 3" in joined
    assert "--lr 1e-05" in joined
    assert "--head_lr 2e-05" in joined
    assert "--lora 8" in joined
    assert "--lora_targets dense" in joined
    assert "--head_dim 128" in joined
    assert "--option_isolation 1" in joined
    assert "--max_state 1024" in joined
    assert "--max_branch 1280" in joined
    assert "--max_packed 1536" in joined
    assert "--dtype bf16" in joined
    assert "--weights_dtype bf16" in joined
    assert "--checkpointing 1" in joined
    assert "--batch 8" in joined
    assert "--accum 1" in joined
    assert "--p_none 0" in joined


def test_gemma_backbone_configuration_is_forwarded_and_recorded(tmp_path):
    command = kev_train_command(
        data=tmp_path / "records.jsonl", output=tmp_path / "checkpoint", seed=0,
        max_state=512, max_branch=768, max_packed=1024, device="cuda", dtype="bf16",
        weights_dtype="bf16", checkpointing=True, batch=1, accum=1,
        base="google/gemma-4-E2B", base_revision="d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        init_from="", lora_targets="all",
    )
    joined = " ".join(command)
    assert "--base google/gemma-4-E2B" in joined
    assert "--base_revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f" in joined
    assert "--init_from" not in command
    assert "--max_state 512 --max_branch 768 --max_packed 1024" in joined
    assert "--dtype bf16 --weights_dtype bf16 --checkpointing 1" in joined


def _record(record_id: str, label: bool = False) -> dict:
    return {
        "state": f"state {record_id}",
        "questions": {"q": {"type": "noul", "instructions": "anomaly?", "label": label}},
        "_meta": {
            "id": record_id,
            "dataset": record_id.split("/")[0],
            "source": record_id.split("/")[0],
            "split": "train",
        },
    }


def test_named_prepared_inputs_and_mixtures_are_deterministic(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(_record(f"a/{i}")) for i in range(2)) + "\n")
    b.write_text(json.dumps(_record("b/0", True)) + "\n")
    assert parse_named_path(f"a={a}") == ("a", a)
    inputs = load_named_inputs([f"a={a}", f"b={b}"])
    assert [item.name for item in inputs] == ["a", "b"]

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    summary1 = materialize_mixture(inputs, first, policy="equal_dataset", weights={})
    summary2 = materialize_mixture(inputs, second, policy="equal_dataset", weights={})
    assert first.read_bytes() == second.read_bytes()
    assert summary1 == summary2
    assert summary1["realized_counts"] == {"a": 2, "b": 2}
    assert [row["_meta"]["id"] for row in map(json.loads, first.read_text().splitlines())] == [
        "a/0",
        "b/0",
        "a/1",
        "b/0",
    ]


def test_proportional_mixture_concatenates_inputs_without_reordering(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(_record(f"a/{i}")) for i in range(2)) + "\n")
    b.write_text("\n".join(json.dumps(_record(f"b/{i}")) for i in range(2)) + "\n")
    inputs = load_named_inputs([f"a={a}", f"b={b}"])

    output = tmp_path / "mix.jsonl"
    materialize_mixture(inputs, output, policy="proportional", weights={})

    assert [row["_meta"]["id"] for row in map(json.loads, output.read_text().splitlines())] == [
        "a/0",
        "a/1",
        "b/0",
        "b/1",
    ]


def test_named_prepared_inputs_accept_missing_and_arbitrary_producer_metadata(tmp_path):
    without_meta = _record("a/0")
    without_meta.pop("_meta")
    with_meta = _record("a/1")
    with_meta["_meta"] = {"split": "producer-defined", "custom": {"keep": True}}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(without_meta) + "\n" + json.dumps(with_meta) + "\n")

    loaded = load_named_inputs([f"a={path}"])[0]

    assert loaded.records == (without_meta, with_meta)
    assert loaded.validated[0]["_meta"]["split"] == "custom"
    assert loaded.validated[1]["_meta"]["split"] == "producer-defined"
    assert loaded.validated[1]["_meta"]["custom"] == {"keep": True}


def test_train_input_rejects_context_before_kev_launch(monkeypatch, tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_record("a/0")) + "\n")
    checkpoint = tmp_path / "run" / "checkpoint"
    launched = False

    def reject_context(*args, **kwargs):
        raise ValueError("record exceeds configured context")

    def fake_run(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr("lev.train.validate_training_context", reject_context)
    monkeypatch.setattr("lev.train.run_and_tee", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lev-train", "--train-input", f"a={source}", "--out", str(checkpoint), "--device", "cpu"],
    )

    with pytest.raises(SystemExit):
        train_main()

    assert launched is False
    assert not (checkpoint.parent / "run.json").exists()


def test_validate_training_context_rejects_records_that_do_not_fit(monkeypatch):
    tokenizer = object()
    records = [_record("a/accepted"), _record("a/rejected")]
    calls = []

    monkeypatch.setattr("kev.model.load_tokenizer", lambda base, revision: tokenizer)

    def fake_fits_context(actual_tokenizer, record, *, max_state, max_branch, max_packed):
        calls.append((actual_tokenizer, record["_meta"]["id"], max_state, max_branch, max_packed))
        return record["_meta"]["id"] != "a/rejected"

    monkeypatch.setattr("kev.train.fits_context", fake_fits_context)

    with pytest.raises(ValueError, match=r"1 training records exceed configured context.*a/rejected"):
        validate_training_context(
            records,
            base="fixture-base",
            revision="fixture-revision",
            limits={"max_state": 10, "max_branch": 20, "max_packed": 30},
        )

    assert calls == [
        (tokenizer, "a/accepted", 10, 20, 30),
        (tokenizer, "a/rejected", 10, 20, 30),
    ]


def test_train_input_preserves_all_questions_and_records_verified_hashes(monkeypatch, tmp_path):
    record = _record("a/0")
    record["questions"]["choice"] = {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "first", "b": "second"},
        "label": "b",
    }
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(record) + "\n")
    checkpoint = tmp_path / "run" / "checkpoint"
    delegated_records = None

    monkeypatch.setattr("lev.train.validate_training_context", lambda *args, **kwargs: None)
    monkeypatch.setattr("lev.train.runtime_identity", lambda device: {"requested_device": device, "gpu": None})

    def fake_run(command, log_path):
        nonlocal delegated_records
        data = Path(command[command.index("--data") + 1])
        delegated_records = [json.loads(line) for line in data.read_text().splitlines()]
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "head.pt", "adapter_config.json"):
            (output / name).write_bytes(name.encode())
        Path(log_path).write_text("trained\n")

    monkeypatch.setattr("lev.train.run_and_tee", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lev-train", "--train-input", f"a={source}", "--out", str(checkpoint), "--device", "cpu"],
    )

    train_main()

    assert delegated_records == [record]
    manifest = json.loads((checkpoint.parent / "run.json").read_text())
    mixture = checkpoint.parent / "train-mixture.jsonl"
    assert manifest["prepared"]["inputs"]["a"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["prepared"]["mixture"]["sha256"] == hashlib.sha256(mixture.read_bytes()).hexdigest()


def test_train_input_does_not_reload_checkpoint_after_training(monkeypatch, tmp_path):
    record = _record("a/0")
    record.pop("_meta")
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(record) + "\n")
    checkpoint = tmp_path / "run" / "checkpoint"

    monkeypatch.setattr("lev.train.validate_training_context", lambda *args, **kwargs: None)
    monkeypatch.setattr("lev.train.runtime_identity", lambda device: {"requested_device": device, "gpu": None})

    def fake_run(command, log_path):
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "head.pt", "adapter_config.json"):
            (output / name).write_bytes(name.encode())
        Path(log_path).write_text("trained\n")

    monkeypatch.setattr("lev.train.run_and_tee", fake_run)
    monkeypatch.setattr(
        "kev.evaluate.load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training must not reload the checkpoint")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lev-train",
            "--train-input",
            f"a={source}",
            "--out",
            str(checkpoint),
            "--device",
            "cpu",
        ],
    )

    train_main()

    manifest = json.loads((checkpoint.parent / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["checkpoint"]) == {
        "adapter_model.safetensors",
        "head.pt",
        "adapter_config.json",
    }


def test_explicit_mixture_and_invalid_named_inputs(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(_record("a/0")) + "\n")
    b.write_text(json.dumps(_record("b/0")) + "\n")
    inputs = load_named_inputs([f"a={a}", f"b={b}"])
    summary = materialize_mixture(inputs, tmp_path / "mix.jsonl", policy="explicit", weights={"a": 1, "b": 3})
    assert summary["realized_counts"] == {"a": 1, "b": 3}
    with pytest.raises(ValueError, match="duplicate input name"):
        load_named_inputs([f"a={a}", f"a={b}"])
    with pytest.raises(ValueError, match="unknown"):
        materialize_mixture(inputs, tmp_path / "bad.jsonl", policy="explicit", weights={"c": 1})


def test_explicit_mixture_uses_every_supplied_record(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(_record(f"a/{i}")) for i in range(2)) + "\n")
    b.write_text(json.dumps(_record("b/0")) + "\n")
    inputs = load_named_inputs([f"a={a}", f"b={b}"])
    summary = materialize_mixture(inputs, tmp_path / "mix.jsonl", policy="explicit", weights={"a": 1, "b": 3})
    assert summary["realized_counts"] == {"a": 2, "b": 6}
    rows = list(map(json.loads, (tmp_path / "mix.jsonl").read_text().splitlines()))
    mixed_ids = {row["_meta"]["id"] for row in rows}
    assert {"a/0", "a/1", "b/0"} <= mixed_ids
    assert [row["_meta"]["id"] for row in rows] == [
        "a/0",
        "b/0",
        "a/1",
        "b/0",
        "b/0",
        "b/0",
        "b/0",
        "b/0",
    ]


def test_checkpoint_identity_requires_and_hashes_load_artifacts(tmp_path):
    for name in ("adapter_model.safetensors", "head.pt", "adapter_config.json"):
        (tmp_path / name).write_bytes(name.encode())
    identity = checkpoint_identity(tmp_path)
    assert set(identity) == {"adapter_model.safetensors", "head.pt", "adapter_config.json"}
    (tmp_path / "head.pt").unlink()
    with pytest.raises(ValueError, match="head.pt"):
        checkpoint_identity(tmp_path)


def test_run_manifest_captures_command_subset_limits_and_runtime(tmp_path):
    records = [_record("a/0"), _record("b/0", True)]
    path = tmp_path / "run.json"
    write_run_manifest(
        path,
        command=["python", "-m", "kev.train", "--dtype", "bf16"],
        records_path=tmp_path / "records.jsonl",
        checkpoint_path=tmp_path / "checkpoint",
        records=records,
        limits={"max_state": 1024, "max_branch": 1280, "max_packed": 1536},
        training={"dtype": "bf16", "weights_dtype": "bf16", "checkpointing": True},
        runtime={"requested_device": "cpu", "gpu": None},
        outer_command=["python", "-m", "lev.train"],
        detached_session="lev-fixture",
        implementation={"git_sha": "c" * 40, "tracked_files_dirty": False},
    )

    manifest = json.loads(path.read_text())
    assert manifest["command_shell"] == "python -m kev.train --dtype bf16"
    assert manifest["subset"]["total"] == 2
    assert manifest["subset"]["by_dataset"] == {"a": 1, "b": 1}
    assert set(manifest["subset"]["record_ids"]) == {"a/0", "b/0"}
    assert manifest["limits"]["max_state"] == 1024
    assert manifest["training"]["checkpointing"] is True
    assert manifest["runtime"] == {"gpu": None, "requested_device": "cpu"}
    assert manifest["outer_command"] == ["python", "-m", "lev.train"]
    assert manifest["delegated_command"] == ["python", "-m", "kev.train", "--dtype", "bf16"]
    assert set(manifest) == {
        "created_at", "status", "detached_session", "outer_command", "delegated_command",
        "command", "command_shell", "environment", "records_path", "checkpoint_path",
        "subset", "limits", "training", "runtime", "implementation", "prepared",
    }
    assert manifest["detached_session"] == "lev-fixture"
    assert manifest["implementation"]["git_sha"] == "c" * 40


def test_run_and_tee_preserves_output(tmp_path, capsys):
    log = tmp_path / "train.log"
    run_and_tee([sys.executable, "-c", "print('initial loss 0.75')"], log)

    assert log.read_text() == "initial loss 0.75\n"
    assert "initial loss 0.75" in capsys.readouterr().out
