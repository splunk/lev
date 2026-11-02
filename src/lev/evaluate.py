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

"""Evaluate a checkpoint on named prepared KEV labelled-request JSONL inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import math

import torch
import torch.nn.functional as F

from kev.data import materialize

from .run import repository_identity
from .prepared import load_named_inputs, sha256_file


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = false_positives = 0
    previous_recall = average_precision = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group = []
        while index < len(ranked) and ranked[index][0] == score:
            group.append(ranked[index][1])
            index += 1
        true_positives += sum(group)
        false_positives += len(group) - sum(group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def aggregate_predictions(rows: list[dict]) -> dict:
    """Aggregate saved predictions; Choice macro-F1 averages over gold-present option keys."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["input"], row["question_type"]), []).append(row)
    result: dict[str, dict[str, dict]] = {}
    for (input_name, question_type), items in sorted(grouped.items()):
        n = len(items)
        accuracy = sum(item["predicted_index"] == item["gold_index"] for item in items) / n
        nll = -sum(math.log(max(float(item["probabilities"][item["gold_index"]]), 1e-12)) for item in items) / n
        metrics = {"count": n, "accuracy": accuracy, "nll": nll}
        if question_type == "choice":
            labels = [(item["option_keys"][item["gold_index"]],
                       item["option_keys"][item["predicted_index"]]) for item in items]
            class_f1 = []
            for class_key in sorted({gold for gold, _ in labels}):
                tp = sum(gold == class_key and predicted == class_key for gold, predicted in labels)
                fp = sum(gold != class_key and predicted == class_key for gold, predicted in labels)
                fn = sum(gold == class_key and predicted != class_key for gold, predicted in labels)
                class_f1.append(2 * tp / (2 * tp + fp + fn))
            metrics["macro_f1_gold_present"] = sum(class_f1) / len(class_f1)
        elif question_type == "noul":
            labels = [int(item["gold_index"]) for item in items]
            scores = [float(item["probabilities"][1]) for item in items]
            predictions = [int(score >= 0.5) for score in scores]
            tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
            fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
            fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            positive_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
            negative_precision = tn / (tn + fn) if tn + fn else 0.0
            negative_recall = tn / (tn + fp) if tn + fp else 0.0
            negative_f1 = (
                2 * negative_precision * negative_recall / (negative_precision + negative_recall)
                if negative_precision + negative_recall
                else 0.0
            )
            metrics.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "positive_f1": positive_f1,
                    "macro_f1": (positive_f1 + negative_f1) / 2,
                    "brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / n,
                    "auprc": _average_precision(labels, scores),
                }
            )
        result.setdefault(input_name, {})[question_type] = metrics
    return result


def resolve_evaluation_dtype(checkpoint_dtype: str, requested: str | None) -> str:
    effective = requested or checkpoint_dtype or "fp32"
    if checkpoint_dtype == "bf16" and effective != "bf16":
        raise ValueError("a BF16-trained checkpoint must be evaluated with --weights-dtype bf16")
    if effective not in ("fp32", "bf16"):
        raise ValueError(f"unsupported checkpoint weights dtype: {effective}")
    return effective


def _prediction_rows(named_inputs, checkpoint: Path, batch_size: int, limits: dict, dtype: str):
    from kev.evaluate import load

    tokenizer, model = load(
        str(checkpoint),
        "cuda",
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float32,
        max_state=limits["max_state"],
        max_branch=limits["max_branch"],
        max_packed=limits["max_packed"],
    )
    model.head.temperature = 1.0
    with torch.inference_mode():
        for named in named_inputs:
            requests_all = named.validated
            for start in range(0, len(requests_all), batch_size):
                requests = list(requests_all[start : start + batch_size])
                records = [materialize(request) for request in requests]
                encoded = [model.encode(tokenizer, record, strict=True) for record in records]
                outputs = model.forward_batch(encoded)
                for offset, (request, record, question_logits) in enumerate(zip(requests, records, outputs, strict=True)):
                    meta = request.get("_meta", {})
                    record_id = str(meta.get("id", f"{named.name}/{start + offset}"))
                    for question, logits in zip(record["questions"], question_logits, strict=True):
                        values = logits.float().cpu()
                        probabilities = F.softmax(values, dim=-1).tolist()
                        predicted = int(torch.argmax(values))
                        row = {
                            "input": named.name,
                            "row_index": start + offset,
                            "record_id": record_id,
                            "question_id": question["qid"],
                            "question_type": question["qtype"],
                            "option_keys": question["keys"],
                            "gold_index": int(question["label"]),
                            "predicted_index": predicted,
                            "predicted_option": question["keys"][predicted],
                            "logits": values.tolist(),
                            "probabilities": probabilities,
                        }
                        if question["qtype"] == "score":
                            row["expected_level"] = sum(index * probability for index, probability in enumerate(probabilities))
                        phrasing = meta.get("phrasing")
                        if isinstance(phrasing, Mapping):
                            row["phrasing"] = phrasing.get(question["qid"])
                        yield row


def _evaluate_checkpoint(args) -> None:
    named_inputs = load_named_inputs(args.records)
    checkpoint_meta = torch.load(args.checkpoint / "head.pt", map_location="cpu", weights_only=False)
    limits = {
        "max_state": args.max_state if args.max_state is not None else int(checkpoint_meta["max_state"]),
        "max_branch": args.max_branch if args.max_branch is not None else int(checkpoint_meta["max_branch"]),
        "max_packed": args.max_packed if args.max_packed is not None else int(checkpoint_meta["max_packed"]),
    }
    dtype = resolve_evaluation_dtype(str(checkpoint_meta.get("weights_dtype", "fp32")), args.weights_dtype)
    if not 0 < limits["max_state"] <= limits["max_branch"] <= limits["max_packed"]:
        raise ValueError("token limits must satisfy 0 < max-state <= max-branch <= max-packed")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.run_dir / "run.json"
    manifest = {
        "role": args.role,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outer_command": [sys.executable, "-m", "lev.evaluate", *sys.argv[1:]],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": {
            name: sha256_file(args.checkpoint / name)
            for name in ("adapter_model.safetensors", "head.pt", "adapter_config.json")
        },
        "inputs": {item.name: {"path": str(item.path.resolve()), "sha256": item.sha256, "records": len(item.records)} for item in named_inputs},
        "limits": limits,
        "weights_dtype": dtype,
        "batch_size": args.batch_size,
        "runtime": _runtime(),
        "implementation": repository_identity(Path(__file__).resolve().parents[2]),
    }
    _write_json_exclusive(manifest_path, manifest)
    predictions_path = args.run_dir / "predictions.jsonl"
    rows = []
    try:
        with predictions_path.open("x", encoding="utf-8") as output:
            for row in _prediction_rows(named_inputs, args.checkpoint, args.batch_size, limits, dtype):
                rows.append(row)
                output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        metrics = aggregate_predictions(rows)
        _write_json_exclusive(args.run_dir / "metrics.json", metrics)
        _update_status(manifest_path, "completed", prediction_count=len(rows), metrics="metrics.json")
    except BaseException as error:
        _update_status(manifest_path, "failed", error=f"{type(error).__name__}: {error}")
        raise


def _runtime() -> dict:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_index": index,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
    }


def _write_json_exclusive(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _update_status(path: Path, status: str, **values) -> None:
    manifest = json.loads(path.read_text())
    manifest.update(values)
    manifest["status"] = status
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--records", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--role", choices=("calibration", "validation"), default="validation")
    parser.add_argument("--max-state", type=int)
    parser.add_argument("--max-branch", type=int)
    parser.add_argument("--max-packed", type=int)
    parser.add_argument("--weights-dtype", choices=("fp32", "bf16"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    try:
        _evaluate_checkpoint(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
