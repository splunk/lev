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

"""Delegate training on prepared KEV records to KEV."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from kev.data import load_records

from .prepared import load_named_inputs, materialize_mixture, parse_weights, sha256_file
from .run import repository_identity


BASE = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
INIT_FROM = ""


def kev_train_command(
    *,
    data: str | Path,
    output: str | Path,
    seed: int,
    max_state: int,
    max_branch: int,
    max_packed: int,
    device: str,
    dtype: str,
    weights_dtype: str,
    checkpointing: bool,
    batch: int,
    accum: int,
    base: str = BASE,
    base_revision: str = BASE_REVISION,
    init_from: str = INIT_FROM,
    epochs: int = 1,
    lr: float = 2e-5,
    head_lr: float = 0.0,
    weight_decay: float = 0.01,
    lora: int = 16,
    lora_targets: str = "all",
    head_dim: int = 256,
    option_isolation: bool = False,
    special_embeddings: bool = False,
    ord_w: float = 0.0,
    perm_kl: float = 0.0,
    perm_frac: float = 0.3,
    anchor: str = "",
    anchor_w: float = 0.0,
    anchor_sources: str = "",
    p_none: float = 0.0,
    p_none_distract: float = 0.0,
    p_distract: float = 0.0,
    p_none_pair: float = 0.0,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "kev.train",
        "--data",
        str(data),
        "--base",
        base,
        "--base_revision",
        base_revision,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--head_lr",
        str(head_lr),
        "--weight_decay",
        str(weight_decay),
        "--lora",
        str(lora),
        "--lora_targets",
        lora_targets,
        "--head_dim",
        str(head_dim),
        "--option_isolation",
        str(int(option_isolation)),
        "--special_embeddings",
        str(int(special_embeddings)),
        "--ord_w",
        str(ord_w),
        "--perm_kl",
        str(perm_kl),
        "--perm_frac",
        str(perm_frac),
        "--batch",
        str(batch),
        "--accum",
        str(accum),
        "--device",
        device,
        "--dtype",
        dtype,
        "--weights_dtype",
        weights_dtype,
        "--checkpointing",
        str(int(checkpointing)),
        "--max_state",
        str(max_state),
        "--max_branch",
        str(max_branch),
        "--max_packed",
        str(max_packed),
        "--p_none",
        str(p_none),
        "--p_none_distract",
        str(p_none_distract),
        "--p_distract",
        str(p_distract),
        "--p_none_pair",
        str(p_none_pair),
        "--seed",
        str(seed),
        "--out",
        str(output),
    ]
    if init_from:
        command.extend(["--init_from", init_from])
    if anchor:
        command += ["--anchor", anchor, "--anchor_w", str(anchor_w)]
        if anchor_sources:
            command += ["--anchor_sources", anchor_sources]
    return command


def runtime_identity(device: str) -> dict:
    """Capture the requested runtime and the visible CUDA hardware without changing it."""
    identity = {
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": None,
    }
    if device != "cuda":
        return identity

    import torch

    identity["torch_cuda_available"] = torch.cuda.is_available()
    identity["torch_cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        identity["gpu"] = {
            "torch_index": 0,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode == 0:
        identity["nvidia_smi"] = [
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
            }
            for line in query.stdout.splitlines()
            if len(fields := [field.strip() for field in line.split(",")]) == 5
        ]
    else:
        identity["nvidia_smi_error"] = query.stderr.strip() or f"exit {query.returncode}"
    return identity


def write_run_manifest(
    path: str | Path,
    *,
    command: list[str],
    records_path: str | Path,
    checkpoint_path: str | Path,
    records: list[dict],
    limits: dict,
    training: dict,
    runtime: dict,
    outer_command: list[str] | None = None,
    detached_session: str | None = None,
    implementation: dict | None = None,
    prepared: dict | None = None,
) -> None:
    """Persist the exact LEV-to-KEV invocation and input subset before training starts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "detached_session": detached_session,
        "outer_command": outer_command or [],
        "delegated_command": command,
        "command": command,
        "command_shell": shlex.join(command),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_DATASETS_CACHE",
                "TORCH_HOME",
                "UV_CACHE_DIR",
                "TRANSFORMERS_OFFLINE",
                "PYTORCH_CUDA_ALLOC_CONF",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "records_path": str(records_path),
        "checkpoint_path": str(checkpoint_path),
        "subset": {
            "total": len(records),
            "by_dataset": dict(
                sorted(
                    Counter(
                        record.get("_meta", {}).get("dataset", record.get("_meta", {}).get("source", "unknown"))
                        for record in records
                    ).items()
                )
            ),
            "record_ids": [record["_meta"]["id"] for record in records],
        },
        "limits": limits,
        "training": training,
        "runtime": runtime,
        "implementation": implementation or {},
        "prepared": prepared,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_identity(path: str | Path) -> dict:
    path = Path(path)
    required = ("adapter_model.safetensors", "head.pt", "adapter_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint is missing required artifacts: {missing}")
    return {name: sha256_file(path / name) for name in required}


def validate_training_context(records: list[dict], *, base: str, revision: str, limits: dict) -> None:
    """Reject records KEV would otherwise silently remove for exceeding context."""
    from kev.model import load_tokenizer
    from kev.train import fits_context

    tokenizer = load_tokenizer(base, revision=revision or None)
    rejected = [
        str(record.get("_meta", {}).get("id", index))
        for index, record in enumerate(records)
        if not fits_context(
            tokenizer,
            record,
            max_state=limits["max_state"],
            max_branch=limits["max_branch"],
            max_packed=limits["max_packed"],
        )
    ]
    if rejected:
        raise ValueError(f"{len(rejected)} training records exceed configured context; first IDs: {rejected[:3]}")


def update_run_manifest(
    path: str | Path,
    *,
    status: str,
    metrics_path: str | Path | None = None,
    checkpoint: dict | None = None,
) -> None:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    if metrics_path is not None and Path(metrics_path).exists():
        manifest["training_metrics"] = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if checkpoint is not None:
        manifest["checkpoint"] = checkpoint
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_and_tee(command: list[str], log_path: str | Path) -> None:
    """Run KEV while preserving its combined stdout/stderr in the run directory."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--mixture", choices=("proportional", "equal_dataset", "explicit"), default="proportional")
    parser.add_argument("--dataset-weight", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--session-name")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-state", type=int, default=1024)
    parser.add_argument("--max-branch", type=int, default=1280)
    parser.add_argument("--max-packed", type=int, default=1536)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--weights-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--init-from", default=INIT_FROM)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora", type=int, default=16)
    parser.add_argument("--lora-targets", choices=("all", "dense", "attn", "qv"), default="all")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--option-isolation", action="store_true")
    parser.add_argument("--special-embeddings", action="store_true")
    parser.add_argument("--ord-w", type=float, default=0.0)
    parser.add_argument("--perm-kl", type=float, default=0.0)
    parser.add_argument("--perm-frac", type=float, default=0.3)
    parser.add_argument("--anchor", default="")
    parser.add_argument("--anchor-w", type=float, default=0.0)
    parser.add_argument("--anchor-sources", default="")
    parser.add_argument("--p-none", type=float, default=0.0)
    parser.add_argument("--p-none-distract", type=float, default=0.0)
    parser.add_argument("--p-distract", type=float, default=0.0)
    parser.add_argument("--p-none-pair", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.batch, args.accum, args.epochs, args.lora, args.head_dim) < 1:
        parser.error("batch, accum, epochs, lora, and head-dim must be positive")
    if args.lr <= 0 or args.head_lr < 0 or args.weight_decay < 0:
        parser.error("invalid optimizer settings")
    if min(args.ord_w, args.perm_kl, args.anchor_w) < 0 or not 0 <= args.perm_frac <= 1:
        parser.error("invalid loss or augmentation settings")
    if any(not 0 <= value <= 1 for value in (args.p_none, args.p_none_distract, args.p_distract, args.p_none_pair)):
        parser.error("none-option and distractor probabilities must be between 0 and 1")
    if args.dtype == "bf16" and args.device != "cuda":
        parser.error("--dtype bf16 requires --device cuda")
    if not 0 < args.max_state <= args.max_branch <= args.max_packed:
        parser.error("token limits must satisfy 0 < max-state <= max-branch <= max-packed")
    if bool(args.anchor) != (args.anchor_w > 0):
        parser.error("--anchor and positive --anchor-w must be supplied together")

    manifest_path = args.out.parent / "run.json"
    log_path = args.out.parent / "train.log"
    records_path = args.out.parent / "train-mixture.jsonl"
    existing = next((path for path in (manifest_path, log_path, records_path, args.out) if path.exists()), None)
    if existing is not None:
        parser.error(f"refusing to overwrite {existing}")
    try:
        named_inputs = load_named_inputs(args.train_input)
        weights = parse_weights(args.dataset_weight)
        if args.mixture != "explicit" and weights:
            raise ValueError("--dataset-weight is only valid with --mixture explicit")
        mixture = materialize_mixture(named_inputs, records_path, policy=args.mixture, weights=weights)
    except ValueError as error:
        parser.error(str(error))
    records = load_records(records_path, source="prepared_mixture")
    prepared = {
        "inputs": {item.name: {"path": str(item.path.resolve()), "sha256": item.sha256, "records": len(item.records)} for item in named_inputs},
        "mixture": mixture,
    }
    print(f"wrote {len(records)} mixed prepared records to {records_path}", flush=True)
    limits = {"max_state": args.max_state, "max_branch": args.max_branch, "max_packed": args.max_packed}
    try:
        validate_training_context(records, base=args.base, revision=args.base_revision, limits=limits)
    except ValueError as error:
        parser.error(str(error))
    command = kev_train_command(
        data=records_path,
        output=args.out,
        seed=args.seed,
        max_state=args.max_state,
        max_branch=args.max_branch,
        max_packed=args.max_packed,
        device=args.device,
        dtype=args.dtype,
        weights_dtype=args.weights_dtype,
        checkpointing=args.checkpointing,
        batch=args.batch,
        accum=args.accum,
        base=args.base,
        base_revision=args.base_revision,
        init_from=args.init_from,
        epochs=args.epochs,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        lora=args.lora,
        lora_targets=args.lora_targets,
        head_dim=args.head_dim,
        option_isolation=args.option_isolation,
        special_embeddings=args.special_embeddings,
        ord_w=args.ord_w,
        perm_kl=args.perm_kl,
        perm_frac=args.perm_frac,
        anchor=args.anchor,
        anchor_w=args.anchor_w,
        anchor_sources=args.anchor_sources,
        p_none=args.p_none,
        p_none_distract=args.p_none_distract,
        p_distract=args.p_distract,
        p_none_pair=args.p_none_pair,
    )
    manifest_path = args.out.parent / "run.json"
    log_path = args.out.parent / "train.log"
    if manifest_path.exists() or log_path.exists():
        parser.error(f"refusing to overwrite {manifest_path if manifest_path.exists() else log_path}")
    write_run_manifest(
        manifest_path,
        command=command,
        records_path=records_path,
        checkpoint_path=args.out,
        records=records,
        limits=limits,
        training={
            "dtype": args.dtype,
            "weights_dtype": args.weights_dtype,
            "checkpointing": args.checkpointing,
            "batch": args.batch,
            "accum": args.accum,
            "base": args.base,
            "base_revision": args.base_revision,
            "init_from": args.init_from,
            "epochs": args.epochs,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "lora": args.lora,
            "lora_targets": args.lora_targets,
            "head_dim": args.head_dim,
            "option_isolation": args.option_isolation,
            "special_embeddings": args.special_embeddings,
            "ord_w": args.ord_w,
            "perm_kl": args.perm_kl,
            "perm_frac": args.perm_frac,
            "anchor": args.anchor,
            "anchor_w": args.anchor_w,
            "anchor_sources": args.anchor_sources,
            "p_none": args.p_none,
            "p_none_distract": args.p_none_distract,
            "p_distract": args.p_distract,
            "p_none_pair": args.p_none_pair,
            "seed": args.seed,
        },
        runtime=runtime_identity(args.device),
        outer_command=[sys.executable, "-m", "lev.train", *sys.argv[1:]],
        detached_session=args.session_name,
        implementation=repository_identity(Path(__file__).resolve().parents[2]),
        prepared=prepared,
    )
    try:
        run_and_tee(command, log_path)
        checkpoint = checkpoint_identity(args.out)
    except Exception:
        update_run_manifest(manifest_path, status="failed")
        raise
    update_run_manifest(
        manifest_path,
        status="completed",
        metrics_path=args.out / "training_metrics.json",
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()
