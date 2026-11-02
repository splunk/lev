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

"""Prepare the pinned, patched KEV dependency."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


UPSTREAM_URL = "https://github.com/jaredpalmer/kev.git"


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-C", str(directory), *args],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def setup_kev(root: Path) -> Path:
    destination = root / "third_party" / "kev"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"KEV destination already exists: {destination}")
    packaging = root / "patches" / "kev"
    revision = (packaging / "upstream-revision.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("KEV upstream revision must be a full lowercase commit SHA")
    patch = (packaging / "changes.patch").resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kev-", dir=destination.parent) as staging:
        source = Path(staging) / "source"
        source.mkdir()
        _git(source, "init", "-q")
        _git(source, "fetch", "--depth=1", "--no-tags", UPSTREAM_URL, revision)
        _git(source, "checkout", "--detach", "FETCH_HEAD")
        if _git(source, "rev-parse", "HEAD") != revision:
            raise RuntimeError("Fetched KEV commit does not match the pinned revision")
        _git(source, "apply", "--check", "--index", "--whitespace=nowarn", str(patch))
        _git(source, "apply", "--index", "--whitespace=nowarn", str(patch))
        shutil.rmtree(source / ".git")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"KEV destination already exists: {destination}")
        source.rename(destination)
    return destination


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        destination = setup_kev(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        detail = exc.output if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"KEV setup failed: {detail.strip()}", file=sys.stderr)
        return 1
    print(f"KEV source ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
