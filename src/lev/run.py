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

"""Repository identity recorded by model runs."""

from __future__ import annotations

from pathlib import Path
import subprocess


def repository_identity(repository: str | Path) -> dict[str, str | bool]:
    repository = Path(repository)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repository,
        check=False,
    ).returncode != 0
    return {"git_sha": sha, "tracked_files_dirty": tracked_dirty}
