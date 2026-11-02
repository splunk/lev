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

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "setup_kev.py"
spec = importlib.util.spec_from_file_location("setup_kev", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.STDOUT
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q")
    git(upstream, "config", "user.email", "test@example.invalid")
    git(upstream, "config", "user.name", "Setup test")
    (upstream / "kev").mkdir()
    (upstream / "kev" / "__init__.py").write_text("VALUE = 'upstream'\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-qm", "Initial source")
    revision = git(upstream, "rev-parse", "HEAD").decode().strip()
    (upstream / "kev" / "__init__.py").write_text("VALUE = 'patched'\n")
    (upstream / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (upstream / "run.sh").chmod(0o755)
    (upstream / "payload.bin").write_bytes(b"\x00\xff\x10")
    git(upstream, "add", ".")
    patch = git(upstream, "diff", "--cached", "--binary", "--full-index")
    git(upstream, "commit", "-qm", "Later upstream revision")

    root = tmp_path / "release"
    packaging = root / "patches" / "kev"
    packaging.mkdir(parents=True)
    (packaging / "upstream-revision.txt").write_text(revision + "\n")
    (packaging / "changes.patch").write_bytes(patch)
    monkeypatch.setattr(setup, "UPSTREAM_URL", str(upstream))
    return root, revision


def test_setup_reconstructs_pinned_source_and_patch(source):
    root, _ = source
    destination = setup.setup_kev(root)
    assert destination == root / "third_party" / "kev"
    assert (destination / "kev" / "__init__.py").read_text() == "VALUE = 'patched'\n"
    assert (destination / "payload.bin").read_bytes() == b"\x00\xff\x10"
    assert (destination / "run.sh").stat().st_mode & 0o111 == 0o111
    assert not (destination / ".git").exists()
    assert sorted(p.name for p in destination.parent.iterdir()) == ["kev"]


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_setup_refuses_existing_destination_without_changes(source, kind):
    root, _ = source
    destination = root / "third_party" / "kev"
    destination.parent.mkdir()
    if kind == "directory":
        destination.mkdir()
        (destination / "keep").write_bytes(b"local work")
    elif kind == "file":
        destination.write_bytes(b"local work")
    else:
        destination.symlink_to(root / "missing")
    with pytest.raises(FileExistsError, match="already exists"):
        setup.setup_kev(root)
    if kind == "directory":
        assert (destination / "keep").read_bytes() == b"local work"
    elif kind == "file":
        assert destination.read_bytes() == b"local work"
    else:
        assert destination.is_symlink()
        assert destination.readlink() == root / "missing"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["revision", "fetch", "patch", "identity"])
def test_setup_failure_does_not_publish_partial_source(source, monkeypatch, failure):
    root, _ = source
    packaging = root / "patches" / "kev"
    if failure == "revision":
        (packaging / "upstream-revision.txt").write_text("main\n")
    elif failure == "fetch":
        monkeypatch.setattr(setup, "UPSTREAM_URL", str(root / "missing-source"))
    elif failure == "patch":
        patch = (packaging / "changes.patch").read_text()
        (packaging / "changes.patch").write_text(patch.replace("'upstream'", "'wrong'"))
    else:
        original_git = setup._git

        def wrong_identity(directory, *args):
            if args == ("rev-parse", "HEAD"):
                return "0" * 40
            return original_git(directory, *args)

        monkeypatch.setattr(setup, "_git", wrong_identity)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        setup.setup_kev(root)
    assert not (root / "third_party" / "kev").exists()
    if (root / "third_party").exists():
        assert list((root / "third_party").iterdir()) == []
    assert (packaging / "upstream-revision.txt").exists()


def test_setup_cli_resolves_paths_independently_of_working_directory(source, tmp_path):
    root, _ = source
    script = root / "scripts" / "setup_kev.py"
    script.parent.mkdir()
    shutil.copyfile(SCRIPT, script)
    destination = root / "third_party" / "kev"
    destination.mkdir(parents=True)
    (destination / "keep").write_text("local work")
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert str(destination) in result.stderr
    assert (destination / "keep").read_text() == "local work"
