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

from pathlib import Path
from types import SimpleNamespace

import pytest

from kev.model import DecisionModel, encode


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=list(range(10, 10 + len(text.split()))))

    def convert_tokens_to_ids(self, token):
        return {"<|fim_prefix|>": 1, "<|fim_middle|>": 2, "<|box_start|>": 3,
                "<|box_end|>": 4, "<|fim_suffix|>": 5}[token]


def record(words=8):
    return {
        "state": " ".join(f"w{i}" for i in range(words)),
        "questions": [{"instr": "anomaly question", "options": ["normal", "anomaly"], "label": 0}],
    }


def test_encode_uses_explicit_state_and_branch_limits():
    tok = FakeTokenizer()
    encoded = encode(tok, record(), max_state=9, max_branch=20, strict=True)
    assert encoded["effective_limits"] == {"max_state": 9, "max_branch": 20}

    with pytest.raises(ValueError, match="state exceeds 5"):
        encode(tok, record(), max_state=5, max_branch=20, strict=True)


def test_decision_model_encode_uses_configured_defaults_without_loading_weights():
    model = DecisionModel.__new__(DecisionModel)
    model.option_isolation = False
    model.max_state = 6
    model.max_branch = 18
    model.max_packed = 24

    encoded = model.encode(FakeTokenizer(), record(words=5), strict=True)
    assert encoded["effective_limits"] == {"max_state": 6, "max_branch": 18}
    with pytest.raises(ValueError, match="state exceeds 6"):
        model.encode(FakeTokenizer(), record(words=6), strict=True)

    model.max_packed = len(encoded["ids"]) - 1
    with pytest.raises(ValueError, match="packed request exceeds"):
        model.encode(FakeTokenizer(), record(words=5), strict=True)


def test_training_cli_exposes_and_records_limits():
    source = Path(__file__).parents[1] / "third_party" / "kev" / "kev" / "train.py"
    text = source.read_text()
    for flag in ("--max_state", "--max_branch", "--max_packed"):
        assert flag in text
    assert '"max_state": a.max_state' in text
    assert '"max_branch": a.max_branch' in text
    assert '"max_packed": a.max_packed' in text
