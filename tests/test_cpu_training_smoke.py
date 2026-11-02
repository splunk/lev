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

from types import SimpleNamespace

import torch

from kev.data import materialize
from kev.model import DecisionModel, PointerHead
from kev.train import question_loss
from lev.data.hdfs import load_hdfs_sessions


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[6 + (index % 20) for index, _ in enumerate(text.split())])

    def convert_tokens_to_ids(self, token):
        return {"<|fim_prefix|>": 1, "<|fim_middle|>": 2, "<|box_start|>": 3,
                "<|box_end|>": 4, "<|fim_suffix|>": 5}[token]


class TinyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.embed = torch.nn.Embedding(32, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        del position_ids, attention_mask
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


def tiny_decision_model():
    model = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(model)
    model.lm = TinyBackbone()
    model.head = PointerHead(12, dp=8)
    model.pad_id = 0
    model.hybrid = False
    model.execution_mode = "packed"
    model.option_isolation = False
    model.device = "cpu"
    model.max_state = 1024
    model.max_branch = 1280
    model.max_packed = 1536
    return model


def test_cpu_forward_backward_checkpoint_reload(tmp_path):
    fixture_dir = __import__("pathlib").Path(__file__).parent / "fixtures"
    request = load_hdfs_sessions(fixture_dir / "hdfs.log", fixture_dir / "hdfs_labels.csv")[0]
    record = materialize(request)
    tokenizer = FakeTokenizer()
    model = tiny_decision_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    encoded = model.encode(tokenizer, record, strict=True)
    logits = model.forward(encoded)
    loss = question_loss(logits[0], record["questions"][0], "cpu", ord_w=0)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()

    checkpoint = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = tiny_decision_model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    model.eval()
    with torch.no_grad():
        expected = model.forward(encoded)[0]
        actual = restored.forward(encoded)[0]
    torch.testing.assert_close(actual, expected)
