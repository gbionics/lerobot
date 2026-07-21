#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from importlib import util
from pathlib import Path

import torch


def _load_modeling_module():
    repo_root = Path(__file__).resolve().parents[4]
    file_path = repo_root / "src" / "lerobot" / "policies" / "pi05_cotrain" / "modeling_pi05.py"
    spec = util.spec_from_file_location("pi05_cotrain_modeling", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modeling = _load_modeling_module()


def test_resolve_image_validity_mask_from_explicit_key():
    batch = {
        "observation.images.wrist_is_valid": torch.tensor([1, 0, 1], dtype=torch.int64),
    }

    mask = modeling.resolve_image_validity_mask(
        batch=batch,
        image_key="observation.images.wrist",
        bsize=3,
        device=torch.device("cpu"),
    )

    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True]


def test_resolve_image_validity_mask_from_is_human_rules():
    batch = {
        "is_human": torch.tensor([1, 0, 1], dtype=torch.int64),
    }

    mask = modeling.resolve_image_validity_mask(
        batch=batch,
        image_key="observation.images.wrist",
        bsize=3,
        device=torch.device("cpu"),
        is_human=batch["is_human"],
        human_missing_image_keys=["observation.images.wrist"],
        robot_missing_image_keys=[],
    )

    assert mask.tolist() == [False, True, False]


def test_apply_image_validity_mask_replaces_invalid_with_minus_one():
    img = torch.zeros((3, 3, 2, 2), dtype=torch.float32)
    mask = torch.tensor([True, False, True], dtype=torch.bool)

    out = modeling.apply_image_validity_mask(img, mask)

    assert torch.all(out[0] == 0.0)
    assert torch.all(out[2] == 0.0)
    assert torch.all(out[1] == -1.0)
