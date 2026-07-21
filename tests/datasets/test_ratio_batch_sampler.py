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

import pytest
from importlib import util
from pathlib import Path


def _load_sampler_module():
    repo_root = Path(__file__).resolve().parents[2]
    sampler_path = repo_root / "src" / "lerobot" / "datasets" / "sampler.py"
    spec = util.spec_from_file_location("ratio_sampler_module", sampler_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sampler module from {sampler_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sampler_module = _load_sampler_module()
HumanRobotRatioBatchSampler = _sampler_module.HumanRobotRatioBatchSampler
split_indices_by_episode_source = _sampler_module.split_indices_by_episode_source


def test_split_indices_by_episode_source_basic():
    indices = [0, 1, 2, 3, 4, 5]
    frame_episode_indices = [0, 0, 1, 1, 2, 2]
    episode_source_map = {0: "human", 1: "robot", 2: "human"}

    human_indices, robot_indices = split_indices_by_episode_source(
        indices,
        frame_episode_indices,
        episode_source_map,
        human_values=["human"],
        robot_values=["robot"],
        source_column="source_type",
    )

    assert human_indices == [0, 1, 4, 5]
    assert robot_indices == [2, 3]


def test_ratio_sampler_strict_composition_per_batch():
    sampler = HumanRobotRatioBatchSampler(
        human_indices=[0, 1, 2, 3],
        robot_indices=[10, 11, 12, 13],
        batch_size=4,
        human_ratio=0.5,
        seed=10,
    )

    for batch in sampler:
        human_count = sum(index in {0, 1, 2, 3} for index in batch)
        robot_count = sum(index in {10, 11, 12, 13} for index in batch)
        assert human_count == 2
        assert robot_count == 2


def test_ratio_sampler_oversamples_minority_pool():
    sampler = HumanRobotRatioBatchSampler(
        human_indices=[0],
        robot_indices=[10, 11, 12, 13],
        batch_size=4,
        human_ratio=0.5,
        seed=123,
    )

    first_batch = next(iter(sampler))
    assert sum(index == 0 for index in first_batch) == 2


def test_ratio_sampler_requires_available_source_pool():
    with pytest.raises(ValueError, match="requires human samples"):
        HumanRobotRatioBatchSampler(
            human_indices=[],
            robot_indices=[10, 11],
            batch_size=4,
            human_ratio=0.5,
        )
