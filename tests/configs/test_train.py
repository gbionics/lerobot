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

from lerobot.configs.train import BatchSamplingConfig


def test_batch_sampling_config_valid():
    cfg = BatchSamplingConfig(
        human_ratio=0.5,
        source_column="source_type",
        human_values=["human"],
        robot_values=["robot"],
        fallback_policy="oversample",
    )

    cfg.validate()


def test_batch_sampling_config_ratio_bounds():
    with pytest.raises(ValueError, match="human_ratio"):
        BatchSamplingConfig(human_ratio=-0.1).validate()

    with pytest.raises(ValueError, match="human_ratio"):
        BatchSamplingConfig(human_ratio=1.1).validate()


def test_batch_sampling_config_overlapping_labels_raises():
    with pytest.raises(ValueError, match="overlap"):
        BatchSamplingConfig(
            human_values=["human", "shared"],
            robot_values=["robot", "shared"],
        ).validate()


def test_batch_sampling_config_unsupported_fallback_raises():
    with pytest.raises(ValueError, match="fallback_policy"):
        BatchSamplingConfig(fallback_policy="relax").validate()
