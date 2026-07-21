#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def build_episode_source_map(
    episodes: Any,
    source_column: str,
) -> dict[int, str]:
    """Build an episode_index -> source label map from episode metadata."""
    features = getattr(episodes, "features", {})
    if source_column not in features:
        available = sorted(features.keys())
        raise ValueError(
            f"Episode metadata column {source_column!r} was not found. "
            f"Available columns: {available}"
        )

    episode_indices = episodes["episode_index"]
    source_values = episodes[source_column]
    if len(episode_indices) != len(source_values):
        raise ValueError(
            f"Episode metadata has inconsistent lengths for 'episode_index' and {source_column!r}."
        )

    source_map: dict[int, str] = {}
    for episode_index, source_value in zip(episode_indices, source_values, strict=True):
        if source_value is None:
            raise ValueError(
                f"Episode {episode_index} has no source label in column {source_column!r}."
            )
        source_map[int(episode_index)] = str(source_value)

    return source_map


def split_indices_by_episode_source(
    indices: list[int],
    frame_episode_indices: list[int],
    episode_source_map: dict[int, str],
    human_values: list[str],
    robot_values: list[str],
    source_column: str,
) -> tuple[list[int], list[int]]:
    """Split dataset frame indices into human and robot pools using episode-level source labels."""
    human_set = {value.strip().lower() for value in human_values}
    robot_set = {value.strip().lower() for value in robot_values}

    human_indices: list[int] = []
    robot_indices: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= len(frame_episode_indices):
            raise ValueError(f"Sample index {idx} is out of bounds for frame episode mapping.")

        episode_index = int(frame_episode_indices[idx])
        if episode_index not in episode_source_map:
            raise ValueError(
                f"Episode {episode_index} was not found in episode source metadata column {source_column!r}."
            )

        source_value = episode_source_map[episode_index].strip().lower()
        if source_value in human_set:
            human_indices.append(idx)
        elif source_value in robot_set:
            robot_indices.append(idx)
        else:
            raise ValueError(
                f"Unsupported episode source value {episode_source_map[episode_index]!r} for episode "
                f"{episode_index} in column {source_column!r}. "
                f"Expected one of human={sorted(human_set)} or robot={sorted(robot_set)}."
            )

    return human_indices, robot_indices


class HumanRobotRatioBatchSampler:
    """Batch sampler enforcing strict per-batch human/robot composition."""

    def __init__(
        self,
        human_indices: list[int],
        robot_indices: list[int],
        batch_size: int,
        human_ratio: float,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if not 0.0 <= human_ratio <= 1.0:
            raise ValueError(f"human_ratio must be in [0, 1], got {human_ratio}")

        self.human_indices = human_indices
        self.robot_indices = robot_indices
        self.batch_size = batch_size
        self.human_ratio = human_ratio
        self.seed = seed

        self.num_human_per_batch = int(round(batch_size * human_ratio))
        self.num_human_per_batch = max(0, min(batch_size, self.num_human_per_batch))
        self.num_robot_per_batch = batch_size - self.num_human_per_batch

        if self.num_human_per_batch > 0 and not self.human_indices:
            raise ValueError("Human ratio requires human samples but none were found in the dataset.")

        if self.num_robot_per_batch > 0 and not self.robot_indices:
            raise ValueError("Robot ratio requires robot samples but none were found in the dataset.")

        total_unique = len(self.human_indices) + len(self.robot_indices)
        self._num_batches = max(1, math.ceil(total_unique / self.batch_size))
        self._epoch = 0

    def _draw_indices(
        self,
        pool: list[int],
        count: int,
        state: dict[str, Any],
        generator: torch.Generator,
    ) -> list[int]:
        if count == 0:
            return []

        out: list[int] = []
        while len(out) < count:
            order = state["order"]
            pos = state["pos"]

            if not order or pos >= len(order):
                perm = torch.randperm(len(pool), generator=generator).tolist()
                order = [pool[i] for i in perm]
                pos = 0

            take = min(count - len(out), len(order) - pos)
            out.extend(order[pos : pos + take])
            pos += take

            state["order"] = order
            state["pos"] = pos

        return out

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed + self._epoch)

        human_state: dict[str, Any] = {"order": [], "pos": 0}
        robot_state: dict[str, Any] = {"order": [], "pos": 0}

        for _ in range(len(self)):
            batch_human = self._draw_indices(
                self.human_indices,
                self.num_human_per_batch,
                human_state,
                generator,
            )
            batch_robot = self._draw_indices(
                self.robot_indices,
                self.num_robot_per_batch,
                robot_state,
                generator,
            )
            batch = batch_human + batch_robot
            if len(batch) > 1:
                shuffled_order = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[i] for i in shuffled_order]
            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        return self._num_batches
