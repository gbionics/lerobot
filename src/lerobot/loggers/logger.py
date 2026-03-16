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
"""
Abstract Logger interface for LeRobot.

This module defines the Logger abstract base class that all logging backends
must implement. The interface provides methods for logging metrics, videos,
and model artifacts.

Example:
    To implement a custom logger:

    >>> from lerobot.loggers.logger import Logger
    >>> from lerobot.loggers.config import LoggerConfig
    >>>
    >>> class MyLogger(Logger):
    ...     def __init__(self, cfg: LoggerConfig, train_cfg):
    ...         super().__init__(cfg, train_cfg)
    ...         # Initialize your logging backend
    ...
    ...     def log_dict(self, d: dict, step: int, mode: str = "train"):
    ...         # Log metrics dictionary
    ...         pass
    ...
    ...     def log_video(self, video_path: str, step: int, mode: str = "train"):
    ...         # Log video file
    ...         pass
    ...
    ...     def log_policy(self, checkpoint_dir):
    ...         # Log model checkpoint
    ...         pass
"""

import abc
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.loggers.config import LoggerConfig


class Logger(abc.ABC):
    """
    Abstract base class for all logging backends in LeRobot.

    This class defines the interface that all loggers must implement.
    It provides methods for logging training metrics, evaluation videos,
    and model checkpoints.

    Attributes:
        config: The logger-specific configuration.
        train_cfg: The training pipeline configuration.
        log_dir: The directory where logs should be saved.
        job_name: The name of the training job.
        env_fps: Frames per second for the environment (used for video logging).

    Note:
        Subclasses must implement all abstract methods. The constructor
        initializes common attributes but subclasses should call super().__init__()
        before setting up their specific logging backend.
    """

    def __init__(self, cfg: "LoggerConfig", train_cfg: "TrainPipelineConfig"):
        """
        Initialize the logger.

        Args:
            cfg: Logger-specific configuration (e.g., WandBLoggerConfig).
            train_cfg: The training pipeline configuration containing
                       output_dir, job_name, and other training settings.
        """
        self.config = cfg
        self.train_cfg = train_cfg
        self.log_dir = train_cfg.output_dir
        self.job_name = train_cfg.job_name
        self.env_fps = train_cfg.env.fps if train_cfg.env else None

    @abc.abstractmethod
    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        """
        Log a dictionary of metrics.

        This method is called during training to log metrics such as loss,
        learning rate, gradient norm, and evaluation results.

        Args:
            d: Dictionary of metric names to values. Values should be
               numeric (int, float) or strings.
            step: The global training step. Required if custom_step_key is None.
            mode: Either "train" or "eval", used to prefix metric names.
            custom_step_key: Optional key in d to use as the step value.
                           Used for asynchronous training where multiple
                           step types exist (e.g., environment step vs training step).

        Raises:
            ValueError: If mode is not "train" or "eval".
            ValueError: If both step and custom_step_key are None.

        Example:
            >>> logger.log_dict({"loss": 0.5, "lr": 1e-4}, step=1000, mode="train")
        """
        pass

    @abc.abstractmethod
    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """
        Log a video file.

        This method is typically called during evaluation to log rollout videos.

        Args:
            video_path: Path to the video file (usually MP4 format).
            step: The global training step at which this video was recorded.
            mode: Either "train" or "eval", used to categorize the video.

        Raises:
            ValueError: If mode is not "train" or "eval".

        Example:
            >>> logger.log_video("/path/to/eval_video.mp4", step=1000, mode="eval")
        """
        pass

    @abc.abstractmethod
    def log_policy(self, checkpoint_dir: Path) -> None:
        """
        Log a model checkpoint/artifact.

        This method is called when saving checkpoints to upload the model
        weights to the logging backend.

        Args:
            checkpoint_dir: Path to the directory containing the checkpoint.
                          Should contain the PRETRAINED_MODEL_DIR subdirectory
                          with model weights (either model.safetensors or
                          adapter_model.safetensors for PEFT models).

        Note:
            If config.disable_artifact is True, this method should return
            without uploading the artifact.

        Example:
            >>> logger.log_policy(Path("outputs/train/step_1000"))
        """
        pass

    @abc.abstractmethod
    def finish(self) -> None:
        """
        Finalize the logging session.

        This method is called at the end of training to properly close
        the logging backend, flush any remaining data, and clean up resources.

        Subclasses should override this method if their logging backend
        requires explicit cleanup (e.g., wandb.finish()).

        The default implementation does nothing.
        """
        pass

    @abc.abstractmethod
    def get_run_url(self) -> str | None:
        """
        Get the URL for viewing this run in the logging backend's UI.

        Returns:
            str | None: URL to the run dashboard, or None if not available.

        Example:
            >>> url = logger.get_run_url()
            >>> if url:
            ...     print(f"Track this run --> {url}")
        """
        return None
