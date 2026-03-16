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
Weights & Biases (WandB) logger implementation.

This module provides the WandB integration for LeRobot's logging system.
It implements the Logger interface for logging metrics, videos, and model
artifacts to the WandB platform.

Example:
    >>> from lerobot.loggers.wandb import WandBLogger, WandBLoggerConfig
    >>>
    >>> config = WandBLoggerConfig(project="my_project", entity="my_team")
    >>> logger = WandBLogger(config, train_cfg)
    >>> logger.log_dict({"loss": 0.5}, step=100)
"""

import logging
import os
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.loggers.config import LoggerConfig
from lerobot.loggers.logger import Logger
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


def cfg_to_group(
    cfg: "TrainPipelineConfig",
    return_list: bool = False,
    truncate_tags: bool = False,
    max_tag_length: int = 64,
) -> list[str] | str:
    """
    Generate a group name for logging based on training configuration.

    Creates tags from the policy type, seed, dataset, and environment
    to identify and group related runs.

    Args:
        cfg: The training pipeline configuration.
        return_list: If True, return tags as a list; otherwise join with "-".
        truncate_tags: If True, truncate tags to max_tag_length characters.
        max_tag_length: Maximum length for each tag (default 64, WandB limit).

    Returns:
        Either a list of tag strings or a single "-" joined string.
    """

    def _maybe_truncate(tag: str) -> str:
        """Truncate tag to max_tag_length characters if required."""
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    lst = [
        f"policy:{cfg.policy.type}",
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    """
    Retrieve the WandB run ID from the filesystem for run resumption.

    Args:
        log_dir: The logging directory containing the wandb folder.

    Returns:
        The WandB run ID string.

    Raises:
        RuntimeError: If the run ID cannot be determined from the filesystem.
    """
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str) -> str:
    """
    Sanitize an artifact name for WandB compatibility.

    WandB artifacts don't accept ":" or "/" in their names.

    Args:
        name: The original artifact name.

    Returns:
        A sanitized artifact name with ":" and "/" replaced by "_".
    """
    return name.replace(":", "_").replace("/", "_")


@LoggerConfig.register_subclass("wandb")
@dataclass
class WandBLoggerConfig(LoggerConfig):
    """
    Configuration for the Weights & Biases logger.

    This configuration class defines all the parameters needed to set up
    WandB logging, including project name, entity, and run settings.

    Attributes:
        enable: Whether WandB logging is enabled. Defaults to True.
        disable_artifact: Skip model artifact uploads. Defaults to False.
        project: WandB project name. Defaults to "lerobot".
        entity: WandB team/user entity. Defaults to None (uses default entity).
        notes: Optional notes for the run. Defaults to None.
        run_id: Specific run ID for resumption. Defaults to None.
        mode: WandB mode: "online", "offline", or "disabled". Defaults to None (online).

    Example:
        >>> config = WandBLoggerConfig(project="my_project", entity="my_team", mode="online")
    """

    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline', 'disabled'


class WandBLogger(Logger):
    """
    Weights & Biases logger implementation.

    This logger integrates with the WandB platform to log training metrics,
    evaluation videos, and model artifacts. It supports both online and
    offline logging modes, as well as run resumption.

    Attributes:
        _wandb: The wandb module instance (lazy loaded).
        _group: Group identifier for this run based on config.
        _wandb_custom_step_key: Set of custom step keys for async training.

    Example:
        >>> config = WandBLoggerConfig(project="lerobot", mode="online")
        >>> logger = WandBLogger(config, train_cfg)
        >>> logger.log_dict({"loss": 0.5, "lr": 1e-4}, step=1000)
        >>> logger.log_video("eval.mp4", step=1000, mode="eval")
    """

    def __init__(self, cfg: WandBLoggerConfig, train_cfg: "TrainPipelineConfig"):
        """
        Initialize the WandB logger.

        Sets up the WandB run with the specified configuration, including
        project, entity, tags, and resume settings.

        Args:
            cfg: WandB-specific configuration.
            train_cfg: The training pipeline configuration.
        """
        super().__init__(cfg, train_cfg)
        self._group = cfg_to_group(train_cfg)

        # Set up WandB with silent mode to reduce console noise
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        # Determine run ID for new or resumed runs
        wandb_run_id = (
            cfg.run_id
            if cfg.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if train_cfg.resume
            else None
        )

        wandb.init(
            id=wandb_run_id,
            project=cfg.project,
            entity=cfg.entity,
            name=self.job_name,
            notes=cfg.notes,
            tags=cfg_to_group(train_cfg, return_list=True, truncate_tags=True),
            dir=self.log_dir,
            config=train_cfg.to_dict(),
            # TODO(rcadene): try set to True
            save_code=False,
            # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
            job_type="train_eval",
            resume="must" if train_cfg.resume else None,
            mode=cfg.mode if cfg.mode in ["online", "offline", "disabled"] else "online",
        )

        run_id = wandb.run.id
        # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
        # This is because we want to be able to resume the run from the wandb run id.
        cfg.run_id = run_id

        # Handle custom step key for RL asynchronous training
        self._wandb_custom_step_key: set[str] | None = None

        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path) -> None:
        """
        Upload a model checkpoint to WandB as an artifact.

        Handles both standard models (model.safetensors) and PEFT models
        (adapter_model.safetensors) appropriately.

        Args:
            checkpoint_dir: Path to the checkpoint directory.
        """
        if self.config.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # Check if this is a PEFT model (has adapter files instead of model.safetensors)
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT model: add adapter files and configs
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # Also add the policy config which is needed for loading
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            # Standard model: add the single safetensors file
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        total_bytes = sum(p.stat().st_size for p in pretrained_model_dir.rglob("*") if p.is_file())
        total_mb = total_bytes / (1024 * 1024)
        logging.info(f"Uploading WandB artifact '{artifact_name}' ({total_mb:.1f} MB) ...")
        queued = self._wandb.log_artifact(artifact)
        queued.wait()
        logging.info(f"WandB artifact upload complete: {artifact_name}")

    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        """
        Log a dictionary of metrics to WandB.

        Args:
            d: Dictionary of metric names to values.
            step: The global training step.
            mode: Either "train" or "eval".
            custom_step_key: Optional key for asynchronous training.

        Raises:
            ValueError: If mode is invalid or step info is missing.
        """
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # NOTE: This is not simple. Wandb step must always monotonically increase and it
        # increases with each wandb.log call, but in the case of asynchronous RL for example,
        # multiple time steps is possible. For example, the interaction step with the environment,
        # the training step, the evaluation step, etc. So we need to define a custom step key
        # to log the correct step for each metric.
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" '
                    "is not handled by this wrapper."
                )
                continue

            # Do not log the custom step key itself
            if self._wandb_custom_step_key is not None and k in self._wandb_custom_step_key:
                continue

            if custom_step_key is not None:
                value_custom_step = d[custom_step_key]
                data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                self._wandb.log(data)
                continue

            self._wandb.log(data={f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """
        Log a video file to WandB.

        Args:
            video_path: Path to the video file.
            step: The global training step.
            mode: Either "train" or "eval".

        Raises:
            ValueError: If mode is not "train" or "eval".
        """
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)

    def finish(self) -> None:
        """Finalize the WandB run."""
        if self._wandb.run is not None:
            self._wandb.finish()

    def get_run_url(self) -> str | None:
        """Get the URL to view this run on WandB."""
        if self._wandb.run is not None:
            return self._wandb.run.get_url()
        return None
