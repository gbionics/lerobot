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
MLflow logger implementation.

This module provides a built-in MLflow backend for LeRobot's logging system.
It implements the Logger interface for logging metrics, videos, and model
artifacts to MLflow.

Example:
    >>> from lerobot.loggers.mlflow import MLFlowLogger, MLFlowLoggerConfig
    >>>
    >>> config = MLFlowLoggerConfig(experiment_name="lerobot")
    >>> logger = MLFlowLogger(config, train_cfg)
    >>> logger.log_dict({"loss": 0.5}, step=100)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.loggers.config import LoggerConfig
from lerobot.loggers.logger import Logger

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


@LoggerConfig.register_subclass("mlflow")
@dataclass
class MLFlowLoggerConfig(LoggerConfig):
    """Configuration for the MLflow logger."""

    experiment_name: str = "lerobot"
    tracking_uri: str | None = None
    run_name: str | None = None
    tags: dict[str, str] | None = None


class MLFlowLogger(Logger):
    """MLflow logger implementation."""

    def __init__(self, cfg: MLFlowLoggerConfig, train_cfg: "TrainPipelineConfig"):
        super().__init__(cfg, train_cfg)

        # Lazy import to keep mlflow optional until this backend is used.
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is not installed. Please install LeRobot with MLflow support via: pip install -e ."
            ) from exc

        self._mlflow = mlflow

        if cfg.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking_uri)

        experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(cfg.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        tags = cfg.tags.copy() if cfg.tags else {}
        tags["policy_type"] = train_cfg.policy.type
        tags["seed"] = str(train_cfg.seed)
        if train_cfg.dataset is not None:
            tags["dataset"] = train_cfg.dataset.repo_id
        if train_cfg.env is not None:
            tags["env_type"] = train_cfg.env.type

        run_name = cfg.run_name or self.job_name
        self._run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        )

        self._log_config_params(train_cfg)

        logging.info("MLflow run started: %s", self._run.info.run_id)
        logging.info("Track this run --> %s", self.get_run_url())

    def _log_config_params(self, train_cfg: "TrainPipelineConfig") -> None:
        """Log key training configuration values as MLflow params."""
        params = {
            "batch_size": train_cfg.batch_size,
            "steps": train_cfg.steps,
            "seed": train_cfg.seed,
            "log_freq": train_cfg.log_freq,
            "eval_freq": train_cfg.eval_freq,
            "save_freq": train_cfg.save_freq,
        }

        if train_cfg.optimizer:
            params["optimizer_type"] = train_cfg.optimizer.type
            params["learning_rate"] = train_cfg.optimizer.lr

        self._mlflow.log_params(params)

    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'.")
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        actual_step = step
        if custom_step_key is not None and custom_step_key in d:
            actual_step = int(d[custom_step_key])

        metrics = {}
        for k, v in d.items():
            if not isinstance(v, int | float):
                continue
            if custom_step_key is not None and k == custom_step_key:
                continue
            metrics[f"{mode}/{k}"] = v

        if metrics:
            self._mlflow.log_metrics(metrics, step=actual_step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'eval'.")

        artifact_path = f"{mode}/videos/step_{step}"
        self._mlflow.log_artifact(video_path, artifact_path=artifact_path)

    def log_policy(self, checkpoint_dir: Path) -> None:
        if self.config.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_path = f"checkpoints/{step_id}"
        self._mlflow.log_artifacts(str(checkpoint_dir), artifact_path=artifact_path)

    def finish(self) -> None:
        if self._run is not None:
            self._mlflow.end_run()
            logging.info("MLflow run ended: %s", self._run.info.run_id)

    def get_run_url(self) -> str | None:
        if self._run is None:
            return None

        tracking_uri = self._mlflow.get_tracking_uri()
        run_id = self._run.info.run_id
        experiment_id = self._run.info.experiment_id

        if tracking_uri.startswith(("http://", "https://")):
            return f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"

        return f"mlruns/{experiment_id}/{run_id}"
