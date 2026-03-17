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
Logger factory for creating logger instances.

This module provides the factory function for instantiating loggers based on
their configuration type. It follows the same pattern as other LeRobot factories
(e.g., make_policy, make_env).

The factory supports automatic discovery of logger implementations through
the plugin system. Custom loggers registered with the LoggerConfig.register_subclass
decorator will be automatically discovered and available.

Example:
    >>> from lerobot.loggers.factory import make_logger
    >>> from lerobot.loggers.wandb import WandBLoggerConfig
    >>>
    >>> config = WandBLoggerConfig(project="my_project")
    >>> logger = make_logger(config, train_cfg)
"""

import importlib
import logging
from typing import TYPE_CHECKING

from lerobot.loggers.config import LoggerConfig
from lerobot.loggers.logger import Logger

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

# Registry mapping logger type names to their implementation module paths
# This allows for lazy loading of logger implementations
_LOGGER_REGISTRY: dict[str, tuple[str, str]] = {
    "mlflow": ("lerobot.loggers.mlflow", "MLFlowLogger"),
    "wandb": ("lerobot.loggers.wandb", "WandBLogger"),
}


def register_logger(type_name: str, module_path: str, class_name: str) -> None:
    """
    Register a logger implementation in the factory registry.

    This function allows external packages to register their logger implementations
    for discovery by the factory. It's typically called during package initialization.

    Args:
        type_name: The type name used in configuration (e.g., "mlflow").
        module_path: The full module path where the logger class is defined.
        class_name: The name of the logger class to import.

    Example:
        >>> register_logger("mlflow", "my_package.loggers.mlflow", "MLFlowLogger")
    """
    _LOGGER_REGISTRY[type_name] = (module_path, class_name)


def get_logger_class(type_name: str) -> type[Logger]:
    """
    Get the logger class for a given type name.

    Args:
        type_name: The registered type name (e.g., "wandb").

    Returns:
        The Logger subclass for the given type.

    Raises:
        ValueError: If the logger type is not registered.
        ImportError: If the logger module cannot be imported.
    """
    if type_name not in _LOGGER_REGISTRY:
        available = list(_LOGGER_REGISTRY.keys())
        raise ValueError(
            f"Unknown logger type: '{type_name}'. "
            f"Available loggers: {available}. "
            "Make sure the logger package is installed and registered."
        )

    module_path, class_name = _LOGGER_REGISTRY[type_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_logger(
    cfg: LoggerConfig,
    train_cfg: "TrainPipelineConfig",
) -> Logger:
    """
    Create a logger instance from configuration.

    This is the main factory function for instantiating loggers. It looks up
    the appropriate logger class based on the configuration type and creates
    an instance with the provided configurations.

    Args:
        cfg: Logger configuration (e.g., WandBLoggerConfig).
        train_cfg: The training pipeline configuration.

    Returns:
        An initialized Logger instance.

    Raises:
        ValueError: If the logger type is not recognized.
        ImportError: If the logger module cannot be imported.

    Example:
        >>> from lerobot.loggers import make_logger, WandBLoggerConfig
        >>>
        >>> config = WandBLoggerConfig(project="my_project", mode="online")
        >>> logger = make_logger(config, train_cfg)
        >>> logger.log_dict({"loss": 0.5}, step=100)
    """
    logger_type = cfg.type
    logging.debug(f"Creating logger of type: {logger_type}")

    # Validate configuration before creating logger
    cfg.validate()

    logger_class = get_logger_class(logger_type)
    return logger_class(cfg, train_cfg)


def make_loggers(
    configs: list[LoggerConfig],
    train_cfg: "TrainPipelineConfig",
) -> list[Logger]:
    """
    Create multiple logger instances from a list of configurations.

    This function supports the use case where multiple logging backends
    are used simultaneously (e.g., WandB and MLflow together).

    Args:
        configs: List of logger configurations.
        train_cfg: The training pipeline configuration.

    Returns:
        List of initialized Logger instances.

    Example:
        >>> configs = [WandBLoggerConfig(), MLFlowLoggerConfig()]
        >>> loggers = make_loggers(configs, train_cfg)
    """
    return [make_logger(cfg, train_cfg) for cfg in configs if cfg.enable]
