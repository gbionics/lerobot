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
Logger configuration classes.

This module defines the base LoggerConfig class that uses draccus.ChoiceRegistry
for automatic subclass registration and discovery. All logger configurations
should inherit from LoggerConfig.

Example:
    To create a custom logger configuration:

    >>> from dataclasses import dataclass
    >>> from lerobot.loggers.config import LoggerConfig
    >>>
    >>> @LoggerConfig.register_subclass("my_logger")
    >>> @dataclass
    >>> class MyLoggerConfig(LoggerConfig):
    ...     api_key: str | None = None
    ...     project: str = "my_project"
"""

import abc
from dataclasses import dataclass

import draccus


@dataclass(kw_only=True)
class LoggerConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    Base configuration class for all loggers.

    This class serves as the foundation for logger configurations in LeRobot.
    It uses draccus.ChoiceRegistry for automatic subclass registration, enabling
    the plugin discovery system to find and load custom logger configurations.

    Subclasses should be registered using the @LoggerConfig.register_subclass
    decorator with a unique name.

    Attributes:
        enable: Whether the logger is enabled. Defaults to True.
        disable_artifact: Whether to skip artifact uploads. Defaults to False.

    Example:
        >>> @LoggerConfig.register_subclass("wandb")
        >>> @dataclass
        >>> class WandBLoggerConfig(LoggerConfig):
        ...     project: str = "lerobot"
        ...     entity: str | None = None
    """

    enable: bool = False
    disable_artifact: bool = False

    @property
    def type(self) -> str:
        """
        Return the registered type name for this configuration.

        Returns:
            str: The type name used during registration (e.g., "wandb").
        """
        return self.get_choice_name(self.__class__)

    def validate(self) -> None:
        """
        Validate the configuration.

        Subclasses can override this method to add custom validation logic.
        This method is called before the logger is instantiated.

        Raises:
            ValueError: If the configuration is invalid.
        """
        pass
