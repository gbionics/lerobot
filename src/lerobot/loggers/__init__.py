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
Logging system for LeRobot.

This module provides a pluggable logging system that allows users to integrate
their own logging backends. The design follows the "Bring Your Own Logger"
philosophy, similar to the existing "Bring Your Own Hardware" functionality.

Built-in loggers:
    - WandBLogger: Integration with Weights & Biases (default)

To implement a custom logger:
    1. Create a config class inheriting from LoggerConfig
    2. Create a logger class inheriting from Logger
    3. Register the config with @LoggerConfig.register_subclass("my_logger")
    4. Package as lerobot_logger_* for automatic discovery
"""

from lerobot.loggers.composite import CompositeLogger
from lerobot.loggers.config import LoggerConfig
from lerobot.loggers.factory import make_logger, make_loggers, register_logger
from lerobot.loggers.logger import Logger
from lerobot.loggers.wandb import WandBLogger, WandBLoggerConfig

__all__ = [
    "CompositeLogger",
    "Logger",
    "LoggerConfig",
    "make_logger",
    "make_loggers",
    "register_logger",
    "WandBLogger",
    "WandBLoggerConfig",
]
