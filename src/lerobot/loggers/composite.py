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
Composite logger that delegates to multiple logging backends.

This module provides CompositeLogger, which wraps a list of Logger instances
and fans out every call to each backend. This allows running several loggers
simultaneously (e.g. WandB + MLflow).
"""

from __future__ import annotations

import logging
from pathlib import Path

from lerobot.loggers.logger import Logger


class CompositeLogger:
    """Fan-out logger that delegates to zero or more backends."""

    def __init__(self, loggers: list[Logger] | None = None):
        self._loggers: list[Logger] = loggers or []

    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        for lg in self._loggers:
            try:
                lg.log_dict(d, step=step, mode=mode, custom_step_key=custom_step_key)
            except Exception:
                logging.exception("Logger %s failed in log_dict", type(lg).__name__)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        for lg in self._loggers:
            try:
                lg.log_video(video_path, step, mode=mode)
            except Exception:
                logging.exception("Logger %s failed in log_video", type(lg).__name__)

    def log_policy(self, checkpoint_dir: Path) -> None:
        for lg in self._loggers:
            try:
                lg.log_policy(checkpoint_dir)
            except Exception:
                logging.exception("Logger %s failed in log_policy", type(lg).__name__)

    def finish(self) -> None:
        for lg in self._loggers:
            try:
                lg.finish()
            except Exception:
                logging.exception("Logger %s failed in finish", type(lg).__name__)
