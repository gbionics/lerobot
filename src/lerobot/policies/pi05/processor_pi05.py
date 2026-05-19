#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
import string
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    ObservationProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _normalize_prompt(text: str) -> str:
    """Normalize a prompt string to match the openpi tokenizer convention.

    Applies: lowercase, replace underscores with spaces, collapse extra whitespace,
    strip trailing punctuation.  Matches ``tokenize_high_low_prompt`` in the JAX
    reference implementation.
    """
    text = text.lower().strip().replace("_", " ").replace("\n", " ")
    if text and text[-1] in string.punctuation:
        text = text[:-1]
    return text


@ProcessorStepRegistry.register(name="pi05_subtask_tokenizer_processor_step")
@dataclass
class Pi05SubtaskTokenizerProcessorStep(ObservationProcessorStep):
    """Tokenize prompt for pi0.5 subtask training.

    Replaces Pi05PrepareStateTokenizerProcessorStep + TokenizerProcessorStep
    for the subtask training pipeline. Builds a "Task: X. Subtask: X" prompt
    (identity subtask by default — swap in real high/low annotations here when
    available), tokenizes it, and produces four observation keys:

    - ``OBS_LANGUAGE_TOKENS``: token IDs [max_length]
    - ``OBS_LANGUAGE_ATTENTION_MASK``: padding mask [max_length]
    - ``token_ar_mask``: 0=bidirectional (prefix), 1=causal (subtask) [max_length]
    - ``token_loss_mask``: True only on subtask + EOS tokens [max_length]

    Note: state information is **not** included in the prompt (matching the JAX
    openpi_with_subtask implementation). Use the standard pipeline's
    Pi05PrepareStateTokenizerProcessorStep if state-in-prompt is needed.
    """

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_length: int = 200
    eos_token_id: int = 1
    task_key: str = "task"

    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def observation(self, observation: dict) -> dict:
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError(f"No '{self.task_key}' found in complementary data for subtask tokenization")

        if isinstance(tasks, str):
            tasks = [tasks]

        # Use real subtask annotations when available, otherwise fall back to identity subtask
        subtasks = complementary_data.get("subtask", None)
        if isinstance(subtasks, str):
            subtasks = [subtasks]

        all_tokens, all_masks, all_ar_masks, all_loss_masks = [], [], [], []

        for i, task in enumerate(tasks):
            if not isinstance(task, str):
                task = str(task)
            task = _normalize_prompt(task)

            # Use real subtask if available, otherwise identity (high = low = task)
            if subtasks is not None and i < len(subtasks):
                raw_low = subtasks[i] if isinstance(subtasks[i], str) else str(subtasks[i])
                low_prompt = _normalize_prompt(raw_low)
            else:
                low_prompt = task

            prefix_str = f"Task: {task}. Subtask: "
            suffix_str = low_prompt

            prefix_ids = self._tokenizer.encode(prefix_str, add_special_tokens=True)
            suffix_ids = self._tokenizer.encode(suffix_str, add_special_tokens=False) + [self.eos_token_id]

            all_ids = prefix_ids + suffix_ids
            prefix_len = len(prefix_ids)
            suffix_len = len(suffix_ids)
            total = len(all_ids)

            ar_mask = [0] * prefix_len + [1] * suffix_len
            loss_mask = [False] * prefix_len + [True] * suffix_len

            # Truncate or pad to max_length
            if total > self.max_length:
                all_ids = all_ids[: self.max_length]
                ar_mask = ar_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                if not any(loss_mask):
                    logging.warning(
                        f"Subtask sequence truncated to {self.max_length} tokens and all suffix "
                        f"(loss) tokens were cut off. CE loss will be zero for this sample. "
                        f"prefix_len={prefix_len}, total={total}. Consider increasing max_length."
                    )
                valid_len = self.max_length
                pad_len = 0
            else:
                valid_len = total
                pad_len = self.max_length - total
                all_ids += [0] * pad_len
                ar_mask += [0] * pad_len
                loss_mask += [False] * pad_len

            att_mask = [True] * valid_len + [False] * pad_len

            all_tokens.append(np.array(all_ids, dtype=np.int64))
            all_masks.append(np.array(att_mask, dtype=bool))
            all_ar_masks.append(np.array(ar_mask, dtype=np.int32))
            all_loss_masks.append(np.array(loss_mask, dtype=bool))

        new_observation = dict(observation)
        new_observation[OBS_LANGUAGE_TOKENS] = torch.tensor(np.stack(all_tokens), dtype=torch.long)
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(np.stack(all_masks), dtype=torch.bool)
        new_observation["token_ar_mask"] = torch.tensor(np.stack(all_ar_masks), dtype=torch.long)
        new_observation["token_loss_mask"] = torch.tensor(np.stack(all_loss_masks), dtype=torch.bool)
        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi05_subtask_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pre/post-processor pipeline for pi0.5 subtask training.

    Identical to the standard pipeline except that
    Pi05PrepareStateTokenizerProcessorStep + TokenizerProcessorStep are
    replaced by Pi05SubtaskTokenizerProcessorStep, which additionally produces
    ``token_ar_mask`` and ``token_loss_mask`` for the subtask CE loss.
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05SubtaskTokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
