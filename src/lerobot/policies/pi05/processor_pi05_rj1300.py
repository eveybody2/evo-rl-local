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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="concat_dual_arm_state_step")
@dataclass
class ConcatDualArmStateStep(ProcessorStep):
    """
    将左右臂的 state 拼接为一个完整的 state 向量
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {})
        # 请替换为你的数据集中实际的 key 名称，比如 "observation.state"
        left_state = obs.get("observation.state")
        right_state = obs.get("observation.rightstate")

        if left_state is not None and right_state is not None:
            # 沿着最后一个维度拼接，6 + 6 = 12
            obs[OBS_STATE] = torch.cat([left_state, right_state], dim=-1)

        return transition

    def transform_features(self, features):
        return features


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

        # 获取状态
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")

        # 1. 获取左右臂的力控数据
        obs_dict = transition.get(TransitionKey.OBSERVATION, {})
        left_force_data = obs_dict.get("observation.force")
        right_force_data = obs_dict.get("observation.rightforce")

        state = deepcopy(state)
        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # Discretize into 256 bins
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        batch_size = state.shape[0]
        full_prompts = []

        for i in range(batch_size):
            # 2. 格式化力控数据：保留 2 位小数并转为字符串
            left_force_str = "None"
            if left_force_data is not None:
                left_force_str = " ".join([f"{v:.2f}" for v in left_force_data[i].tolist()])

            right_force_str = "None"
            if right_force_data is not None:
                right_force_str = " ".join([f"{v:.2f}" for v in right_force_data[i].tolist()])

            # 3. 注入到自定义的英文 Task 描述中
            # custom_task_desc = (
            #     "Use both arms to insert the black cable into the trumpet-shaped opening. "
            #     "Both arms have real-time force control data. "
            #     f"Left arm: {left_force_str}; Right arm: {right_force_str}"
            # )
            custom_task_desc = (
                "Use both arms to insert the black cable into the trumpet-shaped opening. "
            )

            # 状态序列化
            state_str = " ".join(map(str, discretized_states[i]))

            # 4. 组装最终给 PI0.5 的完整 Prompt
            full_prompt = f"Task: {custom_task_desc}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        # 将生成的 Prompt 存入 complementary data 供分词器使用
        if TransitionKey.COMPLEMENTARY_DATA not in transition:
            transition[TransitionKey.COMPLEMENTARY_DATA] = {}

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts

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

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        # 修改这里：将 {} 改为 config.rename_map
        RenameObservationsProcessorStep(rename_map=config.rename_map),
        ConcatDualArmStateStep(),  # <--- 新增在这里
        AddBatchDimensionProcessorStep(),
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
