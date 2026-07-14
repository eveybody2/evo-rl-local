# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import pickle  # nosec
import threading
import time
import os
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from lerobot.configs.types import FeatureType, PolicyFeature  # 🚨 引入官方的核心类

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)

# 绑定 GPU 卡号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.logger.info("======================= [__init__ START] =======================")
        self.config = config
        self.shutdown_event = threading.Event()

        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.observation_queue = Queue(maxsize=1)
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self.last_processed_obs = None

        # ==========================================
        # 🚨 核心硬编码区：彻底切断对 Client 的依赖
        # ==========================================
        self.policy_type = "smolvla"

        # 唯一需要指定的部署目录！(可以是全量模型，也可以是 LoRA 目录)
        self.pretrained_name_or_path = "/models/smolvla_lora2_peft/100000/pretrained_model"

        self.device = "cuda"





        # ==========================================
        # 🚀 STAGE 1: 模型加载与 LoRA 融合阶段
        # ==========================================
        self.logger.info("==================================================")
        self.logger.info("🚀 正在 Server 启动阶段预加载模型...")
        start_load = time.perf_counter()

        from lerobot.configs.policies import PreTrainedConfig

        # 读取目标部署目录的配置
        target_config = PreTrainedConfig.from_pretrained(self.pretrained_name_or_path)
        policy_class = get_policy_class(self.policy_type)

        # 🔍 智能探测：检查配置中的 use_peft，或者目录下是否有 LoRA 特有的 adapter_config.json
        is_lora_model = getattr(target_config, "use_peft", False) or os.path.exists(
            os.path.join(self.pretrained_name_or_path, "adapter_config.json")
        )

        if is_lora_model:
            # ---------------------------------------------------------
            # 🛤️ 路由 A：检测到 LoRA 模型，执行双轨加载法
            # ---------------------------------------------------------
            self.logger.info("⚠️ 检测到 PEFT/LoRA 模型，启用双轨加载机制 (Base Weights + LoRA Config)")

            # 自动从 config 中提取底层 Base 模型路径
            base_model_path = getattr(target_config, "pretrained_path", None)
            if not base_model_path:
                raise ValueError(
                    f"🚨 致命错误：检测到 LoRA 模型，但在其 config.json 中找不到 `pretrained_path` 属性！"
                )

            self.logger.info(f"🧱 1/3 正在加载基础模型权重: {base_model_path}")
            self.policy = policy_class.from_pretrained(base_model_path, config=target_config)

            from peft import PeftModel
            self.logger.info(f"🧩 2/3 正在挂载 LoRA 权重: {self.pretrained_name_or_path}")
            self.policy = PeftModel.from_pretrained(self.policy, self.pretrained_name_or_path)

            self.logger.info("🔄 3/3 正在执行权重合并 (merge_and_unload)...")
            self.policy = self.policy.merge_and_unload()

        else:
            # ---------------------------------------------------------
            # 🛤️ 路由 B：检测到全量模型
            # ---------------------------------------------------------
            self.logger.info("📦 检测到全量模型，执行标准原生加载机制")
            self.logger.info(f"🧱 正在加载全量权重与配置: {self.pretrained_name_or_path}")
            self.policy = policy_class.from_pretrained(self.pretrained_name_or_path)

            from peft import PeftModel
            if isinstance(self.policy, PeftModel):
                self.logger.warning("🚨 [性能警告] 探测到'全量模型'顶层意外残留了 PEFT 包装壳！正在自动修复...")
                self.policy = self.policy.merge_and_unload()
            elif hasattr(self.policy, "model") and isinstance(self.policy.model, PeftModel):
                self.logger.warning("🚨 [性能警告] 探测到'全量模型'内部意外残留了 PEFT 包装壳！正在自动修复...")
                self.policy.model = self.policy.model.merge_and_unload()

        self.policy.to(self.device)
        self.policy.eval()
        self.logger.info(f"✅ 模型预加载整体工作完成！耗时: {time.perf_counter() - start_load:.2f} 秒")

        # ==========================================
        # 🚀 STAGE 2: 处理器组装与参数强校验
        # ==========================================
        self.logger.info("⚙️ 正在执行参数校验与组装 Pre/Post-processor...")

        if hasattr(self.policy.config, "n_action_steps"):
            self.actions_per_chunk = self.policy.config.n_action_steps
        elif hasattr(self.policy.config, "chunk_size"):
            self.actions_per_chunk = self.policy.config.chunk_size
        else:
            raise AttributeError(
                f"🚨 致命错误: 在 {self.policy_type} 的 config 中找不到 `n_action_steps` 或 `chunk_size` 参数！"
            )
        self.logger.info(f"✅ 动作截断长度 (actions_per_chunk) 已锁定为: {self.actions_per_chunk}")

        self.logger.info("======================= [__init__ END] =======================")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"lerobot_features keys (before): {list(policy_specs.lerobot_features.keys())} "
        )

        # 1. 正常接收客户端发来的传感器特征和映射表
        self.rename_map = getattr(policy_specs, "rename_map", {})
        self.lerobot_features = policy_specs.lerobot_features

        # 2. 提前篡改 lerobot_features，将客户端的物理传感器名翻译成模型特征名
        for old_key, new_key in self.rename_map.items():
            if old_key in self.lerobot_features:
                self.lerobot_features[new_key] = self.lerobot_features.pop(old_key)

        # 3. 重新初始化处理器 (这里只做轻量级组装，不到 0.01 秒)
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.pretrained_name_or_path, # 使用你在 __init__ 中硬编码的路径
            preprocessor_overrides={
                "device_processor": device_override,
                # 🚨 核心修改：这里必须传入空字典 {}！
                # 告诉官方预处理器“不需要重命名了”，因为我们在 _predict_action_chunk 已经把活干完了
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        return services_pb2.Empty()


    def SendObservations(self, request_iterator, context):  # noqa: N802
        client_id = context.peer()

        # 🟢 记录接收时间用于监控延迟
        receive_time = time.time()

        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger)
        timed_observation = pickle.loads(received_bytes)  # nosec

        # 🟢 计算并打印 FPS 和 网络延迟
        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)
        self.logger.debug(
            f"Received obs #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")
            pass
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            actions_bytes = pickle.dumps(action_chunk)  # nosec
            actions = services_pb2.Actions(data=actions_bytes)

            time.sleep(max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts)))
            return actions
        except Empty:
            return services_pb2.Empty()
        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps
        if obs.get_timestep() in predicted_timesteps:
            return False
        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            return False
        return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        if obs.must_go or self.last_processed_obs is None or self._obs_sanity_checks(obs, self.last_processed_obs):
            if self.observation_queue.full():
                _ = self.observation_queue.get_nowait()
            self.observation_queue.put(obs)
            return True
        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        raw_obs_dict = observation_t.get_observation()
        print(f"==========rename_map start==========")
        self.logger.info(f"DEBUG: raw_obs_dict (before mapping): {raw_obs_dict}")
        # 🚨 3. 利用保存的 rename_map，篡改每一帧的原始数据 Key
        if hasattr(self, "rename_map") and self.rename_map:
            for old_key, new_key in self.rename_map.items():
                # rename_map 的 key 是 "observation.images.base_0_rgb"
                # 但 raw_obs_dict 里的 key 是 "base_0_rgb"，需要剥离前缀
                raw_old = old_key.split(".")[-1]
                raw_new = new_key.split(".")[-1]
                if raw_old in raw_obs_dict:
                    raw_obs_dict[raw_new] = raw_obs_dict.pop(raw_old)

        self.logger.info(f"DEBUG: raw_obs_dict (after mapping): {raw_obs_dict}")

        # ==========================================
        # 📝 Task Prompt 极简提取 (交由预处理器处理)
        # ==========================================
        raw_task = raw_obs_dict.get("task", "")
        if isinstance(raw_task, list) and len(raw_task) > 0:
            raw_task = raw_task[0]

        if isinstance(raw_task, str):
            raw_obs_dict["task"] = raw_task.strip().replace('_', ' ')

        # 此时传进 raw_observation_to_observation 的数据已完美对齐，不可能再报 KeyError！
        observation: Observation = raw_observation_to_observation(
            raw_obs_dict,
            self.lerobot_features,
            self.policy_image_features,
        )

        # 2. 预处理 (触发缩放裁剪机制)
        # 🔍 【调试插入】检查预处理后的 observation 结构
        self.logger.info(f"DEBUG: Preprocessor input observation: {observation}")
        print(f"==========rename_map end==========")
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t

        # 3. 核心推理 (开启 BF16 加速)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            action_tensor = self._get_action_chunk(observation)

        # 4. 后处理
        # 🚨 遍历 Chunk 维度，因为 LeRobot 反归一化器期望输入是 (Batch, Action_Dim)
        # _, chunk_size, _ = action_tensor.shape
        # processed_actions = []
        # for i in range(chunk_size):
        #     single_action = action_tensor[:, i, :]
        #     processed_action = self.postprocessor(single_action)
        #     processed_actions.append(processed_action)
        #
        # # 重新堆叠回 (Batch, chunk_size, action_dim) 并去掉 Batch 维
        # action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        # action_tensor = action_tensor.detach().cpu()

        B, C, D = action_tensor.shape

        # 1. 压平: (B, C, D) -> (B * C, D)，使用 reshape 替代 view 更安全
        flattened_action = action_tensor.reshape(B * C, D)

        # 2. 一次性进行向量化的反归一化 (充分利用 GPU 并行)
        processed_flattened = self.postprocessor(flattened_action)

        # 3. 恢复形状，并严格对齐原代码的后处理逻辑 (去掉Batch维，移至CPU)
        action_tensor = processed_flattened.reshape(B, C, D)
        action_tensor = action_tensor.squeeze(0).detach().cpu()


        # ==========================================
        # 🤖 物理动作维度安全截断
        # ==========================================
        if action_tensor.shape[-1] > 6:
            action_tensor = action_tensor[:, :6]

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        return action_chunk

    def stop(self):
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = PolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()