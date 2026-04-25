#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

from collections import deque

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.feature_history = deque(maxlen=Config.TEMPORAL_WINDOW)
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1
        self.feature_history.clear()

    def observation_process(self, env_obs, curriculum_stage=None, loot_reward_scale=None):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        frame_feature, legal_action, remain_info = self.preprocessor.feature_process(
            env_obs,
            self.last_action,
            curriculum_stage=curriculum_stage,
            loot_reward_scale=loot_reward_scale,
        )
        feature = self._stack_temporal_feature(frame_feature)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
            action_prior=remain_info.get("action_prior", [0.0] * Config.ACTION_NUM),
        )
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        action_prior = getattr(list_obs_data[0], "action_prior", None)

        logits, value, prob = self._run_model(feature, legal_action, action_prior=action_prior)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        try:
            state_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            if self.logger:
                self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"model file {model_file_path} not found, keep current weights")
        except RuntimeError as exc:
            if self.logger:
                self.logger.warning(
                    f"skip incompatible model {model_file_path}, keep current weights: {exc}"
                )

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action, action_prior=None):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        action_prior_np = self._normalize_action_prior(action_prior, legal_action_np)
        logits_np = logits_np + float(Config.ACTION_PRIOR_LOGIT_SCALE) * action_prior_np
        prob = self._legal_soft_max(logits_np, legal_action_np)
        prob = self._apply_sampling_floor(prob, legal_action_np)
        prob = self._mix_action_prior(prob, action_prior_np, legal_action_np)

        return logits_np, value_np, prob

    def _stack_temporal_feature(self, frame_feature):
        """Build a fixed-length temporal observation from recent frames.

        用最近几帧单帧特征构造固定长度的时序观测。
        """
        frame_feature = np.asarray(frame_feature, dtype=np.float32)

        if not self.feature_history:
            for _ in range(Config.TEMPORAL_WINDOW):
                self.feature_history.append(frame_feature.copy())
        else:
            self.feature_history.append(frame_feature.copy())

        stacked_frames = list(self.feature_history)
        if len(stacked_frames) < Config.TEMPORAL_WINDOW:
            pad_frame = stacked_frames[0]
            stacked_frames = [pad_frame.copy() for _ in range(Config.TEMPORAL_WINDOW - len(stacked_frames))] + stacked_frames

        return np.concatenate(stacked_frames, axis=0).astype(np.float32, copy=False)

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))

    def _apply_sampling_floor(self, probs, legal_action):
        floor = float(Config.SAMPLING_PROB_FLOOR)
        if floor <= 0.0:
            return probs

        legal_mask = np.array(legal_action, dtype=np.float32)
        legal_count = float(np.sum(legal_mask))
        if legal_count <= 1.0:
            return probs

        uniform_prob = legal_mask / legal_count
        mixed_prob = (1.0 - floor) * probs + floor * uniform_prob
        return mixed_prob / (np.sum(mixed_prob, keepdims=True) * 1.00001)

    def _normalize_action_prior(self, action_prior, legal_action):
        if action_prior is None:
            return np.zeros(Config.ACTION_NUM, dtype=np.float32)

        prior = np.array(action_prior, dtype=np.float32)
        if prior.shape[0] != Config.ACTION_NUM:
            prior = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prior = np.clip(prior, 0.0, 1.0) * legal_action
        return prior

    def _mix_action_prior(self, probs, action_prior, legal_action):
        prior = np.array(action_prior, dtype=np.float32) * np.array(legal_action, dtype=np.float32)
        prior_sum = float(np.sum(prior))
        if prior_sum <= 1e-6:
            return probs

        prior_dist = prior / prior_sum
        mix_weight = float(Config.ACTION_PRIOR_MIX_MAX) * float(np.clip(np.max(prior), 0.0, 1.0))
        mixed_prob = (1.0 - mix_weight) * probs + mix_weight * prior_dist
        return mixed_prob / (np.sum(mixed_prob, keepdims=True) * 1.00001)
