#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Create a conv layer with orthogonal initialization.

    创建正交初始化的卷积层。
    """
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


class Model(nn.Module):
    """Hybrid dense + CNN backbone with Actor/Critic heads.

    稠密特征分支 + CNN 空间分支骨干，接 Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite"
        self.device = device

        dense_dim = Config.DENSE_FEATURE_LEN
        spatial_map_size = Config.SPATIAL_MAP_SIZE
        spatial_channels = Config.SPATIAL_CHANNELS
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.dense_backbone = nn.Sequential(
            make_fc_layer(dense_dim, 64),
            nn.ReLU(),
        )

        self.spatial_backbone = nn.Sequential(
            make_conv_layer(spatial_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            make_conv_layer(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            make_conv_layer(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        conv_output_size = spatial_map_size // 4 + 1
        self.spatial_proj = nn.Sequential(
            nn.Flatten(),
            make_fc_layer(32 * conv_output_size * conv_output_size, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            make_fc_layer(128, 128),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(128, action_num)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(128, value_num)

    def forward(self, obs, inference=False):
        dense_obs = obs[:, : Config.DENSE_FEATURE_LEN]
        spatial_obs = obs[:, Config.DENSE_FEATURE_LEN :]
        spatial_obs = spatial_obs.reshape(
            -1,
            Config.SPATIAL_CHANNELS,
            Config.SPATIAL_MAP_SIZE,
            Config.SPATIAL_MAP_SIZE,
        )

        dense_hidden = self.dense_backbone(dense_obs)
        spatial_hidden = self.spatial_proj(self.spatial_backbone(spatial_obs))
        hidden = self.fusion(torch.cat([dense_hidden, spatial_hidden], dim=1))
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
