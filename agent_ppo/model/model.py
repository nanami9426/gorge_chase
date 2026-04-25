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


def make_gru_layer(input_size, hidden_size):
    """Create a GRU layer with orthogonal initialization.

    创建带正交初始化的 GRU。
    """
    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
    )
    for name, param in gru.named_parameters():
        if "weight" in name:
            nn.init.orthogonal_(param.data)
        else:
            nn.init.zeros_(param.data)
    return gru


class Model(nn.Module):
    """Hybrid per-frame encoder + temporal GRU with Actor/Critic heads.

    单帧稠密特征分支 + CNN 空间分支编码，再由 GRU 聚合最近几帧，
    最后接 Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_temporal_gru"
        self.device = device

        dense_dim = Config.DENSE_FEATURE_LEN
        frame_feature_len = Config.FRAME_FEATURE_LEN
        spatial_map_size = Config.SPATIAL_MAP_SIZE
        spatial_channels = Config.SPATIAL_CHANNELS
        temporal_window = Config.TEMPORAL_WINDOW
        action_num = Config.ACTION_NUM
        value_head_num = Config.VALUE_HEAD_NUM
        aux_target_num = Config.AUX_TARGET_NUM
        frame_embed_dim = Config.FRAME_EMBED_DIM
        temporal_hidden_dim = Config.TEMPORAL_HIDDEN_DIM

        self.frame_feature_len = frame_feature_len
        self.temporal_window = temporal_window

        self.dense_backbone = nn.Sequential(
            make_fc_layer(dense_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_fc_layer(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.spatial_backbone = nn.Sequential(
            make_conv_layer(spatial_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            make_conv_layer(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            make_conv_layer(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        conv_output_size = spatial_map_size // 4 + 1
        self.spatial_proj = nn.Sequential(
            nn.Flatten(),
            make_fc_layer(64 * conv_output_size * conv_output_size, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
        )

        self.frame_fusion = nn.Sequential(
            make_fc_layer(224, frame_embed_dim),
            nn.LayerNorm(frame_embed_dim),
            nn.ReLU(),
        )

        self.temporal_encoder = make_gru_layer(frame_embed_dim, temporal_hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=temporal_hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(temporal_hidden_dim)

        self.fusion = nn.Sequential(
            make_fc_layer(frame_embed_dim + temporal_hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            make_fc_layer(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(128, action_num)

        # Multi-head critic / 多头价值
        self.value_head = make_fc_layer(128, value_head_num)

        # Auxiliary prediction head / 辅助预测头
        self.aux_head = nn.Sequential(
            make_fc_layer(128, 64),
            nn.ReLU(),
            make_fc_layer(64, aux_target_num),
            nn.Sigmoid(),
        )

    def forward(self, obs, inference=False):
        obs = obs.reshape(-1, self.temporal_window, self.frame_feature_len)

        dense_obs = obs[:, :, : Config.DENSE_FEATURE_LEN].reshape(-1, Config.DENSE_FEATURE_LEN)
        spatial_obs = obs[:, :, Config.DENSE_FEATURE_LEN :].reshape(
            -1,
            Config.SPATIAL_CHANNELS,
            Config.SPATIAL_MAP_SIZE,
            Config.SPATIAL_MAP_SIZE,
        )

        dense_hidden = self.dense_backbone(dense_obs)
        spatial_hidden = self.spatial_proj(self.spatial_backbone(spatial_obs))
        frame_hidden = self.frame_fusion(torch.cat([dense_hidden, spatial_hidden], dim=1))
        frame_hidden = frame_hidden.reshape(-1, self.temporal_window, Config.FRAME_EMBED_DIM)

        temporal_sequence, temporal_state = self.temporal_encoder(frame_hidden)
        current_hidden = frame_hidden[:, -1, :]
        temporal_hidden = temporal_state[-1]
        attention_query = temporal_sequence[:, -1:, :]
        attention_context, _ = self.temporal_attention(
            query=attention_query,
            key=temporal_sequence,
            value=temporal_sequence,
            need_weights=False,
        )
        attention_context = self.attention_norm(attention_context.squeeze(1) + temporal_hidden)
        hidden = self.fusion(torch.cat([current_hidden, temporal_hidden, attention_context], dim=1))
        logits = self.actor_head(hidden)
        value_heads = self.value_head(hidden)
        aux_pred = self.aux_head(hidden)
        return logits, value_heads, aux_pred

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
