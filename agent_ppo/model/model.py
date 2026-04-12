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


class FeatureProjector(nn.Module):
    def __init__(self, in_dim, token_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = make_fc_layer(in_dim, token_dim)
        self.fc2 = make_fc_layer(token_dim, token_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            make_fc_layer(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            make_fc_layer(ff_hidden_dim, embed_dim),
        )

    def forward(self, x):
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, ff_hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            make_fc_layer(hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            make_fc_layer(ff_hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class Model(nn.Module):
    # 分组特征编码 + 自注意力 + MLP
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_attn"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        self.feature_splits = list(Config.FEATURE_SPLIT_SHAPE)

        token_dim = 64
        attn_heads = 4
        attn_ff_dim = 128
        trunk_dim = 256
        head_dim = 128

        self.feature_projectors = nn.ModuleList(
            [FeatureProjector(split_dim, token_dim) for split_dim in self.feature_splits]
        )
        self.token_pos_embed = nn.Parameter(torch.zeros(1, len(self.feature_splits), token_dim))
        nn.init.normal_(self.token_pos_embed, mean=0.0, std=0.02)

        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(token_dim, attn_heads, attn_ff_dim),
                AttentionBlock(token_dim, attn_heads, attn_ff_dim),
            ]
        )
        self.token_norm = nn.LayerNorm(token_dim)

        fused_dim = len(self.feature_splits) * token_dim + token_dim
        self.obs_skip = nn.Sequential(
            make_fc_layer(input_dim, token_dim),
            nn.ReLU(),
        )
        self.input_fusion = nn.Sequential(
            make_fc_layer(fused_dim + token_dim, trunk_dim),
            nn.ReLU(),
        )
        self.backbone_blocks = nn.ModuleList(
            [
                ResidualMLPBlock(trunk_dim, trunk_dim * 2),
                ResidualMLPBlock(trunk_dim, trunk_dim * 2),
            ]
        )
        self.backbone_out = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            make_fc_layer(trunk_dim, head_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(head_dim, head_dim),
            nn.ReLU(),
            make_fc_layer(head_dim, action_num),
        )
        self.critic_head = nn.Sequential(
            make_fc_layer(head_dim, head_dim),
            nn.ReLU(),
            make_fc_layer(head_dim, value_num),
        )

    def encode_tokens(self, obs):
        split_obs = torch.split(obs, self.feature_splits, dim=1)
        tokens = [projector(segment) for projector, segment in zip(self.feature_projectors, split_obs)]
        tokens = torch.stack(tokens, dim=1)
        tokens = tokens + self.token_pos_embed
        for block in self.attention_blocks:
            tokens = block(tokens)
        return self.token_norm(tokens)

    def forward(self, obs, inference=False):
        del inference
        tokens = self.encode_tokens(obs)
        pooled_token = tokens.mean(dim=1)
        flat_tokens = tokens.flatten(start_dim=1)
        obs_skip = self.obs_skip(obs)
        hidden = self.input_fusion(torch.cat([flat_tokens, pooled_token, obs_skip], dim=1))
        for block in self.backbone_blocks:
            hidden = block(hidden)
        hidden = self.backbone_out(hidden)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
