#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Dense feature dimensions / 稠密特征维度
    DENSE_FEATURES = [
        4,
        5,
        5,
        6,
        8,
        8,
        8,
        8,
        6,
        5,
        30,
        16,
        2,
        5,
    ]

    # Spatial observation branch / 空间观测分支
    SPATIAL_MAP_SIZE = 21
    SPATIAL_CHANNELS = 4

    DENSE_FEATURE_LEN = sum(DENSE_FEATURES)
    SPATIAL_FEATURE_LEN = SPATIAL_CHANNELS * SPATIAL_MAP_SIZE * SPATIAL_MAP_SIZE
    FRAME_FEATURE_LEN = DENSE_FEATURE_LEN + SPATIAL_FEATURE_LEN

    # Temporal observation window / 时序观测窗口：将最近 N 帧拼成一条观测
    TEMPORAL_WINDOW = 10
    TEMPORAL_FEATURE_LEN = FRAME_FEATURE_LEN * TEMPORAL_WINDOW

    # Temporal encoder dims / 时序编码器维度
    FRAME_EMBED_DIM = 128
    TEMPORAL_HIDDEN_DIM = 128

    # Backward-compatible aliases / 兼容旧配置名
    FEATURES = DENSE_FEATURES
    FEATURE_SPLIT_SHAPE = DENSE_FEATURES
    FEATURE_LEN = TEMPORAL_FEATURE_LEN
    DIM_OF_OBSERVATION = TEMPORAL_FEATURE_LEN

    # Action space / 动作空间：16个移动方向
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1
    VALUE_HEAD_NUM = 4
    VALUE_HEAD_TOTAL = 0
    VALUE_HEAD_SURVIVAL = 1
    VALUE_HEAD_LOOT = 2
    VALUE_HEAD_EXPLORE = 3
    AUX_TARGET_NUM = 3

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0002
    BETA_START = 0.001
    BETA_MIN = 0.0002
    BETA_MAX = 0.008
    ENTROPY_TARGET_RATIO = 0.45
    ENTROPY_BETA_ADJUST_RATE = 0.05
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    VALUE_HEAD_COEF = 0.25
    AUX_LOSS_COEF = 0.05
    GRAD_CLIP_RANGE = 0.5
    SAMPLING_PROB_FLOOR = 0.03
    ACTION_PRIOR_LOGIT_SCALE = 1.2
    ACTION_PRIOR_MIX_MAX = 0.15
