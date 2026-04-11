#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

from agent_ppo.feature.rewards import OrganProcessor # type: ignore

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max map Euclidean distance / 地图欧氏距离上限
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.organ_processor = OrganProcessor()
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.organ_processor.reset()

        self.visited_grids = set() # 访问过的格子
        self.grid_size = 8 # 地图128*128，把它切成16*16的格子，每格长8
        self.last_hero_pos = None
        self.stall_steps = 0

    def calc_monster_dist_reward(self, monster_feats) -> float:
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        # 远离最近的怪物，远离->正，接近->负
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        self.last_min_monster_dist_norm = cur_min_dist_norm
        return dist_shaping

    def calc_explore_reward(self, hero_pos) -> float:
        # 探索新的区域
        grid_x = hero_pos["x"] // self.grid_size
        grid_z = hero_pos["z"] // self.grid_size
        grid = (grid_x, grid_z)
        if grid not in self.visited_grids:
            explore_reward = 0.02
            self.visited_grids.add(grid)
        else:
            explore_reward = 0.0

        # 防止原地逗留
        cur_pos = (hero_pos["x"], hero_pos["z"])
        if self.last_hero_pos is None:
            moved = True
        else:
            moved = cur_pos != self.last_hero_pos
        if moved:
            self.stall_steps = 0
        else:
            self.stall_steps += 1
        stall_penalty = -min(0.002 * self.stall_steps, 0.2)
        self.last_hero_pos = cur_pos
        return explore_reward + stall_penalty

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        # 观测信息
        observation = env_obs["observation"]
        frame_state = observation["frame_state"] # 帧状态数据，类型FrameState
        env_info = observation["env_info"]  # 环境信息，类型EnvInfo
        map_info = observation["map_info"] # 局部地图信息（以英雄为中心的视野栅格，1=可通行，0=障碍物），类型int32[][]
        legal_act_raw = observation["legal_action"] # 合法动作掩码（16维，true 表示可执行），类型bool[16]

        self.step_no = observation["step_no"] # 当前步数，类型int32
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"] # 英雄位置 {x, z}（栅格坐标），类型Position
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAX_MAP_DISTANCE)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # OrganState[] 物件状态列表（宝箱、buff）
        organs = frame_state.get("organs", [])
        organs_feat, organ_reward = self.organ_processor.process(env_info=env_info, organs=organs, hero_pos=hero_pos)

        # Local map features (16D) / 局部地图特征
        # 将局部信息转成1*16的向量
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0 # 压平后的下标：4*4->16，即0~15
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # 合法动作掩码 0~7移动，8~15闪现
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                organs_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        monster_dist_reward = self.calc_monster_dist_reward(monster_feats=monster_feats)
        explore_reward = self.calc_explore_reward(hero_pos=hero_pos)

        # 每多活一步固定给奖励，后继考虑删了
        survive_reward = 0.0005
        reward = [survive_reward + monster_dist_reward + explore_reward + organ_reward]

        return feature, legal_action, reward
