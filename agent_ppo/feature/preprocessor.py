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

from agent_ppo.feature.rewards import OrganProcessor
from agent_ppo.feature.rewards import ExploreProcessor
from agent_ppo.feature.rewards import FlashProcessor
from agent_ppo.feature.rewards import MoveProcessor
from agent_ppo.feature.rewards import TerrainProcessor

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max map Euclidean distance / 地图欧氏距离上限
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
# Max monster distance bucket / 怪物距离桶上限
MAX_MONSTER_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Keep step-score shaping strong enough that "survive longer = score higher"
# becomes a first-class objective during PPO training.
STEP_SCORE_REWARD_SCALE = 0.01
SQRT_HALF = float(np.sqrt(0.5))

DIR_TO_VEC = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (SQRT_HALF, SQRT_HALF),
    3: (0.0, 1.0),
    4: (-SQRT_HALF, SQRT_HALF),
    5: (-1.0, 0.0),
    6: (-SQRT_HALF, -SQRT_HALF),
    7: (0.0, -1.0),
    8: (SQRT_HALF, -SQRT_HALF),
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.organ_processor = OrganProcessor()
        self.explore_processor = ExploreProcessor()
        self.flash_processor = FlashProcessor()
        self.move_processor = MoveProcessor()
        self.terrain_processor = TerrainProcessor()
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_step_score = 0.0
        self.organ_processor.reset()
        self.explore_processor.reset()
        self.flash_processor.reset()
        self.move_processor.reset()
        self.terrain_processor.reset()

    def calc_progress_reward(self, env_info) -> float:
        step_score = float(env_info.get("step_score", 0.0))
        step_score_gain = max(0.0, step_score - self.last_step_score)
        self.last_step_score = step_score
        return STEP_SCORE_REWARD_SCALE * step_score_gain

    def direction_to_vector(self, direction_idx):
        return DIR_TO_VEC.get(int(direction_idx), (0.0, 0.0))

    def calc_monster_dist_reward(self, monster_feats) -> float:
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            cur_min_dist_norm = min(cur_min_dist_norm, float(m_feat[4]))

        # 远离最近的怪物，远离->正，接近->负
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        self.last_min_monster_dist_norm = cur_min_dist_norm
        return dist_shaping

    def feature_process(self, env_obs, last_action):
        # last_action：0~7：移动 8~15：闪现
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
                dir_x, dir_z = self.direction_to_vector(m.get("hero_relative_direction", 0))
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                m_pos = m.get("pos", {})
                if is_in_view and isinstance(m_pos, dict) and "x" in m_pos and "z" in m_pos:
                    # In-view monsters use exact distance, direction still uses relative encoding.
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAX_MAP_DISTANCE)
                else:
                    dist_norm = _norm(m.get("hero_l2_distance", MAX_MONSTER_DIST_BUCKET), MAX_MONSTER_DIST_BUCKET)
                monster_feats.append(
                    np.array([is_in_view, dir_x, dir_z, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # OrganState[] 物件状态列表（宝箱、buff）
        organs = frame_state.get("organs", [])
        organs_feat = self.organ_processor.get_feats(organs=organs, hero_pos=hero_pos)

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

        legal_action, move_mask = self.move_processor.mask_legal_action(
            legal_action=legal_action,
            map_info=map_info,
        )
        legal_action, danger_score = self.flash_processor.mask_legal_action(
            legal_action=legal_action,
            monster_feats=monster_feats,
        )
        terrain_feat = self.terrain_processor.get_feats(map_info=map_info, move_mask=move_mask)

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, danger_score], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                organs_feat,
                map_feat,
                terrain_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        monster_dist_reward = self.calc_monster_dist_reward(monster_feats=monster_feats)
        explore_reward = self.explore_processor.calc_reward(hero_pos=hero_pos)
        organ_reward = self.organ_processor.calc_reward(env_info=env_info, organs=organs, hero_pos=hero_pos)
        progress_reward = self.calc_progress_reward(env_info=env_info)
        flash_reward = self.flash_processor.calc_reward(last_action=last_action, danger_score=danger_score)
        move_reward = self.move_processor.calc_reward(last_action=last_action, move_mask=move_mask)
        terrain_reward = self.terrain_processor.calc_reward(
            hero_pos=hero_pos,
            map_info=map_info,
            move_mask=move_mask,
            danger_score=danger_score,
        )

        reward = [
            progress_reward
            + monster_dist_reward
            + explore_reward
            + organ_reward
            + flash_reward
            + move_reward
            + terrain_reward
        ]

        return feature, legal_action, reward
