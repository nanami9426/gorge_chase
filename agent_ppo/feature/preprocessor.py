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
from agent_ppo.feature.rewards import MonsterProcessor
from agent_ppo.feature.rewards import MoveProcessor
from agent_ppo.feature.rewards import TerrainProcessor

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
STEP_SCORE_REWARD_SCALE = 0.003

PHASE_REWARD_WEIGHTS = {
    "phase_0_loot": {
        "progress_reward": 1.00,
        "monster_dist_reward": 0.80,
        "explore_reward": 1.00,
        "treasure_reward": 1.00,
        "buff_reward": 0.85,
        "treasure_stall_penalty": 1.00,
        "terrain_reward": 0.90,
        "flash_reward": 0.90,
        "move_reward": 1.00,
    },
    "phase_1_double_monster": {
        "progress_reward": 1.10,
        "monster_dist_reward": 1.50,
        "explore_reward": 0.55,
        "treasure_reward": 0.65,
        "buff_reward": 0.90,
        "treasure_stall_penalty": 1.15,
        "terrain_reward": 1.30,
        "flash_reward": 1.15,
        "move_reward": 1.00,
    },
    "phase_2_speedup_survival": {
        "progress_reward": 1.20,
        "monster_dist_reward": 1.90,
        "explore_reward": 0.20,
        "treasure_reward": 0.35,
        "buff_reward": 0.75,
        "treasure_stall_penalty": 1.25,
        "terrain_reward": 1.55,
        "flash_reward": 1.30,
        "move_reward": 1.00,
    },
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def get_phase_weights(phase_name):
    """Get fixed reward weights for the current training phase.

    获取当前训练阶段对应的固定奖励权重。
    """
    return dict(PHASE_REWARD_WEIGHTS[str(phase_name)])


class Preprocessor:
    def __init__(self):
        self.organ_processor = OrganProcessor()
        self.explore_processor = ExploreProcessor()
        self.flash_processor = FlashProcessor()
        self.monster_processor = MonsterProcessor()
        self.move_processor = MoveProcessor()
        self.terrain_processor = TerrainProcessor()
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_step_score = 0.0
        self.organ_processor.reset()
        self.explore_processor.reset()
        self.flash_processor.reset()
        self.monster_processor.reset()
        self.move_processor.reset()
        self.terrain_processor.reset()

    def calc_progress_reward(self, env_info) -> float:
        step_score = float(env_info.get("step_score", 0.0))
        step_score_gain = max(0.0, step_score - self.last_step_score)
        self.last_step_score = step_score
        return STEP_SCORE_REWARD_SCALE * step_score_gain

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
        phase_id, phase_name, monster_count, max_monster_speed = self.monster_processor.get_phase_info(monsters)
        monster_feats = self.monster_processor.get_feats(monsters=monsters, hero_pos=hero_pos)
        phase_feat = self.monster_processor.get_phase_feat(
            phase_id=phase_id,
            monster_count=monster_count,
            max_monster_speed=max_monster_speed,
        )
        nearest_monster_vec = self.monster_processor.get_nearest_monster_vector(monster_feats)

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
        terrain_stats = self.terrain_processor.extract_stats(
            map_info=map_info,
            move_mask=move_mask,
            monster_vec=nearest_monster_vec,
        )
        terrain_feat = self.terrain_processor.get_feats(terrain_stats=terrain_stats)
        danger_score = self.flash_processor.calc_danger_score(
            monster_feats=monster_feats,
            wall_pressure=terrain_stats["wall_pressure"],
            corner_pressure=terrain_stats["corner_pressure"],
        )
        legal_action = self.flash_processor.mask_legal_action(
            legal_action=legal_action,
            danger_score=danger_score,
        )

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
                phase_feat,
            ]
        )

        # Step reward / 即时奖励
        monster_dist_reward = self.monster_processor.calc_reward(monster_feats=monster_feats)
        explore_reward = self.explore_processor.calc_reward(hero_pos=hero_pos)
        organ_reward = self.organ_processor.calc_reward(env_info=env_info, organs=organs, hero_pos=hero_pos)
        progress_reward = self.calc_progress_reward(env_info=env_info)
        flash_reward = self.flash_processor.calc_reward(last_action=last_action, danger_score=danger_score)
        move_reward = self.move_processor.calc_reward(last_action=last_action, move_mask=move_mask)
        terrain_reward = self.terrain_processor.calc_reward(
            hero_pos=hero_pos,
            terrain_stats=terrain_stats,
            last_action=last_action,
            danger_score=danger_score,
        )

        # 先记录原始分项奖励，再根据阶段做固定重加权
        raw_reward_breakdown = {
            "progress_reward": float(progress_reward),
            "monster_dist_reward": float(monster_dist_reward),
            "explore_reward": float(explore_reward),
            "treasure_reward": float(organ_reward["treasure_reward"]),
            "buff_reward": float(organ_reward["buff_reward"]),
            "treasure_stall_penalty": float(organ_reward["treasure_stall_penalty"]),
            "terrain_reward": float(terrain_reward),
            "flash_reward": float(flash_reward),
            "move_reward": float(move_reward),
        }
        phase_weights = get_phase_weights(phase_name)
        weighted_reward_breakdown = {
            key: float(raw_reward_breakdown[key] * phase_weights[key]) for key in phase_weights
        }
        total_reward = float(sum(weighted_reward_breakdown.values()))

        # remain_info 除了总奖励，还暴露阶段和奖励拆分，便于训练期诊断
        remain_info = {
            "reward": [total_reward],
            "phase_id": int(phase_id),
            "phase_name": phase_name,
            "monster_count": int(monster_count),
            "max_monster_speed": int(max_monster_speed),
            "reward_breakdown": {
                "raw": raw_reward_breakdown,
                "weights": phase_weights,
                "weighted": weighted_reward_breakdown,
                "total": total_reward,
            },
        }

        return feature, legal_action, remain_info
