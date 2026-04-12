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
STEP_SCORE_REWARD_SCALE = 0.003
SQRT_HALF = float(np.sqrt(0.5))
MAX_MONSTER_COUNT = 2.0
PHASE_LOOT = 0
PHASE_DOUBLE_MONSTER = 1
PHASE_SPEEDUP_SURVIVAL = 2
PHASE_NAME_BY_ID = {
    PHASE_LOOT: "phase_0_loot",
    PHASE_DOUBLE_MONSTER: "phase_1_double_monster",
    PHASE_SPEEDUP_SURVIVAL: "phase_2_speedup_survival",
}
REWARD_KEYS = (
    "progress_reward",
    "monster_dist_reward",
    "explore_reward",
    "treasure_reward",
    "buff_reward",
    "treasure_stall_penalty",
    "terrain_reward",
    "flash_reward",
    "move_reward",
)
PHASE_REWARD_WEIGHTS = {
    PHASE_LOOT: {
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
    PHASE_DOUBLE_MONSTER: {
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
    PHASE_SPEEDUP_SURVIVAL: {
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

    def get_nearest_monster_vector(self, monster_feats):
        if not monster_feats:
            return None

        nearest_feat = min(monster_feats, key=lambda feat: float(feat[4]))
        dir_x = float(nearest_feat[1])
        dir_z = float(nearest_feat[2])
        norm = float(np.sqrt(dir_x * dir_x + dir_z * dir_z))
        if norm <= 1e-6:
            return None
        return (dir_x / norm, dir_z / norm)

    def calc_monster_dist_reward(self, monster_feats) -> float:
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            cur_min_dist_norm = min(cur_min_dist_norm, float(m_feat[4]))

        # 远离最近的怪物，远离->正，接近->负
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)
        self.last_min_monster_dist_norm = cur_min_dist_norm
        return dist_shaping

    def get_phase_info(self, monsters):
        monster_count = min(len(monsters), int(MAX_MONSTER_COUNT))
        max_monster_speed = max((int(m.get("speed", 1)) for m in monsters), default=1)

        if max_monster_speed > 1:
            phase_id = PHASE_SPEEDUP_SURVIVAL
        elif monster_count >= 2:
            phase_id = PHASE_DOUBLE_MONSTER
        else:
            phase_id = PHASE_LOOT

        return phase_id, PHASE_NAME_BY_ID[phase_id], monster_count, max_monster_speed

    def build_phase_feat(self, phase_id, monster_count, max_monster_speed):
        phase_onehot = np.zeros(3, dtype=np.float32)
        phase_onehot[int(phase_id)] = 1.0
        return np.concatenate(
            [
                phase_onehot,
                np.array(
                    [
                        _norm(monster_count, MAX_MONSTER_COUNT, 1.0),
                        _norm(max_monster_speed, MAX_MONSTER_SPEED, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def get_phase_weights(self, phase_id):
        return dict(PHASE_REWARD_WEIGHTS[int(phase_id)])

    def feature_process(self, env_obs, last_action):
        # last_action：0~7：移动 8~15：闪现
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)
        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        monsters = frame_state.get("monsters", [])
        phase_id, phase_name, monster_count, max_monster_speed = self.get_phase_info(monsters)
        phase_feat = self.build_phase_feat(phase_id, monster_count, max_monster_speed)

        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                dir_x, dir_z = self.direction_to_vector(m.get("hero_relative_direction", 0))
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                m_pos = m.get("pos", {})
                if is_in_view and isinstance(m_pos, dict) and "x" in m_pos and "z" in m_pos:
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAX_MAP_DISTANCE)
                else:
                    dist_norm = _norm(m.get("hero_l2_distance", MAX_MONSTER_DIST_BUCKET), MAX_MONSTER_DIST_BUCKET)
                monster_feats.append(
                    np.array([is_in_view, dir_x, dir_z, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        nearest_monster_vec = self.get_nearest_monster_vector(monster_feats)

        organs = frame_state.get("organs", [])
        organs_feat = self.organ_processor.get_feats(organs=organs, hero_pos=hero_pos)

        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

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

        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, danger_score], dtype=np.float32)

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

        monster_dist_reward = self.calc_monster_dist_reward(monster_feats=monster_feats)
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
        phase_weights = self.get_phase_weights(phase_id)
        weighted_reward_breakdown = {
            key: float(raw_reward_breakdown[key] * phase_weights[key]) for key in REWARD_KEYS
        }
        total_reward = float(sum(weighted_reward_breakdown.values()))
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
