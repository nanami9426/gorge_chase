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
from agent_ppo.feature.rewards import PhaseProcessor
from agent_ppo.feature.rewards import TerrainProcessor
from agent_ppo.feature.spatial_encoder import SpatialFeatureEncoder

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
STEP_SCORE_REWARD_SCALE = 0.003
RISK_PRUNE_WALL_PRESSURE_THRESHOLD = 0.55
RISK_PRUNE_CORNER_PRESSURE_THRESHOLD = 0.50
RISK_PRUNE_DELTA = 0.18
RISK_PRUNE_RATIO = 0.55
RISK_PRUNE_MIN_LEGAL_MOVES = 3
RISK_PRUNE_KEEP_TOPK = 2


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
        self.monster_processor = MonsterProcessor()
        self.move_processor = MoveProcessor()
        self.phase_processor = PhaseProcessor()
        self.terrain_processor = TerrainProcessor()
        self.spatial_encoder = SpatialFeatureEncoder()
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
        self.phase_processor.reset()
        self.terrain_processor.reset()

    def calc_progress_reward(self, env_info) -> float:
        step_score = float(env_info.get("step_score", 0.0))
        step_score_gain = max(0.0, step_score - self.last_step_score)
        self.last_step_score = step_score
        return STEP_SCORE_REWARD_SCALE * step_score_gain

    def prune_risky_moves(self, legal_action, terrain_stats):
        masked_action = list(legal_action)
        if terrain_stats["wall_pressure"] < RISK_PRUNE_WALL_PRESSURE_THRESHOLD and (
            terrain_stats["corner_pressure"] < RISK_PRUNE_CORNER_PRESSURE_THRESHOLD
        ):
            return masked_action, 0

        legal_move_indices = [idx for idx in range(8) if masked_action[idx] == 1]
        if len(legal_move_indices) < RISK_PRUNE_MIN_LEGAL_MOVES:
            return masked_action, 0

        escape_dir_scores = terrain_stats["escape_dir_scores"]
        scored_moves = [(idx, float(escape_dir_scores[idx])) for idx in legal_move_indices]
        best_score = max(score for _, score in scored_moves)
        keep_threshold = max(best_score - RISK_PRUNE_DELTA, RISK_PRUNE_RATIO * best_score)
        keep_indices = {idx for idx, score in scored_moves if score >= keep_threshold}
        top_indices = [
            idx
            for idx, _ in sorted(
                scored_moves,
                key=lambda item: (item[1], -item[0]),
                reverse=True,
            )[:RISK_PRUNE_KEEP_TOPK]
        ]
        keep_indices.update(top_indices)

        risk_pruned_moves = 0
        for idx in legal_move_indices:
            if idx not in keep_indices:
                masked_action[idx] = 0
                risk_pruned_moves += 1

        return masked_action, risk_pruned_moves

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
        phase_id, phase_name, monster_count, max_monster_speed = self.phase_processor.get_phase_info(monsters)
        monster_feats = self.monster_processor.get_feats(monsters=monsters, hero_pos=hero_pos)
        phase_feat = self.phase_processor.get_feats(
            phase_id=phase_id,
            monster_count=monster_count,
            max_monster_speed=max_monster_speed,
        )
        nearest_monster_vec = self.monster_processor.get_nearest_monster_vector(monster_feats)

        # OrganState[] 物件状态列表（宝箱、buff）
        organs = frame_state.get("organs", [])
        organs_feat = self.organ_processor.get_feats(organs=organs, hero_pos=hero_pos)
        explore_feat = self.explore_processor.get_feats(hero_pos=hero_pos)
        normalized_map_info, spatial_feat = self.spatial_encoder.encode(
            map_info=map_info,
            monsters=monsters,
            organs=organs,
            hero_pos=hero_pos,
        )
        map_info_for_logic = normalized_map_info if normalized_map_info is not None else map_info

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
            map_info=map_info_for_logic,
        )
        terrain_stats = self.terrain_processor.extract_stats(
            map_info=map_info_for_logic,
            move_mask=move_mask,
            monster_vec=nearest_monster_vec,
        )
        legal_action, risk_pruned_moves = self.prune_risky_moves(
            legal_action=legal_action,
            terrain_stats=terrain_stats,
        )
        terrain_feat = self.terrain_processor.get_feats(terrain_stats=terrain_stats)
        danger_score = self.flash_processor.calc_danger_score(
            monster_feats=monster_feats,
            terrain_stats=terrain_stats,
        )
        legal_action = self.flash_processor.mask_legal_action(
            legal_action=legal_action,
            danger_score=danger_score,
        )

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, danger_score], dtype=np.float32)

        # Dense features stay in front, spatial branch is flattened after them.
        dense_feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                organs_feat,
                explore_feat,
                terrain_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                phase_feat,
            ]
        )
        feature = np.concatenate([dense_feature, spatial_feat.reshape(-1)])

        # Step reward / 即时奖励
        monster_dist_reward = self.monster_processor.calc_reward(monster_feats=monster_feats)
        explore_reward_info = self.explore_processor.calc_reward(
            hero_pos=hero_pos,
            step_no=self.step_no,
            danger_score=danger_score,
        )
        organ_reward = self.organ_processor.calc_reward(
            env_info=env_info,
            organs=organs,
            hero_pos=hero_pos,
            hero=hero,
            terrain_stats=terrain_stats,
            danger_score=danger_score,
        )
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
            "explore_reward": float(explore_reward_info["reward"]),
            "treasure_reward": float(organ_reward["treasure_reward"]),
            "buff_reward": float(organ_reward["buff_reward"]),
            "treasure_stall_penalty": float(organ_reward["treasure_stall_penalty"]),
            "terrain_reward": float(terrain_reward),
            "flash_reward": float(flash_reward),
            "move_reward": float(move_reward),
        }
        phase_weights, weighted_reward_breakdown, total_reward = self.phase_processor.weight_reward_breakdown(
            raw_reward_breakdown=raw_reward_breakdown,
            phase_name=phase_name,
        )

        # remain_info 除了总奖励，还暴露阶段和奖励拆分，便于训练期诊断
        remain_info = {
            "reward": [total_reward],
            "phase_id": int(phase_id),
            "phase_name": phase_name,
            "monster_count": int(monster_count),
            "max_monster_speed": int(max_monster_speed),
            "danger_score": float(danger_score),
            "trap_risk": float(terrain_stats["trap_risk"]),
            "wall_pressure": float(terrain_stats["wall_pressure"]),
            "corner_pressure": float(terrain_stats["corner_pressure"]),
            "risk_pruned_moves": int(risk_pruned_moves),
            "explore_new_grid": int(explore_reward_info["explore_new_grid"]),
            "frontier_bonus": float(explore_reward_info["frontier_bonus"]),
            "explore_streak_bonus": float(explore_reward_info["explore_streak_bonus"]),
            "buff_priority_weight": float(organ_reward["buff_priority_weight"]),
            "buffs_collected": int(organ_reward["buffs_collected"]),
            "reward_breakdown": {
                "raw": raw_reward_breakdown,
                "weights": phase_weights,
                "weighted": weighted_reward_breakdown,
                "total": total_reward,
            },
        }

        return feature, legal_action, remain_info
