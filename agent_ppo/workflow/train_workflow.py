#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

from copy import deepcopy
import os
import time
from collections import defaultdict, deque

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.feature.rewards.phase_processor import CURRICULUM_FULL
from agent_ppo.feature.rewards.phase_processor import CURRICULUM_LOOT_UNLOCK
from agent_ppo.feature.rewards.phase_processor import CURRICULUM_SURVIVAL_BOOTSTRAP
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

MODEL_SAVE_INTERVAL = 60 * 10
SURVIVAL_PROMOTION_WINDOW = 10
SURVIVAL_PROMOTION_STREAK = 3
FULL_PROMOTION_WINDOW = 10
FULL_PROMOTION_STREAK = 3
PHASE_LOG_ORDER = [
    "phase_0_loot",
    "phase_1_double_monster",
    "phase_2_speedup_survival",
]

CURRICULUM_STAGE_CONFIG = {
    CURRICULUM_SURVIVAL_BOOTSTRAP: {
        "env_overrides": {
            "treasure_count": 6,
            "buff_count": 2,
            "monster_interval": 500,
            "monster_speedup": 700,
            "max_step": 2000,
        },
        "loot_reward_scale": 0.70,
    },
    CURRICULUM_LOOT_UNLOCK: {
        "env_overrides": {
            "buff_count": 2,
            "monster_interval": 340,
            "monster_speedup": 560,
            "max_step": 2000,
        },
        "loot_reward_scale": 1.25,
    },
    CURRICULUM_FULL: {
        "env_overrides": {
            "buff_count": 2,
            "monster_interval": 300,
            "monster_speedup": 500,
            "max_step": 2000,
        },
        "loot_reward_scale": 1.15,
    },
}

SURVIVAL_REWARD_KEYS = (
    "progress_reward",
    "monster_dist_reward",
    "terrain_reward",
    "flash_reward",
)
LOOT_REWARD_KEYS = (
    "treasure_reward",
    "buff_reward",
    "treasure_stall_penalty",
)
EXPLORE_REWARD_KEYS = (
    "explore_reward",
    "move_reward",
)


def build_value_head_reward(remain_info, reward):
    weighted = remain_info.get("reward_breakdown", {}).get("weighted", {})
    value_head_reward = np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32)
    value_head_reward[Config.VALUE_HEAD_TOTAL] = float(reward[0])
    value_head_reward[Config.VALUE_HEAD_SURVIVAL] = sum(float(weighted.get(key, 0.0)) for key in SURVIVAL_REWARD_KEYS)
    value_head_reward[Config.VALUE_HEAD_LOOT] = sum(float(weighted.get(key, 0.0)) for key in LOOT_REWARD_KEYS)
    value_head_reward[Config.VALUE_HEAD_EXPLORE] = sum(float(weighted.get(key, 0.0)) for key in EXPLORE_REWARD_KEYS)
    return value_head_reward


def build_aux_target(remain_info):
    return np.array(
        [
            float(np.clip(remain_info.get("danger_score", 0.0), 0.0, 1.0)),
            float(np.clip(remain_info.get("best_route_score", 0.0), 0.0, 1.0)),
            float(np.clip(remain_info.get("treasure_priority", 0.0), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


class CurriculumTracker:
    """Promote difficulty only after the policy can reliably reach midgame."""

    def __init__(self):
        self.curriculum_stage = CURRICULUM_SURVIVAL_BOOTSTRAP
        self.stage_episode_idx = 0
        self.recent_episodes = deque(maxlen=max(SURVIVAL_PROMOTION_WINDOW, FULL_PROMOTION_WINDOW))
        self.promotion_streak = 0

    def get_stage(self):
        return self.curriculum_stage

    def get_stage_episode_idx(self):
        return self.stage_episode_idx

    def _build_metrics(self):
        if not self.recent_episodes:
            return {
                "avg_step_ratio": 0.0,
                "midgame_rate": 0.0,
                "lategame_rate": 0.0,
                "survival_rate": 0.0,
                "phase_2_rate": 0.0,
                "post_speedup_rate": 0.0,
                "avg_phase_2_steps": 0.0,
                "window_size": 0,
                "promotion_streak": self.promotion_streak,
            }

        window = list(self.recent_episodes)
        step_ratios = [episode["step_ratio"] for episode in window]
        return {
            "avg_step_ratio": float(np.mean(step_ratios)),
            "midgame_rate": float(np.mean([ratio >= 0.55 for ratio in step_ratios])),
            "lategame_rate": float(np.mean([ratio >= 0.80 for ratio in step_ratios])),
            "survival_rate": float(np.mean([episode["survived"] for episode in window])),
            "phase_2_rate": float(np.mean([episode["reached_phase_2"] for episode in window])),
            "post_speedup_rate": float(np.mean([episode["phase_2_steps"] >= 100 for episode in window])),
            "avg_phase_2_steps": float(np.mean([episode["phase_2_steps"] for episode in window])),
            "window_size": len(window),
            "promotion_streak": self.promotion_streak,
        }

    def _promote(self, next_stage):
        self.curriculum_stage = next_stage
        self.stage_episode_idx = 0
        self.recent_episodes.clear()
        self.promotion_streak = 0

    def record_episode(self, episode_steps, survived, reached_phase_2, max_step, phase_2_steps=0):
        max_step = max(int(max_step), 1)
        episode_steps = max(int(episode_steps), 0)
        phase_2_steps = max(int(phase_2_steps), 0)
        step_ratio = float(np.clip(float(episode_steps) / float(max_step), 0.0, 1.0))

        self.stage_episode_idx += 1
        self.recent_episodes.append(
            {
                "step_ratio": step_ratio,
                "survived": bool(survived),
                "reached_phase_2": bool(reached_phase_2),
                "phase_2_steps": phase_2_steps,
            }
        )

        promoted = False
        metrics = self._build_metrics()
        if self.curriculum_stage == CURRICULUM_SURVIVAL_BOOTSTRAP:
            ready = (
                metrics["window_size"] >= SURVIVAL_PROMOTION_WINDOW
                and metrics["avg_step_ratio"] >= 0.78
                and metrics["lategame_rate"] >= 0.50
                and metrics["phase_2_rate"] >= 0.50
                and (metrics["post_speedup_rate"] >= 0.30 or metrics["survival_rate"] >= 0.10)
            )
            self.promotion_streak = self.promotion_streak + 1 if ready else 0
            metrics["promotion_streak"] = self.promotion_streak
            if self.promotion_streak >= SURVIVAL_PROMOTION_STREAK:
                metrics["promotion_reason"] = "stable_lategame_bootstrap"
                self._promote(CURRICULUM_LOOT_UNLOCK)
                promoted = True
        elif self.curriculum_stage == CURRICULUM_LOOT_UNLOCK:
            ready = (
                metrics["window_size"] >= FULL_PROMOTION_WINDOW
                and metrics["avg_step_ratio"] >= 0.78
                and metrics["phase_2_rate"] >= 0.50
                and (metrics["post_speedup_rate"] >= 0.40 or metrics["survival_rate"] >= 0.20)
            )
            self.promotion_streak = self.promotion_streak + 1 if ready else 0
            metrics["promotion_streak"] = self.promotion_streak
            if self.promotion_streak >= FULL_PROMOTION_STREAK:
                metrics["promotion_reason"] = "stable_lategame"
                self._promote(CURRICULUM_FULL)
                promoted = True

        metrics["curriculum_stage"] = self.curriculum_stage
        metrics["stage_episode_idx"] = self.stage_episode_idx
        return metrics, promoted


def build_episode_usr_conf(
    base_usr_conf,
    curriculum_stage=CURRICULUM_SURVIVAL_BOOTSTRAP,
    stage_episode_idx=0,
):
    episode_usr_conf = deepcopy(base_usr_conf)
    env_conf = episode_usr_conf.setdefault("env_conf", {})
    stage_conf = CURRICULUM_STAGE_CONFIG.get(
        curriculum_stage,
        CURRICULUM_STAGE_CONFIG[CURRICULUM_SURVIVAL_BOOTSTRAP],
    )
    env_conf.update(deepcopy(stage_conf.get("env_overrides", {})))
    return episode_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= MODEL_SAVE_INTERVAL:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.curriculum_tracker = CurriculumTracker()

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            curriculum_stage = self.curriculum_tracker.get_stage()
            stage_episode_idx = self.curriculum_tracker.get_stage_episode_idx()
            episode_usr_conf = build_episode_usr_conf(
                base_usr_conf=self.usr_conf,
                curriculum_stage=curriculum_stage,
                stage_episode_idx=stage_episode_idx,
            )
            stage_conf = CURRICULUM_STAGE_CONFIG.get(
                curriculum_stage,
                CURRICULUM_STAGE_CONFIG[CURRICULUM_SURVIVAL_BOOTSTRAP],
            )
            loot_reward_scale = stage_conf.get("loot_reward_scale", 1.0)
            # Reset env / 重置环境
            env_obs = self.env.reset(episode_usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(
                env_obs,
                curriculum_stage=curriculum_stage,
                loot_reward_scale=loot_reward_scale,
            )
            remain_info["curriculum_stage"] = curriculum_stage

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            terminal_phase = "unknown"
            phase_step_counts = defaultdict(int)
            reward_breakdown_sum = defaultdict(float)
            risk_metric_sum = defaultdict(float)
            behavior_metric_sum = defaultdict(float)

            self.logger.info(
                f"Episode {self.episode_cnt} start curriculum_stage:{curriculum_stage} "
                f"stage_episode_idx:{stage_episode_idx}"
            )

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                # env_reward 是在当前状态下执行动作 action 所获得的分数。
                # 注意：得分用于衡量模型在环境中的表现，也作为衡量强化学习训练产出模型的评价指标，
                # 与强化学习里的奖励reward要区别开。
                env_reward, env_obs = self.env.step(act)
            
                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated
                bootstrap_value = np.zeros(1, dtype=np.float32)
                bootstrap_value_heads = np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32)

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(
                    env_obs,
                    curriculum_stage=curriculum_stage,
                    loot_reward_scale=loot_reward_scale,
                )
                _remain_info["curriculum_stage"] = curriculum_stage
                if truncated and not terminated:
                    bootstrap_value, bootstrap_value_heads = self.agent.value_process(_obs_data)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                value_head_reward = build_value_head_reward(_remain_info, reward)
                aux_target = build_aux_target(_remain_info)
                total_reward += float(reward[0])
                terminal_phase = _remain_info.get("phase_name", terminal_phase)
                phase_step_counts[terminal_phase] += 1
                weighted_breakdown = _remain_info.get("reward_breakdown", {}).get("weighted", {})
                for key, value in weighted_breakdown.items():
                    reward_breakdown_sum[key] += float(value)
                for key in (
                    "danger_score",
                    "trap_risk",
                    "readiness_score",
                    "dead_end_risk",
                    "route_diversity",
                    "wall_pressure",
                    "corner_pressure",
                    "risk_pruned_moves",
                    "critical_pruned_actions",
                    "max_flash_dir_score",
                    "action_prior_max",
                    "action_prior_sum",
                    "best_route_score",
                    "safe_area_ratio",
                    "treasure_priority",
                ):
                    risk_metric_sum[key] += float(_remain_info.get(key, 0.0))
                for key in (
                    "explore_new_grid",
                    "frontier_bonus",
                    "explore_streak_bonus",
                    "no_progress_penalty",
                    "local_loop_penalty",
                    "window_loop_penalty",
                    "positioning_need",
                    "buff_priority_weight",
                    "available_treasure_count",
                    "available_buff_count",
                    "known_treasure_count",
                    "known_buff_count",
                    "first_seen_treasure_count",
                    "treasure_bfs_dist_norm",
                    "buff_bfs_dist_norm",
                    "treasure_bfs_progress",
                    "buff_bfs_progress",
                    "nearest_buff_dist_norm",
                ):
                    behavior_metric_sum[key] += float(_remain_info.get(key, 0.0))

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                terminal_survival_reward = 0.0
                terminal_loot_reward = 0.0
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    max_step = int(
                        env_info.get(
                            "max_step",
                            episode_usr_conf.get("env_conf", {}).get("max_step", 1000),
                        )
                    )
                    total_score = env_info.get("total_score", 0)
                    treasures_collected = env_info.get("treasures_collected", 0)
                    collected_buff = env_info.get("collected_buff", 0)
                    step_ratio = float(np.clip(float(step) / float(max(max_step, 1)), 0.0, 1.0))

                    if terminated:
                        early_death_extra = 2.0 * max(0.0, 0.60 - step_ratio) / 0.60
                        terminal_survival_reward = -(4.0 + early_death_extra)
                        result_str = "FAIL"
                    else:
                        terminal_survival_reward = 3.0
                        result_str = "TRUNCATED" if truncated else "WIN"
                    terminal_loot_reward = 0.45 * float(treasures_collected) + 0.10 * float(collected_buff)
                    final_reward[0] = terminal_survival_reward + terminal_loot_reward

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"treasures:{treasures_collected} buffs_collected:{collected_buff} "
                        f"total_reward:{total_reward:.3f} final_reward:{final_reward[0]:.2f} "
                        f"terminal_survival_reward:{terminal_survival_reward:.2f} "
                        f"terminal_loot_reward:{terminal_loot_reward:.2f} "
                        f"terminal_phase:{terminal_phase} "
                        f"curriculum_stage:{curriculum_stage}"
                    )
                    phase_step_counts_log = {
                        phase_name: phase_step_counts.get(phase_name, 0) for phase_name in PHASE_LOG_ORDER
                    }
                    reward_breakdown_mean = {
                        key: round(value / max(step, 1), 4)
                        for key, value in sorted(reward_breakdown_sum.items())
                    }
                    risk_metric_mean = {
                        key: round(value / max(step, 1), 4)
                        for key, value in sorted(risk_metric_sum.items())
                    }
                    explore_new_grids = int(round(behavior_metric_sum.get("explore_new_grid", 0.0)))
                    frontier_bonus_mean = round(
                        behavior_metric_sum.get("frontier_bonus", 0.0) / max(step, 1),
                        4,
                    )
                    explore_streak_mean = round(
                        behavior_metric_sum.get("explore_streak_bonus", 0.0) / max(step, 1),
                        4,
                    )
                    no_progress_penalty_mean = round(
                        behavior_metric_sum.get("no_progress_penalty", 0.0) / max(step, 1),
                        4,
                    )
                    local_loop_penalty_mean = round(
                        behavior_metric_sum.get("local_loop_penalty", 0.0) / max(step, 1),
                        4,
                    )
                    window_loop_penalty_mean = round(
                        behavior_metric_sum.get("window_loop_penalty", 0.0) / max(step, 1),
                        4,
                    )
                    positioning_need_mean = round(
                        behavior_metric_sum.get("positioning_need", 0.0) / max(step, 1),
                        4,
                    )
                    buff_priority_weight_mean = round(
                        behavior_metric_sum.get("buff_priority_weight", 0.0) / max(step, 1),
                        4,
                    )
                    available_buff_count_mean = round(
                        behavior_metric_sum.get("available_buff_count", 0.0) / max(step, 1),
                        4,
                    )
                    nearest_buff_dist_mean = round(
                        behavior_metric_sum.get("nearest_buff_dist_norm", 0.0) / max(step, 1),
                        4,
                    )
                    known_treasure_count_mean = round(
                        behavior_metric_sum.get("known_treasure_count", 0.0) / max(step, 1),
                        4,
                    )
                    treasure_bfs_dist_mean = round(
                        behavior_metric_sum.get("treasure_bfs_dist_norm", 0.0) / max(step, 1),
                        4,
                    )
                    buff_bfs_dist_mean = round(
                        behavior_metric_sum.get("buff_bfs_dist_norm", 0.0) / max(step, 1),
                        4,
                    )
                    treasure_bfs_progress_mean = round(
                        behavior_metric_sum.get("treasure_bfs_progress", 0.0) / max(step, 1),
                        4,
                    )
                    buff_bfs_progress_mean = round(
                        behavior_metric_sum.get("buff_bfs_progress", 0.0) / max(step, 1),
                        4,
                    )
                    first_seen_treasure_count = int(
                        round(behavior_metric_sum.get("first_seen_treasure_count", 0.0))
                    )
                    self.logger.info(
                        f"[PHASE] episode:{self.episode_cnt} terminal_phase:{terminal_phase} "
                        f"phase_step_counts:{phase_step_counts_log} "
                        f"reward_breakdown_mean:{reward_breakdown_mean} "
                        f"risk_metric_mean:{risk_metric_mean} "
                        f"explore_new_grids:{explore_new_grids} "
                        f"frontier_bonus_mean:{frontier_bonus_mean} "
                        f"explore_streak_mean:{explore_streak_mean} "
                        f"no_progress_penalty_mean:{no_progress_penalty_mean} "
                        f"local_loop_penalty_mean:{local_loop_penalty_mean} "
                        f"window_loop_penalty_mean:{window_loop_penalty_mean} "
                        f"positioning_need_mean:{positioning_need_mean} "
                        f"buff_priority_weight_mean:{buff_priority_weight_mean} "
                        f"available_buff_count_mean:{available_buff_count_mean} "
                        f"nearest_buff_dist_mean:{nearest_buff_dist_mean} "
                        f"known_treasure_count_mean:{known_treasure_count_mean} "
                        f"treasure_bfs_dist_mean:{treasure_bfs_dist_mean} "
                        f"buff_bfs_dist_mean:{buff_bfs_dist_mean} "
                        f"treasure_bfs_progress_mean:{treasure_bfs_progress_mean} "
                        f"buff_bfs_progress_mean:{buff_bfs_progress_mean} "
                        f"first_seen_treasure_count:{first_seen_treasure_count} "
                        f"curriculum_stage:{curriculum_stage}"
                    )
                    curriculum_metrics, promoted = self.curriculum_tracker.record_episode(
                        episode_steps=step,
                        survived=not terminated,
                        reached_phase_2=phase_step_counts.get("phase_2_speedup_survival", 0) > 0,
                        max_step=max_step,
                        phase_2_steps=phase_step_counts.get("phase_2_speedup_survival", 0),
                    )
                    if promoted:
                        self.logger.info(
                            f"[CURRICULUM] promoted_to:{self.curriculum_tracker.get_stage()} "
                            f"reason:{curriculum_metrics.get('promotion_reason')} "
                            f"metrics:{curriculum_metrics}"
                        )
                    else:
                        self.logger.info(
                            f"[CURRICULUM] stage:{self.curriculum_tracker.get_stage()} "
                            f"metrics:{curriculum_metrics}"
                        )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(terminated)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value_head_reward=value_head_reward,
                    value_head_sum=np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=bootstrap_value,
                    value_heads=np.array(act_data.value_heads, dtype=np.float32).flatten()[: Config.VALUE_HEAD_NUM],
                    next_value_heads=bootstrap_value_heads,
                    advantage=np.zeros(1, dtype=np.float32),
                    aux_target=aux_target,
                    prob=np.array(act_data.prob, dtype=np.float32),
                    action_prior=np.array(obs_data.action_prior, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward
                        collector[-1].value_head_reward[Config.VALUE_HEAD_TOTAL] += float(final_reward[0])
                        collector[-1].value_head_reward[Config.VALUE_HEAD_SURVIVAL] += float(terminal_survival_reward)
                        collector[-1].value_head_reward[Config.VALUE_HEAD_LOOT] += float(terminal_loot_reward)

                    # Monitor report / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "treasures_collected": treasures_collected,
                            "buffs_collected": collected_buff,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
