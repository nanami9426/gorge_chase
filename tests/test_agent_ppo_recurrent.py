#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

import numpy as np
import torch

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.feature.rewards.flash_processor import FlashProcessor
from agent_ppo.feature.rewards.explore_processor import ExploreProcessor
from agent_ppo.feature.rewards.monster_processor import MonsterProcessor
from agent_ppo.feature.rewards.organ_processor import OrganProcessor
from agent_ppo.feature.rewards.phase_processor import CURRICULUM_LOOT_UNLOCK
from agent_ppo.feature.rewards.phase_processor import CURRICULUM_SURVIVAL_BOOTSTRAP
from agent_ppo.feature.rewards.phase_processor import PhaseProcessor
from agent_ppo.feature.rewards.terrain_processor import TerrainProcessor
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from agent_ppo.workflow.train_workflow import CURRICULUM_STAGE_CONFIG
from agent_ppo.workflow.train_workflow import CurriculumTracker
from agent_ppo.workflow.train_workflow import SURVIVAL_PROMOTION_STREAK
from agent_ppo.workflow.train_workflow import SURVIVAL_PROMOTION_WINDOW
from agent_ppo.workflow.train_workflow import build_episode_usr_conf


class _NoOpLogger:
    def info(self, *args, **kwargs):
        return None


class RecurrentPPOTests(unittest.TestCase):
    def test_model_shapes_match_temporal_observation(self):
        model = Model(device="cpu")
        obs = torch.randn(3, Config.DIM_OF_OBSERVATION)

        logits, value_heads, aux_pred = model(obs, inference=True)

        self.assertEqual(logits.shape, (3, Config.ACTION_NUM))
        self.assertEqual(value_heads.shape, (3, Config.VALUE_HEAD_NUM))
        self.assertEqual(aux_pred.shape, (3, Config.AUX_TARGET_NUM))

    def test_sample_process_computes_gae_targets(self):
        rollout = []
        for step_idx in range(4):
            rollout.append(
                SampleData(
                    obs=np.full(Config.DIM_OF_OBSERVATION, step_idx, dtype=np.float32),
                    legal_action=np.ones(Config.ACTION_NUM, dtype=np.float32),
                    act=np.array([step_idx % Config.ACTION_NUM], dtype=np.float32),
                    reward=np.array([1.0], dtype=np.float32),
                    done=np.array([1.0 if step_idx == 3 else 0.0], dtype=np.float32),
                    value=np.array([0.1 * step_idx], dtype=np.float32),
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value_head_reward=np.array([1.0, 0.5, 0.25, 0.25], dtype=np.float32),
                    value_head_sum=np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32),
                    value_heads=np.full(Config.VALUE_HEAD_NUM, 0.1 * step_idx, dtype=np.float32),
                    next_value_heads=np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32),
                    aux_target=np.array([0.1, 0.5, 0.3], dtype=np.float32),
                    prob=np.full(Config.ACTION_NUM, 1.0 / Config.ACTION_NUM, dtype=np.float32),
                    action_prior=np.zeros(Config.ACTION_NUM, dtype=np.float32),
                )
            )

        processed = sample_process(rollout)

        self.assertEqual(len(processed), 4)
        self.assertAlmostEqual(float(processed[-1].next_value[0]), 0.0)
        for sample in processed:
            self.assertTrue(np.isfinite(sample.advantage).all())
            self.assertTrue(np.isfinite(sample.reward_sum).all())
            self.assertTrue(np.isfinite(sample.value_head_sum).all())

    def test_algorithm_learn_runs_multi_epoch_updates(self):
        rollout = []
        for step_idx in range(16):
            reward = 1.0 if step_idx < 15 else -1.0
            rollout.append(
                SampleData(
                    obs=np.random.randn(Config.DIM_OF_OBSERVATION).astype(np.float32),
                    legal_action=np.ones(Config.ACTION_NUM, dtype=np.float32),
                    act=np.array([step_idx % Config.ACTION_NUM], dtype=np.float32),
                    reward=np.array([reward], dtype=np.float32),
                    done=np.array([1.0 if step_idx == 15 else 0.0], dtype=np.float32),
                    value=np.array([0.05 * step_idx], dtype=np.float32),
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value_head_reward=np.array([reward, 0.5 * reward, 0.25 * reward, 0.25 * reward], dtype=np.float32),
                    value_head_sum=np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32),
                    value_heads=np.full(Config.VALUE_HEAD_NUM, 0.05 * step_idx, dtype=np.float32),
                    next_value_heads=np.zeros(Config.VALUE_HEAD_NUM, dtype=np.float32),
                    aux_target=np.array([0.2, 0.6, 0.4], dtype=np.float32),
                    prob=np.full(Config.ACTION_NUM, 1.0 / Config.ACTION_NUM, dtype=np.float32),
                    action_prior=np.zeros(Config.ACTION_NUM, dtype=np.float32),
                )
            )

        samples = sample_process(rollout)
        for sample in samples:
            sample.obs = torch.tensor(sample.obs, dtype=torch.float32)
            sample.legal_action = torch.tensor(sample.legal_action, dtype=torch.float32)
            sample.act = torch.tensor(sample.act, dtype=torch.float32)
            sample.reward = torch.tensor(sample.reward, dtype=torch.float32)
            sample.done = torch.tensor(sample.done, dtype=torch.float32)
            sample.reward_sum = torch.tensor(sample.reward_sum, dtype=torch.float32)
            sample.value_head_reward = torch.tensor(sample.value_head_reward, dtype=torch.float32)
            sample.value_head_sum = torch.tensor(sample.value_head_sum, dtype=torch.float32)
            sample.value = torch.tensor(sample.value, dtype=torch.float32)
            sample.next_value = torch.tensor(sample.next_value, dtype=torch.float32)
            sample.value_heads = torch.tensor(sample.value_heads, dtype=torch.float32)
            sample.next_value_heads = torch.tensor(sample.next_value_heads, dtype=torch.float32)
            sample.advantage = torch.tensor(sample.advantage, dtype=torch.float32)
            sample.aux_target = torch.tensor(sample.aux_target, dtype=torch.float32)
            sample.prob = torch.tensor(sample.prob, dtype=torch.float32)
            sample.action_prior = torch.tensor(sample.action_prior, dtype=torch.float32)
        model = Model(device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        algorithm = Algorithm(model=model, optimizer=optimizer, device="cpu", logger=_NoOpLogger())

        algorithm.learn(samples)

        self.assertEqual(algorithm.train_step, 1)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_entropy_beta_increases_when_entropy_is_low(self):
        model = Model(device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        algorithm = Algorithm(model=model, optimizer=optimizer, device="cpu", logger=_NoOpLogger())
        initial_beta = algorithm.var_beta

        algorithm._update_entropy_beta(
            entropy_loss=torch.tensor(0.1),
            entropy_target=torch.tensor(1.0),
        )

        self.assertGreater(algorithm.var_beta, initial_beta)

    def test_action_prior_changes_masked_policy_distribution(self):
        model = Model(device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        algorithm = Algorithm(model=model, optimizer=optimizer, device="cpu", logger=_NoOpLogger())
        logits = torch.zeros(1, Config.ACTION_NUM)
        legal_action = torch.ones(1, Config.ACTION_NUM)
        action_prior = torch.zeros(1, Config.ACTION_NUM)
        action_prior[0, 8] = 1.0

        prob = algorithm._masked_softmax(
            logits=logits,
            legal_action=legal_action,
            action_prior=action_prior,
        )

        self.assertGreater(float(prob[0, 8]), float(prob[0, 0]))

    def test_dual_monster_context_increases_escape_urgency(self):
        monster_processor = MonsterProcessor()
        terrain_processor = TerrainProcessor()
        flash_processor = FlashProcessor()

        monster_feats = [
            np.array([1.0, 1.0, 0.0, 0.8, 0.25], dtype=np.float32),
            np.array([1.0, -1.0, 0.0, 0.7, 0.30], dtype=np.float32),
        ]
        terrain_stats = terrain_processor.extract_stats(
            map_info=np.ones((13, 13), dtype=np.int32).tolist(),
            move_mask=[1] * 8,
            monster_vec=monster_processor.get_nearest_monster_vector(monster_feats),
        )

        self.assertGreater(terrain_stats["escape_dir_scores"][2], terrain_stats["escape_dir_scores"][0])

        danger_score = flash_processor.calc_danger_score(
            monster_feats=monster_feats,
            terrain_stats=terrain_stats,
        )
        self.assertGreater(danger_score, 0.6)

    def test_flash_route_scores_prefer_wall_cut_escape(self):
        terrain_processor = TerrainProcessor()
        map_info = np.ones((13, 13), dtype=np.int32)
        center = 6
        map_info[center, center + 1] = 0
        map_info[center, center + 2] = 0
        move_mask = [0, 1, 1, 1, 1, 1, 1, 1]
        legal_action = [1] * Config.ACTION_NUM

        terrain_stats = terrain_processor.extract_stats(
            map_info=map_info.tolist(),
            move_mask=move_mask,
            monster_vec=(-1.0, 0.0),
            legal_action=legal_action,
        )

        self.assertEqual(terrain_stats["escape_dir_scores"][0], 0.0)
        self.assertGreater(terrain_stats["flash_dir_scores"][0], 0.60)
        self.assertGreater(terrain_stats["flash_dir_scores"][0], terrain_stats["flash_dir_scores"][4])

    def test_dead_end_has_lower_readiness_than_open_area(self):
        terrain_processor = TerrainProcessor()
        open_stats = terrain_processor.extract_stats(
            map_info=np.ones((13, 13), dtype=np.int32).tolist(),
            move_mask=[1] * 8,
            monster_vec=None,
            legal_action=[1] * Config.ACTION_NUM,
        )

        dead_end_map = np.zeros((13, 13), dtype=np.int32)
        center = 6
        dead_end_map[center, center] = 1
        dead_end_map[center, center + 1] = 1
        dead_end_stats = terrain_processor.extract_stats(
            map_info=dead_end_map.tolist(),
            move_mask=[1, 0, 0, 0, 0, 0, 0, 0],
            monster_vec=None,
            legal_action=[1] * Config.ACTION_NUM,
        )

        self.assertGreater(open_stats["readiness_score"], dead_end_stats["readiness_score"])
        self.assertGreater(dead_end_stats["dead_end_risk"], open_stats["dead_end_risk"])

    def test_explore_penalizes_small_loop_in_bad_terrain(self):
        explore_processor = ExploreProcessor()
        hero_pos = {"x": 10, "z": 10}
        terrain_stats = {"dead_end_risk": 0.9, "readiness_score": 0.1}

        reward_info = None
        for step_idx in range(8):
            reward_info = explore_processor.calc_reward(
                hero_pos=hero_pos,
                step_no=step_idx,
                danger_score=0.1,
                terrain_stats=terrain_stats,
            )

        self.assertLess(reward_info["no_progress_penalty"], 0.0)
        self.assertLess(reward_info["local_loop_penalty"], 0.0)
        self.assertGreater(reward_info["positioning_need"], 0.5)

    def test_speedup_critical_prune_keeps_best_escape_actions(self):
        preprocessor = Preprocessor()
        legal_action = [1] * Config.ACTION_NUM
        terrain_stats = {
            "escape_dir_scores": [0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.0, 0.3],
            "flash_dir_scores": [0.95, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, 0.6],
        }

        masked_action, pruned_count = preprocessor.prune_critical_escape_actions(
            legal_action=legal_action,
            terrain_stats=terrain_stats,
            danger_score=0.7,
            max_monster_speed=2,
        )

        self.assertGreater(pruned_count, 0)
        self.assertEqual(masked_action[0], 1)
        self.assertEqual(masked_action[8], 1)
        self.assertEqual(masked_action[6], 0)

    def test_action_prior_prefers_flash_and_buff_under_speedup_risk(self):
        preprocessor = Preprocessor()
        legal_action = [1] * Config.ACTION_NUM
        terrain_stats = {
            "escape_dir_scores": [0.1, 0.2, 0.3, 0.4, 0.95, 0.4, 0.3, 0.2],
            "flash_dir_scores": [0.2, 0.3, 0.4, 0.5, 0.96, 0.4, 0.3, 0.2],
            "dead_end_risk": 0.8,
            "readiness_score": 0.2,
            "route_diversity": 0.2,
            "best_flash_score": 0.96,
        }
        organs_feat = np.array(
            [
                1.0, 0.3, 1.0, 0.0,
                1.0, 0.2, 0.0, 1.0,
            ],
            dtype=np.float32,
        )

        action_prior = preprocessor.build_action_prior(
            legal_action=legal_action,
            terrain_stats=terrain_stats,
            organs_feat=organs_feat,
            hero={"buff_remaining_time": 0.0},
            danger_score=0.7,
            max_monster_speed=2,
        )

        self.assertGreater(action_prior[12], action_prior[0])
        self.assertGreater(action_prior[8 + 4], 0.5)

    def test_action_prior_keeps_treasure_target_under_moderate_danger(self):
        preprocessor = Preprocessor()
        legal_action = [1] * Config.ACTION_NUM
        terrain_stats = {
            "escape_dir_scores": [0.65, 0.55, 0.45, 0.35, 0.25, 0.30, 0.40, 0.55],
            "flash_dir_scores": [0.2] * 8,
            "dead_end_risk": 0.25,
            "readiness_score": 0.65,
            "route_diversity": 0.7,
            "best_flash_score": 0.20,
        }
        organs_feat = np.array(
            [
                1.0, 0.22, 1.0, 0.0,
                1.0, 0.30, 0.0, 1.0,
            ],
            dtype=np.float32,
        )
        treasure_priority_feat = np.array([1.0, 0.22, 1.0, 0.0, 0.72, 1.0, 0.5, 0.8], dtype=np.float32)

        action_prior = preprocessor.build_action_prior(
            legal_action=legal_action,
            terrain_stats=terrain_stats,
            organs_feat=organs_feat,
            hero={"buff_remaining_time": 20.0},
            danger_score=0.52,
            max_monster_speed=1,
            treasure_priority_feat=treasure_priority_feat,
        )

        self.assertGreater(action_prior[0], 0.65)
        self.assertGreater(action_prior[0], action_prior[2])

    def test_danger_flash_reward_when_it_reduces_pressure(self):
        flash_processor = FlashProcessor()
        flash_processor.last_danger_score = 0.85
        flash_processor.last_trap_risk = 0.60
        flash_processor.last_min_dist_norm = 0.20

        reward = flash_processor.calc_reward(
            last_action=8,
            danger_score=0.35,
            monster_feats=[np.array([1.0, -1.0, 0.0, 0.4, 0.55], dtype=np.float32)],
            terrain_stats={"trap_risk": 0.20},
        )

        self.assertGreater(reward, 0.0)

    def test_strategic_flash_not_penalized_when_dead_end_escape_exists(self):
        flash_processor = FlashProcessor()
        flash_processor.last_danger_score = 0.35
        flash_processor.last_trap_risk = 0.50
        flash_processor.last_dead_end_risk = 0.70
        flash_processor.last_best_flash_score = 0.80
        flash_processor.last_min_dist_norm = 0.25

        reward = flash_processor.calc_reward(
            last_action=8,
            danger_score=0.30,
            monster_feats=[np.array([1.0, -1.0, 0.0, 0.4, 0.40], dtype=np.float32)],
            terrain_stats={"trap_risk": 0.30, "dead_end_risk": 0.30, "best_flash_score": 0.70},
        )

        self.assertGreaterEqual(reward, 0.0)

    def test_curriculum_survival_stage_damps_treasure_reward(self):
        phase_processor = PhaseProcessor()
        raw_reward_breakdown = {
            "progress_reward": 0.2,
            "monster_dist_reward": 0.1,
            "explore_reward": 0.05,
            "treasure_reward": 2.0,
            "buff_reward": 1.5,
            "treasure_stall_penalty": -0.1,
            "terrain_reward": 0.3,
            "flash_reward": 0.15,
            "move_reward": 0.01,
        }

        _, curriculum_weights, combined_weights, weighted_reward_breakdown, _ = phase_processor.weight_reward_breakdown(
            raw_reward_breakdown=raw_reward_breakdown,
            phase_name="phase_0_loot",
            curriculum_stage=CURRICULUM_SURVIVAL_BOOTSTRAP,
            loot_reward_scale=0.55,
        )

        self.assertGreater(curriculum_weights["treasure_reward"], 0.0)
        self.assertGreater(weighted_reward_breakdown["treasure_reward"], 0.0)
        self.assertLess(weighted_reward_breakdown["treasure_reward"], raw_reward_breakdown["treasure_reward"])
        self.assertGreater(weighted_reward_breakdown["monster_dist_reward"], 0.0)
        self.assertGreater(combined_weights["terrain_reward"], 1.0)

    def test_curriculum_tracker_does_not_promote_on_600_step_failures(self):
        tracker = CurriculumTracker()

        promoted = False
        for _ in range(SURVIVAL_PROMOTION_WINDOW + SURVIVAL_PROMOTION_STREAK + 5):
            metrics, promoted = tracker.record_episode(
                episode_steps=600,
                survived=False,
                reached_phase_2=False,
                max_step=1000,
            )

        self.assertEqual(tracker.get_stage(), CURRICULUM_SURVIVAL_BOOTSTRAP)
        self.assertFalse(promoted)
        self.assertLess(metrics["avg_step_ratio"], 0.78)
        self.assertEqual(metrics["phase_2_rate"], 0.0)

    def test_curriculum_tracker_promotes_after_stable_lategame_bootstrap(self):
        tracker = CurriculumTracker()

        promoted = False
        episodes_until_promotion = SURVIVAL_PROMOTION_WINDOW + SURVIVAL_PROMOTION_STREAK - 1
        for idx in range(episodes_until_promotion):
            metrics, promoted = tracker.record_episode(
                episode_steps=850,
                survived=False,
                reached_phase_2=True,
                max_step=1000,
                phase_2_steps=150,
            )
            if idx < episodes_until_promotion - 1:
                self.assertFalse(promoted)

        self.assertEqual(tracker.get_stage(), CURRICULUM_LOOT_UNLOCK)
        self.assertTrue(promoted)
        self.assertEqual(metrics["promotion_reason"], "stable_lategame_bootstrap")
        self.assertGreaterEqual(metrics["avg_step_ratio"], 0.78)
        self.assertGreaterEqual(metrics["lategame_rate"], 0.50)
        self.assertGreaterEqual(metrics["phase_2_rate"], 0.50)

    def test_curriculum_tracker_stays_in_survival_when_metrics_miss(self):
        tracker = CurriculumTracker()

        for _ in range(500):
            tracker.record_episode(
                episode_steps=450,
                survived=False,
                reached_phase_2=False,
                max_step=1000,
            )

        self.assertEqual(tracker.get_stage(), CURRICULUM_SURVIVAL_BOOTSTRAP)

    def test_curriculum_tracker_loot_stage_episode_index_increments_once(self):
        tracker = CurriculumTracker()
        tracker.curriculum_stage = CURRICULUM_LOOT_UNLOCK
        tracker.stage_episode_idx = 0

        tracker.record_episode(
            episode_steps=500,
            survived=False,
            reached_phase_2=False,
            max_step=1000,
        )

        self.assertEqual(tracker.get_stage_episode_idx(), 1)

    def test_build_episode_usr_conf_preserves_configured_maps(self):
        base_usr_conf = {"env_conf": {"map": [7, 8], "map_random": False, "treasure_count": 10}}

        survival_conf = build_episode_usr_conf(
            base_usr_conf=base_usr_conf,
            curriculum_stage=CURRICULUM_SURVIVAL_BOOTSTRAP,
            stage_episode_idx=0,
        )

        self.assertEqual(survival_conf["env_conf"]["map"], [7, 8])
        self.assertFalse(survival_conf["env_conf"]["map_random"])
        self.assertEqual(
            survival_conf["env_conf"]["treasure_count"],
            CURRICULUM_STAGE_CONFIG[CURRICULUM_SURVIVAL_BOOTSTRAP]["env_overrides"]["treasure_count"],
        )

    def test_build_episode_usr_conf_preserves_loot_treasure_count(self):
        base_usr_conf = {"env_conf": {"treasure_count": 10, "map_random": False}}

        loot_conf = build_episode_usr_conf(
            base_usr_conf=base_usr_conf,
            curriculum_stage=CURRICULUM_LOOT_UNLOCK,
            stage_episode_idx=0,
        )

        self.assertEqual(loot_conf["env_conf"]["treasure_count"], 10)
        self.assertFalse(loot_conf["env_conf"]["map_random"])

    def test_organ_processor_damps_treasure_approach_under_extreme_danger(self):
        processor = OrganProcessor()
        env_info = {"treasures_collected": 0, "collected_buff": 0}
        hero = {"buff_remaining_time": 0.0}
        hero_pos = {"x": 0, "z": 0}
        organs_far = [
            {
                "status": 1,
                "sub_type": 1,
                "config_id": 1,
                "hero_relative_direction": 1,
                "pos": {"x": 20, "z": 0},
            }
        ]
        organs_near = [
            {
                "status": 1,
                "sub_type": 1,
                "config_id": 1,
                "hero_relative_direction": 1,
                "pos": {"x": 10, "z": 0},
            }
        ]

        processor.calc_reward(env_info, organs_far, hero_pos, hero=hero, terrain_stats={}, danger_score=0.0)
        safe_reward = processor.calc_reward(env_info, organs_near, hero_pos, hero=hero, terrain_stats={}, danger_score=0.2)

        processor.reset()
        processor.calc_reward(env_info, organs_far, hero_pos, hero=hero, terrain_stats={}, danger_score=0.0)
        moderate_reward = processor.calc_reward(
            env_info,
            organs_near,
            hero_pos,
            hero=hero,
            terrain_stats={},
            danger_score=0.8,
        )

        processor.reset()
        processor.calc_reward(env_info, organs_far, hero_pos, hero=hero, terrain_stats={}, danger_score=0.0)
        danger_reward = processor.calc_reward(env_info, organs_near, hero_pos, hero=hero, terrain_stats={}, danger_score=0.95)

        self.assertGreater(safe_reward["treasure_reward"], 0.0)
        self.assertGreater(moderate_reward["treasure_reward"], 0.0)
        self.assertLess(moderate_reward["treasure_reward"], safe_reward["treasure_reward"])
        self.assertEqual(danger_reward["treasure_reward"], 0.0)

    def test_safe_readiness_boosts_treasure_approach(self):
        processor = OrganProcessor()
        env_info = {"treasures_collected": 0, "collected_buff": 0}
        hero = {"buff_remaining_time": 0.0}
        hero_pos = {"x": 0, "z": 0}
        organs_far = [
            {
                "status": 1,
                "sub_type": 1,
                "config_id": 1,
                "hero_relative_direction": 1,
                "pos": {"x": 20, "z": 0},
            }
        ]
        organs_near = [
            {
                "status": 1,
                "sub_type": 1,
                "config_id": 1,
                "hero_relative_direction": 1,
                "pos": {"x": 10, "z": 0},
            }
        ]

        processor.calc_reward(env_info, organs_far, hero_pos, hero=hero, terrain_stats={}, danger_score=0.0)
        ordinary_reward = processor.calc_reward(
            env_info,
            organs_near,
            hero_pos,
            hero=hero,
            terrain_stats={"readiness_score": 0.2},
            danger_score=0.2,
        )

        processor.reset()
        processor.calc_reward(env_info, organs_far, hero_pos, hero=hero, terrain_stats={}, danger_score=0.0)
        ready_reward = processor.calc_reward(
            env_info,
            organs_near,
            hero_pos,
            hero=hero,
            terrain_stats={"readiness_score": 0.8},
            danger_score=0.2,
        )

        self.assertGreater(ready_reward["treasure_reward"], ordinary_reward["treasure_reward"])

    def test_buff_priority_rises_under_danger(self):
        processor = OrganProcessor()
        hero = {"buff_remaining_time": 0.0}
        treasure = {"dist_norm": 0.10}
        buff = {"dist_norm": 0.12}

        safe_weight = processor.calc_buff_priority_weight(
            hero=hero,
            nearest_treasure=treasure,
            nearest_buff=buff,
            danger_score=0.1,
            trap_risk=0.1,
        )
        danger_weight = processor.calc_buff_priority_weight(
            hero=hero,
            nearest_treasure=treasure,
            nearest_buff=buff,
            danger_score=0.8,
            trap_risk=0.6,
        )

        self.assertGreater(danger_weight, safe_weight)

    def test_monster_prediction_detects_approach_pressure(self):
        processor = MonsterProcessor()
        hero_pos = {"x": 0, "z": 0}
        first_frame = [
            {
                "monster_id": 1,
                "is_in_view": 1,
                "hero_relative_direction": 1,
                "speed": 2,
                "pos": {"x": 20, "z": 0},
            }
        ]
        second_frame = [
            {
                "monster_id": 1,
                "is_in_view": 1,
                "hero_relative_direction": 1,
                "speed": 2,
                "pos": {"x": 18, "z": 0},
            }
        ]

        processor.get_prediction_info(first_frame, hero_pos)
        prediction_feat, future_positions = processor.get_prediction_info(second_frame, hero_pos)

        self.assertEqual(prediction_feat.shape, (6,))
        self.assertGreater(float(prediction_feat[0]), 0.0)
        self.assertGreater(float(prediction_feat[1]), 0.0)
        self.assertGreater(float(prediction_feat[2]), float(prediction_feat[4]))
        self.assertGreaterEqual(float(prediction_feat[5]), float(prediction_feat[2]))
        self.assertIn(20, future_positions)

    def test_route_plan_penalizes_monster_covered_side(self):
        processor = TerrainProcessor()
        map_info = np.ones((21, 21), dtype=np.int32).tolist()
        hero_pos = {"x": 10, "z": 10}
        route_plan = processor.calc_route_plan_scores(
            map_info=map_info,
            move_mask=[1] * 8,
            legal_action=[1] * Config.ACTION_NUM,
            hero_pos=hero_pos,
            future_monster_positions={5: [{"x": 14, "z": 10}]},
        )

        self.assertGreater(route_plan["move_route_scores"][4], route_plan["move_route_scores"][0])
        self.assertGreater(route_plan["best_route_score"], 0.0)
        self.assertGreater(route_plan["safe_area_ratio"], 0.0)

    def test_treasure_priority_damps_under_high_danger(self):
        processor = OrganProcessor()
        hero_pos = {"x": 0, "z": 0}
        organs = [
            {
                "status": 1,
                "sub_type": 1,
                "config_id": 1,
                "hero_relative_direction": 1,
                "pos": {"x": 8, "z": 0},
            }
        ]
        terrain_stats = {"readiness_score": 0.8, "dead_end_risk": 0.1, "trap_risk": 0.1}

        safe_feat = processor.get_priority_feats(organs, hero_pos, terrain_stats=terrain_stats, danger_score=0.1)
        danger_feat = processor.get_priority_feats(organs, hero_pos, terrain_stats=terrain_stats, danger_score=0.9)
        extreme_feat = processor.get_priority_feats(organs, hero_pos, terrain_stats=terrain_stats, danger_score=0.95)

        self.assertEqual(safe_feat.shape, (8,))
        self.assertGreater(float(safe_feat[4]), float(danger_feat[4]))
        self.assertGreater(float(safe_feat[5]), float(extreme_feat[5]))

    def test_memory_features_point_to_known_safe_neighbor(self):
        processor = ExploreProcessor()
        hero_pos = {"x": 16, "z": 16}
        grid = processor.get_grid(hero_pos)
        processor.danger_grid_ema[(grid[0] + 1, grid[1])] = 0.05
        processor.danger_grid_ema[(grid[0] - 1, grid[1])] = 0.95

        memory_feat = processor.get_memory_feats(hero_pos)

        self.assertEqual(memory_feat.shape, (5,))
        self.assertGreater(float(memory_feat[1]), 0.9)
        self.assertEqual(float(memory_feat[2]), 1.0)
        self.assertEqual(float(memory_feat[3]), 0.0)


if __name__ == "__main__":
    unittest.main()
