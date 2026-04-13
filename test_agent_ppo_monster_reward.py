import unittest

import numpy as np

from agent_ppo.feature.rewards.monster_processor import MonsterProcessor


def build_monster_feat(dist_norm):
    return np.array([1.0, 1.0, 0.0, 0.0, float(dist_norm)], dtype=np.float32)


class TestMonsterRewardShaping(unittest.TestCase):
    def test_moving_away_gives_smaller_positive_reward(self):
        processor = MonsterProcessor()
        processor.last_min_monster_dist_norm = 0.40

        reward = processor.calc_reward([build_monster_feat(0.50), build_monster_feat(0.80)])

        self.assertAlmostEqual(reward, 0.004, places=6)

    def test_moving_closer_gets_stronger_negative_penalty(self):
        processor = MonsterProcessor()
        processor.last_min_monster_dist_norm = 0.40

        reward = processor.calc_reward([build_monster_feat(0.30), build_monster_feat(0.80)])

        self.assertAlmostEqual(reward, -0.014, places=6)
        self.assertLess(abs(reward), 0.02)


if __name__ == "__main__":
    unittest.main()
