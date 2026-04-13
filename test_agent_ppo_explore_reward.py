import unittest

from agent_ppo.feature.rewards.explore_processor import ExploreProcessor


class TestExploreRewardShaping(unittest.TestCase):
    def test_non_recent_revisit_gets_small_positive_adjustment(self):
        processor = ExploreProcessor()

        processor.visited_grid_counts[(0, 0)] = 1
        revisit_adjustment = processor.calc_revisit_adjustment(
            grid=(0, 0),
            explore_new_grid=0,
            visit_count_after=2,
        )

        self.assertGreater(revisit_adjustment, 0.0)
        self.assertAlmostEqual(revisit_adjustment, 0.02, places=6)

    def test_non_recent_revisit_reward_decays_with_visit_count(self):
        processor = ExploreProcessor()

        reward_second_visit = processor.calc_revisit_adjustment(
            grid=(0, 0),
            explore_new_grid=0,
            visit_count_after=2,
        )
        reward_fifth_visit = processor.calc_revisit_adjustment(
            grid=(0, 0),
            explore_new_grid=0,
            visit_count_after=5,
        )

        self.assertGreater(reward_second_visit, reward_fifth_visit)
        self.assertGreater(reward_fifth_visit, 0.0)

    def test_recent_revisit_adjustment_still_gets_penalized(self):
        processor = ExploreProcessor()
        processor.recent_grids.append((0, 0))

        revisit_adjustment = processor.calc_revisit_adjustment(
            grid=(0, 0),
            explore_new_grid=0,
            visit_count_after=2,
        )

        self.assertLess(revisit_adjustment, 0.0)


if __name__ == "__main__":
    unittest.main()
