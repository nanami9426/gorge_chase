import unittest

import numpy as np
import torch

from agent_ppo.agent import Agent
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData
from agent_ppo.model.model import Model


class DummyLogger:
    def info(self, message):
        return None

    def warning(self, message):
        return None

    def error(self, message):
        return None


def build_env_obs(hero_x=32, hero_z=64, step_no=10, map_size=13):
    map_info = [[1 for _ in range(map_size)] for _ in range(map_size)]
    center = map_size // 2
    map_info[center - 1][center + 2] = 0
    map_info[center + 2][center - 1] = 0

    return {
        "observation": {
            "frame_state": {
                "heroes": {
                    "pos": {"x": hero_x, "z": hero_z},
                    "flash_cooldown": 12,
                    "buff_remaining_time": 5,
                },
                "monsters": [
                    {
                        "is_in_view": 1,
                        "hero_relative_direction": 2,
                        "speed": 2,
                        "hero_l2_distance": 2,
                        "pos": {"x": hero_x + 2, "z": hero_z + 1},
                    }
                ],
                "organs": [
                    {
                        "status": 1,
                        "sub_type": 1,
                        "config_id": 101,
                        "hero_relative_direction": 4,
                        "hero_l2_distance": 2,
                        "pos": {"x": hero_x - 1, "z": hero_z + 2},
                    }
                ],
            },
            "env_info": {
                "max_step": 200,
                "step_score": 0.0,
                "treasures_collected": 0,
                "collected_buff": 0,
                "total_score": 0.0,
            },
            "map_info": map_info,
            "legal_action": [True] * Config.ACTION_NUM,
            "step_no": step_no,
        },
        "terminated": False,
        "truncated": False,
    }


class TestAgentPpoTemporal(unittest.TestCase):
    def test_observation_process_stacks_recent_frames(self):
        agent = Agent(device=torch.device("cpu"))

        first_obs, _ = agent.observation_process(build_env_obs(hero_x=32, hero_z=64, step_no=10))
        second_obs, _ = agent.observation_process(build_env_obs(hero_x=35, hero_z=67, step_no=11))

        first_seq = np.array(first_obs.feature, dtype=np.float32).reshape(
            Config.TEMPORAL_WINDOW,
            Config.FRAME_FEATURE_LEN,
        )
        second_seq = np.array(second_obs.feature, dtype=np.float32).reshape(
            Config.TEMPORAL_WINDOW,
            Config.FRAME_FEATURE_LEN,
        )

        self.assertEqual(len(first_obs.feature), Config.DIM_OF_OBSERVATION)
        self.assertTrue(np.allclose(first_seq[0], first_seq[-1]))
        self.assertTrue(np.allclose(second_seq[:-1], first_seq[1:]))
        self.assertFalse(np.allclose(second_seq[-1], second_seq[0]))

    def test_reset_clears_temporal_history(self):
        agent = Agent(device=torch.device("cpu"))
        agent.observation_process(build_env_obs(hero_x=32, hero_z=64, step_no=10))
        agent.observation_process(build_env_obs(hero_x=35, hero_z=67, step_no=11))

        agent._business_reset()
        reset_obs, _ = agent.observation_process(build_env_obs(hero_x=40, hero_z=70, step_no=1))
        reset_seq = np.array(reset_obs.feature, dtype=np.float32).reshape(
            Config.TEMPORAL_WINDOW,
            Config.FRAME_FEATURE_LEN,
        )

        for frame_idx in range(1, Config.TEMPORAL_WINDOW):
            self.assertTrue(np.allclose(reset_seq[0], reset_seq[frame_idx]))

    def test_temporal_model_forward_shapes(self):
        model = Model(device=torch.device("cpu"))
        obs = torch.randn(3, Config.DIM_OF_OBSERVATION)
        logits, value = model(obs)

        self.assertEqual(tuple(logits.shape), (3, Config.ACTION_NUM))
        self.assertEqual(tuple(value.shape), (3, Config.VALUE_NUM))

    def test_agent_predict_and_learn_with_temporal_obs(self):
        agent = Agent(device=torch.device("cpu"), logger=DummyLogger())
        obs_data, _ = agent.observation_process(build_env_obs())
        act_data = agent._business_predict([obs_data])[0]

        sample = SampleData(
            obs=torch.tensor(obs_data.feature, dtype=torch.float32),
            legal_action=torch.tensor(obs_data.legal_action, dtype=torch.float32),
            act=torch.tensor([act_data.action[0]], dtype=torch.float32),
            reward=torch.zeros(Config.VALUE_NUM, dtype=torch.float32),
            reward_sum=torch.zeros(Config.VALUE_NUM, dtype=torch.float32),
            done=torch.zeros(1, dtype=torch.float32),
            value=torch.tensor(np.array(act_data.value), dtype=torch.float32).flatten()[:1],
            next_value=torch.zeros(Config.VALUE_NUM, dtype=torch.float32),
            advantage=torch.ones(Config.VALUE_NUM, dtype=torch.float32),
            prob=torch.tensor(act_data.prob, dtype=torch.float32),
        )
        agent._business_learn([sample])

        self.assertEqual(len(obs_data.feature), Config.DIM_OF_OBSERVATION)
        self.assertTrue(0 <= int(act_data.action[0]) < Config.ACTION_NUM)


if __name__ == "__main__":
    unittest.main()
