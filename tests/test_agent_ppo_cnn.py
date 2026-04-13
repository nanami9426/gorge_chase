import os
import tempfile
import unittest

import numpy as np
import torch

from agent_ppo.agent import Agent
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.feature.spatial_encoder import SpatialFeatureEncoder
from agent_ppo.model.model import Model


class DummyLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, message):
        self.infos.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def error(self, message):
        self.errors.append(str(message))


def build_env_obs(map_size=13):
    map_info = [[1 for _ in range(map_size)] for _ in range(map_size)]
    center = map_size // 2
    map_info[center - 2][center] = 0
    map_info[center + 2][center - 3] = 0

    hero_pos = {"x": 32, "z": 64}
    return {
        "observation": {
            "frame_state": {
                "heroes": {
                    "pos": hero_pos,
                    "flash_cooldown": 12,
                    "buff_remaining_time": 5,
                },
                "monsters": [
                    {
                        "is_in_view": 1,
                        "hero_relative_direction": 2,
                        "speed": 2,
                        "hero_l2_distance": 2,
                        "pos": {"x": 34, "z": 65},
                    },
                    {
                        "is_in_view": 0,
                        "hero_relative_direction": 7,
                        "speed": 1,
                        "hero_l2_distance": 4,
                    },
                ],
                "organs": [
                    {
                        "status": 1,
                        "sub_type": 1,
                        "config_id": 101,
                        "hero_relative_direction": 4,
                        "hero_l2_distance": 2,
                        "pos": {"x": 31, "z": 66},
                    },
                    {
                        "status": 1,
                        "sub_type": 2,
                        "config_id": 201,
                        "hero_relative_direction": 7,
                        "hero_l2_distance": 3,
                        "pos": {"x": 32, "z": 61},
                    },
                    {
                        "status": 1,
                        "sub_type": 2,
                        "config_id": 202,
                        "hero_relative_direction": 1,
                        "hero_l2_distance": 2,
                    },
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
            "step_no": 10,
        },
        "terminated": False,
        "truncated": False,
    }


class TestAgentPpoCnn(unittest.TestCase):
    def test_spatial_encoder_projection(self):
        encoder = SpatialFeatureEncoder()
        env_obs = build_env_obs()
        hero_pos = env_obs["observation"]["frame_state"]["heroes"]["pos"]
        monsters = env_obs["observation"]["frame_state"]["monsters"]
        organs = env_obs["observation"]["frame_state"]["organs"]

        normalized_map_info, spatial = encoder.encode(
            map_info=env_obs["observation"]["map_info"],
            monsters=monsters,
            organs=organs,
            hero_pos=hero_pos,
        )

        self.assertEqual(len(normalized_map_info), Config.SPATIAL_MAP_SIZE)
        self.assertEqual(
            tuple(spatial.shape),
            (Config.SPATIAL_CHANNELS, Config.SPATIAL_MAP_SIZE, Config.SPATIAL_MAP_SIZE),
        )
        self.assertEqual(float(spatial[0, 4, 6]), 1.0)
        self.assertEqual(float(spatial[1, 5, 8]), 1.0)
        self.assertEqual(float(spatial[2, 4, 5]), 1.0)
        self.assertEqual(float(spatial[3, 9, 6]), 1.0)

    def test_preprocessor_feature_layout_and_projection(self):
        preprocessor = Preprocessor()
        feature, legal_action, remain_info = preprocessor.feature_process(build_env_obs(), last_action=-1)

        self.assertEqual(len(feature), Config.DIM_OF_OBSERVATION)
        self.assertEqual(len(legal_action), Config.ACTION_NUM)
        self.assertIn("reward", remain_info)

        dense = feature[: Config.DENSE_FEATURE_LEN]
        spatial = feature[Config.DENSE_FEATURE_LEN :].reshape(
            Config.SPATIAL_CHANNELS,
            Config.SPATIAL_MAP_SIZE,
            Config.SPATIAL_MAP_SIZE,
        )
        self.assertEqual(len(dense), Config.DENSE_FEATURE_LEN)

        self.assertEqual(float(spatial[0, 4, 6]), 1.0)
        self.assertEqual(float(spatial[0, 8, 3]), 1.0)
        self.assertEqual(float(spatial[1, 5, 8]), 1.0)
        self.assertEqual(float(spatial[2, 4, 5]), 1.0)
        self.assertEqual(float(spatial[3, 9, 6]), 1.0)
        self.assertEqual(float(spatial[1].sum()), 1.0)
        self.assertEqual(float(spatial[2].sum()), 1.0)
        self.assertEqual(float(spatial[3].sum()), 1.0)

    def test_preprocessor_pads_non_standard_map_with_walls(self):
        preprocessor = Preprocessor()
        feature, _, _ = preprocessor.feature_process(build_env_obs(map_size=11), last_action=-1)
        spatial = feature[Config.DENSE_FEATURE_LEN :].reshape(
            Config.SPATIAL_CHANNELS,
            Config.SPATIAL_MAP_SIZE,
            Config.SPATIAL_MAP_SIZE,
        )

        self.assertEqual(float(spatial[0, 0, 0]), 1.0)
        self.assertEqual(float(spatial[0, -1, -1]), 1.0)

    def test_model_forward_shapes(self):
        model = Model(device=torch.device("cpu"))
        obs = torch.randn(3, Config.DIM_OF_OBSERVATION)
        logits, value = model(obs)

        self.assertEqual(tuple(logits.shape), (3, Config.ACTION_NUM))
        self.assertEqual(tuple(value.shape), (3, Config.VALUE_NUM))

    def test_agent_predict_learn_and_incompatible_checkpoint(self):
        logger = DummyLogger()
        agent = Agent(device=torch.device("cpu"), logger=logger)
        obs_data, _ = agent.observation_process(build_env_obs())
        act_data = agent._business_predict([obs_data])[0]

        self.assertEqual(len(act_data.prob), Config.ACTION_NUM)
        self.assertTrue(0 <= int(act_data.action[0]) < Config.ACTION_NUM)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.ckpt-1.pkl")
            torch.save({"unexpected_key": torch.zeros(1)}, ckpt_path)
            agent.load_model(path=tmpdir, id="1", framework=True)

        self.assertTrue(any("skip incompatible model" in msg for msg in logger.warnings))


if __name__ == "__main__":
    unittest.main()
