import numpy as np


MAX_MONSTER_SPEED = 5.0
MAX_MONSTER_COUNT = 2.0

PHASE_LOOT = 0
PHASE_DOUBLE_MONSTER = 1
PHASE_SPEEDUP_SURVIVAL = 2
PHASE_NAME_BY_ID = {
    PHASE_LOOT: "phase_0_loot",
    PHASE_DOUBLE_MONSTER: "phase_1_double_monster",
    PHASE_SPEEDUP_SURVIVAL: "phase_2_speedup_survival",
}

CURRICULUM_SURVIVAL_BOOTSTRAP = "curriculum_survival_bootstrap"
CURRICULUM_LOOT_UNLOCK = "curriculum_loot_unlock"
CURRICULUM_FULL = "curriculum_full"

PHASE_REWARD_WEIGHTS = {
    "phase_0_loot": {
        "progress_reward": 1.05,
        "monster_dist_reward": 1.05,
        "explore_reward": 1.15,
        "treasure_reward": 1.35,
        "buff_reward": 1.05,
        "treasure_stall_penalty": 1.35,
        "terrain_reward": 1.15,
        "flash_reward": 1.00,
        "move_reward": 1.00,
    },
    "phase_1_double_monster": {
        "progress_reward": 1.25,
        "monster_dist_reward": 2.10,
        "explore_reward": 0.85,
        "treasure_reward": 1.15,
        "buff_reward": 1.25,
        "treasure_stall_penalty": 1.45,
        "terrain_reward": 1.75,
        "flash_reward": 1.35,
        "move_reward": 1.00,
    },
    "phase_2_speedup_survival": {
        "progress_reward": 1.40,
        "monster_dist_reward": 2.50,
        "explore_reward": 0.55,
        "treasure_reward": 0.85,
        "buff_reward": 1.35,
        "treasure_stall_penalty": 1.60,
        "terrain_reward": 2.10,
        "flash_reward": 1.60,
        "move_reward": 1.00,
    },
}

CURRICULUM_REWARD_WEIGHTS = {
    CURRICULUM_SURVIVAL_BOOTSTRAP: {
        "progress_reward": 1.35,
        "monster_dist_reward": 1.45,
        "explore_reward": 0.85,
        "treasure_reward": 0.65,
        "buff_reward": 1.15,
        "treasure_stall_penalty": 0.90,
        "terrain_reward": 1.35,
        "flash_reward": 1.25,
        "move_reward": 1.00,
    },
    CURRICULUM_LOOT_UNLOCK: {
        "progress_reward": 1.10,
        "monster_dist_reward": 1.15,
        "explore_reward": 1.00,
        "treasure_reward": 1.25,
        "buff_reward": 1.10,
        "treasure_stall_penalty": 1.25,
        "terrain_reward": 1.10,
        "flash_reward": 1.05,
        "move_reward": 1.00,
    },
    CURRICULUM_FULL: {
        "progress_reward": 1.00,
        "monster_dist_reward": 1.00,
        "explore_reward": 1.00,
        "treasure_reward": 1.15,
        "buff_reward": 1.00,
        "treasure_stall_penalty": 1.20,
        "terrain_reward": 1.00,
        "flash_reward": 1.00,
        "move_reward": 1.00,
    },
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class PhaseProcessor:
    """阶段识别、阶段特征和阶段奖励重加权处理器。"""

    def reset(self):
        # 当前阶段处理器无跨帧状态，保留 reset 接口与其他 processor 对齐
        return None

    def get_phase_info(self, monsters):
        # 按真实事件切阶段：先看怪物是否加速，再看第二只怪是否出现
        monster_count = min(len(monsters), int(MAX_MONSTER_COUNT))
        max_monster_speed = max((int(m.get("speed", 1)) for m in monsters), default=1)

        if max_monster_speed > 1:
            phase_id = PHASE_SPEEDUP_SURVIVAL
        elif monster_count >= 2:
            phase_id = PHASE_DOUBLE_MONSTER
        else:
            phase_id = PHASE_LOOT

        return phase_id, PHASE_NAME_BY_ID[phase_id], monster_count, max_monster_speed

    def get_feats(self, phase_id, monster_count, max_monster_speed):
        # 3维阶段 one-hot + 怪物数量/速度归一化
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

    def get_reward_weights(self, phase_name):
        return dict(PHASE_REWARD_WEIGHTS[str(phase_name)])

    def get_curriculum_weights(self, curriculum_stage=None, loot_reward_scale=None):
        curriculum_stage = curriculum_stage or CURRICULUM_LOOT_UNLOCK
        weights = dict(
            CURRICULUM_REWARD_WEIGHTS.get(
                str(curriculum_stage),
                CURRICULUM_REWARD_WEIGHTS[CURRICULUM_LOOT_UNLOCK],
            )
        )
        if loot_reward_scale is not None:
            loot_reward_scale = float(np.clip(loot_reward_scale, 0.05, 1.50))
            weights["treasure_reward"] = loot_reward_scale
        return weights

    def weight_reward_breakdown(
        self,
        raw_reward_breakdown,
        phase_name,
        curriculum_stage=None,
        loot_reward_scale=None,
    ):
        phase_weights = self.get_reward_weights(phase_name)
        curriculum_weights = self.get_curriculum_weights(
            curriculum_stage=curriculum_stage,
            loot_reward_scale=loot_reward_scale,
        )
        combined_weights = {
            key: float(phase_weights[key] * curriculum_weights.get(key, 1.0))
            for key in phase_weights
        }
        weighted_reward_breakdown = {
            key: float(raw_reward_breakdown[key] * combined_weights[key])
            for key in phase_weights
        }
        total_reward = float(sum(weighted_reward_breakdown.values()))
        return (
            phase_weights,
            curriculum_weights,
            combined_weights,
            weighted_reward_breakdown,
            total_reward,
        )
