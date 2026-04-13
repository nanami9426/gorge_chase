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

PHASE_REWARD_WEIGHTS = {
    "phase_0_loot": {
        "progress_reward": 1.00,
        "monster_dist_reward": 0.95,
        "explore_reward": 1.35,
        "treasure_reward": 1.00,
        "buff_reward": 1.25,
        "treasure_stall_penalty": 1.15,
        "terrain_reward": 1.10,
        "flash_reward": 0.90,
        "move_reward": 1.00,
    },
    "phase_1_double_monster": {
        "progress_reward": 1.10,
        "monster_dist_reward": 1.60,
        "explore_reward": 0.85,
        "treasure_reward": 0.70,
        "buff_reward": 1.10,
        "treasure_stall_penalty": 1.15,
        "terrain_reward": 1.40,
        "flash_reward": 1.15,
        "move_reward": 1.00,
    },
    "phase_2_speedup_survival": {
        "progress_reward": 1.20,
        "monster_dist_reward": 1.90,
        "explore_reward": 0.35,
        "treasure_reward": 0.35,
        "buff_reward": 0.90,
        "treasure_stall_penalty": 1.25,
        "terrain_reward": 1.65,
        "flash_reward": 1.30,
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

    def weight_reward_breakdown(self, raw_reward_breakdown, phase_name):
        phase_weights = self.get_reward_weights(phase_name)
        weighted_reward_breakdown = {
            key: float(raw_reward_breakdown[key] * phase_weights[key]) for key in phase_weights
        }
        total_reward = float(sum(weighted_reward_breakdown.values()))
        return phase_weights, weighted_reward_breakdown, total_reward
