import numpy as np


MAP_SIZE = 128.0
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
MAX_MONSTER_SPEED = 5.0
MAX_MONSTER_DIST_BUCKET = 5.0
MONSTER_FEATURE_COUNT = 2
MONSTER_DIST_REWARD_AWAY_SCALE = 0.04
MONSTER_DIST_APPROACH_SCALE = 0.14
SQRT_HALF = float(np.sqrt(0.5))

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
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class MonsterProcessor:
    """怪物特征和怪物距离 shaping 处理器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_min_monster_dist_norm = 0.5

    def direction_to_vector(self, direction_idx):
        return DIR_TO_VEC.get(int(direction_idx), (0.0, 0.0))

    def get_feats(self, monsters, hero_pos):
        # Monster features (5D x 2) / 怪物特征
        monster_feats = []
        for i in range(MONSTER_FEATURE_COUNT):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                dir_x, dir_z = self.direction_to_vector(m.get("hero_relative_direction", 0))
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                m_pos = m.get("pos", {})
                if is_in_view and isinstance(m_pos, dict) and "x" in m_pos and "z" in m_pos:
                    # In-view monsters use exact distance, direction still uses relative encoding.
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAX_MAP_DISTANCE)
                else:
                    dist_norm = _norm(m.get("hero_l2_distance", MAX_MONSTER_DIST_BUCKET), MAX_MONSTER_DIST_BUCKET)
                monster_feats.append(
                    np.array([is_in_view, dir_x, dir_z, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return monster_feats

    def get_nearest_monster_vector(self, monster_feats):
        # 用最近怪物的方向构造逃跑参考向量
        if not monster_feats:
            return None

        nearest_feat = min(monster_feats, key=lambda feat: float(feat[4]))
        dir_x = float(nearest_feat[1])
        dir_z = float(nearest_feat[2])
        norm = float(np.sqrt(dir_x * dir_x + dir_z * dir_z))
        if norm <= 1e-6:
            return None
        return (dir_x / norm, dir_z / norm)

    def calc_reward(self, monster_feats) -> float:
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            cur_min_dist_norm = min(cur_min_dist_norm, float(m_feat[4]))

        # 对最近怪物的距离变化做非对称 shaping：
        # 1. 远离时给较小的正奖励，避免挂机
        # 2. 靠近时给更强的负奖励，强调后期贴脸风险
        delta_dist_norm = cur_min_dist_norm - self.last_min_monster_dist_norm
        if delta_dist_norm >= 0.0:
            dist_shaping = MONSTER_DIST_REWARD_AWAY_SCALE * delta_dist_norm
        else:
            dist_shaping = MONSTER_DIST_APPROACH_SCALE * delta_dist_norm

        self.last_min_monster_dist_norm = cur_min_dist_norm
        return dist_shaping
