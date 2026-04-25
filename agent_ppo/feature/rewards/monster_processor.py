import numpy as np


MAP_SIZE = 128.0
MAX_MAP_DISTANCE = MAP_SIZE * 1.41
MAX_MONSTER_SPEED = 5.0
MAX_MONSTER_DIST_BUCKET = 5.0
MONSTER_FEATURE_COUNT = 2
MONSTER_PREDICTION_HORIZONS = (5, 10, 20)
MONSTER_DIST_REWARD_AWAY_SCALE = 0.075
MONSTER_DIST_APPROACH_SCALE = 0.24
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
        self.last_min_monster_dist_norm = None
        self.last_monster_positions = {}

    def direction_to_vector(self, direction_idx):
        return DIR_TO_VEC.get(int(direction_idx), (0.0, 0.0))

    def get_monster_key(self, monster, fallback_idx):
        return int(monster.get("monster_id", fallback_idx))

    def get_monster_pos(self, monster):
        pos = monster.get("pos", {})
        if not isinstance(pos, dict) or "x" not in pos or "z" not in pos:
            return None
        return {
            "x": float(pos["x"]),
            "z": float(pos["z"]),
        }

    def calc_dist_norm(self, hero_pos, monster_pos):
        raw_dist = np.sqrt((float(hero_pos["x"]) - monster_pos["x"]) ** 2 + (float(hero_pos["z"]) - monster_pos["z"]) ** 2)
        return _norm(raw_dist, MAX_MAP_DISTANCE)

    def get_feats(self, monsters, hero_pos):
        # Monster features (5D x 2) / 怪物特征
        monster_feats = []
        for i in range(MONSTER_FEATURE_COUNT):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                dir_x, dir_z = self.direction_to_vector(m.get("hero_relative_direction", 0))
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                m_pos = self.get_monster_pos(m)
                if is_in_view and m_pos is not None:
                    # In-view monsters use exact distance, direction still uses relative encoding.
                    dist_norm = self.calc_dist_norm(hero_pos, m_pos)
                else:
                    dist_norm = _norm(m.get("hero_l2_distance", MAX_MONSTER_DIST_BUCKET), MAX_MONSTER_DIST_BUCKET)
                monster_feats.append(
                    np.array([is_in_view, dir_x, dir_z, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return monster_feats

    def get_prediction_info(self, monsters, hero_pos):
        """Predict short-horizon monster pressure from recent trajectory."""
        next_positions = {}
        future_positions = {horizon: [] for horizon in MONSTER_PREDICTION_HORIZONS}
        cur_min_dist_norm = 1.0
        pred_min_dist_by_horizon = {horizon: 1.0 for horizon in MONSTER_PREDICTION_HORIZONS}
        has_prediction = 0.0

        for idx, monster in enumerate(monsters):
            monster_pos = self.get_monster_pos(monster)
            if monster_pos is None:
                continue

            key = self.get_monster_key(monster, idx)
            next_positions[key] = monster_pos
            cur_dist_norm = self.calc_dist_norm(hero_pos, monster_pos)
            cur_min_dist_norm = min(cur_min_dist_norm, cur_dist_norm)

            prev_pos = self.last_monster_positions.get(key)
            if prev_pos is not None:
                vel_x = monster_pos["x"] - prev_pos["x"]
                vel_z = monster_pos["z"] - prev_pos["z"]
                has_prediction = 1.0
            else:
                dir_x, dir_z = self.direction_to_vector(monster.get("hero_relative_direction", 0))
                speed = float(np.clip(monster.get("speed", 1.0), 0.0, MAX_MONSTER_SPEED))
                # hero_relative_direction points from hero to monster; chase motion roughly points back to hero.
                vel_x = -dir_x * speed
                vel_z = -dir_z * speed

            vel_x = float(np.clip(vel_x, -MAX_MONSTER_SPEED, MAX_MONSTER_SPEED))
            vel_z = float(np.clip(vel_z, -MAX_MONSTER_SPEED, MAX_MONSTER_SPEED))
            for horizon in MONSTER_PREDICTION_HORIZONS:
                future_pos = {
                    "x": monster_pos["x"] + vel_x * float(horizon),
                    "z": monster_pos["z"] + vel_z * float(horizon),
                }
                future_positions[horizon].append(future_pos)
                pred_dist_norm = self.calc_dist_norm(hero_pos, future_pos)
                pred_min_dist_by_horizon[horizon] = min(pred_min_dist_by_horizon[horizon], pred_dist_norm)

        if next_positions:
            self.last_monster_positions = next_positions
        else:
            self.last_monster_positions = {}

        pred_dist_5 = pred_min_dist_by_horizon[MONSTER_PREDICTION_HORIZONS[0]]
        max_closing_delta = max(
            MONSTER_PREDICTION_HORIZONS[0] * MAX_MONSTER_SPEED / MAX_MAP_DISTANCE,
            1e-6,
        )
        approach_pressure = float(np.clip((cur_min_dist_norm - pred_dist_5) / max_closing_delta, 0.0, 1.0))
        pred_dangers = [1.0 - pred_min_dist_by_horizon[horizon] for horizon in MONSTER_PREDICTION_HORIZONS]
        worst_pred_danger = float(max(pred_dangers, default=0.0))
        prediction_feat = np.array(
            [
                has_prediction,
                approach_pressure,
                *pred_dangers,
                worst_pred_danger,
            ],
            dtype=np.float32,
        )
        return prediction_feat, future_positions

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

        if self.last_min_monster_dist_norm is None:
            self.last_min_monster_dist_norm = cur_min_dist_norm
            return 0.0

        # 对最近怪物的距离变化做非对称 shaping：
        # 1. 远离时给较小的正奖励，避免挂机
        # 2. 靠近时给更强的负奖励，强调后期贴脸风险
        delta_dist_norm = cur_min_dist_norm - self.last_min_monster_dist_norm
        close_pressure = 1.0 - cur_min_dist_norm
        urgency_scale = 1.0 + 0.75 * close_pressure
        if delta_dist_norm >= 0.0:
            dist_shaping = urgency_scale * MONSTER_DIST_REWARD_AWAY_SCALE * delta_dist_norm
        else:
            dist_shaping = urgency_scale * MONSTER_DIST_APPROACH_SCALE * delta_dist_norm

        self.last_min_monster_dist_norm = cur_min_dist_norm
        return dist_shaping
