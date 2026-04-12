import numpy as np


MOVE_ACTION_NUM = 8
MAX_CLEARANCE_STEPS = 4.0
SQRT_HALF = float(np.sqrt(0.5))
OPEN_FIELD_TARGET = 0.65
# 开阔度相较上一帧提升时的奖励系数
OPENNESS_DELTA_REWARD_SCALE = 0.06
CLEARANCE_DELTA_REWARD_SCALE = 0.04

# 已经处在较开阔区域时的额外驻留奖励
OPEN_FIELD_BONUS_SCALE = 0.012

# 贴墙压力超过该阈值后开始处罚
WALL_PRESSURE_THRESHOLD = 0.42

# 贴墙压力处罚强度
WALL_PRESSURE_PENALTY_SCALE = 0.024

# 可逃方向比例过低时的阈值
LOW_ESCAPE_THRESHOLD = 0.375

# 可逃方向太少时的处罚强度
LOW_ESCAPE_PENALTY_SCALE = 0.018

# 中等危险开始逐步偏向显式逃跑
ESCAPE_WEIGHT_START = 0.35
ESCAPE_WEIGHT_RANGE = 0.35

# 逃跑方向和动作对齐奖励强度
ESCAPE_ACTION_REWARD_SCALE = 0.08

# 贴墙停滞处罚参数
WALL_STALL_PENALTY_SCALE = 0.012
WALL_STALL_PENALTY_CAP = 0.20

MOVE_DIRS = [
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MOVE_VECS = [
    (1.0, 0.0),
    (SQRT_HALF, SQRT_HALF),
    (0.0, 1.0),
    (-SQRT_HALF, SQRT_HALF),
    (-1.0, 0.0),
    (-SQRT_HALF, -SQRT_HALF),
    (0.0, -1.0),
    (SQRT_HALF, -SQRT_HALF),
]


class TerrainProcessor:
    """基于局部地图计算开阔度相关特征与奖励。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_hero_pos = None
        self.last_openness = None
        self.last_clearance = None
        self.last_escape_dir_scores = None
        self.last_escape_weight = 0.0

        # 连续贴墙停住的步数
        self.wall_stall_steps = 0

    def is_cell_passable(self, map_info, row, col) -> bool:
        if map_info is None or not map_info or not map_info[0]:
            return True
        if row < 0 or row >= len(map_info):
            return False
        if col < 0 or col >= len(map_info[0]):
            return False
        return bool(map_info[row][col] != 0)

    def calc_ring_ratio(self, map_info, center_row, center_col, radius) -> float:
        # 统计以英雄为中心、指定半径那一圈格子的可通行比例。
        if map_info is None or not map_info or not map_info[0]:
            return 1.0

        passable = 0
        total = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                if max(abs(dr), abs(dc)) != radius:
                    # 只算边框的
                    continue
                total += 1
                if self.is_cell_passable(map_info, center_row + dr, center_col + dc):
                    passable += 1
        return float(passable) / float(total) if total > 0 else 1.0

    def calc_openness_ratio(self, map_info) -> float:
        # 计算当前位置的综合开阔度，把 1/2/3 圈的可通行比例做加权平均
        if map_info is None or not map_info or not map_info[0]:
            return 1.0

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2
        ring1 = self.calc_ring_ratio(map_info, center_row, center_col, radius=1)
        ring2 = self.calc_ring_ratio(map_info, center_row, center_col, radius=2)
        ring3 = self.calc_ring_ratio(map_info, center_row, center_col, radius=3)
        openness = 0.50 * ring1 + 0.30 * ring2 + 0.20 * ring3
        return float(np.clip(openness, 0.0, 1.0))

    def calc_directional_clearances(self, map_info):
        # 计算 8 个方向上的净空情况，能连续走多远不撞墙
        if map_info is None or not map_info or not map_info[0]:
            return [1.0] * MOVE_ACTION_NUM

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2
        clearances = []
        for dr, dc in MOVE_DIRS:
            steps = 0
            for dist in range(1, int(MAX_CLEARANCE_STEPS) + 1):
                row = center_row + dr * dist
                col = center_col + dc * dist
                if not self.is_cell_passable(map_info, row, col):
                    break
                steps += 1
            clearances.append(float(steps) / MAX_CLEARANCE_STEPS)
        return clearances

    def calc_clearance_stats(self, map_info):
        clearances = self.calc_directional_clearances(map_info)
        avg_clearance = float(sum(clearances) / len(clearances)) if clearances else 1.0
        min_clearance = float(min(clearances)) if clearances else 1.0
        return clearances, avg_clearance, min_clearance

    def normalize_monster_vec(self, monster_vec):
        if monster_vec is None:
            return None

        vec_x = float(monster_vec[0])
        vec_z = float(monster_vec[1])
        norm = float(np.sqrt(vec_x * vec_x + vec_z * vec_z))
        if norm <= 1e-6:
            return None
        return (vec_x / norm, vec_z / norm)

    def calc_escape_dir_scores(self, clearances, move_mask, monster_vec):
        monster_vec = self.normalize_monster_vec(monster_vec)
        move_mask = list(move_mask[:MOVE_ACTION_NUM]) if move_mask is not None else [1] * MOVE_ACTION_NUM

        escape_dir_scores = []
        for idx, clearance in enumerate(clearances):
            left_adj = clearances[(idx - 1) % MOVE_ACTION_NUM]
            right_adj = clearances[(idx + 1) % MOVE_ACTION_NUM]
            sector_open = 0.50 * clearance + 0.25 * left_adj + 0.25 * right_adj

            if monster_vec is None:
                away = 0.5
            else:
                move_vec = MOVE_VECS[idx]
                away = (1.0 - (move_vec[0] * monster_vec[0] + move_vec[1] * monster_vec[1])) / 2.0
                away = float(np.clip(away, 0.0, 1.0))

            score = float(move_mask[idx]) * (0.45 * sector_open + 0.35 * clearance + 0.20 * away)
            escape_dir_scores.append(float(np.clip(score, 0.0, 1.0)))
        return escape_dir_scores

    def calc_positioning_weight(self, danger_score) -> float:
        return float(max(0.25, 1.0 - float(np.clip(danger_score, 0.0, 1.0))))

    def calc_escape_weight(self, danger_score) -> float:
        return float(
            np.clip(
                (float(danger_score) - ESCAPE_WEIGHT_START) / ESCAPE_WEIGHT_RANGE,
                0.0,
                1.0,
            )
        )

    def extract_stats(self, map_info, move_mask, monster_vec=None):
        openness = self.calc_openness_ratio(map_info)
        escape_ratio = (
            float(sum(move_mask[:MOVE_ACTION_NUM])) / float(MOVE_ACTION_NUM)
            if move_mask is not None
            else 1.0
        )
        clearances, avg_clearance, min_clearance = self.calc_clearance_stats(map_info)
        wall_pressure = 1.0 - (0.45 * openness + 0.35 * escape_ratio + 0.20 * avg_clearance)
        wall_pressure = float(np.clip(wall_pressure, 0.0, 1.0))
        corner_pressure = 1.0 - 0.5 * (escape_ratio + min_clearance)
        corner_pressure = float(np.clip(corner_pressure, 0.0, 1.0))
        escape_dir_scores = self.calc_escape_dir_scores(clearances, move_mask, monster_vec)
        return {
            "openness": openness,
            "escape_ratio": escape_ratio,
            "avg_clearance": avg_clearance,
            "min_clearance": min_clearance,
            "wall_pressure": wall_pressure,
            "corner_pressure": corner_pressure,
            "clearances": clearances,
            "escape_dir_scores": escape_dir_scores,
        }

    def get_feats(self, terrain_stats):
        return np.array(
            [
                terrain_stats["openness"],
                terrain_stats["escape_ratio"],
                terrain_stats["avg_clearance"],
                *terrain_stats["escape_dir_scores"],
            ],
            dtype=np.float32,
        )

    def calc_reward(self, hero_pos, terrain_stats, last_action, danger_score) -> float:
        openness = terrain_stats["openness"]
        escape_ratio = terrain_stats["escape_ratio"]
        avg_clearance = terrain_stats["avg_clearance"]
        wall_pressure = terrain_stats["wall_pressure"]
        corner_pressure = terrain_stats["corner_pressure"]
        escape_dir_scores = terrain_stats["escape_dir_scores"]

        positioning_weight = self.calc_positioning_weight(danger_score)
        escape_weight = self.calc_escape_weight(danger_score)
        reward = 0.0

        if self.last_openness is not None:
            # 当前比上一帧更开阔
            reward += positioning_weight * OPENNESS_DELTA_REWARD_SCALE * (openness - self.last_openness)
        if self.last_clearance is not None:
            # 当前平均净空更大
            reward += positioning_weight * CLEARANCE_DELTA_REWARD_SCALE * (avg_clearance - self.last_clearance)

        # 如果已经身处较开阔区域，给少量正反馈
        reward += positioning_weight * OPEN_FIELD_BONUS_SCALE * max(0.0, openness - OPEN_FIELD_TARGET)

        if wall_pressure > WALL_PRESSURE_THRESHOLD:
            # 贴墙压力过大时，按超过阈值的程度处罚
            pressure_ratio = (wall_pressure - WALL_PRESSURE_THRESHOLD) / max(1.0 - WALL_PRESSURE_THRESHOLD, 1e-6)
            reward -= positioning_weight * WALL_PRESSURE_PENALTY_SCALE * pressure_ratio

        if escape_ratio < LOW_ESCAPE_THRESHOLD:
            # 能走的方向太少，说明位置偏狭窄，不利于后续拉扯
            escape_gap = (LOW_ESCAPE_THRESHOLD - escape_ratio) / max(LOW_ESCAPE_THRESHOLD, 1e-6)
            reward -= positioning_weight * LOW_ESCAPE_PENALTY_SCALE * escape_gap

        # 避免模型偏好死角
        reward -= positioning_weight * 0.012 * corner_pressure

        if self.last_escape_dir_scores is not None and last_action is not None:
            action_idx = int(last_action)
            if 0 <= action_idx < 16:
                dir_idx = action_idx % MOVE_ACTION_NUM
                mean_escape_score = float(np.mean(self.last_escape_dir_scores))
                reward += self.last_escape_weight * ESCAPE_ACTION_REWARD_SCALE * (
                    float(self.last_escape_dir_scores[dir_idx]) - mean_escape_score
                )

        cur_pos = (hero_pos["x"], hero_pos["z"])
        moved = True
        if self.last_hero_pos is not None:
            moved = cur_pos != self.last_hero_pos

        if moved or wall_pressure <= WALL_PRESSURE_THRESHOLD:
            # 一旦移动了，或者已经不算贴墙，就清空停滞计数
            self.wall_stall_steps = 0
        else:
            # 贴墙停滞会在危险时继续加重，避免沿墙发呆直到怪物贴脸
            self.wall_stall_steps += 1
            stall_penalty = min(
                WALL_STALL_PENALTY_SCALE
                * self.wall_stall_steps
                * wall_pressure
                * (0.4 + 0.6 * float(np.clip(danger_score, 0.0, 1.0))),
                WALL_STALL_PENALTY_CAP,
            )
            reward -= stall_penalty

        self.last_hero_pos = cur_pos
        self.last_openness = openness
        self.last_clearance = avg_clearance
        self.last_escape_dir_scores = np.array(escape_dir_scores, dtype=np.float32)
        self.last_escape_weight = escape_weight
        return reward
